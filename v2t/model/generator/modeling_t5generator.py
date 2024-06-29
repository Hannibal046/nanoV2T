import torch
import torch.nn as nn
import re
from transformers import T5ForConditionalGeneration,T5Config 
from typing import Optional,Union


class T5GeneratorConfig(T5Config):
    def __init__(
        self,
        projector_type = 'mlp2x_gelu',
        embedder_hidden_size = 128,
        embedder_name_or_path = None,
        expanding_factor = 16,
        project_draft_embedding = False,
        layer_norm_after_projection = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.projector_type = projector_type
        self.embedder_hidden_size = embedder_hidden_size
        self.embedder_name_or_path = embedder_name_or_path
        self.expanding_factor = expanding_factor
        self.project_draft_embedding = project_draft_embedding
        self.layer_norm_after_projection = layer_norm_after_projection


class Projector(nn.Module):
    def __init__(self,config):
        super().__init__()
        projector_type = config.projector_type
        self.expanding_factor = config.expanding_factor
        mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
        if mlp_gelu_match:
            mlp_depth = int(mlp_gelu_match.group(1))
            modules = [nn.Linear(config.embedder_hidden_size, config.hidden_size),nn.Dropout(config.dropout_rate)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(config.hidden_size, config.hidden_size * self.expanding_factor))
            self.projector = nn.Sequential(*modules)
        
        self.init_weight()
    
    def init_weight(self):
        for module in self.projector:
            if isinstance(module,nn.Linear):
                in_feature = module.weight.shape[1] 
                k = 1 / (in_feature ** 0.5)
                module.weight.data.uniform_(-k,k)
                if hasattr(module, "bias") and module.bias is not None:
                    module.bias.data.uniform_(-k,k)

    def forward(self,embeddings):
        ## [batch_size,d_model]
        batch_size,_ = embeddings.shape
        return self.projector(embeddings).view(batch_size,self.expanding_factor,-1)

class T5Generator(T5ForConditionalGeneration):
    def __init__(self,config):
        super().__init__(config)
        self.expanding_factor = config.expanding_factor
        self.target_projector = Projector(config)
        self.embedder_hidden_size = config.embedder_hidden_size
            
        if config.project_draft_embedding:
            self.draft_projector = Projector(config)
            self.diff_projector = Projector(config)
        
        if config.layer_norm_after_projection:
            self.proj_ln = nn.LayerNorm(self.config.hidden_size)
        
        self.post_init()

    def prepare_inputs_embeds(self,input_ids,target_embeddings,draft_embeddings=None,attention_mask=None):
        """
        input_ids + input_embeds == > model(input_embeds)
        input_ids:    [batch_size,seq_length]
        embeddings:   [batch_size,num_embeds,embedder_hidden_size]
        """        
        ## project target_embeddings
        num_sep_tokens = 0
        batch_size,_ = target_embeddings.shape
        device = target_embeddings.device
        input_embeds = self.target_projector(target_embeddings)
        sep_token = torch.ones((batch_size, 1), dtype=torch.long, device=target_embeddings.device) * self.config.eos_token_id
        sep_token = self.shared(sep_token)
        ## project draft_embeddings
        if draft_embeddings is not None:
            input_embeds = torch.concat(
                (
                    sep_token,
                    input_embeds,
                    sep_token,
                    self.draft_projector(draft_embeddings),
                ),dim=1
            )
            num_sep_tokens += 2

            input_embeds = torch.concat(
                (
                    input_embeds,
                    sep_token,
                    self.diff_projector(target_embeddings-draft_embeddings),
                ),dim=1
            )
            num_sep_tokens += 1

        ## input_ids --> embeddings
        if input_ids is not None:
            token_embeds = self.shared(input_ids) # [batch_size,seq_length, d_model]
            input_embeds = torch.concat((input_embeds,sep_token,token_embeds),dim=1)
            num_sep_tokens += 1

        ## prepare attention_mask
        multiplier = 3 if draft_embeddings is not None else 1
        extended_attention_mask = torch.ones((batch_size,self.expanding_factor * multiplier + num_sep_tokens),device=device)

        if attention_mask is not None:
            extended_attention_mask = torch.concat((extended_attention_mask,attention_mask),dim=1)
            assert extended_attention_mask.shape[:2] == input_embeds.shape[:2], (extended_attention_mask.shape,input_embeds.shape)

        if hasattr(self,"proj_ln"):
            input_embeds = self.proj_ln(input_embeds)

        return input_embeds,extended_attention_mask

    def forward(
        self,
        input_ids = None,
        target_embeddings = None,
        draft_embeddings = None,
        attention_mask = None,
        encoder_outputs = None,
        **kwargs,
    ):
        ## == Training Mode == ##
        if encoder_outputs is None:
            inputs_embeds,attention_mask = self.prepare_inputs_embeds(input_ids,target_embeddings,draft_embeddings,attention_mask)
            input_ids = None
            return super().forward(
                input_ids = input_ids,
                inputs_embeds = inputs_embeds,
                attention_mask = attention_mask,
                **kwargs,
            )
        
        ## == Generation Mode == ##
        else:
            assert target_embeddings is None
            assert not self.training
            return super().forward(
                input_ids = input_ids,
                attention_mask=attention_mask,
                encoder_outputs=encoder_outputs,
                **kwargs,
            )

    @torch.no_grad()
    def generate(
        self,
        input_ids = None,
        target_embeddings = None,
        draft_embeddings = None,
        **kwargs,
    ):
        attention_mask = kwargs.pop("attention_mask",None)
        # assert attention_mask is not None
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported for generate")
        
        inputs_embeds=None
        if target_embeddings is not None:
            inputs_embeds, attention_mask = self.prepare_inputs_embeds(input_ids,target_embeddings,draft_embeddings,attention_mask)
            input_ids = None
            if attention_mask is not None:
                assert inputs_embeds.shape[1] == attention_mask.shape[1],(inputs_embeds.shape,attention_mask.shape)
            return super().generate(
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                **kwargs
            )
        
        else:
            return super().generate(
                attention_mask=attention_mask,
                input_ids=input_ids,
                **kwargs
            )
    