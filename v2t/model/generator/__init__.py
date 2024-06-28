import torch
import os
from transformers import AutoTokenizer,AutoConfig
from ..generator.modeling_t5generator import T5Generator,T5GeneratorConfig

from v2t.overwatch import initialize_overwatch
overwatch = initialize_overwatch(__name__)

hf_id2class = {
    "google-t5/t5-base":  [T5GeneratorConfig,T5Generator],
}

v2t_id2class = {
    "T5Generator":[T5GeneratorConfig,T5Generator],
}

retriever2dim = {
    "sentence-transformers/gtr-t5-base":768,
}

def load_generator(
    model_name_or_path: str,
    embedder_name_or_path: float = None,
    expanding_factor=1,
    torch_compile=True,
    project_draft_embedding=None,
    layer_norm_after_projection=False,
):
    overwatch.info(f"Loading Generator: {model_name_or_path}")
    ## == Define Model and Config Class == ##
    if os.path.exists(model_name_or_path):
        config = AutoConfig.from_pretrained(model_name_or_path)
        model_family = config.architectures[0]
        CONFIG_CLASS,MODEL_CLASS = v2t_id2class[model_family]
        config = CONFIG_CLASS.from_pretrained(model_name_or_path)
    else:
        CONFIG_CLASS,MODEL_CLASS = hf_id2class[model_name_or_path]
        config = CONFIG_CLASS.from_pretrained(
            model_name_or_path,
            embedder_name_or_path=embedder_name_or_path,
            embedder_hidden_size=retriever2dim[embedder_name_or_path],
            expanding_factor=expanding_factor,
            project_draft_embedding=project_draft_embedding,
            layer_norm_after_projection=layer_norm_after_projection,
        )
    ## == Load == ##
    model = MODEL_CLASS.from_pretrained(
        model_name_or_path,
        config=config,
        # attn_implementation='flash_attention_2',
        # torch_dtype = torch_precisions[load_precision],
    )
    if torch_compile: model = torch.compile(model)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,use_fast=True)
    return model,tokenizer
