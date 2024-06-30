import json
import copy
import os
from functools import partial
from collections import defaultdict
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm
from accelerate import PartialState
import torch.distributed as dist
import transformers

from v2t.utils import eval_v2t,write_jsonl,is_equal
from v2t.data import collator,load_data
from v2t.model import load_embedder,load_generator
from v2t.overwatch import initialize_overwatch
overwatch = initialize_overwatch(__name__)

def move_to_device(batch,device):
    for k in batch:
        if isinstance(batch[k],torch.Tensor):
            batch[k] = batch[k].to(device)
    return batch

def single_step_generation(dataset,generator,tokenizer,generation_kwargs,embedder,batch_size,enable_progress_bar=False):

    device = generator.device
    collate_fn = partial(collator,tokenizer=tokenizer)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size = batch_size,
        collate_fn = collate_fn,
    ) 
    progress_bar = tqdm(range(len(dataloader)), disable=not enable_progress_bar, ncols=100)

    id2hyps = defaultdict(list)
    for batch in dataloader:
        progress_bar.update(1)
        batch = move_to_device(batch,device)

        ## == generate with transformers generation method == ##
        generated_ids = generator.generate(
            input_ids = batch['draft_input_ids'],
            attention_mask = batch['draft_attention_mask'],
            draft_embeddings = batch['draft_embeddings'],
            target_embeddings = batch['target_embeddings'], 
            **generation_kwargs,
        )
        generated_hyps = tokenizer.batch_decode(generated_ids,skip_special_tokens=True, clean_up_tokenization_spaces=True)

        ## embed generated hyps and compute cos similarity 
        hyps_embeddings = embedder.encode(generated_hyps,show_progress_bar=False,convert_to_tensor=True).to(device)
        target_embedding = batch['target_embeddings'].repeat_interleave(generation_kwargs['num_return_sequences'],dim=0)
        cos_sim_scores = F.cosine_similarity(target_embedding,hyps_embeddings,dim=1).tolist()


        ids = [x for x in batch['ids'] for _ in range(generation_kwargs["num_return_sequences"])]
        for id,cos_sim_score,generated_hyp,hyp_embedding in zip(ids,cos_sim_scores,generated_hyps,hyps_embeddings,strict=True):
            id2hyps[int(id)].append({
                "cos_sim":cos_sim_score,
                "hyp":generated_hyp,
                "hyp_embedding":hyp_embedding,
            })
    
    return id2hyps

def maybe_gather_from_other_gpu(list_of_text,distributed_state):
    if distributed_state.num_processes > 1:
        all_ranks_objects = [None for _ in range(distributed_state.num_processes)]
        dist.gather_object(
            list_of_text,
            all_ranks_objects if distributed_state.is_local_main_process else None,
            dst=0
        )
        if distributed_state.is_local_main_process:
            list_of_text = [x for y in all_ranks_objects for x in y]
    return list_of_text

class GTREmbedder:
    def __init__(self,device):
        model_kwargs = {
                "low_cpu_mem_usage": True,  # Not compatible with DeepSpeed
                "output_hidden_states": False,
            }
        self.embedder = transformers.AutoModel.from_pretrained(
            "sentence-transformers/gtr-t5-base", **model_kwargs,
        ).encoder.to(device)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            "sentence-transformers/gtr-t5-base"
        )
        self.embedder.eval()
    @torch.inference_mode()
    def encode(self,list_of_text,**kwargs):
        device = self.embedder.device
        inputs = self.tokenizer(list_of_text,max_length=32,truncation=True,padding="max_length",return_tensors='pt').to(device)
        model_output = self.embedder(input_ids = inputs.input_ids,attention_mask = inputs.attention_mask)
        hidden_state = model_output.last_hidden_state
        embeddings = self.mean_pool(hidden_state, inputs.attention_mask)
        return embeddings
    
    def mean_pool(self,hidden_states, attention_mask):
        B, S, D = hidden_states.shape
        unmasked_outputs = hidden_states * attention_mask[..., None]
        pooled_outputs = unmasked_outputs.sum(dim=1) / attention_mask.sum(dim=1)[:, None]
        assert pooled_outputs.shape == (B, D)
        return pooled_outputs
    
def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft_dir")
    parser.add_argument("--dataset_name_or_path")
    parser.add_argument("--embedder_name_or_path")
    parser.add_argument("--max_seq_length",type=int,default=32)
    parser.add_argument("--num_beams",type=int,default=8)
    parser.add_argument("--num_beam_groups",type=int,default=1)
    parser.add_argument("--diversity_penalty",type=float,default=1.0)   
    parser.add_argument("--num_iters",type=int,default=50)
    parser.add_argument("--enable_progress_bar",type=eval,default=False)
    parser.add_argument("--eval_batch_size",type=int,default=64)
    parser.add_argument("--max_eval_samples",default=None,type=int)
    parser.add_argument("--output_dir")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    distributed_state = PartialState()
    device = distributed_state.device
    num_return_sequences = args.num_beams
    generation_kwargs = {
        "early_stopping": False,
        "num_beams": args.num_beams,
        "do_sample": False,
        "no_repeat_ngram_size": 0,
        "max_length": args.max_seq_length, ## T5 would add <bos> and <eos>
        "num_return_sequences":num_return_sequences,
    }
    if args.num_beam_groups > 1:
        generation_kwargs['num_beam_groups'] = args.num_beam_groups
        generation_kwargs['diversity_penalty'] = args.diversity_penalty
    
    ## == load embedder and generator == ##
    
    ##  =====MODIFY FOR OFFICIAL CHECKPOINT==== ##
    embedder = GTREmbedder(device)
    import vec2text
    corrector_model = vec2text.models.CorrectorEncoderModel.from_pretrained("jxm/gtr__nq__32__correct")
    generator,tokenizer = load_generator(
        "google-t5/t5-base",
        "sentence-transformers/gtr-t5-base",
        expanding_factor=16,
        torch_compile=False,
        project_draft_embedding=True,
        layer_norm_after_projection=True,
        init_new_module=True
    )

    state_dict = corrector_model.state_dict()
    keys = list(state_dict.keys())
    for key in keys:
        state_dict[key.removeprefix("encoder_decoder.")] = state_dict.pop(key)

    keys = list(state_dict.keys())
    for key in keys:
        if "embedding_transform_1" in key:
            state_dict[key.replace("embedding_transform_1","target_projector.projector")] = state_dict.pop(key)
        if "embedding_transform_2" in key:
            state_dict[key.replace("embedding_transform_2","diff_projector.projector")] = state_dict.pop(key)
        if "embedding_transform_3" in key:
            state_dict[key.replace("embedding_transform_3","draft_projector.projector")] = state_dict.pop(key)
        if "layernorm" in key:
            state_dict[key.replace("layernorm","proj_ln")] = state_dict.pop(key)
    
    generator.load_state_dict(state_dict,strict=True)
    ##  =====MODIFY FOR OFFICIAL CHECKPOINT==== ##
    generator = generator.to(device).eval()

    ## == load dataset == ##
    test_dataset = load_data(
        dataset_name_or_path=args.dataset_name_or_path,
        tokenizer=tokenizer,
        partial_state = distributed_state,
        max_seq_length=args.max_seq_length,
        embedder=load_embedder("sentence-transformers/gtr-t5-base").to(device),
        load_split=['test'],
        draft_dir=args.draft_dir,
    )
    test_dataset = list(test_dataset)[:args.max_eval_samples]
    for sample in test_dataset:
        sample['text'] = tokenizer.decode(tokenizer(sample['text'],add_special_tokens=False,max_length=31,truncation=True).input_ids,skip_special_tokens=True)

    refs = [test_dataset[idx]['text'].strip() for idx in range(len(test_dataset))]

    ## == split dataset across multiple GPUs (if possible), also compatible with single GPU setting == ##
    with distributed_state.split_between_processes(test_dataset) as sharded_dataset:
        
        ##  =====MODIFY FOR OFFICIAL CHECKPOINT==== ##
        for idx,sample in enumerate(sharded_dataset):
            sample['target_embedding'] = embedder.encode([sample['text']])[0]
            draft_input_text = tokenizer.decode(sample['draft_input_ids'],skip_special_tokens=True)
            sample['draft_input_ids'] = tokenizer("<pad>" + draft_input_text, max_length=32,truncation=True,padding=True,return_tensors='pt',add_special_tokens=False).input_ids[0]
            sample['draft_embedding'] = embedder.encode([draft_input_text])[0]
            

        ## core data structure, sample_id --> same sample with different draft
        dataset_dict = {}
        for sample in sharded_dataset:
            dataset_dict[int(sample['idx'].item())] = [sample]

        metrics_over_iter = []
        hyps_over_iter = []
        id2best_hyp = {sample_idx: None for sample_idx in dataset_dict.keys()}

        ## ==start iterative generation with sequence-level beam search== ##
        for iter_idx in range(args.num_iters):
            ## set up dataset from dataset_dict 
            dataset_for_current_iter = []
            for sample in dataset_dict.values():
                dataset_for_current_iter.extend(sample[:args.num_beams])

            ## Generation
            start_time = time.time()
            id2hyps = single_step_generation(
                dataset_for_current_iter,generator,tokenizer,generation_kwargs,embedder,
                enable_progress_bar=distributed_state.is_local_main_process and args.enable_progress_bar,
                batch_size = args.eval_batch_size,
                )
            torch.cuda.synchronize()
            duration = time.time() - start_time
            
            ## sort generated hyps by their distance to target_embedding
            ## pick top-k unique hyps
            for sample_id in id2hyps:
                sorted_hyps = sorted(id2hyps[sample_id],key=lambda x:x["cos_sim"],reverse=True)
                unique_list,unique_set = [],set()
                for hyp_dict in sorted_hyps:
                    if hyp_dict['hyp'] not in unique_set:
                        unique_set.add(hyp_dict['hyp'])
                        unique_list.append(hyp_dict)
                        if len(unique_list) == num_return_sequences:
                            break
                id2hyps[sample_id] = unique_list
            
            ##== early stop when cos_sim == 1.0 ==## 
            for sample_id in list(id2hyps.keys()):
                id2best_hyp[sample_id] = id2hyps[sample_id][0]['hyp'] ## top-1 ranked hyp
                if is_equal(id2hyps[sample_id][0]['cos_sim'],1.0):
                    id2hyps.pop(sample_id)
                    dataset_dict.pop(sample_id)

            ## == evaluation == ##
            hyps_for_eval = list(id2best_hyp.values())
            hyps_for_eval = maybe_gather_from_other_gpu(hyps_for_eval,distributed_state)
            if distributed_state.is_local_main_process:
                eval_result = eval_v2t(hyps_for_eval,refs,embedder)
                metrics_over_iter.append(eval_result)
                hyps_over_iter.append(hyps_for_eval)
                overwatch.info(f"{iter_idx+1:2}# {duration:4.1f}s {eval_result}")

            ## == update dataset_dict for next round generation == ##
            for sample_id in dataset_dict:
                if len(dataset_dict[sample_id]) != len(id2hyps[sample_id]):
                    ## copy.deepcopy super important!
                    dataset_dict[sample_id] = [copy.deepcopy(dataset_dict[sample_id][0]) for _ in range(len(id2hyps[sample_id]))]
                assert len(dataset_dict[sample_id]) == len(id2hyps[sample_id])
                
                for rank in range(len(id2hyps[sample_id])):
                    dataset_dict[sample_id][rank]['draft_input_ids'] = tokenizer("<pad>" + id2hyps[sample_id][rank]['hyp'],max_length=args.max_seq_length,truncation=True,padding=True,return_tensors='pt',add_special_tokens=False).input_ids[0]
                    dataset_dict[sample_id][rank]['draft_embedding'] = id2hyps[sample_id][rank]['hyp_embedding']

    ##== write back to disk ==##
    if args.output_dir is not None:
        if distributed_state.is_local_main_process:
            os.makedirs(args.output_dir,exist_ok=True)
            metrics_path = os.path.join(args.output_dir,"metrics.jsonl")
            write_jsonl(metrics_over_iter,metrics_path)

            hyps_path = os.path.join(args.output_dir,"results.json")
            results = {"hyps":hyps_over_iter,'refs':refs}
            with open(hyps_path,'w') as f:
                json.dump(results,f,indent=4)