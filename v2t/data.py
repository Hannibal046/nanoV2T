from functools import partial
import random
import copy,os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import datasets
import json
import numpy as np
from pathlib import Path
## Torch
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
## HF
from datasets import load_dataset

from v2t.utils import md5_hash
from v2t.overwatch import initialize_overwatch
overwatch = initialize_overwatch(__name__)

CACHE_DIR=os.path.join(Path.home(),".cache/v2t")

def collator(        
        samples,
        tokenizer,
    ):
    """
    collate tokenized input_ids and labels with left and right side padding supported
    
    Args:
        samples (dict): a dict contains input_ids, labels and text
        llm_tokenizer: tokenizer for llm
    
    Returns:
        xrag_input_ids: input_ids with xrag_token_id (xrag_labels,xrag_attention_mask)
        vanilla_input_ids: input_ids for llm without xrag_token_id, vanilla rag (labels,attention_mask)
        retriever_input_ids: input_ids for retriever (retriever_attention_mask)

    """
    def padding(ids,padding_value,padding_side='right'):
        if padding_side == 'right':
            return torch.nn.utils.rnn.pad_sequence(ids,batch_first=True,padding_value=padding_value)
        
        elif padding_side == 'left':
            flipped_ids = [torch.flip(x, dims=[0]) for x in ids]  
            return torch.flip(
                torch.nn.utils.rnn.pad_sequence(flipped_ids,batch_first=True,padding_value=padding_value),
                dims=[1],
            )

    ret = {
        "ids":[x['idx'] for x in samples]
    }
    padding_side = tokenizer.padding_side
    
    ## Pad Labels
    labels = padding([x['labels'] for x in samples],-100,padding_side)
    ret['labels'] = labels

    ## Pad Draft Input IDs
    if "draft_input_ids" in samples[0].keys():
        draft_input_ids = [x["draft_input_ids"] for x in samples]
        draft_input_ids = padding(draft_input_ids,tokenizer.pad_token_id,padding_side)
        draft_attention_mask = (draft_input_ids != tokenizer.pad_token_id).long()
        ret['draft_input_ids'] = draft_input_ids
        ret['draft_attention_mask'] = draft_attention_mask
        ret['draft_embeddings'] = torch.stack([x['draft_embedding'] for x in samples])
    
    ## Target Embedding
    if 'target_embedding' in samples[0].keys():
        ret['target_embeddings'] = torch.stack([x['target_embedding'] for x in samples])
    else:
        ret['text'] = [x['text'] for x in samples]
    
    return ret

def v2t_encode(
    examples,
    tokenizer,
    max_seq_length,    
):
    tokenized_examples = tokenizer(
        examples['text'],
        max_length=max_seq_length,
        truncation=True,
        add_special_tokens=False,
    )

    truncated_input_text = tokenizer.batch_decode(tokenized_examples['input_ids'],skip_special_tokens=True)
    input_ids = tokenizer(
        truncated_input_text,
        max_length = max_seq_length + 1, ## EOS for T5
        truncation=True,
        padding=True,
        return_tensors='pt',
    ).input_ids
    labels = input_ids.clone()
    labels[labels==tokenizer.pad_token_id]=-100

    ret =  {
        "labels":labels,
        "text":truncated_input_text,
    }

    if "draft" in examples:
        draft_input_ids = tokenizer(
            examples['draft'],
            max_length = max_seq_length + 1, ## EOS for T5
            truncation=True,
            padding=True,
            return_tensors='pt',
        ).input_ids

        ret['draft_input_ids'] = draft_input_ids

    return ret

def load_data_from_hf(dataset_name_or_path,max_eval_samples,*args):
    if dataset_name_or_path == 'jxm/nq_corpus_dpr':
        ## train/dev
        dataset = load_dataset(dataset_name_or_path)
        max_eval_samples = min(len(dataset["dev"]), max_eval_samples)
        assert len(dataset['dev']) > 2 * max_eval_samples 
        dataset['test'] = dataset["dev"].select(range(max_eval_samples))
        dataset['dev']  = dataset["dev"].select(range(max_eval_samples,max_eval_samples*2))
    
    elif dataset_name_or_path == "Tevatron/msmarco-passage-corpus":
        ## only train split (#8841823)
        dataset = load_dataset(dataset_name_or_path)
        num_samples = len(dataset['train'])
        dataset['dev']   = dataset['train'].select(range(num_samples - 2*max_eval_samples,num_samples - max_eval_samples))
        dataset['test']  = dataset['train'].select(range(num_samples - max_eval_samples,num_samples))
        dataset['train'] = dataset['train'].select(range(0,num_samples-2*max_eval_samples))
    return dataset

def build_embeddings(text,accelerator,embedder):
    """
    Using Embedder to Build Embeddings (possibly with DDP)
    Args:
        text: List[str]
        accelerator:
        embedder: emebdding model with encode interface
    Return:
        List[np.array]
    """

    with accelerator.split_between_processes(text) as sharded_text:
        embeddings = embedder.encode(
            sharded_text,batch_size=128,
            show_progress_bar=accelerator.is_local_main_process,
        )
        if accelerator.num_processes > 1:
            embeddings_from_all_ranks = [None for _ in range(accelerator.num_processes)]
            dist.all_gather_object(embeddings_from_all_ranks,embeddings)
            embeddings = [x for y in embeddings_from_all_ranks for x in y]
    return embeddings

def load_data(
        dataset_name_or_path,
        tokenizer,
        partial_state=None,
        max_train_samples = None,
        max_seq_length = 32,
        embedding_construction = 'offline',
        embedder = None,
        max_eval_samples=1000,
        draft_dir = None,
        load_split = ['train','dev','test'],
    ):
    hash_code = md5_hash(
        embedder_name_or_path=embedder.model_card_data.base_model,
        dataset_name_or_path=dataset_name_or_path,
        max_seq_length=max_seq_length,
        max_train_samples=max_train_samples,
        embedding_construction=embedding_construction,
        max_eval_samples = max_eval_samples,
        draft_dir = draft_dir,
    )
    cache_path = os.path.join(CACHE_DIR,hash_code)
    if os.path.exists(cache_path):
        overwatch.info(f"Loading data from cached path: {cache_path}")
        v2t_datasets = datasets.load_from_disk(cache_path)
    else:
        overwatch.info(f"Loading train data from {dataset_name_or_path}")
        v2t_datasets = load_data_from_hf(dataset_name_or_path,max_eval_samples)

        ### Add draft here
        if draft_dir is not None:
            overwatch.info(f"Loading pre-generated draft from: {draft_dir}")
            for _split in load_split:
                _drafts = [json.loads(x) for x in open(os.path.join(draft_dir,f"{_split}.hyps.jsonl")).readlines()]
                assert len(_drafts) == len(v2t_datasets[_split])
                drafts = [None for _ in range(len(_drafts))]
                for sample in _drafts:
                    idx = int(sample['idx'])
                    drafts[idx] = sample['draft']
                assert None not in drafts
                v2t_datasets[_split] = v2t_datasets[_split].add_column(name="draft", column=drafts)

        ### select N samples
        if (
            max_train_samples is not None 
            and len(v2t_datasets['train']) > max_train_samples
            and "train" in load_split
        ):
            overwatch.info(f"Randomly select {max_train_samples} samples from dataset of size {len(v2t_datasets['train'])}")
            selected_indices = random.sample(range(len(v2t_datasets['train'])),max_train_samples)
            v2t_datasets['train'] = v2t_datasets['train'].select(selected_indices)
        
        ## Tokenize to Max Length
        encode_function = partial(
            v2t_encode,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
        )
        with partial_state.main_process_first():
            for _split in load_split:
                v2t_datasets[_split] = v2t_datasets[_split].map(
                    encode_function,
                    batched=True,
                    num_proc=16,
                    desc=f"Tokenizing and reformatting data on rank: {partial_state.local_process_index}",
                )
                v2t_datasets.set_format(type="pt")
                if _split == 'train':
                    v2t_datasets['train'] = v2t_datasets['train'].filter(lambda example: (example['labels'] != -100).any())

        partial_state.wait_for_everyone()

        ## Construct Embeddings
        if embedding_construction == 'offline':
            ## Construct Embeddings for Target Text
            for _split in load_split:
                overwatch.info(f"Pre-Build [bold]{_split}[/bold] Target Embeddings...")
                embeddings = build_embeddings(v2t_datasets[_split]['text'],partial_state,embedder)
                v2t_datasets[_split] = v2t_datasets[_split].add_column(name="target_embedding", column=embeddings)
                v2t_datasets[_split].set_format(type="pt")

            ## Construct Embeddings for Draft Text
            if draft_dir is not None:
                for _split in load_split:
                    overwatch.info(f"Pre-Build [bold]{_split}[/bold] Darft Embeddings...")
                    embeddings = build_embeddings(v2t_datasets[_split]['draft'],partial_state,embedder)
                    v2t_datasets[_split] = v2t_datasets[_split].add_column(name="draft_embedding", column=embeddings)
                    v2t_datasets[_split].set_format(type="pt")
        
        ## Add index
        for _split in load_split:
            v2t_datasets[_split] = v2t_datasets[_split].add_column("idx", range(len(v2t_datasets[_split])))

        if partial_state.is_local_main_process and len(load_split)==3:
            overwatch.info(f"Saving data to cache path: {cache_path}")
            v2t_datasets.save_to_disk(cache_path)
        partial_state.wait_for_everyone()
    
    rets = [v2t_datasets[_split] for _split in load_split]
    return rets[0] if len(rets)==1 else rets