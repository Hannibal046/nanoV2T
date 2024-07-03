## nanoV2T
Simple replication of [Text Embeddings Reveal (Almost) As Much As Text](https://arxiv.org/abs/2310.06816) (for pedagogy and fun). The original repo is [here](https://github.com/jxmorris12/vec2text). 

If you want to know more analysis about vec2text, we also recommend this paper: [Understanding and Mitigating the Threat of Vec2Text to Dense Retrieval Systems](https://arxiv.org/abs/2402.12784v1).

## Get Started
```
conda create -n v2t python=3.11 -y && conda activate v2t
conda install pytorch==2.1.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install transformers==4.41.2 accelerate==0.31.0 datasets sentencepiece wandb rich ipywidgets gpustat wget tiktoken pytest evaluate sacrebleu nltk sentence_transformers numpy==1.26.4
pip install -e .
python -c 'import nltk;nltk.download("punkt")'
```

## Stage1 Training
```bash
accelerate launch --num_processes 8 --mixed_precision bf16 \
    v2t/train.py \
        --config config/stage1.yaml
```

## Stage2 Training
```bash
accelerate launch --num_processes 8 --mixed_precision bf16 \
    v2t/train.py \
        --config config/stage2.yaml \
        --draft_dir checkpoints/gtr_t5_nq_32_stage1/wandb/latest-run/files/hyps
```
## Inference
We provide stage1 output in the `output` folder and trained stage2 model at [Hannibal046/gtr_t5_nq_32_stage2](https://huggingface.co/Hannibal046/gtr_t5_nq_32_stage2), so we could do inference like this:
```bash
accelerate launch --num_processes 8 \
    v2t/inference.py \
        --draft_dir output/gtr_t5_nq_32_stage1/hyps \
        --generator_name_or_path Hannibal046/gtr_t5_nq_32_stage2 \
        --dataset_name_or_path jxm/nq_corpus_dpr \
        --embedder_name_or_path sentence-transformers/gtr-t5-base \
        --max_seq_length 32 --max_eval_samples 500
```

This is the expected results:
```
{"bleu": 97.8, "token_f1": 99.5, "em": 93.2, "cos_sim": 1.0}
```

## Notes
Difference from original implementation:
- We do not count special tokens (bos,eos,pad) as tokens that need to be recovered.
- We optimize the inference process with (1) early stop when `cos_sim==1` and (2) distributed inference across GPUs.
- We use the first 1000 samples from `jxm/nq_corpus_dpr` as test set.
- We use larger batch size and correspondingly larger learning rate in both training stages.


## How to add new embedding models?
We support all models following the API of `Sentence Transformers`. Across this project, the embedder loading logic is defined at `v2t/model/emebdder/__init__.py`
```python
def load_embedder(
    model_name_or_path: str,
):
    overwatch.info(f"Loading Retriever from: {model_name_or_path}")
    if model_name_or_path == "sentence-transformers/gtr-t5-base":
        embedder = SentenceTransformer(model_name_or_path,device='cpu')
        tokenizer = embedder.tokenizer
    return embedder
```

## How to add new generation models?
Currently, we only support T5 models (from `T5-base` to `T5-11B`) which is defined at `v2t/model/generator/modeling_t5generator.py`.
