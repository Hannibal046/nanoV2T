## nanoV2T
Simple replication of (for pedagogy and fun):
- [Text Embeddings Reveal (Almost) As Much As Text](https://arxiv.org/abs/2310.06816)
- [Language Model Inversion](https://arxiv.org/abs/2311.13647)

## Get Started
```
conda create -n v2t python=3.11 -y && conda activate v2t
conda install pytorch==2.1.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install transformers==4.41.2 accelerate==0.31.0 datasets sentencepiece wandb rich ipywidgets gpustat wget tiktoken pytest evaluate sacrebleu nltk sentence_transformers
pip install -e .
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