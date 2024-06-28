## nanoV2T (ongoing)
Simple replication of (for pedagogy and fun):
- [Text Embeddings Reveal (Almost) As Much As Text](https://arxiv.org/abs/2310.06816)
- [Language Model Inversion](https://arxiv.org/abs/2311.13647)

## Get Started
```
conda create -n v2t python=3.11 -y && conda activate v2t
conda install pytorch==2.1.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y
conda install ipykernel -y
pip install transformers accelerate datasets sentencepiece wandb rich ipywidgets gpustat wget tiktoken pytest evaluate sacrebleu nltk sentence_transformers
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
```bash
accelerate launch --num_processes 8 \
    v2t/inference.py \
        --draft_dir checkpoints/gtr_t5_nq_32_stage1/wandb/latest-run/files/hyps \
        --generator_name_or_path checkpoints/gtr_t5_nq_32_stage2/wandb/latest-run/files/final \
        --dataset_name_or_path jxm/nq_corpus_dpr \
        --embedder_name_or_path sentence-transformers/gtr-t5-base \
        --max_seq_length 32 \
        --num_beams 5 --num_iters 50
```