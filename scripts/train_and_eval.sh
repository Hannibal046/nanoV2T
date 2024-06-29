## stage 1 training
accelerate launch --num_processes 8 --mixed_precision bf16 \
    v2t/train.py \
        --config config/stage1.yaml

## stage 2 training
accelerate launch --num_processes 8 --mixed_precision bf16 \
    v2t/train.py \
        --config config/stage2.yaml \
        --draft_dir checkpoints/gtr_t5_nq_32_stage1/wandb/latest-run/files/hyps 


## inference with sbeam
accelerate launch --num_processes 8 \
    v2t/inference.py \
        --draft_dir checkpoints/gtr_t5_nq_32_stage1/wandb/latest-run/files/hyps \
        --generator_name_or_path checkpoints/gtr_t5_nq_32_stage2/wandb/latest-run/files/final \
        --dataset_name_or_path jxm/nq_corpus_dpr \
        --embedder_name_or_path sentence-transformers/gtr-t5-base \
        --max_seq_length 32 \
        --num_beams 5 --num_iters 50

# accelerate launch --num_processes 8 \
#     v2t/inference.py \
#         --draft_dir checkpoints/gtr_t5_nq_32_stage1/wandb/latest-run/files/hyps \
#         --generator_name_or_path tmp/temp_model \
#         --dataset_name_or_path jxm/nq_corpus_dpr \
#         --embedder_name_or_path sentence-transformers/gtr-t5-base \
#         --max_seq_length 32 \
#         --num_beams 5 --num_iters 50 --guiding_criteria cos_score --output_dir tmp/new_model_cos_score

accelerate launch --num_processes 8 \
    v2t/inference.py \
        --draft_dir checkpoints/gtr_t5_nq_32_stage1/wandb/latest-run/files/hyps \
        --generator_name_or_path tmp/temp_model \
        --dataset_name_or_path jxm/nq_corpus_dpr \
        --embedder_name_or_path sentence-transformers/gtr-t5-base \
        --max_seq_length 32 \
        --num_beams 5 --num_iters 50 --guiding_criteria gen_score --output_dir tmp/new_model_gen_score