#!/bin/bash
model_path="/home/wenxuansong/chenjy/project/vlas/LLaVA/checkpoints/llava-v1.5-7b-calvin-rel-obs-reduce5-v1_zhaobo/checkpoint-21572"
target_model_path="/home/wenxuansong/chenjy/project/vlas/LLaVA/checkpoints/llava-v1.5-7b-calvin-rel-obs-reduce5-v1_zhaobo/checkpoint-21572"
max_new_tokens=16
export CUDA_VISIBLE_DEVICES=2
# test model is tested and we use the tokenizer of teacher model because the tokenizer of test model has something to fix
python3 /home/wenxuansong/chenjy/project/Consistency_LLM/eval/vla/speedup.py \
    --test_model_path $model_path \
    --teacher_model_path $target_model_path \
    --test_filename /home/wenxuansong/chenjy/project/vlas/LLaVA/playground/calvin_data/test.json \
    --action_stat  "/mnt/sda/wenxuansong/data/dataset/task_ABC_D/training/statistics.yaml" \
    --image_folder "/home/wenxuansong/chenjy/data/calvin_cvla/task_ABC_D/vla_processed_r5" \
    --model_path "/home/wenxuansong/chenjy/project/vlas/LLaVA/checkpoints/llava-v1.5-7b-calvin-rel-obs-reduce5-v1-abc2d_2024_01_29" \
    --max_new_tokens $max_new_tokens
