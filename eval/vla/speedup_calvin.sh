#!/bin/bash
model_path="/home/wenxuansong/chenjy/project/Consistency_LLM/models/cvla_model/llava-v1.5-7b-abcd-checkpoint-40000"
target_model_path=/home/wenxuansong/chenjy/project/Consistency_LLM/models/cvla_model/llava-v1.5-7b-abcd-checkpoint-40000
max_new_tokens=16
export CUDA_VISIBLE_DEVICES=1
# test model is tested and we use the tokenizer of teacher model because the tokenizer of test model has something to fix
python3 /home/wenxuansong/chenjy/project/Consistency_LLM/eval/vla/speedup_calvin.py \
    --test_model_path $model_path \
    --teacher_model_path $target_model_path \
    --action_stat  "/mnt/sda/wenxuansong/data/dataset/task_ABCD_D/training/statistics.yaml" \
    --image_folder "/home/wenxuansong/chenjy/data/calvin_cvla/task_ABCD_D/vla_processed_r5" \
    --model_path $model_path \
    --max_new_tokens $max_new_tokens
