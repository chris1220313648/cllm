#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
python3 /home/wenxuansong/chenjy/project/Consistency_LLM/eval/vla/acc.py \
    --test_file "/home/wenxuansong/chenjy/project/vlas/LLaVA/playground/calvin_data/test.json" \
    --max_tokens 2048 \
    --action_stat  "/mnt/sda/wenxuansong/data/dataset/task_ABC_D/training/statistics.yaml" \
    --image_folder "/home/wenxuansong/chenjy/data/calvin_cvla/task_ABC_D/vla_processed_r5" \
    --model_path "/home/wenxuansong/chenjy/project/vlas/LLaVA/checkpoints/llava-v1.5-7b-calvin-rel-obs-reduce5-v1-abc2d_2024_01_29" \
    --port 9010 \
    --max_new_tokens_for_consistency 16 \
    # --use_consistency_decoding
    # --eval_only \


