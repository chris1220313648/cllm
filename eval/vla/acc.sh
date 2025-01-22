#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
python3 /home/wenxuansong/chenjy/project/Consistency_LLM/eval/vla/acc.py \
    --test_file "/home/wenxuansong/chenjy/project/vlas/LLaVA/playground/calvin_data/test.json" \
    --max_tokens 2048 \
    --max_new_tokens_for_consistency 16 \
    # --use_consistency_decoding
    # --eval_only \


