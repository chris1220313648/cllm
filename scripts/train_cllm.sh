export CUDA_VISIBLE_DEVICES=1
export WANDB_PROJECT=consistency_llm
model_path=/home/wenxuansong/chenjy/project/Consistency_LLM/models/Abel-7B-001
trajectory_file=/home/wenxuansong/chenjy/project/Consistency_LLM/data/collected_jacobi_trajectory/cleaned_gsm8k_train.jsonl_jacobi_max_new_tokens16_augFalse_labels_False_max_seq_len_64.json
output_path=/home/wenxuansong/chenjy/project/Consistency_LLM/models/cllm_model
n_token_size=16
qlora=$5
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_id=101 --rdzv_endpoint='localhost:8871' \
    --master_port 10000 \
    cllm/train_cllm_global.py \
    --target_model_path ${model_path} \
    --data_path ${trajectory_file} \
    --output_dir ${output_path} \
    --max_new_tokens ${n_token_size} \
    --bf16 True \
    --report_to wandb \
    --do_train \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing True \
    --save_strategy "steps" \
    --save_steps 100000 \
    --save_total_limit 50 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --model_max_length 2048 \
    --lazy_preprocess True \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --qlora ${qlora}
