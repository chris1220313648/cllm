python3 data/generate_trajectory.py \
  --filename //home/wenxuansong/chenjy/project/Consistency_LLM/data/raw_data/gsm8k_test.jsonl \
  --model /home/wenxuansong/chenjy/project/Consistency_LLM/models/Abel-7B-001 \
  --max_new_tokens 16 \
  --max_new_seq_len 256 
  # --use_labels
#python3 data/generate_trajectory.py   --filename /home/wenxuansong/chenjy/project/Consistency_LLM/data/raw_data/gsm8k_train.jsonl   --model /home/wenxuansong/chenjy/project/Consistency_LLM/models/Abel-7B-001   --max_new_tokens 16   --max_new_seq_len 64   > /home/wenxuansong/chenjy/project/Consistency_LLM/logs/generate_trajectory.log 2>&1