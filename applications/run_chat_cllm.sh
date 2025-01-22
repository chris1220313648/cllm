model_path=$1
cllm_type=$2

python3 applications/chat_cli_cllm.py --model_path /home/wenxuansong/chenjy/project/Consistency_LLM/models/consistency-llm-7b-sharegpt48k --cllm_type sharegpt --chat --debug
