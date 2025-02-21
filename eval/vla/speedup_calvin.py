from dataclasses import dataclass, field
import json
import math
import pathlib
import functools
from typing import Dict, Optional, Sequence, List, Tuple
import random
from tqdm import tqdm
import torch.nn.functional as F
import sqlite3
import time
import numpy as np
import torch
from torch.utils.data import Dataset
import transformers
from transformers.trainer_pt_utils import LabelSmoother, get_module_class_from_name
from fastchat.model.model_adapter import get_conversation_template
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers import LlamaModel, LlamaForCausalLM, GenerationConfig
import argparse

import os

import sys
from pathlib import Path
from flask import Flask, jsonify, request, Response

path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

from cllm.utils import detect_repetitive_patterns
# from cllm.cllm_llama_modeling import delete_false_key_value, jacobi_forward_profiling
from cllm.cvla_llama_modeling import delete_false_key_value, cvla_jacobi_forward_profiling
from cllm.train_cvla_global import CLLMRobotgenerate
###
import numpy as np

from PIL import Image
from dataclasses import dataclass, field
import json
import math
import pathlib
from typing import Dict, Optional

import os
import sys
import torch
from torch.utils.data import Dataset
import transformers
from transformers.trainer_pt_utils import LabelSmoother, get_module_class_from_name
import datasets

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from typing import Dict



from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

import logging

import os
from llava import conversation as conversation_lib
from llava.model.builder import load_pretrained_model_cvla, load_pretrained_model
from llava.mm_utils import (
    tokenizer_image_token,
    process_images,
    get_model_name_from_path,
)
from llava.action_tokenizer import ActionTokenizer, encode_robot_obs
from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_AUDIO_TOKEN,IMAGE_TOKEN_INDEX
from llava.mm_utils import tokenizer_image_token
from llava.utils import disable_torch_init
###
DynamicCache.delete_false_key_value = delete_false_key_value
LlamaForCausalLM.jacobi_forward = cvla_jacobi_forward_profiling

def clva_jacobi_generate(input_ids, images_tensor,model, tokenizer, max_new_tokens, max_new_seq_len):
    converge_step = []
    forward_times = 0

    all_jacobian_trajectory = []
    prompt_len = input_ids.shape[1]  # 直接获取序列长度
    # print("prompt_len:",prompt_len)
    generation = input_ids
    ### prefill the kv-cache
    past_key_values, first_correct_token = model.jacobi_forward(input_ids=input_ids,images_tensor=images_tensor, max_new_tokens=max_new_tokens, past_key_values=None, use_cache = True, prefill_phase = True)#返回键值对缓存，和第一个预测的元素
    ### generation phase
    itr = 0
    eos_reached = False
    while True:
        itr+=1
        bsz = 1 # only support batch_size = 1 now
        # randomly initialize the first point of jacobian trajectory
        valid_tokens = generation[0][generation[0] != IMAGE_TOKEN_INDEX].tolist()
        random_point = torch.tensor(random.choices(valid_tokens, k=(max_new_tokens-1)), device="cuda").view(1,-1)
        input_ids = torch.cat((first_correct_token.view(1,-1), random_point),dim=-1)#[1,16]
        # jacobian_trajectory整个轨迹 n_gram_generation收敛点 iter_steps迭代步数
        jacobian_trajectory, n_gram_generation, first_correct_token, iter_steps = model.jacobi_forward(input_ids=input_ids,images_tensor=images_tensor, max_new_tokens=max_new_tokens, past_key_values=past_key_values, use_cache = True, prefill_phase = False)
        forward_times += iter_steps
        all_jacobian_trajectory.append(jacobian_trajectory)#4维度 jacobian_trajectory3维[n,1,16]
        eos_positions = torch.where(n_gram_generation[0]==tokenizer.eos_token_id)[0]

        if len(eos_positions)>0:
            eos_reached = True
        
        ### see if next max_new_tokens should be generated & if True, update weights and prepare new input_id 
        generation = torch.cat((generation, n_gram_generation), dim=-1)
        if eos_reached or itr*max_new_tokens > max_new_seq_len:
            break
    
    # to support bsz > 1
    converge_step.append(forward_times / itr)#平均每次小循环的迭代步数
    # print("generation[0]:",generation[0])

    return generation[0, prompt_len:], converge_step, all_jacobian_trajectory#这里的all 指一条数据的所有迭代轨迹 [3,n,1,16]

def jacobian_speed_evaluate(input_ids, image_tensor ,model, tokenizer, max_new_tokens, max_new_seq_len,action_tokenizer):#

    time_speed = []
    eos_reached = False
    # inputs = tokenizer([processed_prompt], return_tensors="pt").to(model.device)
    t1 = torch.cuda.Event(enable_timing=True)
    t2 = torch.cuda.Event(enable_timing=True)
    t1.record()
    #jacobi_generation不包括prompt  最终结果   all_jacobian_trajectory整个轨迹 3次小循环 3维
    jacobi_generation, converge_step, all_jacobian_trajectory = clva_jacobi_generate(input_ids, image_tensor,model, tokenizer, max_new_tokens, max_new_seq_len)
    t2.record()
    torch.cuda.synchronize()
    
    t = t1.elapsed_time(t2) / 1000
    # print(f'jacobian time: {t}')
    prompt_token_len = torch.sum(input_ids, dim=-1)
    eos_positions = torch.where(jacobi_generation==tokenizer.eos_token_id)[0]#a=行索引
    # print("jacobi_generation",jacobi_generation)
    if len(eos_positions)>0:
        eos_reached = True
        actions=[]
        print("jacobi_generation[:int(eos_positions[0]).shape]：",jacobi_generation[:int(eos_positions[0])].shape)
        output_ids=jacobi_generation[int(eos_positions[0])-35:int(eos_positions[0])].cpu().numpy().tolist()
        
        for elem in output_ids:
            actions.append(action_tokenizer.decode_token_ids_to_actions(elem))
        total_generation_len = jacobi_generation.shape[0]
        decoded_generation = actions# 
    else:
        total_generation_len = jacobi_generation.shape[0]
        decoded_generation = tokenizer.decode(jacobi_generation)
    time_speed.append(total_generation_len/t)#平均每秒推理token数 （最终答案）
    print(f'jacobian generation: {decoded_generation}')

    return eos_reached, time_speed, converge_step, jacobi_generation, decoded_generation, all_jacobian_trajectory
    
def speed_compare(input_ids, image_tensor, vla_model_generate, args):
    # Load model and tokenizer
    CLLMRobotgenerate_implementation=vla_model_generate
    model=CLLMRobotgenerate_implementation.vla_model
    tokenizer = CLLMRobotgenerate_implementation.tokenizer
    action_tokenizer = CLLMRobotgenerate_implementation.action_tokenizer
    ##### compare speed of jacobian and ar #####
    converge_step = []#收敛步数
    ar_time_speed = []#ar速度
    jacobian_time_speed = []#jacobian速度
    per_request_meta_trajectory_records = []
    # data_lst=min(data_lst,len(data))
    # only support batch size ==1 now
    max_new_tokens = args.max_new_tokens
    

    input_ids = input_ids.to(device=model.device, dtype=torch.int)
   
    image_tensor = image_tensor.to(device=model.device)
    #jacobian_time_speed_lst 平均每秒推理token数 （最终答案），[1]; jacobian_itr_step_lst 平均每次小循环的迭代步数 16token是3个给[3]，all_jacobian_trajectory [3,itr,1,16]
    eos_reached, jacobian_time_speed_lst, jacobian_itr_step_lst, decoded_ids, decoded_result, all_jacobian_trajectory = jacobian_speed_evaluate(input_ids,image_tensor, model, tokenizer, max_new_tokens, args.max_new_seq_len,action_tokenizer)
    
    # if not detect_repetitive_patterns(tokenizer, decoded_ids, repeat_ngram_size=10):
    per_request_meta_trajectory_records.append(all_jacobian_trajectory)#记录每一条数据的迭代轨迹 all_jacobian_trajectory一个大循环，3个小循环 3维数组  per_request_meta_trajectory_records5维度 [num_samples,3,小循环迭代数，1，16]

    jacobian_time_speed.append(*jacobian_time_speed_lst)
    converge_step.append(*jacobian_itr_step_lst)

    # inputs = tokenizer([processed_prompt], return_tensors="pt").to(model.device)

    gen_cfg = GenerationConfig.from_model_config(model.config)

    ar_begin = torch.cuda.Event(enable_timing=True)
    ar_end = torch.cuda.Event(enable_timing=True)
    ar_begin.record()
    ar_generated = vla_model_generate.robot_action_generate(
            input_ids,
            image_tensor
            # do_sample=False,
            # temperature=args.temperature,
            # max_new_tokens=max_new_token,
        )
    # ar_generated = model.generate(**inputs, use_cache=True, max_new_tokens=512)[0][inputs.input_ids.shape[-1]:-1]
    ar_end.record()
    torch.cuda.synchronize()
    
    #print(ar_generated)
    print(f'ar generated length: {len(ar_generated)}')
    ar_time = ar_begin.elapsed_time(ar_end) / 1000
    # print(f'ar time: {ar_time}')
    ar_time_speed.append(len(ar_generated)/ar_time)
    
    # all trajectory analsis for speedup interpretability
    fast_forward_and_fix_points_statistics = {}
    # initialize dict for all stats
    fast_forward_and_fix_points_statistics['fix_points'] = 0
    fast_forward_and_fix_points_statistics['fast_forward'] = 0
    fast_forward_and_fix_points_statistics['fix_points_per_gram'] = []
    all_jacobian_trajectory=all_jacobian_trajectory
    # iterate over all requests
    
    fast_forward_metrics = []

    fix_points_metrics = 0

    effective_trajectory_length = args.max_new_tokens
    # iterate over all n-grams, across the sequence length dimension
    # last trajectory contains eos, we need to keep track
    last_traj_flag = False
    for n_gram_id in range(len(all_jacobian_trajectory)):#每一条数据的大迭代次数 3次
        # initialize fix_points_tracker
        fix_points_tracker = {}
        for pos_ind in range(args.max_new_tokens):
            # to track how many times right token is predicted right
            fix_points_tracker[pos_ind] = 0

        # initialize single_fast_forward_metrics
        single_fast_forward_metrics = []

        generation_trajectory = all_jacobian_trajectory[n_gram_id]

        if n_gram_id == len(all_jacobian_trajectory) - 1:
            last_traj_flag = True

        correct_token_cnt = 0
        fix_points_per_gram = 0
        # look at a single n-gram trajectory
        # iterate over all points in the trajectory (with the same dimension)
        eos_reached = False
        eos_pos = None
        steps_to_convergence = 0
        for id, generation_ids in enumerate(generation_trajectory):#小迭代  generation_ids：[1,16]
            # skip initialiation
            if id == 0:
                continue
            if eos_reached == True:
                break
            assert len(generation_ids[0]) == args.max_new_tokens

            # iterate over all tokens
            fast_forward_cnt = 0

            contiguous_correct_flag = True

            for i in range(len(generation_ids[0])):
                token_generated = generation_ids[0][i]
                if generation_ids[0][i] == generation_trajectory[-1][0][i]:#遇到正确toekn
                    #print(BLUE + tokenizer.decode([token_generated]) + RESET, end=" ")  # print blue token
                    # update fix point tracker
                    fix_points_tracker[i] += 1 #固定token次数

                    # update fast-forward tracker
                    # first (i + 1) is to offset index
                    if (i + 1) > correct_token_cnt and contiguous_correct_flag:
                        fast_forward_cnt += 1

                    # check whether eos has been reached as a contiguous sentence
                    if last_traj_flag and token_generated == tokenizer.eos_token_id and contiguous_correct_flag:
                        effective_trajectory_length = i + 1

                        eos_reached = True
                        eos_pos = i

                        # before break out of the loop, uppdate values
                        correct_token_cnt += fast_forward_cnt

                        break
                else:
                    #print(RED + tokenizer.decode([token_generated]) + RESET, end=" ")  # print red token
                    if fix_points_tracker[i] > 0:
                            fix_points_tracker[i] = 0

                    if contiguous_correct_flag:
                        correct_token_cnt += fast_forward_cnt
                        contiguous_correct_flag = False

            single_fast_forward_metrics.append(fast_forward_cnt)#记录每次小循环的fast forward次数  连续遇到正确token次数

            steps_to_convergence += 1#收敛步数

        ff_baseline_cnt = {}
        for pos_ind in range(effective_trajectory_length):
            # to track how many times right token should be predicted right, if there is only fast_forward
            ff_baseline_cnt[pos_ind] = 0

        fast_forward_ptr = 0
        next_ff_flag = True
        for pos_ind in range(effective_trajectory_length):
            if next_ff_flag:
                fast_forward_offset = single_fast_forward_metrics[fast_forward_ptr]
                next_ff_flag = False

            ff_baseline_cnt[pos_ind] = steps_to_convergence - fast_forward_ptr #从当前位置到 收敛点需要的步数

            fast_forward_offset -= 1
            if fast_forward_offset == 0:
                next_ff_flag = True
                fast_forward_ptr += 1

        for pos_ind in fix_points_tracker.keys():
            cnt = fix_points_tracker[pos_ind]
            ff_baseline = ff_baseline_cnt[pos_ind]
            if cnt > ff_baseline:
                fix_points_metrics += 1#超前预测
                fix_points_per_gram += 1

            if last_traj_flag and pos_ind == eos_pos:
                break

        # record avg fast forward count over a single n-gram
        fast_forward_metrics.append(np.average(single_fast_forward_metrics))
        fast_forward_and_fix_points_statistics['fix_points_per_gram'].append(fix_points_per_gram)



    avg_fast_forward = np.average(fast_forward_metrics)
    fast_forward_and_fix_points_statistics['fast_forward']=avg_fast_forward
    fast_forward_and_fix_points_statistics['fix_points']=fix_points_metrics
    fast_forward_and_fix_points_statistics['ar_speed']=ar_time_speed[0]
    fast_forward_and_fix_points_statistics['jacobian_speed']=jacobian_time_speed[0]

    print(f"global average fast forward cnt: {fast_forward_and_fix_points_statistics['fast_forward']}")
    print(f"global average fix-point cnt: {fast_forward_and_fix_points_statistics['fix_points']}")
    print(f"global average fix-point per gram cnt: {np.average(fast_forward_and_fix_points_statistics['fix_points_per_gram'])}")
    ckpt_name = os.path.basename(args.test_model_path)
    save_path = 'data/speedup_profiling_results/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    new_file_path= f'{script_name}_profiling_results_{ckpt_name}_{args.max_new_tokens}_{args.max_new_seq_len}_{args.data_size}_{server_start_time}_stats.jsonl'
    # save_path = 'data/speedup_profiling_results/'
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)

    # new_file_path= f'calvin_speedup_profiling_results_{args.max_new_tokens}_{args.max_new_seq_len}_{args.data_size}_{server_start_time}_stats.json'
    fast_forward_and_fix_points_statistics_file = os.path.join(save_path, new_file_path)

    with open(fast_forward_and_fix_points_statistics_file, 'a') as f:
        json.dump(fast_forward_and_fix_points_statistics, f)
        f.write('\n') 
    
    ar_time_speed = ar_time_speed[0]
    jacobian_time_speed = jacobian_time_speed[0]
    print(f'ar speed: {ar_time_speed}')
    print(f'jacobian speed: {jacobian_time_speed}')
    # print(f'The max speed of model {args.test_model_path} using jacobian iteration (max_new_tokens: {max_new_tokens}) is {max(jacobian_time_speed)}')
    # print(f'The min speed of model {args.test_model_path} using jacobian iteration (max_new_tokens: {max_new_tokens}) is {min(jacobian_time_speed)}')
    # print(f'The avg speed of model {args.test_model_path} using jacobian iteration (max_new_tokens: {max_new_tokens}) is {sum(jacobian_time_speed)/len(jacobian_time_speed)}')
    # print(f'The max speed of model {args.test_model_path} using ar is {max(ar_time_speed)}')
    # print(f'The min speed of model {args.test_model_path} using ar is {min(ar_time_speed)}')
    # print(f'The avg speed of model {args.test_model_path} using ar is {sum(ar_time_speed)/len(ar_time_speed)}')
    return decoded_result

if __name__ == "__main__":  
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_filename", type=str,
                        default="eval/gsm8k/test.jsonl")
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--port", type=int, default=9011)
    parser.add_argument("--max_new_seq_len", type=int, default=1024)
    parser.add_argument("--test_model_path", type=str,
                        default="models/vicuna-7b-sharegpt-gpt4-48k")
                        
    parser.add_argument("--teacher_model_path", type=str,
                        default="cllm/consistency-llm-7b-sharegpt48k")
    parser.add_argument("--data_size", type=str,
                        default=500)
    parser.add_argument("--action_stat", type=str,
                        default="/mnt/sda/wenxuansong/data/dataset/task_ABC_D/training/statistics.yaml")
    parser.add_argument("--image_folder", type=str,
                        default="/home/wenxuansong/chenjy/data/calvin_cvla/task_ABC_D/vla_processed_r5")
    parser.add_argument("--model_path", type=str,
                        default="/home/wenxuansong/chenjy/project/vlas/LLaVA/checkpoints/llava-v1.5-7b-calvin-rel-obs-reduce5-v1-abc2d_2024_01_29")   
    args = parser.parse_args() 
    action_stat = args.action_stat
    image_folder = args.image_folder
    model_path=args.model_path
    vla_model_generate = CLLMRobotgenerate(model_path,action_stat,image_folder)
    model= vla_model_generate.vla_model
    tokenizer = vla_model_generate.tokenizer
    action_tokenizer = vla_model_generate.action_tokenizer
    script_name = os.path.basename(__file__).replace('.py', '')
    flask_app = Flask(__name__)
    import datetime
    # 获取服务器开始运行的时间，并格式化为字符串
    server_start_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    @flask_app.route("/predict", methods=["POST"])
    def predict():
        if request.method == "POST":
            # 接收图片和指令数据
            img_static = np.frombuffer(request.files["img_static"].read(), dtype=np.uint8)
            img_static = img_static.reshape((200, 200, 3))
            img_gripper = np.frombuffer(request.files["img_gripper"].read(), dtype=np.uint8)
            img_gripper = img_gripper.reshape((84, 84, 3))

            content = request.files["json"].read()
            content = json.loads(content)
            instruction = content["instruction"]
            robot_obs = content["robot_obs"]

            img_static = Image.fromarray(img_static)
            img_gripper = Image.fromarray(img_gripper)

            input_ids, image_tensor = vla_model_generate.compose_robot_input_for_calvin(
                img_static, img_gripper, instruction, robot_obs
            )

            # 调用速度比较函数进行推理计算
            result = speed_compare(input_ids, image_tensor, vla_model_generate, args)

            return jsonify(result)
    flask_app.run(host="0.0.0.0", port=args.port)