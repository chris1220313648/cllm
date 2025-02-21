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
        valid_tokens = generation[0][input_ids[0] != IMAGE_TOKEN_INDEX].tolist()
        random_point = torch.tensor(random.choices(valid_tokens, k=(max_new_tokens-1)), device="cuda").view(1,-1)
        input_ids = torch.cat((first_correct_token.view(1,-1), random_point),dim=-1)#[1,16]
        # jacobian_trajectory整个轨迹 n_gram_generation收敛点 iter_steps迭代步数
        jacobian_trajectory, n_gram_generation, first_correct_token, iter_steps = model.jacobi_forward(input_ids=input_ids,images_tensor=images_tensor, max_new_tokens=max_new_tokens, past_key_values=past_key_values, use_cache = True, prefill_phase = False)
        forward_times += iter_steps
        all_jacobian_trajectory.append(jacobian_trajectory)#4维度 jacobian_trajectory3维
        eos_positions = torch.where(n_gram_generation[0]==tokenizer.eos_token_id)[0]

        if len(eos_positions)>0:
            eos_reached = True
        
        ### see if next max_new_tokens should be generated & if True, update weights and prepare new input_id 
        generation = torch.cat((generation, n_gram_generation), dim=-1)
        if eos_reached or itr*max_new_tokens > max_new_seq_len:
            break
    
    # to support bsz > 1
    converge_step.append(forward_times / itr)#平均每次小循环的迭代步数

    return generation[0, prompt_len:], converge_step, all_jacobian_trajectory#这里的all 指一条数据的所有迭代轨迹 

def jacobian_speed_evaluate(input_ids, image_tensor ,model, tokenizer, max_new_tokens, max_new_seq_len):#

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
    prompt_token_len = torch.sum(input_ids, dim=-1)
    eos_positions = torch.where(jacobi_generation==tokenizer.eos_token_id)[0]
    if len(eos_positions)>0:
        eos_reached = True
        total_generation_len = jacobi_generation[:int(eos_positions[0])].shape[0]
        decoded_generation = tokenizer.decode(jacobi_generation[:int(eos_positions[0])])# 可以修改成动作解码 现在这样会生成汉字
    else:
        total_generation_len = jacobi_generation.shape[0]
        decoded_generation = tokenizer.decode(jacobi_generation)
    time_speed.append(total_generation_len/t)#平均每秒推理token数 （最终答案）

    return eos_reached, time_speed, converge_step, jacobi_generation, decoded_generation, all_jacobian_trajectory
    
def speed_compare(args,CLLMRobotgenerate_implementation):
    # Load model and tokenizer
    model=CLLMRobotgenerate_implementation.vla_model
    tokenizer = CLLMRobotgenerate_implementation.tokenizer
    action_tokenizer = CLLMRobotgenerate_implementation.action_tokenizer
    ##### compare speed of jacobian and ar #####
    converge_step = []#收敛步数
    ar_time_speed = []#ar速度
    jacobian_time_speed = []#jacobian速度
    filename = args.test_filename
    data = []
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    data=CLLMRobotgenerate_implementation.preprocess_vla_data(data,tokenizer,action_tokenizer)
    per_request_meta_trajectory_records = []
    data_len=min(args.data_size,len(data))
    data_lst = range(data_len)
    # data_lst=min(data_lst,len(data))
    # only support batch size ==1 now
    max_new_tokens = args.max_new_tokens
    for i in tqdm(data_lst): 
        raw_data = data[i]
        input_ids = torch.Tensor(raw_data['sources_input_ids']).squeeze(0).to(device=model.device, dtype=torch.int)
        label_ids=raw_data["labels_ids"]
        image_tensor=raw_data["image_tensor"].to(device=model.device)
        ar_begin = time.time()
        output_ids_vla = vla_model_initialization.robot_action_generate(
                input_ids,
                image_tensor
                # do_sample=False,
                # temperature=args.temperature,
                # max_new_tokens=max_new_token,
            )
        # ar_generated = model.generate(**inputs, use_cache=True, max_new_tokens=1024, do_sample=False)[0][inputs['input_ids'].shape[-1]:-1]
        ar_end = time.time()
        print(f'ar generated length: {len(output_ids_vla)}')#[35]
        #jacobian_time_speed_lst 平均每秒推理token数 （最终答案），[1]; jacobian_itr_step_lst 平均每次小循环的迭代步数 16token是3个给[3]
        eos_reached, jacobian_time_speed_lst, jacobian_itr_step_lst, decoded_ids, decoded_result, all_jacobian_trajectory = jacobian_speed_evaluate(input_ids,image_tensor, model, tokenizer, max_new_tokens, args.max_new_seq_len)
        
        # if not detect_repetitive_patterns(tokenizer, decoded_ids, repeat_ngram_size=10):
        per_request_meta_trajectory_records.append(all_jacobian_trajectory)#记录每一条数据的迭代轨迹 all_jacobian_trajectory一个大循环，3个小循环 3维数组  per_request_meta_trajectory_records5维度 [num_samples,3,小循环迭代数，1，16]

        jacobian_time_speed.append(*jacobian_time_speed_lst)
        converge_step.append(*jacobian_itr_step_lst)

        # inputs = tokenizer([processed_prompt], return_tensors="pt").to(model.device)

        gen_cfg = GenerationConfig.from_model_config(model.config)

        ar_begin = torch.cuda.Event(enable_timing=True)
        ar_end = torch.cuda.Event(enable_timing=True)
        ar_begin.record()
        ar_generated = vla_model_initialization.robot_action_generate(
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
        print(f'ar speed: {len(ar_generated)/(ar_time)}')
        ar_time_speed.append(len(ar_generated)/ar_time)
    
    # all trajectory analsis for speedup interpretability
    fast_forward_and_fix_points_statistics = {}
    # initialize dict for all stats
    fast_forward_and_fix_points_statistics['fix_points'] = []
    fast_forward_and_fix_points_statistics['fast_forward'] = []
    fast_forward_and_fix_points_statistics['fix_points_per_gram'] = []

    # iterate over all requests
    for all_generation_trajectory in per_request_meta_trajectory_records:
        fast_forward_metrics = []

        fix_points_metrics = 0

        effective_trajectory_length = args.max_new_tokens
        # iterate over all n-grams, across the sequence length dimension
        # last trajectory contains eos, we need to keep track
        last_traj_flag = False
        for n_gram_id in range(len(all_generation_trajectory)):#每一条数据的大迭代次数 3次
            # initialize fix_points_tracker
            fix_points_tracker = {}#记录n次小迭代中每个位置遇到正确token的次数
            for pos_ind in range(args.max_new_tokens):
                # to track how many times right token is predicted right
                fix_points_tracker[pos_ind] = 0

            # initialize single_fast_forward_metrics
            single_fast_forward_metrics = []#记录每次小循环的fast forward的值

            generation_trajectory = all_generation_trajectory[n_gram_id]

            if n_gram_id == len(all_generation_trajectory) - 1:
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
                        if (i + 1) > correct_token_cnt and contiguous_correct_flag:#连续遇到正确token
                            fast_forward_cnt += 1#这小迭代比上一次快进了多少

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
                            contiguous_correct_flag = False#一旦遇到不正确的token，连续标志位就变成False，fast_forward_cnt也再不能增加

                single_fast_forward_metrics.append(fast_forward_cnt)#记录每次小循环的fast forward次数  相对于上次迭代来说

                steps_to_convergence += 1#收敛步数

            ff_baseline_cnt = {}
            for pos_ind in range(effective_trajectory_length):
                # to track how many times right token should be predicted right, if there is only fast_forward
                ff_baseline_cnt[pos_ind] = 0

            fast_forward_ptr = 0#指向小循环的迭代id
            next_ff_flag = True
            for pos_ind in range(effective_trajectory_length):
                if next_ff_flag:#第一次正确token进入
                    fast_forward_offset = single_fast_forward_metrics[fast_forward_ptr]
                    next_ff_flag = False

                ff_baseline_cnt[pos_ind] = steps_to_convergence - fast_forward_ptr #从当前token位置到 收敛点需要的步数  88 77 66 55 44 33 22 11 

                fast_forward_offset -= 1
                if fast_forward_offset == 0:#最后一次正确token退出
                    next_ff_flag = True
                    fast_forward_ptr += 1

            for pos_ind in fix_points_tracker.keys():#012345678910112131415
                cnt = fix_points_tracker[pos_ind]#固定次数
                ff_baseline = ff_baseline_cnt[pos_ind]#从当前token位置到 收敛点需要的步数
                if cnt > ff_baseline:#说明提前预测到 true false true这种情况
                    fix_points_metrics += 1
                    fix_points_per_gram += 1

                if last_traj_flag and pos_ind == eos_pos:
                    break

            # record avg fast forward count over a single n-gram
            fast_forward_metrics.append(np.average(single_fast_forward_metrics))#记录每个小循环的平均 fast forward次数/每次迭代
            fast_forward_and_fix_points_statistics['fix_points_per_gram'].append(fix_points_per_gram)#每个小循环的fix-point次数 n_sample*3


        all_fast_forward = fast_forward_and_fix_points_statistics['fast_forward']
        all_fix_points = fast_forward_and_fix_points_statistics['fix_points']

        avg_fast_forward = np.average(fast_forward_metrics)
        all_fast_forward.append(avg_fast_forward)#加入一次大循环的  平均fast forward次数/每次迭代
        all_fix_points.append(fix_points_metrics)#加入一次大循环的  总共fix-point次数 n_sample个


    print(f"global average fast forward cnt: {np.average(fast_forward_and_fix_points_statistics['fast_forward'])}")#按大循环平均
    print(f"global average fix-point cnt: {np.average(fast_forward_and_fix_points_statistics['fix_points'])}")#按大循环平均
    print(f"global average fix-point per gram cnt: {np.average(fast_forward_and_fix_points_statistics['fix_points_per_gram'])}")#按小循环平均
    
    save_path = 'data/speedup_profiling_results/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    new_file_path= f'calvin_speedup_profiling_results_{args.max_new_tokens}_{args.max_new_seq_len}_{args.data_size}_stats.json'
    fast_forward_and_fix_points_statistics_file = os.path.join(save_path, new_file_path)

    with open(fast_forward_and_fix_points_statistics_file, 'w') as f:
        json.dump(fast_forward_and_fix_points_statistics, f, indent=4)
    
    ar_time_speed = ar_time_speed[1:]
    jacobian_time_speed = jacobian_time_speed[1:]
    print(f'ar speed: {ar_time_speed}')
    print(f'jacobian speed: {jacobian_time_speed}')
    print(f'The max speed of model {args.test_model_path} using jacobian iteration (max_new_tokens: {max_new_tokens}) is {max(jacobian_time_speed)}')
    print(f'The min speed of model {args.test_model_path} using jacobian iteration (max_new_tokens: {max_new_tokens}) is {min(jacobian_time_speed)}')
    print(f'The avg speed of model {args.test_model_path} using jacobian iteration (max_new_tokens: {max_new_tokens}) is {sum(jacobian_time_speed)/len(jacobian_time_speed)}')
    print(f'The max speed of model {args.test_model_path} using ar is {max(ar_time_speed)}')
    print(f'The min speed of model {args.test_model_path} using ar is {min(ar_time_speed)}')
    print(f'The avg speed of model {args.test_model_path} using ar is {sum(ar_time_speed)/len(ar_time_speed)}')

if __name__ == "__main__":  
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_filename", type=str,
                        default="eval/gsm8k/test.jsonl")
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--max_new_seq_len", type=int, default=1024)
    parser.add_argument("--test_model_path", type=str,
                        default="models/vicuna-7b-sharegpt-gpt4-48k")
    parser.add_argument("--action_stat", type=str,
                        default="/mnt/sda/wenxuansong/data/dataset/task_ABC_D/training/statistics.yaml")
    parser.add_argument("--image_folder", type=str,
                        default="/home/wenxuansong/chenjy/data/calvin_cvla/task_ABC_D/vla_processed_r5")
    parser.add_argument("--model_path", type=str,
                        default="/home/wenxuansong/chenjy/project/vlas/LLaVA/checkpoints/llava-v1.5-7b-calvin-rel-obs-reduce5-v1-abc2d_2024_01_29")                
    parser.add_argument("--teacher_model_path", type=str,
                        default="cllm/consistency-llm-7b-sharegpt48k")
    parser.add_argument("--data_size", type=str,
                        default=500)
    args = parser.parse_args() 
    action_stat = args.action_stat
    image_folder = args.image_folder
    model_path=args.model_path
    vla_model_initialization = CLLMRobotgenerate(model_path,action_stat,image_folder)
    
    speed_compare(args,vla_model_initialization)