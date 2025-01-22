import json
from transformers import AutoTokenizer, LlamaForCausalLM
import torch
from tqdm import tqdm
import random
import argparse
import transformers
import json
from typing import Optional, Dict, Sequence
import os, sys
import json
import argparse
import numpy as np

def get_default_question(cllm_type):
    if cllm_type == 'sharegpt':
        return "Which methods did Socrates employ to challenge the prevailing thoughts of his time?"
    elif cllm_type == 'spider':
        return "The SQL database has table named vehicle with columns ['Vehicle_ID', 'Model', 'Build_Year', 'Top_Speed', 'Power', 'Builder', 'Total_Production'], table named driver with columns ['Driver_ID', 'Name', 'Citizenship', 'Racing_Series'], table named vehicle_driver with columns ['Driver_ID', 'Vehicle_ID'], Question: What are the vehicle ids and models which have been driven by more than 2 drivers or been driven by the driver named 'Jeff Gordon'?"
    elif cllm_type == 'python':
        return "Implement the Conway's Game of Life. You should start with a 2D grid initialized with some configuration of live and dead cells. 1 for live cell and -1 for dead cell. The simulation should update the grid state by applying the rules for each cell simultaneously: any live cell with fewer than two live neighbors dies, as if by underpopulation. Any live cell with two or three live neighbors lives on to the next generation. Any live cell with more than three live neighbors dies, as if by overpopulation. Any dead cell with exactly three live neighbors becomes a live cell, as if by reproduction. initial_grid = [[0, 1, 0], [0, 0, 1], [1, 1, 1], [0, 0, 0]]"
    elif cllm_type == 'gsm8k':
        return "Poppy is solving a 1000-piece jigsaw puzzle. She places a quarter of the pieces on the board, then her mom places a third of the remaining pieces. How many jigsaw pieces are left to be placed?"
    else:
        return "Tell me a short story."

def get_system_prompt(cllm_type):
    if cllm_type == 'sharegpt':
        return "Answer in English unless other language is used. A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
    elif cllm_type == 'spider':
        return "Could you translate the following question into SQL. Please only generate SQL, don't include explanation in the answer.\n"
    elif cllm_type == 'python':
        return "Please generate code based on the following doc:\n"
    elif cllm_type == 'gsm8k':
        return ""
    else:
        return "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n"

def get_instruction_template(system_prompt, roles, model_input, cllm_type):
    if cllm_type == 'sharegpt':
        return system_prompt + f"{roles[0]}: " + f"{model_input}\n{roles[1]}: "
    if cllm_type == 'spider' or 'python':
        return f"### Instruction:\n" + system_prompt + f"{model_input}\n" + f"### Response:\n"
    if cllm_type == 'gsm8k':
        prompt_mapping = "Question:\n{input}\nAnswer:\nLet's think step by step.\n"
        return prompt_mapping.format(input=model_input)
    else:
        return system_prompt + f"{roles[0]}: " + f"{model_input}\n{roles[1]}: "
    

def detect_repetitive_patterns(tokenizer, prompt_ids, repeat_ngram_size):

    if len(prompt_ids.shape)==1:
        prompt_ids = prompt_ids
    elif len(prompt_ids.shape)==2:
        prompt_ids = prompt_ids[0]
    elif len(prompt_ids.shape)==3:
        prompt_ids = prompt_ids[0][0]
    else:
        print(f'Unexpected shape {prompt_ids.shape}! Please check prompt ids')
        assert False

    count = 1
    for i in range(1, len(prompt_ids)):
        if prompt_ids[i] == tokenizer.eos_token_id:
            break
        if prompt_ids[i] == prompt_ids[i - 1]:
            count += 1
            if count == repeat_ngram_size:
                return True
        else:
            count = 1

    return False

# def jacobian_generated_data_postprocessed(generated_data, model_path):
#     tokenizer = AutoTokenizer.from_pretrained(model_path)
#     low_quality_data_id_lst = []#记录低质量数据序号
#     # delete low quality data with repetitive pattern
#     for i, d in enumerate(generated_data):
#         if detect_repetitive_patterns(tokenizer, np.array(d['teacher_output_ids']), repeat_ngram_size=10):
#             prompt_ids = np.array(d['teacher_output_ids'])
#             if len(prompt_ids.shape)==2:
#                 prompt_ids = prompt_ids[0]
#             elif len(prompt_ids.shape)==3:
#                 prompt_ids = prompt_ids[0][0]
#             print(f'Low quality generation detected: {tokenizer.decode(prompt_ids)}')
#             low_quality_data_id_lst.append(i)
#     print(f'{len(low_quality_data_id_lst)} low quality data detected. {len(low_quality_data_id_lst)/len(generated_data)} percent of low quality data.')

#     # add complete teacher outputs
#     teacher_output_inspector = {}#记录第data_i的teacher_output，记录所有itr的
#     for d in generated_data: 
#         data_id = d["data_id"]
#         if data_id in teacher_output_inspector.keys():
#             all_teacher_output_map = teacher_output_inspector[data_id]
#         else:
#             all_teacher_output_map = {}#记录第itr的teacher_output
#             #print(data_id)
#         itr = d["jacobian_itr_id"]
#         # handle bsz=1 case only
#         all_teacher_output_map[itr] = d["teacher_output_ids"][0]#增加记录
#         teacher_output_inspector[data_id] = all_teacher_output_map
# # teacher_output_inspector = {
# #     "data_1": {
# #         "itr_1": [1, 2, 3, 4, 5, 6],
# #         "itr_2": [7, 8, 9, 10, 11, ，1，1，1，1，1，1，，]
# #     },
# #     "data_2": {
# #         "itr_1": [13, 14, 15, 16, 17, 18],
# #         "itr_2": [19, 20, 21, 22, 23, 24]
# #     }
# # }
#     teacher_output_collector = {}#记录记录第data_i的teacher_output，只记录最后itr的
#     for d_id in teacher_output_inspector.keys():
#         all_teacher_output_map = teacher_output_inspector[d_id]
#         all_itr = [int(s.split('_')[1]) for s in all_teacher_output_map.keys()]
#         print(all_itr)
#         max_itr = max(all_itr)
#         max_itr_s = "itr_" + str(max_itr)
#         complete_teacher_output = all_teacher_output_map[max_itr_s]
#         teacher_output_collector[d_id] = complete_teacher_output
# # teacher_output_collector = {
# #     "data_1": [7, 8, 9, 10, 11, 12，1，1，1，1，1，1，],
# #     "data_2": [19, 20, 21, 22, 23, 24]
# # }

#     f_result = []
#     for d in generated_data:#增加complete_teacher_output_ids项，理论上和label_id一样
#         data_id = d["data_id"]
#         complete_teacher_output = teacher_output_collector[data_id]
#         d["complete_teacher_output_ids"] = complete_teacher_output
#         f_result.append(d)
    
#     cleaned_f_result = []
#     for i, d in enumerate(generated_data):#清楚带有重复项目的数据
#         if i in low_quality_data_id_lst:
#             continue
#         cleaned_f_result.append(d)


#     return cleaned_f_result

#优化
def jacobian_generated_data_postprocessed(generated_data, model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    low_quality_data_id_lst = []  # 记录低质量数据序号
    
    # 删除低质量数据：检测重复模式
    for i, d in enumerate(generated_data):
        if detect_repetitive_patterns(tokenizer, np.array(d['teacher_output_ids']), repeat_ngram_size=10):
            prompt_ids = np.array(d['teacher_output_ids'])
            if len(prompt_ids.shape) == 2:
                prompt_ids = prompt_ids[0]
            elif len(prompt_ids.shape) == 3:
                prompt_ids = prompt_ids[0][0]
            
            print(f'Low quality generation detected: {tokenizer.decode(prompt_ids)}')
            low_quality_data_id_lst.append(i)
    
    print(f'{len(low_quality_data_id_lst)} low quality data detected. {len(low_quality_data_id_lst) / len(generated_data):.2%} percent of low quality data.')

    # 记录每个 data_id 对应的所有迭代的 teacher output
    teacher_output_inspector = {}  # 存储每个data_id对应所有itr的teacher_output
    for d in generated_data:
        data_id = d["data_id"]
        itr = d["jacobian_itr_id"]
        if data_id not in teacher_output_inspector:
            teacher_output_inspector[data_id] = {}
        teacher_output_inspector[data_id][itr] = d["teacher_output_ids"][0]
    
    # 记录每个 data_id 对应的最后一个 teacher output
    teacher_output_collector = {}  # 存储每个data_id对应最后一个itr的teacher_output
    for data_id, all_teacher_output_map in teacher_output_inspector.items():
        max_itr = max(int(s.split('_')[1]) for s in all_teacher_output_map.keys())
        max_itr_s = f"itr_{max_itr}"
        teacher_output_collector[data_id] = all_teacher_output_map[max_itr_s]

    # 处理生成的数据，清理低质量数据
    cleaned_f_result = []
    for i, d in enumerate(generated_data):
        if i not in low_quality_data_id_lst:
            data_id = d["data_id"]
            d["complete_teacher_output_ids"] = teacher_output_collector[data_id]
            cleaned_f_result.append(d)

    return cleaned_f_result
