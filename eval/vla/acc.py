import transformers
import os
import re
import json
import jsonlines
import argparse
import torch
from tqdm import tqdm
import sys
import pdb
import random
from math_normalization import *

###
from cllm.train_cvla_global import CLLMRobotgenerate
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
def consistency_generate(
    model,
    tokenizer,
    inputs,
    max_new_tokens,
    max_new_seq_len
    ):
    itr = 0
    while True:
        if itr == 0:
            input_ids = inputs['input_ids']
            input_masks = inputs['attention_mask']
        else:
            input_masks = torch.ones_like(input_ids).to(input_ids.device)
            for j in range(bsz):
                input_masks[j][torch.sum(inputs["attention_mask"], dim=-1)[j] + itr*max_new_tokens:] = 0

        bsz = input_ids.shape[0]
        eos_reached = torch.tensor([False] * bsz, device="cuda")
        generation = get_jacobian_trajectory(model, tokenizer, input_ids, input_masks, max_new_tokens)
        ### tokens generated after <eos> are set to <pad>
        for j in range(bsz):
            prompt_len = torch.sum(input_masks, dim=-1)
            eos_positions = torch.where(generation[j]==tokenizer.eos_token_id)[0]
            if len(eos_positions)==0:
                # no EOS, continue to the next item in the batch
                total_token_len = prompt_len + max_new_tokens
                continue
            # otherwise, set tokens coming after EOS as pad 
            eos_reached[j] = True
            total_token_len = int(eos_positions[0])

        ### see if next max_new_tokens should be generated & if True, update weights and prepare new input_ids
        itr+=1      
        if all(eos_reached) or itr*max_new_tokens >= max_new_seq_len:
            return generation[0, :total_token_len]
        input_ids = generation

def consistency_generate_cvla(
        raw_data, 
        model, 
        tokenizer,
        max_new_tokens, 
        max_new_seq_len, 
        vla_model_initialization, 
        device, 
        ):
    

    input_ids = torch.Tensor(raw_data['sources_input_ids']).squeeze(0).to(device=device, dtype=torch.int)
    image_tensor = raw_data['image_tensor'].unsqueeze(0).to(device=device, dtype=torch.int)
    bsz = input_ids.shape[0]
    itr = 0
    eos_reached = False
    while itr * max_new_tokens < max_new_seq_len and not eos_reached:


        attention_mask = torch.full_like(input_ids, 1, dtype=torch.int).to(device)
        print("input_ids.shape:",input_ids.shape)


        jacobian_trajectory_ids, teacher_logits, eos_reached = get_vla_jacobian_trajectory(
            model, tokenizer, input_ids, attention_mask, max_new_tokens, image_tensor, vla_model_initialization
        )
        # print("jacobian_trajectory_ids:",jacobian_trajectory_ids)
        generation=jacobian_trajectory_ids[-1]#[bsz,len]
        # print("generation.shape:",generation.shape)
               ### tokens generated after <eos> are set to <pad>
        for j in range(bsz):
            prompt_len = torch.sum(input_ids, dim=-1)
            eos_positions = torch.where(generation[j]==tokenizer.eos_token_id)[0]
            print("eos_positions:",eos_positions)
            if len(eos_positions)==0:
                # no EOS, continue to the next item in the batch
                total_token_len = prompt_len + max_new_tokens
                continue
            # otherwise, set tokens coming after EOS as pad 
            # eos_reached[j] = True
            total_token_len = int(eos_positions[0])

        ### see if next max_new_tokens should be generated & if True, update weights and prepare new input_ids
        itr+=1      
        if eos_reached or itr*max_new_tokens >= max_new_seq_len:
            return generation[0, :total_token_len]
        input_ids = generation
    
@torch.inference_mode()
def get_jacobian_trajectory(
    model,
    tokenizer,
    input_ids,
    attention_mask,
    max_new_tokens
):

    bsz = input_ids.shape[0] 
    prompt_len = [torch.sum(t) for t in attention_mask]
    max_prompt_len = max(prompt_len)
    total_len = max_prompt_len + max_new_tokens

    # initialize the first point of jacobian trajectory
    tokens = torch.full((bsz, total_len), tokenizer.pad_token_id, dtype=torch.long, device="cuda")
    for i in range(bsz):
        tokens[i, :] = torch.tensor(random.choices(input_ids[i][attention_mask[i]==1], k=total_len), dtype=torch.long, device="cuda")
        tokens[i, : prompt_len[i]] = torch.tensor(input_ids[i][: prompt_len[i]], dtype=torch.long, device="cuda")
    itr = 0
    next_generation = tokens
    generate_attention_mask = torch.full_like(next_generation, 1).to(tokens.device)
    while True:

        current_generation = next_generation
        with torch.no_grad():
            logits = model(current_generation, generate_attention_mask).logits
        next_generation = torch.argmax(torch.nn.functional.softmax(logits, dim=-1), dim=-1)

        # hold prompt unchanged and update generated tokens
        for i in range(bsz):
            next_generation[i, :] = torch.cat((tokens[i, :prompt_len[i]], next_generation[i, prompt_len[i]-1:total_len-1]), dim=0)
        if torch.all(torch.eq(next_generation, current_generation)).item():
            print(f"Iteration steps: {itr}")
            return next_generation # right generation is saved twice so we delete the last element of trajectory list
        itr+=1
####### Get jacobian trajectory #######
@torch.inference_mode()
def get_vla_jacobian_trajectory(#bsz一般是1,这里传进来的是vla模型
    model,
    tokenizer,
    input_ids,
    attention_mask,
    max_new_tokens,
    image_tensor,
    vla_model_initialization
    ):

    bsz = input_ids.shape[0]
    prompt_len = [torch.sum(t) for t in attention_mask]
    max_prompt_len = max(prompt_len)
    # print("max_prompt_len:",max_prompt_len)
    total_len = max_prompt_len + max_new_tokens

    # initialize the first point of jacobian trajectory
    tokens = torch.full((bsz, total_len), tokenizer.pad_token_id, dtype=torch.long, device="cuda")
    valid_tokens = input_ids[0][input_ids[0] != IMAGE_TOKEN_INDEX].tolist()
    # print("valid_tokens",len(valid_tokens))
    #     tokens[0, :] = torch.tensor(random.choices(valid_tokens, k=total_len)).to(dtype=torch.long, device="cuda")
    #     tokens[0, : prompt_len] = torch.tensor(input_ids[0][: prompt_len], dtype=torch.long, device="cuda")
    for i in range(bsz):
        valid_tokens = input_ids[i][input_ids[i] != IMAGE_TOKEN_INDEX].tolist()
        tokens[i, :] = torch.tensor(random.choices(valid_tokens, k=total_len)).to(dtype=torch.long, device="cuda")#把顺序打乱
        tokens[i, : prompt_len[i]] = torch.tensor(input_ids[i][: prompt_len[i]], dtype=torch.long, device="cuda")#保持prompt不变
        # tokens[i, :] = torch.tensor(random.choices(input_ids[i][attention_mask[i]==1], k=total_len)).to(dtype=torch.long, device="cuda")#把顺序打乱
        # tokens[i, : prompt_len[i]] = torch.tensor(input_ids[i][: prompt_len[i]], dtype=torch.long, device="cuda")#保持prompt不变
    trajectory = []
    logits_trajectory = []
    next_generation = tokens
    generate_attention_mask = torch.full_like(next_generation, 1).to(model.device)
    trajectory.append(tokens)
    itr=0
    while True:
        
        # current_generation = next_generation
        current_generation=next_generation.clone()
        # print("current_generation.dtype:",current_generation.dtype)
        input_embeds = vla_model_initialization.embedding_generate(current_generation, image_tensor)
        logits = model(input_ids=None, inputs_embeds=input_embeds).logits #这里的model是一个llavallamaforcasualLM
        logits_trajectory.append(logits)
        next_generation = torch.argmax(torch.nn.functional.softmax(logits, dim=-1) / 0.01, dim=-1)
        # print("next_generation.shape:",next_generation.shape)
        # print("next_generation.shape:",next_generation.shape)
        # print("prompt_len:",prompt_len)
        # print("total_len:",total_len)
        # hold prompt unchanged and update generated tokens
        tmp = torch.zeros((bsz, total_len),device=next_generation.device).to(torch.long)
        for i in range(bsz):
            # import pdb;pdb.set_trace()
            # next_generation[i, :] = torch.cat((tokens[i, :prompt_len[i]], next_generation[i, prompt_len[i]-1:total_len-1]), dim=0)
            tmp[i, :] = torch.cat((tokens[i, :prompt_len[i]], next_generation[i, -max_new_tokens-1:-1]), dim=0)
        next_generation=tmp
        trajectory.append(next_generation)
        if torch.all(torch.eq(next_generation, current_generation)).item():
            eos_reached = len(torch.where(trajectory[-1] == tokenizer.eos_token_id)[0])>0
            # print(trajectory[:-1])
            return trajectory[:-1], logits_trajectory[-1], eos_reached # converged generation is saved twice so we delete the last element of trajectory list
        itr+=1

def get_results(pred_file, dev_set):
    def test_answer(pred_str, ans_str):
        pattern = "#### (.*)$"

        if "Question" in pred_str:
            pred_str = pred_str.split("Question")[0]

        preds = re.findall(pattern, pred_str)
        pred = preds[-1] if len(preds) >= 1 else ""
        if "</s>" in pred:
            pred = pred[:-4]

        gold = ans_str
        pred = normalize_final_answer(pred)
        gold = normalize_final_answer(gold)
        return check_sympy_equivalence(gold, pred), pred, gold

    def parse_pred_ans(preds_str, golds_str, properties_list):
        num_q = 0
        acc = 0
        results = []
        preds = []
        golds = []
        correct_table = {}
        cnt_table = {}
        source_set = set()
        for pred_str, gold_str, properties in tqdm(zip(preds_str, golds_str, properties_list), total=len(preds_str)):
            num_q += 1
            result, pred, gold = test_answer(pred_str, gold_str)
            results.append(result)
            preds.append(pred)
            golds.append(gold)
            if result:
                acc += 1
            source = properties['source']
            tag = properties['tag']
            source_set.add(source)
            if source not in correct_table.keys():
                correct_table[source] = 1 if result else 0
                cnt_table[source] = 1
            else:
                correct_table[source] = (correct_table[source] + 1) if result else correct_table[source]
                cnt_table[source] += 1
            for key in tag.keys():
                value = tag[key]
                value = source+","+key+"__"+value
                if value not in correct_table.keys():
                    correct_table[value] = 1 if result else 0
                    cnt_table[value] = 1
                else:
                    correct_table[value] = (correct_table[value] + 1) if result else correct_table[value]
                    cnt_table[value] += 1
        print('num_q %d correct %d ratio %.4f' % (num_q, acc, float(acc / num_q)))
        acc_table = {}
        for key in correct_table.keys():
            acc_table[key] = correct_table[key] / cnt_table[key]
        acc_table = list(zip(acc_table.keys(), acc_table.values()))
        acc_table.sort(key=lambda x: x[0])
        for key, acc in acc_table:
            if key in source_set:
                print(key+" : "+str(acc))
            else:
                print("    " + key.split(",")[-1]+ " : " + str(acc))
        return results, preds, golds

    if dev_set in ['all', 'gsm8k', 'math', 'mathgpt', 'gsm8k_robust']:
        golds_str = []
        properties = []
        with open(f'test.jsonl', 'r', encoding='utf-8') as f:
            for line in f:
                if dev_set != "all":
                    if json.loads(line)['source'].lower() == dev_set:
                        golds_str.append(json.loads(line)['target'])
                        properties.append({"source": json.loads(line)['source'], "tag": json.loads(line)['tag']})
                else:
                    golds_str.append(json.loads(line)['target'])
                    properties.append({"source": json.loads(line)['source'], "tag": json.loads(line)['tag']})
        preds_str = []
        with open(pred_file, 'r', encoding='utf-8') as f:
            for line in f:
                preds_str.append(json.loads(line)['response'])
        results, preds, golds = parse_pred_ans(preds_str, golds_str, properties)
        with open(pred_file, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
        for i, line in enumerate(data):
            line['pred'] = preds[i]
            line['gold'] = golds[i]
            line['result'] = results[i]

        # Save the updated list of dictionaries back to the jsonl file
        with open(pred_file, 'w') as file:
            for item in data:
                file.write(json.dumps(item) + '\n')

    else:
        raise NotImplementedError("Evaluation not supported.")


def get_raw_inputs(dev_set):
    # in this function, we will get the raw queries for a target dev set
    data = []
    if dev_set in ['all', 'gsm8k', 'math', 'mathgpt', 'gsm8k_robust']:
        with open(f'test.jsonl') as f:
            for line in jsonlines.Reader(f):
                data.append(line)
        if dev_set != 'all':
            data = [line for line in data if line['source'].lower() == dev_set]
    else:
        raise ValueError

    prompt_list = [line['question'] for line in data]
    return prompt_list


prompt_mapping = {
    "math-single": "Question:\n{input}\nAnswer:\nLet's think step by step.\n",
}
# 计算差的绝对值的和
def calculate_absolute_difference(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        json_lines = file.readlines()
    id_diffs = []

    for line in json_lines:
        # 解析每行的JSON数据
        data = json.loads(line)
        id = data["id"]
        output_cvla = data["output_ids_cvla"][0]
        output_vla = data["output_ids_vla"][0]
        labels = data["label_ids"]

        # 计算output_cvla与label_ids之间的差的绝对值和
        cvla_diff = sum(abs(o - l) for o, l in zip(output_cvla, labels))
        # 计算output_vla与label_ids之间的差的绝对值和
        vla_diff = sum(abs(o - l) for o, l in zip(output_vla, labels))

        # 保存每个id的差异结果
        id_diffs.append({
            "id": id,
            "cvla_diff": cvla_diff,
            "vla_diff": vla_diff
        })
    
    return id_diffs
if __name__ == '__main__':
    # set args
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--max_tokens', type=int, default=2048)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--presence_penalty', type=float, default=0.0)
    parser.add_argument('--frequency_penalty', type=float, default=0.0)
    parser.add_argument('--output_file_name', type=str, default='/home/wenxuansong/chenjy/project/Consistency_LLM/eval/vla/sava_file/output.jsonl')
    parser.add_argument('--test_file', type=str, default="/home/wenxuansong/chenjy/project/vlas/LLaVA/playground/calvin_data/test.json")
    parser.add_argument('--stop', type=str, nargs='+', default=[], help="you can pass one or multiple stop strings to halt the generation process.")
    parser.add_argument('--dev_set', type=str, default='all')
    parser.add_argument('--prompt_type', type=str, default='math-single')
    parser.add_argument('--sample_num', type=int, default=-1, )
    parser.add_argument('--eval_only', action="store_true")
    parser.add_argument('--max_num_batched_tokens', type=int, default=2048)
    parser.add_argument(
        "--use_consistency_decoding",
        action="store_true",
        help="Whether to use consistency decoding",
    )
    parser.add_argument(
        "--max_new_tokens_for_consistency",
        type=int,
        default=16,
        help="The n-gram for consistency decoding.",
    ) 
    args = parser.parse_args()
    max_new_token = args.max_tokens
    action_stat = "/mnt/sda/wenxuansong/data/dataset/task_ABC_D/training/statistics.yaml"
    image_folder = "/home/wenxuansong/chenjy/data/calvin_cvla/task_ABC_D/vla_processed_r5"
    model_path="/home/wenxuansong/chenjy/project/vlas/LLaVA/checkpoints/llava-v1.5-7b-calvin-rel-obs-reduce5-v1-abc2d_2024_12_25"
    vla_model_initialization = CLLMRobotgenerate(model_path,action_stat,image_folder)
    if args.eval_only == False:
        # part 1 we set the model and tokenizer
        # model = transformers.AutoModelForCausalLM.from_pretrained(
        #     args.model_dir,
        #     torch_dtype=torch.bfloat16,
        #     low_cpu_mem_usage=True,
        #     device_map='cuda',
        # )
        # tokenizer = transformers.AutoTokenizer.from_pretrained(
        #     args.model_dir,
        #     padding_side="right",
        #     use_fast=False,
        # )
        model=vla_model_initialization.vla_model
        action_tokenizer=vla_model_initialization.action_tokenizer
        tokenizer=vla_model_initialization.tokenizer
        print('>>>>>> model and tokenizer loaded')

        # part 2 we prepare raw queries and wrap them with target prompt
        test_file="/home/wenxuansong/chenjy/project/vlas/LLaVA/playground/calvin_data/test.json"
        # raw_queries = get_raw_inputs(args.dev_set)
        # prompt = prompt_mapping[args.prompt_type]
        # processed_prompts = [prompt.format(input=query) for query in raw_queries]
        # processed_prompts = processed_prompts[:args.sample_num] if args.sample_num > 0 else processed_prompts
        with open(test_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    
        processed_data=vla_model_initialization.preprocess_vla_data(data, tokenizer,action_tokenizer)
        results=[]
        # part 3 we generate answers
        for raw_data in tqdm(processed_data):
            # print(raw_data)
            input_ids = torch.Tensor(raw_data['sources_input_ids']).squeeze(0).to(device=model.device, dtype=torch.int)
            label_ids=raw_data["labels_ids"]
            image_tensor=raw_data["image_tensor"].to(device=model.device, dtype=torch.int)
           
            # if args.use_consistency_decoding:
            output_ids_cvla = consistency_generate_cvla(
                raw_data=raw_data,
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=args.max_new_tokens_for_consistency,
                max_new_seq_len=max_new_token,
                vla_model_initialization=vla_model_initialization,
                device=model.device
            )
            output_ids_cvla = output_ids_cvla.unsqueeze(dim=0)
                # print("output_ids:",output_ids)
            # else:
            output_ids_vla = vla_model_initialization.robot_action_generate(
                input_ids,
                image_tensor
                # do_sample=False,
                # temperature=args.temperature,
                # max_new_tokens=max_new_token,
            )
            output_ids_vla=torch.tensor(output_ids_vla).unsqueeze(dim=0)
            # print("output_ids.shape:",output_ids.shape)
            # output_ids_tmp=output_ids
            # if model.config.is_encoder_decoder:
            #     output_ids = output_ids[0]
            # else:
            #     output_ids = output_ids[0][len(input_ids[0]) :]

            # output = tokenizer.decode(
            #     output_ids,
            #     spaces_between_special_tokens=False,
            # )
            # for special_token in tokenizer.special_tokens_map.values():
            #     if isinstance(special_token, list):
            #         for special_tok in special_token:
            #             output = output.replace(special_tok, "")
            #     else:
            #         output = output.replace(special_token, "")
            # output = output.strip()
            results.append({'output_ids_cvla': output_ids_cvla[:,-35:],'output_ids_vla':output_ids_vla ,'label_ids': label_ids})
        print('>>>>>> generation done')

        # part 5 we save the results, always be {'id':id,'response':response}
        # if dir of output file is not exist, it will be created automatically
        # Writing predictions to the output file
        with open(args.output_file_name, "w") as f:
            for idx, result in enumerate(results):
                # Convert tensors to lists for JSON serialization
                output_data = {
                    "id": idx,
                    "output_ids_cvla": result["output_ids_cvla"].tolist(),
                    "output_ids_vla":result["output_ids_vla"].tolist(),
                    "label_ids": result["label_ids"].tolist()[:-1],
                }
                # Write formatted JSON with indentation
                f.write(json.dumps(output_data) + "\n")

            print(">>>>>> Writing predictions completed successfully.")

    # part 6 evaluate, I guess this should be done in a separate script
    # 读取jsonline文件的步骤

        id_diffs = calculate_absolute_difference(args.output_file_name)
        print(id_diffs)


    # get_results(args.output_file_name, args.dev_set)
    # print('>>>>>> evaluation done')


# CUDA_VISIBLE_DEVICES=0 acc.py --model_dir path_to_cllm --temperature 0.0 --top_p 1.0 --output_file_name 'cllm_generated_gsm8k.jsonl' --dev_set "gsm8k" --prompt_type math-single --max_new_tokens_for_consistency 16 --max_tokens 1024 --use_consistency_decoding