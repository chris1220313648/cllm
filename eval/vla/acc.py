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
from flask import Flask, jsonify, request, Response
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
import time


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
        input_ids, 
        image_tensor,
        model, 
        tokenizer,
        max_new_tokens, 
        max_new_seq_len, 
        vla_model_initialization, 
        device, 
        ):
    
    # print("input_ids.shape:",input_ids.shape)
    # print("image_tensor.shape:",image_tensor.shape)
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
            actions = []
            # print("total_token_len:",total_token_len)
            output_ids=generation[0,total_token_len-35 :total_token_len].cpu().numpy().tolist()
            for elem in output_ids:
                actions.append(vla_model_initialization.action_tokenizer.decode_token_ids_to_actions(elem))
            # actions = np.array(actions)
            return actions
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
    parser.add_argument("--port", type=int, default=9010)
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
    parser.add_argument("--action_stat", type=str,
                        default="/mnt/sda/wenxuansong/data/dataset/task_ABC_D/training/statistics.yaml")
    parser.add_argument("--image_folder", type=str,
                        default="/home/wenxuansong/chenjy/data/calvin_cvla/task_ABC_D/vla_processed_r5")
    parser.add_argument("--model_path", type=str,
                        default="/home/wenxuansong/chenjy/project/vlas/LLaVA/checkpoints/llava-v1.5-7b-calvin-rel-obs-reduce5-v1-abc2d_2024_01_29")   
    args = parser.parse_args()
    max_new_token = args.max_tokens
    action_stat = args.action_stat
    image_folder = args.image_folder
    model_path=args.model_path
    vla_model_initialization = CLLMRobotgenerate(model_path,action_stat,image_folder)
    model=vla_model_initialization.vla_model
    action_tokenizer=vla_model_initialization.action_tokenizer
    tokenizer=vla_model_initialization.tokenizer
    print('>>>>>> model and tokenizer loaded')
    not_cvla=False
    flask_app = Flask(__name__)
    @flask_app.route("/predict", methods=["POST"])
    def predict():
        if request.method == "POST":
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

            input_ids, image_tensor = vla_model_initialization.compose_robot_input_for_calvin(
                img_static, img_gripper, instruction, robot_obs
            )
            if not_cvla:
                action = vla_model_initialization.robot_action_generate(input_ids, image_tensor)
                print(action)
                return jsonify(action.tolist())
            else :
                time1 = time.time()
                output_ids_cvla = consistency_generate_cvla(
                    input_ids=input_ids,
                    image_tensor=image_tensor,
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=args.max_new_tokens_for_consistency,
                    max_new_seq_len=max_new_token,
                    vla_model_initialization=vla_model_initialization,
                    device=model.device
                )
                time2 = time.time()
                print("一致性生成时间:",time2-time1)
                # print("output_ids_cvla.dtype:",output_ids_cvla.dtype)
                output_ids_cvla_list = output_ids_cvla
                print(output_ids_cvla_list)
                return jsonify(output_ids_cvla_list)

    flask_app.run(host="0.0.0.0", port=args.port)
    # if args.eval_only == False:
    #     # part 1 we set the model and tokenizer
    #     # model = transformers.AutoModelForCausalLM.from_pretrained(
    #     #     args.model_dir,
    #     #     torch_dtype=torch.bfloat16,
    #     #     low_cpu_mem_usage=True,
    #     #     device_map='cuda',
    #     # )
    #     # tokenizer = transformers.AutoTokenizer.from_pretrained(
    #     #     args.model_dir,
    #     #     padding_side="right",
    #     #     use_fast=False,
    #     # )
    #     model=vla_model_initialization.vla_model
    #     action_tokenizer=vla_model_initialization.action_tokenizer
    #     tokenizer=vla_model_initialization.tokenizer
    #     print('>>>>>> model and tokenizer loaded')
 
    #     # part 2 we prepare raw queries and wrap them with target prompt
    #     test_file="/home/wenxuansong/chenjy/project/vlas/LLaVA/playground/calvin_data/test.json"
    #     # raw_queries = get_raw_inputs(args.dev_set)
    #     # prompt = prompt_mapping[args.prompt_type]
    #     # processed_prompts = [prompt.format(input=query) for query in raw_queries]
    #     # processed_prompts = processed_prompts[:args.sample_num] if args.sample_num > 0 else processed_prompts
    #     with open(test_file, "r", encoding="utf-8") as f:
    #         data = json.load(f)
    
    #     processed_data=vla_model_initialization.preprocess_vla_data(data, tokenizer,action_tokenizer)
        
    #     results=[]
    #     # part 3 we generate answers
    #     for raw_data in tqdm(processed_data):
    #         # print(raw_data)
    #         input_ids = torch.Tensor(raw_data['sources_input_ids']).squeeze(0).to(device=model.device, dtype=torch.int)#2d
    #         label_ids=raw_data["labels_ids"]
    #         image_tensor=raw_data["image_tensor"].to(device=model.device, dtype=torch.float32)
           
    #         # if args.use_consistency_decoding:
    #         output_ids_cvla = consistency_generate_cvla(
    #             raw_data=raw_data,
    #             model=model,
    #             tokenizer=tokenizer,
    #             max_new_tokens=args.max_new_tokens_for_consistency,
    #             max_new_seq_len=max_new_token,
    #             vla_model_initialization=vla_model_initialization,
    #             device=model.device
    #         )
    #         output_ids_cvla = output_ids_cvla.unsqueeze(dim=0)
    #             # print("output_ids:",output_ids)
    #         # else:
    #         output_ids_vla = vla_model_initialization.robot_action_generate(
    #             input_ids,
    #             image_tensor
    #             # do_sample=False,
    #             # temperature=args.temperature,
    #             # max_new_tokens=max_new_token,
    #         )
    #         output_ids_vla=torch.tensor(output_ids_vla).unsqueeze(dim=0)
    #         # print("output_ids.shape:",output_ids.shape)
    #         # output_ids_tmp=output_ids
    #         # if model.config.is_encoder_decoder:
    #         #     output_ids = output_ids[0]
    #         # else:
    #         #     output_ids = output_ids[0][len(input_ids[0]) :]

    #         # output = tokenizer.decode(
    #         #     output_ids,
    #         #     spaces_between_special_tokens=False,
    #         # )
    #         # for special_token in tokenizer.special_tokens_map.values():
    #         #     if isinstance(special_token, list):
    #         #         for special_tok in special_token:
    #         #             output = output.replace(special_tok, "")
    #         #     else:
    #         #         output = output.replace(special_token, "")
    #         # output = output.strip()
    #         results.append({'output_ids_cvla': output_ids_cvla[:,-35:],'output_ids_vla':output_ids_vla ,'label_ids': label_ids})
    #     print('>>>>>> generation done')

    #     # part 5 we save the results, always be {'id':id,'response':response}
    #     # if dir of output file is not exist, it will be created automatically
    #     # Writing predictions to the output file
    #     with open(args.output_file_name, "w") as f:
    #         for idx, result in enumerate(results):
    #             # Convert tensors to lists for JSON serialization
    #             output_data = {
    #                 "id": idx,
    #                 "output_ids_cvla": result["output_ids_cvla"].tolist(),
    #                 "output_ids_vla":result["output_ids_vla"].tolist(),
    #                 "label_ids": result["label_ids"].tolist()[:-1],
    #             }
    #             # Write formatted JSON with indentation
    #             f.write(json.dumps(output_data) + "\n")

    #         print(">>>>>> Writing predictions completed successfully.")

    # # part 6 evaluate, I guess this should be done in a separate script
    # # 读取jsonline文件的步骤

    #     id_diffs = calculate_absolute_difference(args.output_file_name)
    #     print(id_diffs)


    # # get_results(args.output_file_name, args.dev_set)
    # # print('>>>>>> evaluation done')


# CUDA_VISIBLE_DEVICES=0 acc.py --model_dir path_to_cllm --temperature 0.0 --top_p 1.0 --output_file_name 'cllm_generated_gsm8k.jsonl' --dev_set "gsm8k" --prompt_type math-single --max_new_tokens_for_consistency 16 --max_tokens 1024 --use_consistency_decoding