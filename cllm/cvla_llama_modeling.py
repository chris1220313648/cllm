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

import torch.nn.functional as F
from transformers import LlamaModel,LlamaForCausalLM
import argparse

def delete_false_key_value(#删除后面错误的token
        self,
        num_of_false_tokens,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
   
        for layer_idx in range(len(self.key_cache)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx][..., :-num_of_false_tokens, :]
            self.value_cache[layer_idx] = self.value_cache[layer_idx][..., :-num_of_false_tokens, :]

@torch.inference_mode()
def jacobi_forward(
    self,
    input_ids: torch.LongTensor = None,
    tokenizer=None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    use_cache: Optional[bool] = None,
    max_new_tokens: Optional[int] = None,
    prefill_phase: Optional[bool] = False,
    chat: Optional[bool] = False,
):
    assert use_cache == True

    if input_ids is not None:
        batch_size, seq_length = input_ids.shape[:2]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")
    
    if prefill_phase: # prefill phase, just compute the keys & values of prompt 获取并缓存prompt的key和value
        # self.model is the instance of class LlamaModel
        inputs_embeds = self.model.embed_tokens(input_ids)
        past_key_values_length = 0
        if use_cache:#获取转换后缓存的可用长度，通常这是指缓存中有效的历史信息的长度
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:#转换旧缓存
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length) 

        if position_ids is None:#生成新的位置编码
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if self.model._use_flash_attention_2:#生成注意力掩码
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self.model._use_sdpa :
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )
        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        for decoder_layer in self.model.layers:

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[1]

        hidden_states = self.model.norm(hidden_states)#最后一层输出

        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        predict_next_tokens = torch.argmax(torch.nn.functional.softmax(logits, dim=-1) / 0.001,  dim=-1)
        first_correct_token = predict_next_tokens[:, -1]#获取prompt最后一个生成的token，通常是结尾输入token的预测值，用作第一个预测对的值
        return next_decoder_cache, first_correct_token
    else: # generation phase, input as random_initilized point and output as fixed point
        jacobian_trajectory = []
        accurate_n_gram = torch.zeros_like(input_ids).to(input_ids.device)#maxtoken
        accurate_length = 0

        next_point = input_ids
        jacobian_trajectory.append(next_point)

        iter_counter = 0

        prev_len = 0
        while True:

            current_point = next_point
            inputs_embeds = self.model.embed_tokens(current_point)
            attention_mask = None
            position_ids = None
            seq_length = current_point.shape[1]
            if use_cache:#更新key value缓存
                use_legacy_cache = not isinstance(past_key_values, Cache)
                if use_legacy_cache:
                    past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                past_key_values_length = past_key_values.get_usable_length(seq_length) 


            # print("past_key_values_length:",past_key_values_length) # return previous_seq_length
            if position_ids is None:
                device = input_ids.device if input_ids is not None else inputs_embeds.device
                position_ids = torch.arange(
                    past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
                )#获取位置编码
                position_ids = position_ids.unsqueeze(0)

            if self.model._use_flash_attention_2:
                # 2d mask is passed through the layers
                attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
            elif self.model._use_sdpa :
                # output_attentions=True can not be supported when using SDPA, and we fall back on
                # the manual implementation that requires a 4D causal mask in all cases.
                attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                    attention_mask,
                    (batch_size, seq_length),
                    inputs_embeds,
                    past_key_values_length,
                )
            else:
                # 4d mask is passed through the layers
                attention_mask = _prepare_4d_causal_attention_mask(
                    attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
                )
            # embed positions
            hidden_states = inputs_embeds

            # decoder layers            
            for decoder_layer in self.model.layers:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    use_cache=use_cache,
                )

                hidden_states = layer_outputs[0]

            hidden_states = self.model.norm(hidden_states)

            if self.config.pretraining_tp > 1:#分布式预测
                lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
                logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
                logits = torch.cat(logits, dim=-1)
            else:
                logits = self.lm_head(hidden_states)

            logits = logits.float()
            all_shift_one_token = torch.argmax(torch.nn.functional.softmax(logits, dim=-1) / 0.001, dim=-1)
            #获取最后一层输出
            next_point = torch.cat((current_point[0, 0].view(1,-1), all_shift_one_token[0, :seq_length-1].view(1,-1)), dim=-1)
            #拼接current_point[0, 0]这个一定是对的 在加上预测的n-1个
            first_false_index = torch.where(torch.eq(current_point[0], next_point[0]) == False)[0]
            
            jacobian_trajectory.append(next_point)

            if len(first_false_index) > 0:
                fast_forward_cnt = first_false_index[0].item()

                past_key_values.delete_false_key_value(seq_length - fast_forward_cnt) # delete the false keys & values
            else:#没有遇到错误token
                fast_forward_cnt = torch.sum(torch.eq(current_point, next_point)).item()

                accurate_n_gram[0, accurate_length : accurate_length + fast_forward_cnt] = next_point[0, :fast_forward_cnt]         
                first_correct_token = all_shift_one_token[:,-1]   
                if chat:
                    if tokenizer.eos_token_id in accurate_n_gram[0, :accurate_length + fast_forward_cnt]:
                        eos_positions = torch.where(accurate_n_gram[0]==tokenizer.eos_token_id)[0]
                        eos_position = eos_positions[0]
                        generated_str = tokenizer.decode(accurate_n_gram[0, :eos_position], skip_special_tokens=True)
                    else:
                        generated_str = tokenizer.decode(accurate_n_gram[0, :accurate_length + fast_forward_cnt], skip_special_tokens=True)

                    print(generated_str[prev_len:], flush=True, end="")
                    prev_len = len(generated_str)
                break 

            accurate_n_gram[0, accurate_length : accurate_length + fast_forward_cnt] = next_point[0, :fast_forward_cnt]#保存已经正确的部分
            accurate_length += fast_forward_cnt
            next_point = next_point[0, fast_forward_cnt:].view(1,-1) # only false tokens should be re-generated更新 next_point，只保留错误的部分（即不一致的 token）用于下一次生成。

            if chat:
                if tokenizer.eos_token_id in accurate_n_gram[0, :accurate_length]:
                    eos_positions = torch.where(accurate_n_gram[0]==tokenizer.eos_token_id)[0]
                    eos_position = eos_positions[0]

                    generated_str = tokenizer.decode(accurate_n_gram[0, :eos_position], skip_special_tokens=True)
                else:
                    generated_str = tokenizer.decode(accurate_n_gram[0, :accurate_length], skip_special_tokens=True)

                print(generated_str[prev_len:], flush=True, end="")
                prev_len = len(generated_str)

            iter_counter += 1

        return accurate_n_gram, first_correct_token, iter_counter, accurate_length


@torch.inference_mode()
def cvla_jacobi_forward_profiling(
    self,
    input_ids: torch.LongTensor = None,
    images_tensor: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    use_cache: Optional[bool] = None,
    max_new_tokens: Optional[int] = None,
    prefill_phase: Optional[bool] = False,
):
    def embedding_generate( vla_model,input_ids, images_tensor,image_sizes: Optional[torch.Tensor] = None):
        position_ids=None
        attention_mask=None
        (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            )=  vla_model.prepare_inputs_labels_for_multimodal(                
                input_ids.cuda(),
                position_ids,
                attention_mask,
                None,
                None,
                images=images_tensor.to(dtype=torch.float16, device="cuda", non_blocking=True),#本来是float16
                image_sizes=image_sizes)
        return inputs_embeds,attention_mask
    
    assert use_cache == True

    if input_ids is not None:
        batch_size, seq_length = input_ids.shape[:2]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")
    
    if prefill_phase: # prefill phase, just compute the keys & values of prompt
        # self.model is the instance of class LlamaModel
        inputs_embeds,attention_mask = embedding_generate(self,input_ids,images_tensor)
        seq_length= inputs_embeds.shape[1]
        # attention_mask=attention_mask.squeeze(1)
        print("seq_length:",seq_length)
        # print("attention_mask:",attention_mask.shape)
        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length) #根据当前序列长度和缓存中的历史状态来计算出可以复用的部分

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if self.model._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self.model._use_sdpa :
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )
        # embed positions
        hidden_states = inputs_embeds

        # # decoder layers
        for decoder_layer in self.model.layers:

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[1]

        hidden_states = self.model.norm(hidden_states)

        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        predict_next_tokens = torch.argmax(torch.nn.functional.softmax(logits, dim=-1), dim=-1)
        first_correct_token = predict_next_tokens[:, -1]
        return next_decoder_cache, first_correct_token
    else: # generation phase, input as random_initilized point and output as fixed point
        jacobian_trajectory = []
        accurate_n_gram = torch.zeros_like(input_ids).to(input_ids.device)
        accurate_length = 0
        next_point = input_ids
        jacobian_trajectory.append(next_point)
        inputs_embeds,attention_mask = embedding_generate(self,input_ids,images_tensor)
        iter_counter = 0
        while True:           
            current_point = next_point
            inputs_embeds,attention_mask = embedding_generate(self,current_point,images_tensor)
            # inputs_embeds = self.model.embed_tokens(current_point)
            # attention_mask = None #原来是none
            position_ids = None
            # seq_length = current_point.shape[1]
            seq_length = inputs_embeds.shape[1]
            if use_cache:
                use_legacy_cache = not isinstance(past_key_values, Cache)
                if use_legacy_cache:
                    past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                past_key_values_length = past_key_values.get_usable_length(seq_length) 
            # print(past_key_values_length) # return previous_seq_length
            if position_ids is None:
                device = input_ids.device if input_ids is not None else inputs_embeds.device
                position_ids = torch.arange(
                    past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
                )
                position_ids = position_ids.unsqueeze(0)

            if self.model._use_flash_attention_2:
                # 2d mask is passed through the layers
                attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
            elif self.model._use_sdpa :
                # output_attentions=True can not be supported when using SDPA, and we fall back on
                # the manual implementation that requires a 4D causal mask in all cases.
                attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                    attention_mask,
                    (batch_size, seq_length),
                    inputs_embeds,
                    past_key_values_length,
                )
            else:
                # 4d mask is passed through the layers
                attention_mask = _prepare_4d_causal_attention_mask(
                    attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
                )
            # embed positions
            hidden_states = inputs_embeds

            # decoder layers            
            for decoder_layer in self.model.layers:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    use_cache=use_cache,
                )

                hidden_states = layer_outputs[0]

            hidden_states = self.model.norm(hidden_states)

            if self.config.pretraining_tp > 1:
                lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
                logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
                logits = torch.cat(logits, dim=-1)
            else:
                logits = self.lm_head(hidden_states)

            logits = logits.float()
            all_shift_one_token = torch.argmax(torch.nn.functional.softmax(logits, dim=-1) / 0.01, dim=-1)
            next_point= torch.cat((current_point[0, 0].view(1,-1), all_shift_one_token[0, -max_new_tokens:-1].view(1,-1)), dim=-1)#2维度 [1,16]
            print("next_point.shape:",next_point.shape)
            jacobian_trajectory.append(next_point)
            
            if torch.all(torch.eq(current_point, next_point)).item():    
                print('Successfully break!')
                # print(next_point)#16
                first_correct_token = torch.argmax(torch.nn.functional.softmax(logits, dim=-1), dim=-1)[:,-1]#2维
                break
            past_key_values.delete_false_key_value(seq_length)

            iter_counter += 1

        return jacobian_trajectory[:-1], next_point, first_correct_token, iter_counter