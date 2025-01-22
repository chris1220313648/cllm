import transformers
import torch
from transformers import Trainer
# from transformers.trainer_pt_utils import LabelSmoother
import wandb
import random
from torch.utils.data import DataLoader
import numpy as np
from typing import Optional, Dict, Sequence
import os

from llava import conversation as conversation_lib
from llava.model.builder import load_pretrained_model
from llava.mm_utils import (
    tokenizer_image_token,
    process_images,
    get_model_name_from_path,
)
from llava.action_tokenizer import ActionTokenizer, encode_robot_obs
from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_AUDIO_TOKEN,IMAGE_TOKEN_INDEX
from llava.utils import disable_torch_init

class LabelSmoother:
    """
    Adds label-smoothing on a pre-computed output from a Transformers model.

    Args:
        epsilon (`float`, *optional*, defaults to 0.1):
            The label smoothing factor.
        ignore_index (`int`, *optional*, defaults to -100):
            The index in the labels to ignore when computing the loss.
    """

    epsilon: float = 0.1
    ignore_index: int = -100

    def __call__(self, logits, labels, shift_labels=False):
        # logits = model_output["logits"] if isinstance(model_output, dict) else model_output[0]
        if shift_labels:
            logits = logits[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()

        log_probs = -torch.nn.functional.log_softmax(logits, dim=-1)
        if labels.dim() == log_probs.dim() - 1:
            labels = labels.unsqueeze(-1)

        padding_mask = labels.eq(self.ignore_index)
        # In case the ignore_index is -100, the gather will fail, so we replace labels by 0. The padding_mask
        # will ignore them in any case.
        labels = torch.clamp(labels, min=0)
        nll_loss = log_probs.gather(dim=-1, index=labels)
        # works for fp16 input tensor too, by internally upcasting it to fp32
        smoothed_loss = log_probs.sum(dim=-1, keepdim=True, dtype=torch.float32)

        nll_loss.masked_fill_(padding_mask, 0.0)
        smoothed_loss.masked_fill_(padding_mask, 0.0)

        # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
        num_active_elements = padding_mask.numel() - padding_mask.long().sum()
        nll_loss = nll_loss.sum() / num_active_elements
        smoothed_loss = smoothed_loss.sum() / (num_active_elements * log_probs.shape[-1])
        return (1 - self.epsilon) * nll_loss + self.epsilon * smoothed_loss

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

class CllmTrainer(Trainer):
    def __init__(self, vla_model_initialization, *args, **kwargs):
        super().__init__(*args, **kwargs)
        args = kwargs["args"]
        self.train_step_cnt = 0
        self.max_new_tokens = args.max_new_tokens
        self.use_gt_labels = args.use_gt_labels
        self.vla_model_initialization = vla_model_initialization

    def training_step(self, model, inputs):
        self.train_step_cnt += 1
        return self.consistency_training_step(model, inputs)

    # def consistency_training_step(self, model, inputs):

    #     max_new_tokens = self.max_new_tokens      

    #     jacobian_trajectory = inputs["jacobian_trajectory"]
    #     # print("lenjacobian_trajectory:",len(jacobian_trajectory))#17
    #     # print("lenjacobian_trajectory[0]:",len(jacobian_trajectory[0]))#1
    #     # print("lenjacobian_trajectory[0][0]:",len(jacobian_trajectory[0][0]))#84
    #     # 打印 jacobian_trajectory 的维度
    #     # jacobian_trajectory_tensor = torch.tensor(jacobian_trajectory)
    #     # 打印 tensor 的维度
    #     # print(jacobian_trajectory_tensor.shape)
    #     input_masks = inputs["attention_mask"]
    #     image_tensor = inputs["image_tensor"]
    #     bsz = jacobian_trajectory[0].shape[0]#
    #     eos_reached = torch.tensor([False] * bsz).to(model.device)

    #     ### tokens generated after <eos> are set to <pad>
    #     for i in range(len(jacobian_trajectory)):
    #         for j in range(bsz):
    #             trajectory_len = torch.sum(input_masks, dim=-1)
    #             # find the first accurate <EOS> 好像有问题这里
    #             eos_positions = torch.where(jacobian_trajectory[i][j, :(trajectory_len[j]-max_new_tokens)]==self.tokenizer.eos_token_id)[0]
    #             # print("eos_positions:",eos_positions)
    #             if len(eos_positions)==0:
    #                 continue
    #             # otherwise, set tokens coming after the accurate <EOS> as pad 
    #             eos_reached[j] = True
    #             trajectory_copy = jacobian_trajectory[i].clone().detach()
    #             eos_pos = eos_positions[0]
    #             trajectory_copy[j, int(eos_pos)+1:] = self.tokenizer.pad_token_id
    #             jacobian_trajectory[i] = trajectory_copy  

    #     ### compute AutoRegression loss ###
    #     # use labels to avoid pattern collapse
    #     if self.use_gt_labels:
    #         labels = inputs['labels_ids']
    #     else:
    #         labels = inputs['teacher_output_ids']
    #     # TODO: check if it's right when batch size > 1
    #     labels = torch.tensor(labels).to(model.device)#这里相当于input_ids label是80个
    #     attention_mask = torch.full_like(labels, 1).to(model.device)
    #     vla_model_initialization = CLLMRobotgenerate()
    #     input_embeds = vla_model_initialization.embedding_generate(labels.clone(), image_tensor)
    #     # for name, param in model.named_parameters():
    #     #     print(f"Parameter: {name}, Device: {param.device}, Dtype: {param.dtype}")
    #     #model设备都没问题
    #     # import pdb;pdb.set_trace()
    #     label_student_model_output = model(input_ids=None, inputs_embeds=input_embeds, attention_mask=attention_mask)#将标签输入模型 注意力可以看到所有prompt
    #     # print("label_student_model_output.shape:",label_student_model_output.shape)
    #     # print(type(label_student_model_output))  # 查看输出对象的类型
    #     # print(label_student_model_output.keys())  # 查看输出对象的所有字段
    #     # print(type(label_student_model_output['loss']))  # 检查实际类
    #     print("label_student_model_output['logits'].shape:",label_student_model_output['logits'].shape)# 655个通道输出      torch.Size([1, 84, 32032])
    #     # print(label_student_model_output['loss']['logits'].shape)#torch.Size([1, 84, 32032])
        
    #     print("labels.shape:",labels.shape)#84
    #     # print("label_student_model_outputlen",len(label_student_model_output[0]))
    #     _, image_indices = torch.where(labels == IMAGE_TOKEN_INDEX)
    #     output_length = label_student_model_output['logits'].shape[1]#655
    #     label_length = labels.shape[1]#84
    #     part1 = label_student_model_output['logits'][0, :image_indices[0], :]
    #     part2 = label_student_model_output['logits'][0, -(label_length-image_indices[0]):, :]
    #     logits = torch.cat((part1, part2), dim=0).unsqueeze(0)
    #     # 
    #     attention_mask = torch.full_like(jacobian_trajectory[0], 1).to(model.device)
    #     attention_mask = jacobian_trajectory[-1] != self.tokenizer.pad_token_id
    #     logits_last =  self.get_logits(model, jacobian_trajectory[-1].clone().detach(), attention_mask, image_tensor)#最后一次迭代输入模型 
    #     # print("logits_last[0]len:",len(logits_last[0]))#84
    #     # label_smoother = LabelSmoother(epsilon=0.1, ignore_index= -100)
    #     label_smoother = LabelSmoother()
    #     # import pdb;pdb.set_trace()
    #     loss_ar = label_smoother(logits, labels, shift_labels=True)#计算负对数似然
    #     loss_ar*=10
    #     if self.args.qlora:
    #         loss_ar.requires_grad = True
    #     # print(f'loss ar: {loss_ar} computed! performing backward pass...')
    #     with self.accelerator.accumulate(model):
    #         self.accelerator.backward(loss_ar)

    #     ### compute Consistency loss (global) ###
    #     # random select one point from trajectory
    #     i = random.choice(range(len(jacobian_trajectory))[:-1])
    #     print("model.device",model.device)
    #     print("jacobian_trajectory[0].device",jacobian_trajectory[0].device)
    #     attention_mask = torch.full_like(jacobian_trajectory[0], 1).to(jacobian_trajectory[0].device)
    #     attention_mask = jacobian_trajectory[i] != self.tokenizer.pad_token_id
    #     logits_i = self.get_logits(model, jacobian_trajectory[i].clone().detach(), attention_mask, image_tensor)#TODO: 检查这里直接输入image_tensor是否正确，注意这个logits_i和上面的logits_last的区别
    #     print("logits_i.device:",logits_i.device)
    #     output_mask = jacobian_trajectory[i][..., 1:] == self.tokenizer.pad_token_id#跳过开始标记
    #     # We do not calculate the cross entrophy of same logits to alleviate misleading gradients
    #     for j in range(bsz):
    #         end_of_mask_position = torch.where(jacobian_trajectory[i][j, 1:] != jacobian_trajectory[-1][j, 1:])[0]#第一个不一样的token
    #         if len(end_of_mask_position)==0:
    #             output_mask[j, :] = True
    #         else:
    #             output_mask[j, :end_of_mask_position[0]] = True#只看第一个不一样的token之前的
        
    #     loss_global = self.soft_cross_entropy(
    #                 logits_i[..., :-1, :].float(), # logits generated by the last token is dropped
    #                 logits_last[..., :-1, :].to(logits_i.device).clone().detach().float(),
    #                 output_mask.to(logits_i.device)
    #     )
    #     if self.args.qlora:
    #         loss_global.requires_grad = True
    #     # print(f'loss global {loss_global} computed! performing backward pass...')
    #     with self.accelerator.accumulate(model):
    #         self.accelerator.backward(loss_global)
        
    #     if self.args.local_rank == 0:
    #         wandb.log({"ar loss": loss_ar})
    #         wandb.log({"consistency loss": loss_global})

    #     # sync processes
    #     torch.distributed.barrier()
    #     print("loss_ar.device:",loss_ar.device)
    #     print("loss_global.device:",loss_global.device)
    #     # total loss = ar_loss + consistency_global_loss
    #     loss = loss_ar.detach() + loss_global.detach()

    #     return loss
  

    #     max_new_tokens = self.max_new_tokens      

    #     jacobian_trajectory = inputs["jacobian_trajectory"]
    #     input_masks = inputs["attention_mask"]
    #     image_tensor = inputs["image_tensor"]

    #     # 打印 jacobian_trajectory 的 dtype 和 device
    #     print("jacobian_trajectory[0] dtype:", jacobian_trajectory[0].dtype, "device:", jacobian_trajectory[0].device)

    #     bsz = jacobian_trajectory[0].shape[0]
    #     eos_reached = torch.tensor([False] * bsz).to(model.device)

    #     # 打印 eos_reached 的 dtype 和 device
    #     print("eos_reached dtype:", eos_reached.dtype, "device:", eos_reached.device)

    #     # 打印 input_masks 的 dtype 和 device
    #     print("input_masks dtype:", input_masks.dtype, "device:", input_masks.device)

    #     # 打印 image_tensor 的 dtype 和 device
    #     print("image_tensor dtype:", image_tensor.dtype, "device:", image_tensor.device)

    #     ### tokens generated after <eos> are set to <pad>
    #     for i in range(len(jacobian_trajectory)):
    #         for j in range(bsz):
    #             trajectory_len = torch.sum(input_masks, dim=-1)
    #             eos_positions = torch.where(jacobian_trajectory[i][j, :(trajectory_len[j]-max_new_tokens)] == self.tokenizer.eos_token_id)[0]
    #             if len(eos_positions) == 0:
    #                 continue
    #             eos_reached[j] = True
    #             trajectory_copy = jacobian_trajectory[i].clone().detach()
    #             eos_pos = eos_positions[0]
    #             trajectory_copy[j, int(eos_pos)+1:] = self.tokenizer.pad_token_id
    #             jacobian_trajectory[i] = trajectory_copy  

    #     ### compute AutoRegression loss ###
    #     if self.use_gt_labels:
    #         labels = inputs['labels_ids']
    #     else:
    #         labels = inputs['teacher_output_ids']

    #     labels = torch.tensor(labels).to(model.device)

    #     # 打印 labels 的 dtype 和 device
    #     print("labels dtype:", labels.dtype, "device:", labels.device)

    #     attention_mask = torch.full_like(labels, 1).to(model.device)

    #     # 打印 attention_mask 的 dtype 和 device
    #     print("attention_mask dtype:", attention_mask.dtype, "device:", attention_mask.device)

    #     vla_model_initialization = CLLMRobotgenerate()
    #     input_embeds = vla_model_initialization.embedding_generate(labels.clone(), image_tensor)

    #     # 打印 input_embeds 的 dtype 和 device
    #     print("input_embeds dtype:", input_embeds.dtype, "device:", input_embeds.device)

    #     label_student_model_output = model(input_ids=None, inputs_embeds=input_embeds, attention_mask=attention_mask)

    #     # 打印 label_student_model_output 的 logits dtype 和 device
    #     print("label_student_model_output['logits'] dtype:", label_student_model_output['logits'].dtype, "device:", label_student_model_output['logits'].device)

    #     logits_last = self.get_logits(model, jacobian_trajectory[-1].clone().detach(), attention_mask, image_tensor)

    #     # 打印 logits_last 的 dtype 和 device
    #     print("logits_last dtype:", logits_last.dtype, "device:", logits_last.device)

    #     logits_i = self.get_logits(model, jacobian_trajectory[0].clone().detach(), attention_mask, image_tensor)

    #     # 打印 logits_i 的 dtype 和 device
    #     print("logits_i dtype:", logits_i.dtype, "device:", logits_i.device)

    #     output_mask = jacobian_trajectory[0][..., 1:] == self.tokenizer.pad_token_id

    #     # 打印 output_mask 的 dtype 和 device
    #     print("output_mask dtype:", output_mask.dtype, "device:", output_mask.device)

    #     loss_global = self.soft_cross_entropy(
    #         logits_i[..., :-1, :].float(),
    #         logits_last[..., :-1, :].to(logits_i.device).clone().detach().float(),
    #         output_mask.to(logits_i.device)
    #     )
    #     if self.args.qlora:
    #         loss_global.requires_grad = True

    #     # 打印 loss_global 的 dtype 和 device
    #     print("loss_global dtype:", loss_global.dtype, "device:", loss_global.device)

    #     loss_ar = label_smoother(logits, labels, shift_labels=True) * 10

    #     # 打印 loss_ar 的 dtype 和 device
    #     print("loss_ar dtype:", loss_ar.dtype, "device:", loss_ar.device)

    #     with self.accelerator.accumulate(model):
    #         self.accelerator.backward(loss_ar)

    #     i = random.choice(range(len(jacobian_trajectory))[:-1])
    #     attention_mask = torch.full_like(jacobian_trajectory[0], 1).to(jacobian_trajectory[0].device)
    #     attention_mask = jacobian_trajectory[i] != self.tokenizer.pad_token_id

    #     logits_i = self.get_logits(model, jacobian_trajectory[i].clone().detach(), attention_mask, image_tensor)

    #     output_mask = jacobian_trajectory[i][..., 1:] == self.tokenizer.pad_token_id
    #     for j in range(bsz):
    #         end_of_mask_position = torch.where(jacobian_trajectory[i][j, 1:] != jacobian_trajectory[-1][j, 1:])[0]
    #         if len(end_of_mask_position) == 0:
    #             output_mask[j, :] = True
    #         else:
    #             output_mask[j, :end_of_mask_position[0]] = True

    #     loss_global = self.soft_cross_entropy(
    #         logits_i[..., :-1, :].float(),
    #         logits_last[..., :-1, :].to(logits_i.device).clone().detach().float(),
    #         output_mask.to(logits_i.device)
    #     )
    #     if self.args.qlora:
    #         loss_global.requires_grad = True

    #     with self.accelerator.accumulate(model):
    #         self.accelerator.backward(loss_global)

    #     if self.args.local_rank == 0:
    #         wandb.log({"ar loss": loss_ar})
    #         wandb.log({"consistency loss": loss_global})

    #     torch.distributed.barrier()
    #     print("loss_ar.device:", loss_ar.device)
    #     print("loss_global.device:", loss_global.device)

    #     loss = loss_ar.detach() + loss_global.detach()

    #     return loss
    def consistency_training_step(self, model, inputs):

        max_new_tokens = self.max_new_tokens      

        jacobian_trajectory = inputs["jacobian_trajectory"]
        # print("jacobian_trajectory[0] dtype:", jacobian_trajectory[0].dtype, "device:", jacobian_trajectory[0].device)

        input_masks = inputs["attention_mask"]
        # print("input_masks dtype:", input_masks.dtype, "device:", input_masks.device)

        image_tensor = inputs["image_tensor"]
        # print("image_tensor dtype:", image_tensor.dtype, "device:", image_tensor.device)

        bsz = jacobian_trajectory[0].shape[0]
        eos_reached = torch.tensor([False] * bsz).to(model.device)
        # print("eos_reached dtype:", eos_reached.dtype, "device:", eos_reached.device)

        ### tokens generated after <eos> are set to <pad>
        for i in range(len(jacobian_trajectory)):
            for j in range(bsz):
                trajectory_len = torch.sum(input_masks, dim=-1)
                eos_positions = torch.where(jacobian_trajectory[i][j, :(trajectory_len[j]-max_new_tokens)]==self.tokenizer.eos_token_id)[0]
                if len(eos_positions) == 0:
                    continue
                eos_reached[j] = True
                trajectory_copy = jacobian_trajectory[i].clone().detach()
                eos_pos = eos_positions[0]
                trajectory_copy[j, int(eos_pos)+1:] = self.tokenizer.pad_token_id
                jacobian_trajectory[i] = trajectory_copy  

        ### compute AutoRegression loss ###
        if self.use_gt_labels:
            labels = inputs['labels_ids']
        else:
            labels = inputs['teacher_output_ids']
        labels = torch.tensor(labels).to(model.device)
        # print("labels dtype:", labels.dtype, "device:", labels.device)

        attention_mask = torch.full_like(labels, 1).to(model.device)
        # print("attention_mask shape:", attention_mask.shape)
        # print("attention_mask dtype:", attention_mask.dtype, "device:", attention_mask.device)

        vla_model_initialization = self.vla_model_initialization
        input_embeds = vla_model_initialization.embedding_generate(labels.clone(), image_tensor)
        # print("input_embeds shape:", input_embeds.shape)
        # print("input_embeds dtype:", input_embeds.dtype, "device:", input_embeds.device)

        label_student_model_output = model(input_ids=None, inputs_embeds=input_embeds, attention_mask=attention_mask)
        # print("label_student_model_output['logits'] dtype:", label_student_model_output['logits'].dtype, "device:", label_student_model_output['logits'].device)
        _, image_indices = torch.where(labels == IMAGE_TOKEN_INDEX)
        output_length = label_student_model_output['logits'].shape[1]#655
        label_length = labels.shape[1]#84
        part1 = label_student_model_output['logits'][0, :image_indices[0], :]
        part2 = label_student_model_output['logits'][0, -(label_length-image_indices[0]):, :]
        logits = torch.cat((part1, part2), dim=0).unsqueeze(0)
        logits_last = self.get_logits(model, jacobian_trajectory[-1].clone().detach(), attention_mask, image_tensor, vla_model_initialization)
        # print("logits_last dtype:", logits_last.dtype, "device:", logits_last.device)

  

        output_mask = jacobian_trajectory[0][..., 1:] == self.tokenizer.pad_token_id
        # print("output_mask dtype:", output_mask.dtype, "device:", output_mask.device)

        label_smoother = LabelSmoother()
        loss_ar = label_smoother(logits, labels, shift_labels=True) * 10
        # print("loss_ar dtype:", loss_ar.dtype, "device:", loss_ar.device)

        with self.accelerator.accumulate(model):
            self.accelerator.backward(loss_ar)

        ### compute Consistency loss (global) ###
        i = random.choice(range(len(jacobian_trajectory))[:-1])
        attention_mask = torch.full_like(jacobian_trajectory[0], 1).to(jacobian_trajectory[0].device)
        attention_mask = jacobian_trajectory[i] != self.tokenizer.pad_token_id

        logits_i = self.get_logits(model, jacobian_trajectory[i].clone().detach(), attention_mask, image_tensor, vla_model_initialization)
        # print("logits_i dtype:", logits_i.dtype, "device:", logits_i.device)

        output_mask = jacobian_trajectory[i][..., 1:] == self.tokenizer.pad_token_id
        # print("output_mask dtype:", output_mask.dtype, "device:", output_mask.device)

        loss_global = self.soft_cross_entropy(
            logits_i[..., :-1, :].float(),
            logits_last[..., :-1, :].to(logits_i.device).clone().detach().float(),
            output_mask.to(logits_i.device)
        )
        # print("loss_global dtype:", loss_global.dtype, "device:", loss_global.device)

        if self.args.qlora:
            loss_global.requires_grad = True

        with self.accelerator.accumulate(model):
            self.accelerator.backward(loss_global)

        if self.args.local_rank == 0:
            wandb.log({"ar loss": loss_ar})
            wandb.log({"consistency loss": loss_global})

        torch.distributed.barrier()
        # print("loss_ar.device:", loss_ar.device)
        # print("loss_global.device:", loss_global.device)

        loss = loss_ar.detach() + loss_global.detach()

        return loss



    def log(self, logs):
        # Remove the 'loss' entry with value 0 before calling the superclass method
        if 'loss' in logs and logs['loss'] == -1:
            del logs['loss']

        # Call the original `log` method of the `Trainer` class
        super().log(logs)

    def get_train_dataloader(self):
        # Create custom DataLoader with shuffle set to False
        shuffle = True
        dataloader_params = {
            "batch_size": self.args.per_device_train_batch_size,
            "shuffle": shuffle,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

        return self.accelerator.prepare(DataLoader(self.train_dataset, **dataloader_params))

    ###################### Helper Functions #############################
    def soft_cross_entropy(self, predicts, targets, padding_mask):
        #(Pdb) predicts.shape
        # torch.Size([1, 654, 32000])
        # TODO: support batch_size >1 here.
        if (~padding_mask).sum() == 0:
            return 0*predicts[0][0][0]
        predict_log_prob = torch.nn.functional.log_softmax(predicts, dim=-1)
        targets_prob = torch.nn.functional.softmax(targets, dim=-1)
        entropy = -targets_prob * predict_log_prob
        # import pdb;pdb.set_trace()

        #对齐长度
        # 假设需要扩展的长度
        required_length = entropy.shape[1]  # 654

        # 当前长度
        current_length = padding_mask.shape[1]  # 当前 padding_mask 的长度

        # 计算需要添加的数量
        additional_trues = required_length - current_length

        if additional_trues > 0:
            # 构造额外的 True 张量
            extra_trues = torch.ones((padding_mask.shape[0], additional_trues), dtype=torch.bool, device=padding_mask.device)
            
            # 拼接到 padding_mask 的开头
            padding_mask = torch.cat((extra_trues, padding_mask), dim=1)



        # print(padding_mask.shape)  # 现在的 shape 应该是 [1, 654]
        expand_mask = padding_mask.unsqueeze(-1).expand_as(entropy)
        # (Pdb) padding_mask.shape
        # torch.Size([1, 79])
        # (Pdb) entropy.shape
        # torch.Size([1, 654, 32000])
        entropy.masked_fill_(expand_mask, 0)
        mean_entropy = entropy.sum() / (~padding_mask).sum()
        return mean_entropy

    def get_logits(self, model, input_ids, attention_mask, image_tensor, vla_model_initialization):
        
        # vla_model_initialization = CLLMRobotgenerate()
        input_embeds = vla_model_initialization.embedding_generate(input_ids.clone().to(device=model.device), image_tensor.to(device=model.device))
        return model(
            input_ids=None, inputs_embeds=input_embeds, attention_mask=attention_mask
        ).logits

