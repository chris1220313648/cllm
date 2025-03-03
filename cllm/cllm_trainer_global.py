import transformers
import torch
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother
import wandb
import random
from torch.utils.data import DataLoader
import numpy as np
IGNORE_TOKEN_ID = LabelSmoother.ignore_index

class CllmTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        args = kwargs["args"]
        self.train_step_cnt = 0
        self.max_new_tokens = args.max_new_tokens
        self.use_gt_labels = args.use_gt_labels

    def training_step(self, model, inputs):
        self.train_step_cnt += 1
        return self.consistency_training_step(model, inputs)

    def consistency_training_step(self, model, inputs):

        max_new_tokens = self.max_new_tokens      

        jacobian_trajectory = inputs["jacobian_trajectory"]
        # print("lenjacobian_trajectory:",len(jacobian_trajectory))#17
        # print("lenjacobian_trajectory[0]:",len(jacobian_trajectory[0]))#1
        # print("lenjacobian_trajectory[0][0]:",len(jacobian_trajectory[0][0]))#84
        # 打印 jacobian_trajectory 的维度
        # jacobian_trajectory_tensor = torch.tensor(jacobian_trajectory)
        # 打印 tensor 的维度
        # print(jacobian_trajectory_tensor.shape)
        input_masks = inputs["attention_mask"]
        bsz = jacobian_trajectory[0].shape[0]#
        eos_reached = torch.tensor([False] * bsz).to(model.device)

        ### tokens generated after <eos> are set to <pad>
        for i in range(len(jacobian_trajectory)):
            for j in range(bsz):
                trajectory_len = torch.sum(input_masks, dim=-1)
                # find the first accurate <EOS> 好像有问题这里
                eos_positions = torch.where(jacobian_trajectory[i][j, :(trajectory_len[j]-max_new_tokens)]==self.tokenizer.eos_token_id)[0]
                # print("eos_positions:",eos_positions)
                if len(eos_positions)==0:
                    continue
                # otherwise, set tokens coming after the accurate <EOS> as pad 
                eos_reached[j] = True
                trajectory_copy = jacobian_trajectory[i].clone().detach()
                eos_pos = eos_positions[0]
                trajectory_copy[j, int(eos_pos)+1:] = self.tokenizer.pad_token_id
                jacobian_trajectory[i] = trajectory_copy  

        ### compute AutoRegression loss ###
        # use labels to avoid pattern collapse
        if self.use_gt_labels:
            labels = inputs['labels_ids']
        else:
            labels = inputs['teacher_output_ids']
        # TODO: check if it's right when batch size > 1
        labels = torch.tensor(labels).to(model.device)
        attention_mask = torch.full_like(labels, 1).to(model.device)
        label_student_model_output = model(labels, attention_mask)#将标签输入模型 注意力可以看到所有prompt
        # print("label_student_model_output.shape:",label_student_model_output.shape)
        # print(type(label_student_model_output))  # 查看输出对象的类型
        # print(label_student_model_output.keys())  # 查看输出对象的所有字段
        # print(type(label_student_model_output['loss']))  # 检查实际类
        # print(label_student_model_output['logits'].shape)#torch.Size([1, 84, 32032])
        # print(label_student_model_output['loss']['logits'].shape)#torch.Size([1, 84, 32032])
        
        # print("labels[0]len:",len(labels[0]))#84
        # print("label_student_model_outputlen",len(label_student_model_output[0]))

        attention_mask = torch.full_like(jacobian_trajectory[0], 1).to(model.device)
        attention_mask = jacobian_trajectory[-1] != self.tokenizer.pad_token_id
        logits_last =  self.get_logits(model, jacobian_trajectory[-1].clone().detach(), attention_mask)#最后一次迭代输入模型 
        # print("logits_last[0]len:",len(logits_last[0]))#84
        label_smoother = LabelSmoother(epsilon=0.1, ignore_index= -100)
        loss_ar = label_smoother(label_student_model_output, labels, shift_labels=True)#计算负对数似然
        loss_ar*=10
        if self.args.qlora:
            loss_ar.requires_grad = True
        print(f'loss ar: {loss_ar} computed! performing backward pass...')
        with self.accelerator.accumulate(model):
            self.accelerator.backward(loss_ar)

        ### compute Consistency loss (global) ###
        # random select one point from trajectory
        i = random.choice(range(len(jacobian_trajectory))[:-1])

        attention_mask = torch.full_like(jacobian_trajectory[0], 1).to(jacobian_trajectory[0].device)
        attention_mask = jacobian_trajectory[i] != self.tokenizer.pad_token_id
        logits_i = self.get_logits(model, jacobian_trajectory[i].clone().detach(), attention_mask)

        output_mask = jacobian_trajectory[i][..., 1:] == self.tokenizer.pad_token_id#跳过开始标记
        # We do not calculate the cross entrophy of same logits to alleviate misleading gradients
        for j in range(bsz):
            end_of_mask_position = torch.where(jacobian_trajectory[i][j, 1:] != jacobian_trajectory[-1][j, 1:])[0]#第一个不一样的token
            if len(end_of_mask_position)==0:
                output_mask[j, :] = True
            else:
                output_mask[j, :end_of_mask_position[0]] = True#只看第一个不一样的token之前的
        
        loss_global = self.soft_cross_entropy(
                    logits_i[..., :-1, :].float(), # logits generated by the last token is dropped
                    logits_last[..., :-1, :].to(logits_i.device).clone().detach().float(),
                    output_mask.to(logits_i.device)
        )
        if self.args.qlora:
            loss_global.requires_grad = True
        print(f'loss global {loss_global} computed! performing backward pass...')
        with self.accelerator.accumulate(model):
            self.accelerator.backward(loss_global)
        
        if self.args.local_rank == 0:
            wandb.log({"ar loss": loss_ar})
            wandb.log({"consistency loss": loss_global})

        # sync processes
        torch.distributed.barrier()
        # total loss = ar_loss + consistency_global_loss
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
        # TODO: support batch_size >1 here.
        if (~padding_mask).sum() == 0:
            return 0*predicts[0][0][0]
        predict_log_prob = torch.nn.functional.log_softmax(predicts, dim=-1)
        targets_prob = torch.nn.functional.softmax(targets, dim=-1)
        entropy = -targets_prob * predict_log_prob
        expand_mask = padding_mask.unsqueeze(-1).expand_as(entropy)
        entropy.masked_fill_(expand_mask, 0)
        mean_entropy = entropy.sum() / (~padding_mask).sum()
        return mean_entropy

    def get_logits(self, model, input_ids, attention_mask):
        return model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits

