# WX在模型加载方面对本来的vla做的两个改动：1.device_map='cuda' 2.使用flash_attn2
# This code is based on tatsu-lab/stanford_alpaca. Below is the original copyright:
#
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import numpy as np
from tqdm import tqdm
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

from cllm.cvla_trainer_global import CllmTrainer

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
from llava.action_tokenizer import ActionTokenizer, encode_robot_obs,encode_actions
from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_AUDIO_TOKEN,IMAGE_TOKEN_INDEX
from llava.mm_utils import tokenizer_image_token
from llava.utils import disable_torch_init

logger = logging.getLogger(__name__)

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

class CLLMRobotgenerate:
    def __init__(self,model_path,action_stat,image_folder):
        model_path = os.path.expanduser(model_path)
        model_name = get_model_name_from_path(model_path)
        model_base = None
        self.tokenizer, self.vla_model, self.image_processor, self.context_len = (
            load_pretrained_model_cvla(model_path, model_base, model_name)
        )
        self.temperature = 0.0
        self.top_p = None
        self.num_beams = 1
        self.max_new_tokens = 16
        self.action_tokenizer = ActionTokenizer(self.tokenizer)
        # print("action_stat:",action_stat)
        self.action_stat = "/mnt/sda/wenxuansong/data/dataset/task_ABC_D/training/statistics.yaml"
        self.image_folder = image_folder

    def compose_robot_input(
        self, instruction, image, debug=True
    ):
        # print("instruction:",instruction)
        img_concat=image
        image_tensor = self.image_processor.preprocess(img_concat, return_tensors="pt")[
            "pixel_values"
        ][0]
        image_tensor = image_tensor[None, :]
        robot_obs=instruction.split('\n')[-1]
        # print("robot_obs:",robot_obs)
        # robot_obs = [str(elem) for elem in robot_obs]
        # robot_obs = " ".join(robot_obs)
        # print(self.action_stat)
        robot_obs = encode_robot_obs(robot_obs, self.action_tokenizer, self.action_stat)
        instruction=instruction.split('\n')[-2]
        # print("instruction:",instruction)
        instruction = DEFAULT_IMAGE_TOKEN + "\n" + instruction + "\n" + robot_obs
        # print("instruction:",instruction)
        conv = conversation_lib.default_conversation.copy()
        conv.system = "A chat between a curious user and an artificial intelligence robot. The robot provides actions to follow out the user's instructions."
        conv.append_message(conv.roles[0], instruction)
        conv.append_message(conv.roles[1], None)
        instruction = conv.get_prompt()

        input_ids = torch.stack(
            [tokenizer_image_token(instruction, self.tokenizer, return_tensors="pt")],
            dim=0,
        )
        return input_ids, image_tensor
    
    def embedding_generate(self, input_ids, images,image_sizes: Optional[torch.Tensor] = None):
        position_ids=None
        attention_mask=None
        (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            )=self.vla_model.prepare_inputs_labels_for_multimodal(                
                input_ids.cuda(),
                position_ids,
                attention_mask,
                None,
                None,
                images=images.to(dtype=torch.float16, device="cuda", non_blocking=True),#本来是float16
                image_sizes=image_sizes)
        # print("inputs:",inputs)
        # print("inputs_embeds:",inputs_embeds)
        # print("inputs_embeds.shape",inputs_embeds.shape)
        return inputs_embeds

    def preprocess_vla_data(self, data, tokenizer,vla_action_tokenizer):#输出的ipuntid是3维度，lable是一维
    #     data = [
    #     {
    #         "conversations": [
    #             {"value": "What is AI?"},
    #             {"value": "AI stands for Artificial Intelligence."}
    #         ]
    #     }
    # ]
        train_dataset = []
        for i in tqdm(range(len(data))):
            # d = data[i]
            # #if len(d["conversations"]) > 2:
            # #    continue
            # prompt = d["conversations"][0]["value"]
            # image_file = d["image"]
            # image_folder = self.image_folder
            # image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            # image_tensor = self.image_processor.preprocess(image, return_tensors="pt")[
            #     "pixel_values"
            # ][0]
            # image_tensor = image_tensor[None, :]
            
            # if len(prompt) > 1024:
            #     # exclude prompts that are too long
            #     continue
            d = data[i]
            #if len(d["conversations"]) > 2:
            #    continue
            prompt = d["conversations"][0]["value"]
            image_file = d["image"]
            image_folder = self.image_folder
            image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            input_ids, image_tensor = self.compose_robot_input( prompt,image)#input_id: torch.Size([1, 64])和原来的input_id长度不一样
            # print("image_tensor.shape",image_tensor.shape)
            
            if len(prompt) > 1024:
                # exclude prompts that are too long
                continue

            # conv = get_conversation_template(model_path)
            # conv = conversation_lib.default_conversation.copy()
            # conv.system = "A chat between a curious user and an artificial intelligence robot. The robot provides actions to follow out the user's instructions."
            # conv.append_message(conv.roles[0], prompt)
            # conv.append_message(conv.roles[1], "")
            # prompt_with_template = conv.get_prompt()

            #jacobian_prompt = prompt_with_template
            # prompt_with_template_ids = tokenizer(prompt_with_template, return_tensors="pt")['input_ids'] #torch.Size([1, 324])
            # import pdb;pdb.set_trace()
            inputs = torch.Tensor(input_ids).unsqueeze(0).to(dtype=torch.int)#torch.Size([1, 1, 324])
            labels=self.encode_actions(d["conversations"][1]["value"],vla_action_tokenizer)
            # labels = tokenizer(d["conversations"][1]["value"], return_tensors="pt")['input_ids'][0]#改成了action_tokenizer
            # answer_labels = encode_actions(d["conversations"][1]["value"], vla_action_tokenizer)
            # print("answer_labels:",answer_labels)
            labels = torch.tensor(labels)
            labels_ids = torch.concat((labels, torch.tensor([tokenizer.eos_token_id])), dim=-1).to(dtype=torch.int)
            
            train_dataset.append(dict(sources_input_ids=inputs,image_file=image_file, sources_len=[
                input.ne(tokenizer.pad_token_id).sum().item() for input in inputs], labels_ids=labels_ids, image_tensor=image_tensor))
        #         {
        #     'sources_input_ids': 3D tensor,  # 输入的 token IDs（3维）
        #     'sources_len': List[int],        # 输入的长度（1维）
        #     'labels_ids': 1D tensor          # 标签的 token IDs（1维）
        #     'image_tensor': 3D tensor               # 图像（4维）
        # }
        #这个返回的应该是一个batch

        return train_dataset
    def encode_actions(self,sentence, action_tokenizer, statistics=None):
        actions = sentence.split(" ")
        actions = [float(action) for action in actions]
        # actions_lang = action_tokenizer(actions)
        actions = np.clip(actions, a_min=float(action_tokenizer.min_action), a_max=float(action_tokenizer.max_action))
        discretized_action = np.digitize(actions, action_tokenizer.bins)
        actions_inputids=list(action_tokenizer.tokenizer.vocab_size - discretized_action)
        return actions_inputids
    def robot_action_generate(self, input_ids, images):
        """_summary_

        Args:
            input_ids : shape of (1, L)
            images : shape of (1, C, H, W)

        Returns:
            _type_: _description_
        """
        # print("images.shape",images.shape)
        # time0 = time.time()
        with torch.inference_mode():
            output_ids = self.vla_model.generate(
                input_ids.cuda(),
                images=images.to(dtype=torch.float16, device="cuda", non_blocking=True),
                do_sample=True if self.temperature > 0 else False,
                temperature=self.temperature,
                top_p=self.top_p,
                num_beams=self.num_beams,
                max_new_tokens=128,
                use_cache=True,
            )
        memory_used = torch.cuda.memory_allocated() / 1024**3  # 当前显存使用（GB）
        # time1 = time.time()
        # memory_samples.append(memory_used)
        # generate_time = time1 - time0
        # inference_times.append(generate_time)
        # print("推理一次需要的时间为", generate_time)
        
        # skip the <s> and </s> special token
        # print("output_ids.shape",output_ids.shape)
        output_ids = output_ids[0].cpu().numpy().tolist()[2:-1]
        actions = output_ids
        # for elem in output_ids:
            # actions.append(self.action_tokenizer.decode_token_ids_to_actions(elem))
        # actions = np.array(actions)
        return actions
@dataclass
class ModelArguments:
    target_model_path: Optional[str] = field(
        default="models/vicuna-7b-v1.5",  metadata={"help": "Path to target model"})
    qlora: Optional[bool] = field(default=False, metadata={"help": "Enable QLoRA processing"})

@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    lazy_preprocess: bool = False

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    max_new_tokens: int = field(
        default=16,
        metadata={
            "help": "Size of n_token_sequence in Jacobi trajectory."
        },
    )
    use_gt_labels: bool = False
    report_to: str = field(
        default='wandb',
        metadata={
            'help': 'The list of integrations to report the results and logs to.'
        }
    )

def rank0_print(local_rank, *args):
    if local_rank == 0:
        print(*args)

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu()
                          for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

def preprocess_distill_data(
    prompt_ids,
    answer_trajectory_ids,
    teacher_output_ids,
    complete_teacher_output_ids,
    image_path,
    tokenizer: transformers.PreTrainedTokenizer,
    model: str,
    image_processor,
    image_folder,
    labels_ids=None
) -> Dict:

    image = Image.open(os.path.join(image_folder, image_path)).convert('RGB')
    image_tensor = image_processor.preprocess(image, return_tensors="pt")[
        "pixel_values"
    ][0]
    image_tensor = image_tensor[None, :]

    jacobian_trajectory_ids = []
    # only take batch size 1 for now
    # TODO: support bsz > 1 from the generation script. for now, only prompt ids is in (bsz, seq_len)
    jacobian_prompt_ids = torch.tensor(prompt_ids[0], dtype=torch.int64)#2维，原来3维
    teacher_output_ids = torch.tensor(teacher_output_ids[0], dtype=torch.int64)#1维度 原来2维
    complete_teacher_output_ids = torch.tensor(complete_teacher_output_ids, dtype=torch.int64)#1维
    for answer_ids in answer_trajectory_ids:
        answer_ids = torch.tensor(answer_ids, dtype=torch.int64)
        #print(answer_ids)
        #print(jacobian_prompt_ids)
        if len(jacobian_prompt_ids.shape) == len(answer_ids.shape):
            trajectory_ids = torch.cat((jacobian_prompt_ids, answer_ids), dim=-1)
        elif len(jacobian_prompt_ids.shape) > len(answer_ids.shape):#进入这个循环
            # print(f'prompt: {jacobian_prompt_ids.shape}')
            # print(f'answer: {answer_ids.shape}')
            # prompt: torch.Size([1, 1, 64])
            # answer: torch.Size([16])
            trajectory_ids = torch.cat((jacobian_prompt_ids[0][0], answer_ids), dim=-1)#
        # print(trajectory_ids.shape) # torch.Size([228])
        jacobian_trajectory_ids.append(trajectory_ids)#重新构建，包括prompt+gernerate n token  2维
   
    if labels_ids:
        return dict(
            jacobian_trajectory=jacobian_trajectory_ids,#包括prompt的雅可比序列2维度
            attention_mask=jacobian_trajectory_ids[0].ne(tokenizer.pad_token_id),
            labels_ids=labels_ids,#2维
            teacher_output_ids=teacher_output_ids,#1维
            complete_teacher_output_ids=complete_teacher_output_ids,#1维
            image_tensor=image_tensor
        )
    else:
        return dict(
            jacobian_trajectory=jacobian_trajectory_ids,
            attention_mask=jacobian_trajectory_ids[0].ne(tokenizer.pad_token_id),
            teacher_output_ids=teacher_output_ids,
            complete_teacher_output_ids=complete_teacher_output_ids,
            image_tensor=image_tensor
        )
    
class JacobianDataset(Dataset):
    """Dataset for consistency training."""

    def __init__(self, raw_data,
                 tokenizer: transformers.PreTrainedTokenizer,
                 model: str,
                 do_eval: bool = False,
                 local_rank: int = -1,
                 image_processor=None,
                 image_folder=None):
        super(JacobianDataset, self).__init__()
        self.tokenizer = tokenizer

        rank0_print(local_rank, "Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}
        self.do_eval = do_eval
        self.model = model
        self.image_processor = image_processor
        self.image_folder = image_folder
        # import pdb;pdb.set_trace()

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]
        if 'labels_ids' in self.raw_data[i].keys():
            ret = preprocess_distill_data(self.raw_data[i]["prompt_ids"],
                         self.raw_data[i]["answer_trajectory_ids"],
                         self.raw_data[i]["teacher_output_ids"],
                         self.raw_data[i]["complete_teacher_output_ids"],
                         self.raw_data[i]["image"],
                         self.tokenizer,
                         self.model,
                         self.image_processor,
                         self.image_folder,
                         labels_ids=self.raw_data[i]["labels_ids"])
        else:
            ret = preprocess_distill_data(self.raw_data[i]["prompt_ids"],
                            self.raw_data[i]["answer_trajectory_ids"],
                            self.raw_data[i]["teacher_output_ids"],
                            self.raw_data[i]["complete_teacher_output_ids"],
                            self.raw_data[i]["image"],
                            self.tokenizer,
                            self.model,
                            self.image_processor,
                            self.image_folder)
        self.cached_data_dict[i] = ret # cjy 20250108 怀疑是爆内存导致训练失败，为了节省内存，不缓存数据

        return ret


def make_jacobian_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    trajectory_path,
    data_args,
    model: str,
    local_rank: int,
    image_processor,
    image_folder
) -> Dict:
    """Make dataset and collator for consistency training."""
    assert data_args.lazy_preprocess, "only support lazy process"
    dataset_cls = JacobianDataset #这里只定义类，下面传参
    
    rank0_print("Loading data...")
    # with open(trajectory_path, 'r', encoding='utf-8') as file:
    #     train_json = file.readlines()
    train_json=load_jsonl_files_from_directory(trajectory_path)
    # train_json = json.load(open(trajectory_path, "r"))
    print("train_json;",type(train_json))
    print("len(train_json);",len(train_json))

    truncated_train_json = []
    
    for data in train_json:
        data_dict = json.loads(data)
        truncated_train_json.append(data_dict)
       
        
    train_dataset = dataset_cls(truncated_train_json,
                                tokenizer=tokenizer,
                                model=model,
                                local_rank=local_rank,
                                image_processor=image_processor,
                                image_folder=image_folder)
    # print(type(train_dataset[0]["jacobian_trajectory"]))
    # answer_trajectory_ids_np = np.array(train_dataset[0]["jacobian_trajectory"])
    # print(answer_trajectory_ids_np.shape)
    eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)
def load_jsonl_files_from_directory(directory_path):
    # 获取目录下所有的.jsonl文件
    jsonl_files = [f for f in os.listdir(directory_path) if f.endswith('.jsonl')]
    
    # 用于存储所有文件的内容
    all_json_data = []
    
    # 遍历每个文件并读取内容
    for jsonl_file in jsonl_files:
        file_path = os.path.join(directory_path, jsonl_file)
        with open(file_path, 'r', encoding='utf-8') as file:
            # 读取文件内容并将每行内容作为一个字典存储
            file_data = file.readlines()
            all_json_data.extend(file_data)  # 将内容添加到总数据中

    return all_json_data



def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print("model_args:",model_args)
    print("data_args:",data_args)
    print("training_args:",training_args)
    local_rank = int(os.environ["LOCAL_RANK"])
    training_args.local_rank = local_rank
    print("local_rank:",local_rank)
    training_args.qlora = model_args.qlora
    
    torch.set_default_dtype(torch.float32)#debug过程中这里从32改成16

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set RoPE scaling factor, 这是llama系列所采取的位置编码方式，猜测可能在早期的llama训练代码里需要显式设置一下，后面我们llava基本上都是封装好了的
    config = transformers.AutoConfig.from_pretrained(
        model_args.target_model_path,
        cache_dir=training_args.cache_dir,
    )
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = float(
            math.ceil(training_args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    config.use_cache = False
    
    # Load model and tokenizer
    action_stat = "/mnt/sda/wenxuansong/data/dataset/task_ABC_D/training/statistics.yaml"
    image_folder = "/home/wenxuansong/chenjy/data/calvin_cvla/task_ABC_D/vla_processed_r5"
    model_path="/home/wenxuansong/chenjy/project/vlas/LLaVA/checkpoints/llava-v1.5-7b-calvin-rel-obs-reduce5-v1_zhaobo/checkpoint-21572"
    vla_model_initialization = CLLMRobotgenerate(model_path,action_stat,image_folder)
    image_folder = vla_model_initialization.image_folder
    tokenizer = vla_model_initialization.tokenizer
    image_processor = vla_model_initialization.image_processor
    model = vla_model_initialization.vla_model


    # if 'vicuna' in model_args.target_model_path:
    tokenizer.pad_token = tokenizer.unk_token

    if model_args.qlora:
        # Runs w/ qLoRA when qlora tag is enabled is enabled
        model = prepare_model_for_kbit_training(model)
        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=32,
            lora_alpha=16,
            lora_dropout=0.05,
        )
    
        model = get_peft_model(model, config)
        model.config.use_cache = False

    # Load data
    data_module = make_jacobian_data_module(tokenizer=tokenizer,
                                              trajectory_path=data_args.data_path,
                                              data_args=data_args,
                                              model=model_args.target_model_path,
                                              local_rank=training_args.local_rank,
                                              image_processor=image_processor,
                                              image_folder=image_folder)    

    trainer = CllmTrainer(
        vla_model_initialization=vla_model_initialization, model=model, tokenizer=tokenizer, args=training_args, **data_module
    )
    #### cjydebug
    # for name, param in model.named_parameters():
    #     print(f"Parameter {name}: device={param.device}, dtype={param.dtype}")
    model = model.to(dtype=torch.bfloat16)
    ####
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=False)
    else:
        trainer.train()
    model.config.use_cache = True# 原来是true
    trainer.save_state()
    safe_save_model_for_hf_trainer(
        trainer=trainer, output_dir=training_args.output_dir)
    print("model成功保存至",training_args.output_dir)


if __name__ == "__main__":
    train()
