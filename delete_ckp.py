import os
import shutil

def delete_directories_before_tmp(base_dir: str, tmp_dir_name: str):
    # 获取目录中的所有子目录
    subdirs = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))], key=lambda x: os.path.getctime(os.path.join(base_dir, x)))

    # 遍历子目录并删除tmp之前的目录
    for subdir in subdirs:
        subdir_path = os.path.join(base_dir, subdir)
        if subdir == tmp_dir_name:
            break  # 一旦找到tmp目录，停止删除
        if os.path.isdir(subdir_path):  # 确保是目录而非文件
            print(f"Deleting directory: {subdir_path}")
            shutil.rmtree(subdir_path)  # 删除目录及其内容

# 示例调用
base_dir = '/home/wenxuansong/chenjy/project/Consistency_LLM/models/cllm_model'  # 基础目录路径
tmp_dir_name = 'tmp-checkpoint-5500'                    # tmp目录名
delete_directories_before_tmp(base_dir, tmp_dir_name)
