# import torch
# print(torch.cuda.is_available())

# import os
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.dirname(current_dir)
# project_root1 = os.path.dirname(project_root)
# tokenizer_path = os.path.join(project_root, "model")
# print(current_dir, project_root, project_root1, tokenizer_path)

from datasets import load_dataset
data = load_dataset("json", data_files='/home/mizuki/code/my_minimind/dataset/pretrain_t2t_mini.jsonl' ,split="train")
print(data[0])