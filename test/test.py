import torch
# print(torch.cuda.is_available())

# import os
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# from model.MizukiModel import MizukiMindConfig
# from trainer.trainer_utils import (  # 训练工具函数
#     lm_checkpoint,
# )

# lm_config = MizukiMindConfig(
#     hidden_size=512,
#     num_hidden_layers=8,
#     use_moe=False,
# )

# ckp_data = lm_checkpoint(
#         lm_config, weight='pretrain', save_dir="../checkpoints"
#     )

# print(ckp_data['epoch'], ckp_data['step']) # type: ignore

# a = torch.tensor([1, 2, 0, 1, 2, 3, 0, 1, 2], dtype=torch.int32)
# b = a.bincount()
# print(b)
# c = a.bincount().cpu().numpy().cumsum(0)
# print(c)


torch.manual_seed(42)
indices = torch.randperm(10).tolist()
indices1 = torch.randperm(10).tolist()
print(indices)
print(indices1)
