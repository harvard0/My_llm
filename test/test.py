# import torch
# print(torch.cuda.is_available())

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.MizukiModel import MizukiMindConfig
from trainer.trainer_utils import (  # 训练工具函数
    lm_checkpoint,
)

lm_config = MizukiMindConfig(
    hidden_size=512,
    num_hidden_layers=8,
    use_moe=False,
)

ckp_data = lm_checkpoint(
        lm_config, weight='pretrain', save_dir="../checkpoints"
    ) 

print(ckp_data['epoch'], ckp_data['step']) # type: ignore