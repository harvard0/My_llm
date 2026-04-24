import os
import random
import math
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Sampler


# def get_model_params(model, config):
#     total = sum(p.numel() for p in model.parameters()) / 1e6
#     n_routed = getattr(config, "n_routed_experts", getattr(config, "num_experts", 0))
#     n_active = getattr(config, "num_experts_per_tok", 0)
#     n_shared = getattr(config, "n_shared_experts", 0)
#     expert = (
#         sum(p.numel() for n, p in model.named_parameters() if "mlp.experts.0." in n)
#         / 1e6
#     )
#     shared_expert = (
#         sum(
#             p.numel()
#             for n, p in model.named_parameters()
#             if "mlp.shared_experts.0." in n
#         )
#         / 1e6
#     )
#     base = total - (expert * n_routed) - (shared_expert * n_shared)
#     active = base + (expert * n_active) + (shared_expert * n_shared)
#     if active < total:
#         Logger(f"Model Params: {total:.2f}M-A{active:.2f}M")
#     else:
#         Logger(f"Model Params: {total:.2f}M")


# 检查是否是主进程
def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


# 日志
def Logger(content):
    if is_main_process():
        print(content)


# 动态学习率计算
def get_lr(current_step, total_steps, lr):
    return lr * (
        0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps))
    )  # step=0→lr, step=end→0.1*lr


# 初始化分布式
def init_distributed_mode():
    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # 非DDP模式

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


# 设置种子
def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    # CPU 级别的随机性，CPU 上使用 torch.rand()、torch.randn()，
    # 或者在模型还没被 .to("cuda") 之前进行的权重初始化操作。
    torch.manual_seed(seed)
    # 当前单块 GPU 的随机性,数据或模型转移到 GPU 后，
    # 所有的随机操作（比如 GPU 上的 Dropout、GPU 上的随机采样）
    torch.cuda.manual_seed(seed)
    # 所有 GPU 的随机性,DDP 多卡分布式训练，
    # 这行代码会一次性把所有显卡的随机数种子全部设为相同的值。
    torch.cuda.manual_seed_all(seed)
    
    # 强制 cuDNN 使用“确定性算法
    # 不用快但是结果随机的算法，使用计算顺序都绝对严格一致的算法，消除了底层硬件级别的微小数值波动。
    torch.backends.cudnn.deterministic = True
    # 关闭 cuDNN 的“自动调优寻路”功能
    # 默认情况下，如果这个值为 True，cuDNN 会在你训练的第一个 Batch 时，
    # 把所有的卷积算法都跑一遍（Benchmark），看看在你的特定输入尺寸和特定型号显卡下，哪种算法最快。
    # 然后后续的训练都沿用这个最快的算法。
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    """
    DataLoader 多进程数据加载的随机数种子重置函数。

    【为什么需要这个函数？(问题背景)】
    当 DataLoader 的 num_workers > 0 时，OS 会通过 Fork 机制创建子进程。
    这会导致所有子进程完美克隆主进程的 Python `random` 和 `numpy.random` 的全局状态。
    如果不加干预，所有数据处理子进程（Worker）将会生成一模一样的随机数序列，
    导致数据增强（如随机插入 System Prompt）失去真正的随机性。

    【底层原理与逻辑】
    PyTorch 为了避免多线程同态化，会在底层自动给每个子进程的 `torch` 派发一个独立的专属种子
    （生成规则为：根据主进程seed随机出来的 base_seed + worker_id）。
    本函数的作用就是提取 PyTorch 生成的这个专属种子，并同步给 Python 和 Numpy。

    【为什么要有 % 2**32？】
    PyTorch 派发的种子是 64 位大整数。而 Python 原生的 random 以及 Numpy (MT19937算法)
    底层只接受 32 位无符号整数（范围 0 ~ 4294967295）。
    直接传入 64 位整数会导致程序抛出异常，因此必须进行取模截断适配。
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# 设置检查点
def lm_checkpoint(
    lm_config,
    weight="full_sft",
    model=None,
    optimizer=None,
    epoch=0,
    step=0,
    wandb=None,
    save_dir="checkpoints",
    **kwargs,
):
    os.makedirs(save_dir, exist_ok=True)

    moe_path = "_moe" if hasattr(lm_config, "use_moe") and lm_config.use_moe else ""
    ckp_path = f"{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}.pth"
    resume_path = f"{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}_resume.pth"

    if model is not None:
        from torch.nn.parallel import DistributedDataParallel

        if isinstance(model, DistributedDataParallel):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()

        ckp_tmp = ckp_path + ".tmp"
        torch.save({k: v.half() for k, v in state_dict.items()}, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)

        wandb_id = None
        if wandb:
            if hasattr(wandb, "get_run"):
                run = wandb.get_run()
                wandb_id = getattr(run, "id", None) if run else None
            else:
                wandb_id = getattr(wandb, "id", None)

        resume_data = {
            "model": state_dict,
            "optimizer": optimizer.state_dict(),  # type: ignore
            "epoch": epoch,
            "step": step,
            "world_size": dist.get_world_size() if dist.is_initialized() else 1,
            "wandb_id": wandb_id,
        }

        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, "state_dict"):
                    if isinstance(value, DistributedDataParallel):
                        resume_data[key] = value.module.state_dict()
                    else:
                        resume_data[key] = value.state_dict()
                else:
                    resume_data[key] = value

        resume_tmp = resume_path + ".tmp"
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)

    else:  # 加载模式
        if os.path.exists(resume_path):
            ckp_data = torch.load(resume_path, map_location="cpu")
            saved_ws = ckp_data.get("world_size", 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1

            if saved_ws != current_ws:
                ckp_data["step"] = ckp_data["step"] * saved_ws // current_ws
                Logger(
                    f"GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data['step']}"
                )

            return ckp_data
        return None


# 初始化模型
def init_model(
    lm_config,
    from_weight="pretrain",
    tokenizer_path=None,
    save_dir="../out",
    device="cuda",
):
    from transformers import AutoTokenizer
    from model.MizukiModel import MizukiMindForCausalLM

    # 如果没有指定 tokenizer_path，使用项目根目录下的 model 文件夹
    if tokenizer_path is None:
        # 获取当前文件所在目录的父目录（项目根目录）
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        tokenizer_path = os.path.join(project_root, "model")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    model = MizukiMindForCausalLM(lm_config).to(device)  # type: ignore

    if from_weight != "none":
        moe_suffix = (
            "_moe" if hasattr(lm_config, "use_moe") and lm_config.use_moe else ""
        )
        weight_path = (
            f"{save_dir}/{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth"
        )

        weights = torch.load(weight_path, map_location=device)

        model.load_state_dict(weights, strict=False)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    Logger(f"所加载Model可训练参数：{total_params / 1e6:.3f} 百万")

    return model, tokenizer


class SkipBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler  #
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        batch = []  # 当前批次
        skipped = 0  # 已跳过的批次数

        for idx in self.sampler:
            batch.append(idx)  # 添加样本到当前批次

            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1  # 增加跳过计数
                    batch = []  # 清空批次，不返回
                    continue  # 跳过这个批次

                yield batch
                batch = []  # 重置批次

        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    def __len__(self):
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size

        return max(0, total_batches - self.skip_batches)
