import os
import sys

# 📚 Python模块系统
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse  # 命令行参数解析
import time  # 时间统计
import torch  # PyTorch深度学习框架
import torch.nn.functional as F  # 神经网络函数
import torch.distributed as dist  # 分布式训练支持
from contextlib import nullcontext  # 上下文管理器
from torch import optim  # 优化器
from torch.nn.parallel import DistributedDataParallel  # 分布式数据并行
from torch.utils.data import DataLoader, DistributedSampler  # 数据加载

# Mizuki Mind相关组件
from model.MizukiModel import MizukiMindConfig  # 模型配置
from dataset.lm_dataset import DPODataset  # DPO数据集
from trainer.trainer_utils import (  # 训练工具函数
    get_lr,
    Logger,
    is_main_process,
    lm_checkpoint,
    init_distributed_mode,
    seed_worker,
    setup_seed,
    init_model,
    SkipBatchSampler,
)


def logits_to_log_probs(logits, labels):
    # logits shape: (batch_size, seq_len, vocab_size)
    # labels shape: (batch_size, seq_len)
    # log_probs shape: (batch_size, seq_len)
    # 词表logits做softmax操作得到普通概率，再包裹log函数转换为log概率
    log_probs = F.log_softmax(logits, dim=2)
    # 从log词表概率里选出label对应的log概率
    # 也就是从拿到token在其对应位置的概率
    log_probs_per_token = torch.gather(
        log_probs, dim=2, index=labels.unsqueeze(2)
    ).squeeze(-1)
    return log_probs_per_token


# 公式：L = -log(σ(β * (π(y_w) - π(y_l) - (π_ref(y_w) - π_ref(y_l)))))
def dpo_loss(ref_log_probs, policy_log_probs, mask, beta):
    # ref_log_probs 和 policy_log_probs  shape: (batch_size, seq_len)
    ref_log_probs = (ref_log_probs * mask).sum(dim=1)
    policy_log_probs = (policy_log_probs * mask).sum(dim=1)

    # 分别获取chosen和rejected的ref和policy的log概率
    batch_size = ref_log_probs.shape[0]
    chosen_ref_log_probs = ref_log_probs[: batch_size // 2]
    reject_ref_log_probs = ref_log_probs[batch_size // 2 :]
    chosen_policy_log_probs = policy_log_probs[: batch_size // 2]
    reject_policy_log_probs = policy_log_probs[batch_size // 2 :]
    # 计算策略模型的log概率差异
    # log(π(y_w)) - log(π(y_l)) = log(π(y_w) / π(y_l))
    pi_logratios = chosen_policy_log_probs - reject_policy_log_probs
    # 参考模型的log概率差异
    # log(π_ref(y_w)) - log(π_ref(y_l))
    ref_logratios = chosen_ref_log_probs - reject_ref_log_probs
    # DPO损失计算
    # [log(π(y_w)/π(y_l)) - log(π_ref(y_w)/π_ref(y_l))]
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(
        beta * logits
    )  # # β:温度系数，控制对偏好的敏感度（通常 0.1~0.5）
    return loss.mean()


def train_epoch(
    epoch, loader, iters, ref_model, lm_config, start_step=0, wandb=None, beta=0.1
):
    start_time = time.time()
    last_step = start_step

    for step, batch in enumerate(loader, start=start_step + 1):
        x_chosen = batch["x_chosen"].to(args.device)
        x_rejected = batch["x_rejected"].to(args.device)
        y_chosen = batch["y_chosen"].to(args.device)
        y_rejected = batch["y_rejected"].to(args.device)
        mask_chosen = batch["mask_chosen"].to(args.device)
        mask_rejected = batch["mask_rejected"].to(args.device)
        attention_mask_chosen = batch["attention_mask_chosen"].to(args.device)
        attention_mask_rejected = batch["attention_mask_rejected"].to(args.device)

        x = torch.cat([x_chosen, x_rejected], dim=0)
        y = torch.cat([y_chosen, y_rejected], dim=0)
        mask = torch.cat([mask_chosen, mask_rejected], dim=0)
        attention_mask = torch.cat(
            [attention_mask_chosen, attention_mask_rejected], dim=0
        )

        # 📚 学习率调度
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        with autocast_ctx:
            # 📚 参考模型前向传播
            # 参考模型冻结，只用于计算baseline概率
            with torch.no_grad():
                ref_outputs = ref_model(x, attention_mask=attention_mask)
                ref_logits = ref_outputs.logits
            ref_log_probs = logits_to_log_probs(ref_logits, y)

            # 📚 策略模型前向传播
            # 策略模型是需要优化的主要模型
            outputs = model(x, attention_mask=attention_mask)
            logits = outputs.logits
            policy_log_probs = logits_to_log_probs(logits, y)

            # 📚 DPO损失计算
            dpo_loss_val = dpo_loss(ref_log_probs, policy_log_probs, mask, beta=beta)
            loss = dpo_loss_val + outputs.aux_loss
            loss = loss / args.accumulation_steps

        # 📚 反向传播
        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # 📚 训练日志
        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_dpo_loss = dpo_loss_val.item()
            current_aux_loss = outputs.aux_loss.item()
            current_lr = optimizer.param_groups[-1]["lr"]
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60

            Logger(
                f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}) loss:{current_loss:.4f} dpo_loss:{current_dpo_loss:.4f} aux_loss:{current_aux_loss:.4f} lr:{current_lr:.8f} epoch_Time:{eta_min}min:"
            )

            if wandb:
                wandb.log(
                    {
                        "loss": current_loss,
                        "dpo_loss": current_dpo_loss,
                        "aux_loss": current_aux_loss,
                        "lr": current_lr,
                        "epoch_Time": eta_min,
                    }
                )

        # 📚 模型保存
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = "_moe" if lm_config.use_moe else ""
            ckp = f"{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth"
            raw_model = (
                model.module if isinstance(model, DistributedDataParallel) else model
            )
            raw_model = getattr(raw_model, "_orig_mod", raw_model)
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)

            lm_checkpoint(
                lm_config,
                weight=args.save_weight,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                step=step,
                wandb=wandb,
                save_dir="../checkpoints",
            )
            model.train()
            del state_dict  # 释放内存

        # 释放显存，加快垃圾回收
        del (
            x_chosen,
            x_rejected,
            y_chosen,
            y_rejected,
            mask_chosen,
            mask_rejected,
            x,
            y,
            mask,
        )
        del (
            ref_outputs,
            ref_logits,
            ref_log_probs,
            outputs,
            logits,
            policy_log_probs,
            loss,
        )

    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    """
    DPO主函数：直接偏好优化脚本的入口点
    
    📚 DPO训练流程：
    1. 准备策略模型和参考模型
    2. 加载偏好数据（chosen vs rejected）
    3. 同时前向传播计算两种模型的概率
    4. 计算DPO损失并优化策略模型
    5. 迭代直到收敛
    """

    # 📚 命令行参数解析
    parser = argparse.ArgumentParser(
        description="MizukiMind DPO (Direct Preference Optimization)"
    )

    # ========== 基础训练参数 ==========
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument(
        "--save_weight", default="dpo", type=str, help="保存权重的前缀名"
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="训练轮数（DPO通常1-2轮）"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="batch size（DPO batch较小）"
    )

    # 📚 DPO学习率知识点
    # DPO学习率通常很小，避免过度优化导致遗忘
    # 建议不超过5e-8
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=4e-8,
        help="初始学习率（建议<=5e-8避免遗忘）",
    )

    # ========== 硬件配置 ==========
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="训练设备",
    )
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")

    # ========== 训练策略 ==========
    parser.add_argument(
        "--accumulation_steps", type=int, default=1, help="梯度累积步数"
    )
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存间隔")

    # ========== 模型架构参数 ==========
    parser.add_argument("--hidden_size", default=768, type=int, help="隐藏层维度")
    parser.add_argument("--num_hidden_layers", default=8, type=int, help="隐藏层数量")
    parser.add_argument(
        "--max_seq_len", default=1024, type=int, help="训练的最大截断长度"
    )
    parser.add_argument(
        "--use_moe",
        default=0,
        type=int,
        choices=[0, 1],
        help="是否使用MoE架构（0=否，1=是）",
    )

    # ========== DPO数据和模型参数 ==========
    # 📚 DPO数据格式知识点
    # 数据包含chosen（偏好）和rejected（不偏好）回答配对
    parser.add_argument(
        "--data_path", type=str, default="../dataset/dpo.jsonl", help="DPO训练数据路径"
    )

    # 📚 DPO权重继承知识点
    # DPO通常基于SFT模型进行对齐优化
    parser.add_argument(
        "--from_weight",
        default="full_sft",
        type=str,
        help="基于哪个权重训练（通常是SFT模型）",
    )
    parser.add_argument(
        "--from_resume",
        default=0,
        type=int,
        choices=[0, 1],
        help="是否自动检测&续训（0=否，1=是）",
    )

    # 📚 DPO beta参数知识点
    # beta控制优化强度，0.1-0.5是常见范围
    parser.add_argument(
        "--beta", default=0.15, type=float, help="DPO中的beta参数（控制优化强度）"
    )

    # ========== 实验跟踪 ==========
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument(
        "--wandb_project", type=str, default="MizukiMind-DPO", help="wandb项目名"
    )

    # ========== 其他参数 ==========
    parser.add_argument(
        "--use_compile",
        default=0,
        type=int,
        choices=[0, 1],
        help="是否使用torch.compile加速（0=否，1=是）",
    )
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MizukiMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
    )
    ckp_data = (
        lm_checkpoint(lm_config, weight=args.save_weight, save_dir="../checkpoints")
        if args.from_resume == 1
        else None
    )

    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = (
        nullcontext() if device_type == "cpu" else torch.amp.autocast(dtype=dtype)
    )

    # ========== 4. 配置wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb

        wandb_id = ckp_data.get("wandb_id") if ckp_data else None
        resume = "must" if wandb_id else None
        wandb_run_name = f"MizukiMind-DPO-{'MoE' if args.use_moe else ''}-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(
            project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume
        )

    # ========== 5. 定义模型和参考模型 ==========
    # 📚 DPO双模型架构
    # 策略模型：需要优化的模型
    # 参考模型：冻结的baseline模型
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    Logger(
        f"策略模型总参数量：{sum(p.numel() for p in model.parameters()) / 1e6:.3f} M"
    )

    # 📚 参考模型初始化
    # 参考模型与策略模型初始权重相同，但完全冻结
    ref_model, _ = init_model(lm_config, args.from_weight, device=args.device)
    ref_model.eval()  # 设为评估模式
    ref_model.requires_grad_(False)  # 冻结所有参数
    Logger(
        f"参考模型总参数量：{sum(p.numel() for p in ref_model.parameters()) / 1e6:.3f} M"
    )

    # 📚 DPO数据集
    train_ds = DPODataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.amp.GradScaler(enabled=(args.dtype == "float16"))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data["model"])
        optimizer.load_state_dict(ckp_data["optimizer"])
        scaler.load_state_dict(ckp_data["scaler"])
        start_epoch = ckp_data["epoch"]
        start_step = ckp_data.get("step", 0)

    # ========== 7. 编译和分布式包装 ==========
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger("torch.compile enabled")
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        # 1. 设置分布式采样器的 epoch（DDP 专属，自带安全的隔离洗牌逻辑）
        if train_sampler:
            train_sampler.set_epoch(epoch)

        # 2. 📚 局部隔离洗牌（单卡/自定义 Sampler 专属）
        g = torch.Generator()
        g.manual_seed(42 + epoch)
        # 只让 randperm 使用这个专属的 Generator，绝不污染全局环境！
        indices = torch.randperm(len(train_ds), generator=g).tolist()

        # 📚 断点续训处理：
        # 第一个epoch且有检查点时，跳过已训练的step
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(
            train_sampler or indices, args.batch_size, skip
        )
        loader = DataLoader(
            train_ds,
            batch_sampler=batch_sampler,
            num_workers=args.num_workers,
            worker_init_fn=seed_worker,
            pin_memory=True,  # 加速CPU->GPU数据传输
            drop_last=False,
        )

        if skip > 0:
            Logger(
                f"Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始"
            )
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)

    # ========== 9. 清理分布式进程 ==========
    if dist.is_initialized():
        dist.destroy_process_group()
