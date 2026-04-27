import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import GenerationMixin, PreTrainedModel, PretrainedConfig
from transformers.activations import ACT2FN
from transformers.modeling_outputs import CausalLMOutputWithPast


class MizukiMindConfig(PretrainedConfig):
    model_type = "mizukimind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 768, # 512 -> 768
        intermediate_size: int | None = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 4, # 2 -> 4
        head_dim: int | None = None,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05, 
        rope_theta: int = 1000000,
        inference_rope_scaling: bool = False,
        flash_attn: bool = True,
        ############ MoE ############
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.01,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = math.ceil(hidden_size * math.pi / 64) * 64 # 复现minimind3
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim or hidden_size // num_attention_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attn = flash_attn
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.scoring_func = scoring_func

        self.rope_scaling = (
            {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 16,
                "original_max_position_embeddings": 2048,
                "attention_factor": 1.0,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-05):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    # _norm
    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    # forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * self._norm(x.float()).type_as(x)


def precompute_freqs_cis(
    dim: int,
    end: int = int(32 * 1024),
    rope_base: float = 1e6,
    rope_scaling: Optional[dict] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute RoPE frequency encoding.
    Args:
        dim (int): 嵌入维度
        end (int, optional): 最大位置编码长度. Defaults to int(32 * 1024).
        rope_base (float, optional): 基础频率. Defaults to 1e6.
        rope_scaling (Optional[dict], optional): YaRN 配置字典. Defaults to None.

    Returns:
        tuple: (freqs_cos, freqs_sin)
            freqs_cos (torch.Tensor): 余弦频率编码
            freqs_sin (torch.Tensor): 正弦频率编码
    """
    # 1. 初始化标准 RoPE 频率。
    # torch.arange(0, dim, 2) 生成 [0, 2, 4, ... dim-2]
    # 计算出的 freqs 就是标准的 1 / (base ** (2i / d))
    freqs, attn_factor = (
        1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)),
        1.0,
    )

    if rope_scaling is not None:
        # 2. 从配置字典中提取 YaRN 的超参数
        # orig_max: 模型预训练时的原始最大长度（例如 Llama-2 是 2048 或 4096）
        # factor: 要扩展的倍数 s (比如从 2k 扩展到 32k，factor 就是 16)
        # beta_fast (对应论文中的 α): 高频边界，波长比例大于此值的维度不缩放
        # beta_slow (对应论文中的 β): 低频边界，波长比例小于此值的维度全量缩放
        # attn_factor: 注意力温度补偿，由于距离拉长导致注意力分布发散（变平缓），需要乘上一个系数让注意力重新“聚焦”
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048),
            rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0),
            rope_scaling.get("beta_slow", 1.0),
            rope_scaling.get("attention_factor", 1.0),
        )

        # 只有当要推断的长度大于原始训练长度时，才应用缩放
        if end / orig_max > 1.0:
            # 3. 使用前文推导的公式，定义波长比例 b 到维度索引 i 的映射函数
            # b = orig_max / lambda
            # lambda = 2 * π * (rope_base ** (2i / dim))
            # i (inv_dim(b)) = dim * log(orig_max / (b * 2 * π)) / (2 * log(rope_base))
            def inv_dim(b):
                return (dim * math.log(orig_max / (b * 2 * math.pi))) / (
                    2 * math.log(rope_base)
                )

            # 4. 计算高频区和低频区的维度切分点
            # low: 不需要缩放的高频部分的最高索引
            # high: 需要完全缩放的低频部分的最低索引
            low, high = (
                max(math.floor(inv_dim(beta_fast)), 0),
                min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1),
            )

            # 5. 计算混合因子 γ (Ramp)
            # 在 low 之前，ramp 为 0；在 high 之后，ramp 为 1；在 low 和 high 之间，线性过渡。
            # clamp 函数限制了数值只能在 [0, 1] 之间。
            ramp = torch.clamp(
                (torch.arange(dim // 2, device=freqs.device).float() - low)
                / max(high - low, 0.001),
                0,
                1,
            )

            # 6. 频率融合公式：f'(i) = f(i) * ((1-γ) + γ/s)
            # 当 ramp=0 时（高频）：系数为 1，保持原频率不变。
            # 当 ramp=1 时（低频）：系数为 1/factor，即对频率进行线性插值缩放。
            # ramp在0-1之间时：平滑过渡。
            freqs = freqs * (1 - ramp + ramp / factor)

    # 7. 根据目标长度 end，生成位置索引向量 t
    t = torch.arange(end, device=freqs.device)

    # 8. 计算外积：将位置 t 与处理好的频率 freqs 相乘，得到每个位置的旋转角度 θ
    # [end, dim//2]
    freqs = torch.outer(t, freqs).float()

    # 9. 计算 Cos 和 Sin，并应用注意力补偿系数 (attn_factor)
    # [end, dim//2*2]
    # 这里是对半交错配对,不是相邻配对
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor

    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(
    q, k, cos, sin, position_ids=None, unsqueeze_dim=1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    应用 RoPE 旋转编码到查询和键向量
    Args:
        q (torch.Tensor): 查询向量
        k (torch.Tensor): 键向量
        cos (torch.Tensor): 余弦频率编码
        sin (torch.Tensor): 正弦频率编码
        position_ids (torch.Tensor, optional): 位置索引向量. Defaults to None.
        unsqueeze_dim (int, optional): 未压缩维度. Defaults to 1.

    Returns:
        tuple: (q_embed, k_embed)
            q_embed (torch.Tensor): 查询向量旋转后的结果
            k_embed (torch.Tensor): 键向量旋转后的结果
    """

    # 1. 初始化标准 RoPE 频率。
    # [a:b]->[-b:a]
    def rotate_half(x):
        return torch.cat(
            (-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]), dim=-1
        )

    # x_rotated = x * cos + rotated_half(x) * sin
    # [batch_size, seq_len, head, head_dim]
    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (
        rotate_half(q) * sin.unsqueeze(unsqueeze_dim)
    )
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (
        rotate_half(k) * sin.unsqueeze(unsqueeze_dim)
    )
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    GQA: Repeat key-value pairs.
    Args:
        x (torch.Tensor): 键值对张量
        [batch_size, seq_len, head, head_dim]
        n_rep (int): 重复次数

    Returns:
        torch.Tensor: 重复后的键值对
        [batch_size, seq_len, head * n_rep, head_dim]
    """
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, num_key_value_heads, n_rep, head_dim)
        .reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    """
    Multi-Head Attention layer.
    Supports Group Query Attention (GQA) and Flash Attention optimization.
    """

    def __init__(self, config: MizukiMindConfig):
        super().__init__()

        # 处理GQA：如果没有指定kv头数，则使用与query相同的头数
        # 三元运算符：condition ? value1 : value2
        self.num_key_value_heads = (
            config.num_key_value_heads
            if config.num_key_value_heads is not None
            else config.num_attention_heads
        )

        assert self.num_key_value_heads % 2 == 0, "num_key_value_heads must be even"

        # 注意力头配置
        self.n_local_heads = config.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.head_dim

        # 定义线性投影层 (无偏置，节省参数)
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=False
        )

        # RMSNorm层用于归一化
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # Dropout层用于正则化
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout

        # 检查是否支持Flash Attention
        self.flash = (
            hasattr(torch.nn.functional, "scaled_dot_product_attention")
            and config.flash_attn
        )

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],  # 修改为接收cos和sin
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache=False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # x: [batch_size, seq_len, hidden]
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        # Normalization
        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        # kv cache
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        # [bsz, seq_len, head, head_dim] -> [bsz, head * n_rep, seq_len, head_dim]
        xq, xk, xv = (
            xq.transpose(1, 2),
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2),
        )

        # 优先使用PyTorch 2.0+的scaled_dot_product_attention（Flash Attention实现）
        if (
            self.flash
            and seq_len > 1
            and (attention_mask is None or torch.all(attention_mask == 1))
        ):
            # 如果没有显式的attention_mask，直接传None让底层高效实现
            attn_mask = (
                None
                if attention_mask is None
                else attention_mask.view(bsz, 1, 1, -1)
                .expand(bsz, self.n_local_heads, seq_len, -1)
                .bool()
            )
            # F.scaled_dot_product_attention是PyTorch在新版本中提供的高效实现
            output = F.scaled_dot_product_attention(
                xq,
                xk,
                xv,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,  # 自回归（因果）注意力
            )
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)

            # causal mask: 上三角（对角线以上）置为 -inf，防止看到未来信息
            scores[:, :, :, -seq_len:] += torch.full(
                (seq_len, seq_len), float("-inf"), device=scores.device
            ).triu(1)

            # 如果有attention_mask(0/1)，将其扩展后转为 -1e9 的加性mask（掩掉pad位置）
            if attention_mask is not None:
                attn_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                scores += (1.0 - attn_mask) * -1e9

            output = (
                self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq)) @ xv
            )

        # [bsz, head * n_rep, seq_len, head_dim] -> [bsz, seq_len, n_local_heads * head_dim]
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv


class FeedForward(nn.Module):
    """
    Up-Down FeedForward layer.
    """

    def __init__(self, config: MizukiMindConfig):
        super().__init__()
        intermediate_size = config.intermediate_size or int(config.hidden_size * 8 / 3)
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class MoEGate(nn.Module):
    def __init__(self, config: MizukiMindConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts

        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux

        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(
            torch.empty((self.n_routed_experts, self.gating_dim))
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        # shape: [bsz * seq_len, hidden]
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states, self.weight, None)

        if self.scoring_func == "softmax":
            # shape: [bsz * seq_len, n_routed_experts]
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(
                f"insupportable scoring function for MoE gating: {self.scoring_func}"
            )

        # shape: [bsz * seq_len, top_k]
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        # Normalization：由于只取了 K 个专家，这 K 个专家的概率和不为 1，需要重新归一化
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        # Load Balancing（负载均衡）：惩罚那些分配极度不均的情况，鼓励模型将 Token 均匀地分发给所有专家
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            # shape: [bsz, seq_len * aux_topk]
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            # Sequence-level 序列级负载均衡
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                
                # shape: [bsz, n_routed_experts]，统计每条序列里各个专家被选中的次数
                ce = torch.zeros(
                    bsz, self.n_routed_experts, device=hidden_states.device
                ) 
                # 归一化为负载因子
                ce.scatter_add_(
                    dim=1,
                    index=topk_idx_for_aux_loss,
                    src=torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device),
                ).div_(seq_len * aux_topk / self.n_routed_experts)
                # 负载率 * 平均概率，高频选中（ce）且概率高的专家产生的loss大
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(
                    dim=1
                ).mean() * self.alpha
            # Batch-level 批次级负载均衡
            else:
                # shape: [bsz * seq_len * aux_topk, n_routed_experts]
                mask_ce = F.one_hot(
                    topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts
                )
                # shape: [n_routed_experts]，统计每个专家在批次中被选中的次数
                ce = mask_ce.float().mean(0)
                # shape: [n_routed_experts]，统计每个专家的平均概率
                Pi = scores_for_aux.mean(0)
                # shape: [n_routed_experts]，计算每个专家的负载因子fi = ce * n_routed_experts
                fi = ce * self.n_routed_experts
                # shape: [n_routed_experts]，计算每个专家的负载均衡损失
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = scores.new_zeros(1).squeeze()
        return topk_idx, topk_weight, aux_loss


class MoEFeedForward(nn.Module): 
    def __init__(self, config: MizukiMindConfig):
        super().__init__()
        self.config = config
        # 专家层
        self.experts = nn.ModuleList(
            [FeedForward(config) for _ in range(config.n_routed_experts)]
        )
        # 门控层
        self.gate = MoEGate(config)
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList(
                [FeedForward(config) for _ in range(config.n_shared_experts)]
            )

    def forward(self, x):
        identity = x
        orig_shape = x.shape

        # 使用门控机制选择专家
        topk_idx, topk_weight, aux_loss = self.gate(x)
        # 展开x以便处理 [bsz * seq_len, hidden]
        x = x.view(-1, x.shape[-1])

        # shape: [bsz * seq_len * top_k]
        flat_topk_idx = topk_idx.view(-1)
        
        if self.training:
            # 按照定义的num_experts_per_tok重复输入token
            # 每个token安排num_experts_per_tok个专家处理
            # shape: [bsz * seq_len * num_experts_per_tok, hidden]
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
            # y是空张量，和x形状相同
            y = torch.empty_like(x, dtype=x.dtype)
            # 遍历所有专家
            for i, expert in enumerate(self.experts):
                # 找到所有指向专家i的token
                # 然后将这些token输入专家i进行处理
                # 最后将结果放回y对应位置
                expert_out = expert(x[flat_topk_idx == i])
                if expert_out.shape[0] > 0:
                    y[flat_topk_idx == i] = expert_out.to(y.dtype)
                # 如果专家没分到 Token，会执行一段“空转”代码，防止分布式训练（DDP）时梯度计算图断裂导致报错。
                else:
                    y[flat_topk_idx == i] = expert_out.to(y.dtype) + 0 * sum(
                        p.sum() for p in expert.parameters()
                    )
            # 加权求和
            # 最后的y意义是每个token经过专家处理后的加权结果
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape)
        # 如果是推理阶段
        else:
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(
                *orig_shape
            )
        if self.config.n_shared_experts > 0:
            for expert in self.shared_experts:
                y = y + expert(identity)
        self.aux_loss = aux_loss
        return y

    @torch.no_grad()
    # MoE推理方法
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        # 使用cache，创建一个和x形状相同的零张量
        expert_cache = torch.zeros_like(x)
        # 对专家索引进行排序，最后是[0,0,0,1,1,2,2,2,...]这样的顺序
        # 分拣
        idxs = flat_expert_indices.argsort()
        # 统计每个专家被分配到的token数量
        # 打包
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        # 计算每个token对应的专家索引
        token_idxs = idxs // self.config.num_experts_per_tok
        # 对每个打包好的包进行处理
        for i, end_idx in enumerate(tokens_per_expert):
            # 计算当前包的起始位置
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            # 取出当前包对应的专家
            expert = self.experts[i]
            # 取出token对应的原始id
            exp_token_idx = token_idxs[start_idx:end_idx]
            # 取出token对应的数据
            expert_tokens = x[exp_token_idx]
            # 计算专家输出，一次性处理当前包的所有token
            # shape: [num_experts_per_tok, hidden]
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            # 加权
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            # 将结果散点加到缓存中对应位置
            # tips：既然 exp_token_idx 里已经记录了这 N 个 Token 的原先行号（比如 [0, 5, 12]），
            # 直接写 expert_cache[exp_token_idx] += expert_out 不行吗？
            # 在普通的 Python 逻辑里可以，但在追求极致并行的底层 CUDA 算子 scatter_add_ 中不行。
            # scatter_add_ 强制要求 index 的形状必须和数据源 src（也就是 expert_out）的形状完全一样大！ 
            # 因为 expert_out 是一个二维矩阵 [N, HiddenSize]，包含了大量的特征维度，
            # 所以我们必须把只有一维的行号 [N]，也强行拉伸成 [N, HiddenSize] 的矩阵。
            expert_cache.scatter_add_(
                0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out
            )

        return expert_cache

class MizukiMindBlock(nn.Module):
    """
    Connect Attention and FeedForward layers.
    """

    def __init__(self, layer_id: int, config: MizukiMindConfig):
        super().__init__()
        self.layer_id = layer_id

        self.attn = Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = FeedForward(config) if not config.use_moe else MoEFeedForward(config)

    def forward(
        self,
        hidden_state: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],  # 修改为接收cos和sin
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache=False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        residual = hidden_state
        hidden_state, present_key_value = self.attn(
            self.input_layernorm(hidden_state),
            position_embeddings,
            attention_mask,
            past_key_value,
            use_cache,
        )

        hidden_state += residual
        hidden_state = (
            self.mlp(self.post_attention_layernorm(hidden_state)) + hidden_state
        )
        return hidden_state, present_key_value


class MizukiMindModel(nn.Module):
    """
    MizukiMind model.
    """

    def __init__(self, config: MizukiMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.hidden_size = config.vocab_size, config.hidden_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(
            [
                MizukiMindBlock(layer_id, config)
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # precompute RoPE
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.head_dim,
            end=config.max_position_embeddings,
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )

        # register RoPE buffers
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[
            List[Tuple[torch.Tensor, torch.Tensor]] | List[None]
        ] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]] | List[None], int]:
        batch_size, seq_length = input_ids.shape

        # 兼容性检查：某些框架会传入包含.layers属性的对象，视为不携带past信息
        if hasattr(past_key_values, "layers"):
            past_key_values = None

        # past_key_values为每层的(past_k, past_v)列表，如果为None则创建与层数相同的None列表
        past_key_values = past_key_values or [None] * len(self.layers)

        # past_key_values[0] 形如 (k, v)，k.shape = [bsz, past_seq_len, n_kv_heads, head_dim]
        start_pos = (
            past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        )

        hidden_states = self.dropout(self.embed_tokens(input_ids))

        # self.freqs_cos/freqs_sin的shape为 [max_pos, head_dim]
        position_embeddings = (
            self.freqs_cos[start_pos : start_pos + seq_length],  # type: ignore
            self.freqs_sin[start_pos : start_pos + seq_length],  # type: ignore
        )

        # 逐层前向，通过zip把layer和对应的past_key_value配对
        presents = []
        for layer, past_key_value in zip(self.layers, past_key_values):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
            )
            presents.append(present)

        # normalization
        hidden_states = self.norm(hidden_states)

        aux_loss = sum(
            [layer.mlp.aux_loss for layer in self.layers if isinstance(layer.mlp, MoEFeedForward)],
            hidden_states.new_zeros(1).squeeze(),
        )

        return hidden_states, presents, aux_loss


class MizukiMindForCausalLM(PreTrainedModel, GenerationMixin):
    """
    MizukiMind model for causal language.
    """

    def __init__(self, config: Optional[MizukiMindConfig] = None):
        self.config = config or MizukiMindConfig()
        super().__init__(self.config)

        self.model = MizukiMindModel(self.config)
        self.lm_head = nn.Linear(
            self.config.hidden_size, self.config.vocab_size, bias=False
        )

        self.model.embed_tokens.weight = self.lm_head.weight

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[
            List[Tuple[torch.Tensor, torch.Tensor]] | List[None]
        ] = None,
        use_cache: bool = False,
        labels: Optional[torch.Tensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ):
        hidden_states, past_key_values, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )

        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            ).float()
            
        if past_key_values is not None and all(pkv is None for pkv in past_key_values):
            past_key_values = None
            
        output = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
        )
        output.aux_loss = aux_loss
        return output
