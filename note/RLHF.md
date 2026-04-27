# RLHF 与偏好对齐算法笔记 (DPO / PPO / GRPO)

## 一、 DPO (Direct Preference Optimization)

DPO 的核心思想是通过数学推导，将原本需要 Reward Model 的强化学习过程，等价替换为一个简单的分类交叉熵损失函数，直接在偏好数据集上微调策略模型。

### 1. Log Ratio（对数几率）

$$
\text{pi\_logratios} = \log \pi(y_w) - \log \pi(y_l) = \log \frac{\pi(y_w)}{\pi(y_l)}
$$

$$
\text{ref\_logratios} = \log \pi_{ref}(y_w) - \log \pi_{ref}(y_l) = \log \frac{\pi_{ref}(y_w)}{\pi_{ref}(y_l)}
$$

其中：
* $y_w$ = chosen（人类偏好的回答，对应的整个序列概率）
* $y_l$ = rejected（人类不偏好的回答，对应的整个序列概率）
* $\pi$ = 策略模型（当前训练的 Actor 模型）
* $\pi_{ref}$ = 参考模型（冻结的预训练/SFT 模型）

### 2. DPO 核心损失函数

$$
\mathcal{L}_{DPO} = -\log \sigma\left( \beta \cdot \left( \text{pi\_logratios} - \text{ref\_logratios} \right) \right)
$$

展开后：

$$
\mathcal{L}_{DPO} = -\log \sigma\left( \beta \cdot \left( \log \frac{\pi(y_w)}{\pi(y_l)} - \log \frac{\pi_{ref}(y_w)}{\pi_{ref}(y_l)} \right) \right)
$$

### 3. Sigmoid 函数

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

### 4. 损失简化形式

令 $\Delta = \text{pi\_logratios} - \text{ref\_logratios}$，表示**“策略模型相比于参考模型，有多大程度更倾向于选择 chosen 而不是 rejected”**，则：

$$
\mathcal{L}_{DPO} = -\log \sigma(\beta \cdot \Delta) = \log(1 + e^{-\beta \cdot \Delta})
$$

### 5. 序列总对数概率

在标准 DPO 中，必须计算整个句子的**联合概率（连乘）**，转换到对数空间即为**求和**，**绝不能除以序列长度取均值**：

$$
\log P(Y) = \sum_{i=1}^{T} \mathbb{1}_{mask_i} \cdot \log p_i
$$

其中：
* $T$ = 序列长度
* $\mathbb{1}_{mask_i}$ = 掩码（有效 token 为 1，无效为 0）
* *注：如果除以长度（Mean），会破坏 DPO 假定的 Bradley-Terry 奖励模型，导致模型无法正确识别长短句的真实偏好（除非使用 SimPO 等专门修改了底层公式的变体算法）。*

### 6. 参数说明

| 符号 | 含义 | 典型值 |
| :--- | :--- | :--- |
| $\beta$ | 温度/KL惩罚系数，控制模型偏离参考模型的程度 | 0.1 ~ 0.5 |
| $y_w$ | 偏好的回答 (chosen) | - |
| $y_l$ | 不偏好的回答 (rejected) | - |
| $\pi$ | 当前策略模型 (Policy) | - |
| $\pi_{ref}$ | 参考模型 (Reference) | - |

---

## 二、 PPO (Proximal Policy Optimization)

PPO 是 OpenAI 在 InstructGPT 和 ChatGPT 中使用的经典 RLHF 算法。它的特点是极其消耗显存，需要同时在显存中维持 4 个模型。

### 1. PPO 的四大模型
* **Actor Model ($\pi_\theta$)**：正在训练的策略模型。
* **Reference Model ($\pi_{ref}$)**：冻结的 SFT 模型，用于计算 KL 散度，防止 Actor 彻底放飞自我。
* **Reward Model ($r_\phi$)**：打分模型，输入 prompt + response，输出一个标量分数（Reward）。
* **Critic/Value Model ($V_\omega$)**：价值模型，预测当前状态（已生成的 token）未来能拿多少总分，用于计算 Advantage（优势函数）。

### 2. 优势函数 (Advantage)
Advantage 衡量的是“当前生成的回复，比 Critic 预测的平均水平要好多少”：
$$
\hat{A}_t = R(y) - V_\omega(x)
$$

### 3. PPO 核心损失函数
PPO 通过截断（Clip）机制，保证每次模型更新的步子不会迈得太大（即 Trust Region）：

$$
\mathcal{L}^{CLIP}(\theta) = -\mathbb{E} \left[ \min \left( \frac{\pi_\theta(y_t|x)}{\pi_{old}(y_t|x)} \hat{A}_t, \ \text{clip}\left(\frac{\pi_\theta(y_t|x)}{\pi_{old}(y_t|x)}, 1-\epsilon, 1+\epsilon\right) \hat{A}_t \right) \right]
$$

### 4. 完整的 PPO 目标函数
最终的训练目标是最大化奖励，同时最小化与 Reference 模型的差距：

$$
\text{Obj}_{PPO} = \mathcal{L}^{CLIP}(\theta) - \beta \cdot \mathbb{D}_{KL}[\pi_\theta || \pi_{ref}]
$$

---

## 三、 GRPO (Group Relative Policy Optimization)

GRPO 是 DeepSeek (如 DeepSeekMath, DeepSeek-R1) 提出并带火的革命性强化学习算法。它的核心贡献是**彻底干掉了极其占用显存的 Critic (Value) 模型**，极大地降低了训练 RL 的硬件门槛。

### 1. 组内相对评分 (Group Relative Advantage)
GRPO 不再依赖 Critic 模型来预测基准分数（Baseline），而是对同一个 Prompt $x$，让模型并行生成 $G$ 个不同的回复 $\{y_1, y_2, \dots, y_G\}$。

通过 Reward Model（或基于规则的判题系统，如 R1 中的数学答案比对）计算出这 $G$ 个回复的奖励分 $\{r_1, r_2, \dots, r_G\}$，然后**在组内进行 Z-score 标准化**，直接得到优势函数：

$$
\tilde{A}_i = \frac{r_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r})}
$$
*如果 $\tilde{A}_i > 0$，说明这个回复比同组的其他回答要好，模型应该增加生成它的概率。*

### 2. GRPO 核心损失函数
对组内的 $G$ 个样本求平均损失，去掉了 Value 模型的约束，完全依赖组内相对优势进行更新：

$$
\mathcal{L}_{GRPO}(\theta) = -\frac{1}{G} \sum_{i=1}^{G} \left[ \min \left( \frac{\pi_\theta(y_i|x)}{\pi_{old}(y_i|x)} \tilde{A}_i, \ \text{clip}\left(\dots, 1-\epsilon, 1+\epsilon\right) \tilde{A}_i \right) - \beta \mathbb{D}_{KL}\left(\pi_\theta(y_i|x) || \pi_{ref}(y_i|x)\right) \right]
$$

### 3. KL 散度惩罚的精确计算
在 GRPO 中，DeepSeek 通常使用更精确的 KL 散度无偏估计公式，而不是简单的对数差：

$$
\mathbb{D}_{KL}[\pi_\theta || \pi_{ref}] = \frac{\pi_{ref}(y_i|x)}{\pi_\theta(y_i|x)} - \log \frac{\pi_{ref}(y_i|x)}{\pi_\theta(y_i|x)} - 1
$$