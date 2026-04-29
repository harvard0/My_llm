# 为什么DPO不需要随机 `SYSTEM PROMPT`
**SFT / PPO / GRPO** 是在教模型“如何回答”，所以需要用随机 `SYSTEM PROMPT` 来做数据增强，见多识广。

**DPO** 是在教模型“明辨是非”，所以必须保持案发现场（`Prompt`）的绝对静态和公平，严禁任何形式的随机篡改。所以在DPO之前不加入`SYSTEM PROMPT`。