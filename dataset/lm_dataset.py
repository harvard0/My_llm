from torch.utils.data import Dataset
import torch
import json
import os
import random
from datasets import load_dataset, Features, Sequence, Value

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class PretrainDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int = 128):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = load_dataset("json", data_files=self.data_path, split="train")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        tokens = self.tokenizer(
            str(sample["text"]),
            add_special_tokens=False,
            max_length=self.max_length - 2,
            truncation=True,
        ).input_ids

        # Add special tokens
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]

        # Pad to max length
        input_ids = tokens + [self.tokenizer.pad_token_id] * (
            self.max_length - len(tokens)
        )

        # Convert to tensor
        input_ids = torch.tensor(input_ids, dtype=torch.long)

        # Create labels
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100

        # Create attention mask
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        return input_ids, attention_mask, labels
