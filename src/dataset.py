import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class HuggingDataset(Dataset):
    def __init__(
        self, dataset_name, text_field, seq_len, device, separate_token, tokenizer
    ):
        self.dataset_name = dataset_name
        self.text_field = text_field
        self.seq_len = seq_len
        self.device = device
        self.separate_token = separate_token
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.data = self._load_data()

    def _load_data(self):
        dataset = load_dataset(self.dataset_name)["train"]
        return dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx][self.text_field]
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        separate_token_id = self.tokenizer.encode(
            self.separate_token, add_special_tokens=False
        )[0]

        while len(tokens) < self.seq_len - 1:
            tokens.append(separate_token_id)
            next_text_idx = (idx + 1) % len(self.data)
            next_text = self.data[next_text_idx][self.text_field]
            next_tokens = self.tokenizer.encode(next_text, add_special_tokens=False)
            tokens.extend(next_tokens)
            idx = next_text_idx

        tokens = tokens[: self.seq_len - 1]
        tokens.append(separate_token_id)

        input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(self.device)

        return input_ids[0]
