import torch
from torch.utils.data import Dataset


class MTDataset(Dataset):
    def __init__(self, tokenized_source_list, tokenized_target_list, dev):
        self.tokenized_source_list = tokenized_source_list
        self.tokenized_target_list = tokenized_target_list
        self.device = dev

    def __len__(self):
        return len(self.tokenized_source_list["input_ids"])

    def __getitem__(self, idx):
        source_ids, attention_mask, target_ids = self.tokenized_source_list.input_ids[idx].clone().detach().to(self.device), \
                                 self.tokenized_source_list.attention_mask[idx].clone().detach().to(self.device), \
                                 self.tokenized_target_list.input_ids[idx].clone().detach().to(self.device)
        return source_ids, attention_mask, target_ids
