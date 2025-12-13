import torch
from .base import BaseDataset

class InteractionDataset(BaseDataset):
    def __init__(self, data_path, **kwargs):
        super().__init__(data_path)

        self.user_ids = torch.tensor(self.df['user_id'].values, dtype=torch.long)
        self.item_ids = torch.tensor(self.df['item_id'].values, dtype=torch.long)
        self.labels = torch.tensor(self.df['rating'].values, dtype=torch.float32)

    def __getitem__(self, idx):
        x = {
            'user_id': self.user_ids[idx],
            'item_id': self.item_ids[idx]
        }

        return x, self.labels[idx]
