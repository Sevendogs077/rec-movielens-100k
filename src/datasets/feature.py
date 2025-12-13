import torch
from .base import BaseDataset

class FeatureDataset(BaseDataset):
    def __init__(self, data_path: str, **kwargs):
        super().__init__(data_path)

        self.user_ids = torch.tensor(self.df['user_id'].values, dtype=torch.long)
        self.item_ids = torch.tensor(self.df['item_id'].values, dtype=torch.long)
        self.ratings = torch.tensor(self.df['rating'].values, dtype=torch.float32)

