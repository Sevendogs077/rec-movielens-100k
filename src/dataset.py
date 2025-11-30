import os
import pandas as pd
import torch
from torch.utils.data import Dataset

class MovieLensDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()

        abs_path = os.path.abspath(data_path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"找不到数据文件！\n程序试图寻找: {abs_path}\n请检查你的'当前运行目录'是否在项目根目录下。")

        print(f"成功加载数据: {abs_path}")

        names = ['user_id', 'item_id', 'rating', 'timestamp']
        df = pd.read_csv(abs_path, sep='\t', header=None, names=names)

        # 将 user_id, item_id 和 0 对齐
        df['user_id'] -= 1
        df['item_id'] -= 1

        self.users = torch.tensor(df['user_id'].values, dtype=torch.long)
        self.items = torch.tensor(df['item_id'].values, dtype=torch.long)
        self.ratings = torch.tensor(df['rating'].values, dtype=torch.float32)
        self.timestamps = torch.tensor(df['timestamp'].values, dtype=torch.long)

        self.num_users = int(df['user_id'].max()) + 1
        self.num_items = int(df['item_id'].max()) + 1

    def __len__(self):
        return self.ratings.shape[0]

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]
