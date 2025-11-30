import torch
import torch.nn as nn

class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, num_features):
        super().__init__()
        # 在矩阵分解中，user 和 item 的特征向量长度必须相等才可以进行点积，因此 embedding_dim 为一个统一的值
        self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=num_features)
        self.item_embedding = nn.Embedding(num_embeddings=num_items, embedding_dim=num_features)
        # 正态分布初始化
        nn.init.normal_(self.user_embedding.weight, mean=0, std=0.01)
        nn.init.normal_(self.item_embedding.weight, mean=0, std=0.01)

    def forward(self, user_ids, item_ids):
        user_embedding = self.user_embedding(user_ids)
        item_embedding = self.item_embedding(item_ids)
        output = torch.sum(user_embedding * item_embedding, dim=1)
        return output



