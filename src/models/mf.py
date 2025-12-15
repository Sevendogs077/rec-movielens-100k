import torch
import torch.nn as nn
from .base import BaseModel

class MatrixFactorization(BaseModel):

    REQUIRED_FEATURES = ['user_id', 'item_id']

    def __init__(self, feature_dims, embedding_dim, **kwargs):
        super().__init__(feature_dims)

        # Lock feature names with its index
        self.feature_names = self.REQUIRED_FEATURES

        # Initial embedding
        self.num_embeddings = int(sum(feature_dims.values()))
        self.embedding_dim = int(embedding_dim)
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)

        # Compute offsets
        # Shift indices to separate User IDs and Item IDs in the global table
        # e.g., User 0 -> Index 0; Item 0 -> Index N_users
        feature_sizes = [feature_dims[name] for name in self.feature_names]
        # [0, num_users]
        offsets = torch.tensor((0, *feature_sizes[:-1]), dtype=torch.long)
        # Not trained
        self.register_buffer('offsets', torch.cumsum(offsets, dim=0))

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, mean=0, std=0.01)

    def forward(self, inputs):
        # Stack inputs
        feature_ids = [inputs[name] for name in self.feature_names]
        feature_ids = torch.stack(feature_ids, dim=1) # stack at dim 1

        ''' 
        note:
        stack's dim means the index where the new dimension is inserted.

        N: the number of tensors in the list (len(inputs)).
        S: the shape of each tensor (e.g., [3]).

        >>> inputs = [tensor([1,2,3]), tensor([10,20,30])] 
        # N=2 (samples), S=[3] (features)

        # Case A: Stack at dim 0
        # New Shape: [<insert N>, S] -> [2, 3]
        >>> output = torch.stack(inputs, dim=0) 
            [tensor[1, 2, 3],
             tensor[10, 20, 30]]

        # Case B: Stack at dim 1
        # New Shape: [S, <insert N>] -> [3, 2]
        >>> output = torch.stack(inputs, dim=1) 
            [tensor([1, 10]),
             tensor([2, 20]),
             tensor([3, 30])]
        '''

        # Add offsets (Broadcast)
        # feature_ids: [B, 2] + offsets: [2] -> feature_ids: [B, 2]
        feature_ids = feature_ids + self.offsets

        # Embedding lookup
        # [B, 2] -> [B, 2, Dim]
        emb = self.embedding(feature_ids)

        # Split
        # [B, 2, Dim] -> [B, Dim]
        user_emb = emb[:, 0, :]
        item_emb = emb[:, 1, :]

        # Dot product
        output = torch.sum(user_emb * item_emb, dim=1)

        return output