import torch
import torch.nn as nn
from .base import BaseModel
from .layers import MLP

class NeuMF(BaseModel):

    REQUIRED_FEATURES = ['user_id', 'item_id']

    def __init__(self, feature_dims, embedding_dim, mlp_layers, dropout):
        super().__init__(feature_dims)

        self.feature_name = self.REQUIRED_FEATURES

        self.embedding_dim = int(embedding_dim)
        self.num_embeddings = int(sum(feature_dims.values()) * 2) # NeuMF needs 2 embeddings
        self.embedding = nn.Embedding(embedding_dim=self.embedding_dim, num_embeddings=self.num_embeddings)

        # Offsets
        feature_sizes = list(feature_dims[name] for name in self.feature_names)
        offsets = torch.tensor(data=(0,feature_sizes[:-1]),dtype=torch.long)
        self.register_buffer('offsets', torch.cumsum(offsets, dim=0))

        self.mlp = MLP(
                    input_dim=embedding_dim,
                    hidden_dims=mlp_layers,
                    dropout=dropout,
                    output_layer=False,
                )

        predict_input_dim = embedding_dim + mlp_layers[-1]
        self.predict_layer = nn.Linear(predict_input_dim, 1, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.embedding.weight)

        nn.init.constant_(self.predict_layer.weight, 1.0)

        # Kaiming Normal Initialization
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        # inputs: {'f1': [idx1, idx2], 'f2': [idx1, idx2]}
        feature_ids = [inputs[name] for name in self.feature_names]
        feature_ids = torch.stack(feature_ids, dim=1)












# import torch
# import torch.nn as nn
# from .layers import MLP
#
# class NeuMF(nn.Module):
#     def __init__(self, num_users, num_items, embedding_dim, mlp_layers=None, dropout=0.2, **kwargs):
#         super().__init__()
#         self.user_embedding_mlp = nn.Embedding(num_embeddings=num_users, embedding_dim=embedding_dim)
#         self.item_embedding_mlp = nn.Embedding(num_embeddings=num_items, embedding_dim=embedding_dim)
#         self.user_embedding_gmf = nn.Embedding(num_embeddings=num_users, embedding_dim=embedding_dim)
#         self.item_embedding_gmf = nn.Embedding(num_embeddings=num_items, embedding_dim=embedding_dim)
#
#         predict_input_dim = embedding_dim + mlp_layers[-1]
#         self.predict_layer = nn.Linear(predict_input_dim, 1, bias=False)
#
#         self.mlp = MLP(
#             input_dim=embedding_dim * 2,
#             hidden_dims=mlp_layers,
#             dropout=dropout,
#             output_layer=False,
#         )
#
#         self._init_weights()
#
#     def _init_weights(self):
#         nn.init.normal_(self.user_embedding_mlp.weight, mean=0, std=0.01)
#         nn.init.normal_(self.item_embedding_mlp.weight, mean=0, std=0.01)
#         nn.init.normal_(self.user_embedding_gmf.weight, mean=0, std=0.01)
#         nn.init.normal_(self.item_embedding_gmf.weight, mean=0, std=0.01)
#
#         nn.init.constant_(self.predict_layer.weight, 1.0)
#
#         # Kaiming Normal Initialization
#         for m in self.mlp.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.kaiming_normal_(m.weight)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#
#     def forward(self, inputs):
#         user_ids = inputs['user_id']
#         item_ids = inputs['item_id']
#
#         user_emb_mlp = self.user_embedding_mlp(user_ids)
#         item_emb_mlp = self.item_embedding_mlp(item_ids)
#         user_emb_gmf = self.user_embedding_gmf(user_ids)
#         item_emb_gmf = self.item_embedding_gmf(item_ids)
#
#         # Left: GMF
#         gmf_output = user_emb_gmf * item_emb_gmf
#
#         # Right: MLP
#         mlp_input = torch.cat([user_emb_mlp, item_emb_mlp], dim=1)
#         mlp_output = self.mlp(mlp_input)
#
#         # Concatenation
#         fusion_vector = torch.cat([gmf_output, mlp_output], dim=1)
#         logits = self.predict_layer(fusion_vector)
#
#         # Squeeze
#         output = logits.view(-1)
#         return output