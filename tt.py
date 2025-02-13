from torch import nn
import torch

temp_emb = nn.Embedding(num_embeddings=5, embedding_dim=4)
print(temp_emb.weight)

temp = torch.randint(5, size=(3,4))
print(temp)

a = temp_emb(temp)
print(a)
print(a.shape)