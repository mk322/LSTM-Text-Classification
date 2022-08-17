import torch.nn as nn
import torch

class LSTM(torch.nn.Module) :
    def __init__(self, vocab_size, embedding_dim, hidden_dim, glove_weights, dropout=0.5) :
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embeddings.weight.data.copy_(torch.from_numpy(glove_weights))
        self.embeddings.weight.requires_grad = False ## freeze embeddings
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 2)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, l):
        x = self.embeddings(x)
        x = self.dropout(x)
        lstm_out, (ht, ct) = self.lstm(x)
        return self.linear(ht[-1])


a = torch.randn(4, 4)
print(a)
#tensor([[-1.2360, -0.2942, -0.1222,  0.8475],
#        [ 1.1949, -1.1127, -2.2379, -0.6702],
 #       [ 1.5717, -0.9207,  0.1297, -1.8768],
 #       [-0.6172,  1.0036, -0.6060, -0.2432]])
print(torch.max(a, 1))
print(torch.max(a, 1)[0])
#torch.return_types.max(values=tensor([0.8475, 1.1949, 1.5717, 1.0036]), indices=tensor([3, 0, 0, 1]))