
import torch.nn as nn
import torch


class BiLSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, pad_idx):
        super(BiLSTMModel, self).__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
      embedded = self.embedding(x)
      lstm_out, _ = self.lstm(embedded)
      attn_scores = self.attention(lstm_out).squeeze(-1)
      mask = (x != pad_idx)
      attn_scores[~mask] = float('-inf')
      attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)
      context = torch.sum(attn_weights * lstm_out, dim=1)
      output = self.fc(self.dropout(context))
      return output


