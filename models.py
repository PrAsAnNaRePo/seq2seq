from requests import ReadTimeout
from torch import nn
import data
import torch

class Encoder(nn.Module):
    def __init__(self, input_size, embed_dim, hidden_size, num_layers, dropout):
        super().__init__()
        self.embed = nn.Embedding(input_size, embed_dim)
        self.lstms = nn.LSTM(input_size=embed_dim, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(self.embed(x))
        outputs, (hidden, cell) = self.lstms(x)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, input_size, embed_dim, hidden_size, num_layers, dropout, output_size):
        super().__init__()
        self.embed = nn.Embedding(input_size, embed_dim)
        self.lstms = nn.LSTM(input_size=embed_dim, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, out_features=output_size)
    
    def forward(self, x, hidden, cell):
        x = x.unsqueeze(0)
        x = self.dropout(self.embed(x))
        output, (hidden, cell) = self.lstms(x, (hidden, cell))
        return self.fc(output).squeeze(0), hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, target):
        hidden, cell = self.encoder(x)
        outputs = torch.zeros(target.shape[0], x.shape[1], len(data.desc_field.vocab), device='cuda')

        x = target[0]
        for t in range(1, target.shape[0]):
            output, hidden, cell = self.decoder(x, hidden, cell)
            outputs[t] = output
            x = output.argmax(1)
        return outputs

