from torch import nn
import torch

class StationSeqEmbedding(nn.Module):
    def __init__(self, station_num=10, embedding_dim=6, seq_len=24):
        super().__init__()
        self.embedding = nn.Embedding(station_num, embedding_dim)
        self.seq_len = seq_len

    def forward(self, x, station):
        station = station.reshape(-1, 1)
        station = self.embedding(station)
        station = station.reshape(-1, 1, 1).repeat(1, self.seq_len, 1)
        x = torch.cat((x, station), dim=-1)
        return x