import torch
import torch.nn as nn

class LstmModule(nn.Module):
    def __init__(self, input_feature=24, input_seq_len=24, output_seq_len=96, hidden_size=128, num_layers=2, dropout=0.01):
        super().__init__()
        self.input_linear = nn.Linear(input_feature, hidden_size)
        self.encoder_lstm = nn.LSTM(hidden_size, hidden_size, num_layers, dropout=dropout)
        self.decoder_linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_seq_len * hidden_size, hidden_size),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(hidden_size, output_seq_len),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.input_linear(x)
        out, (_, _) = self.encoder_lstm(x)
        return self.decoder_linear(out)

class LstmModelCustomDecoder(LstmModule):
    def __init__(self, custom_decoder, **kwargs):
        super().__init__(**kwargs)
        self.decoder_linear = custom_decoder

class LstmUpsampleModule(nn.Module):
    def __init__(self, input_size, num_layers=2, upscale=4):
        super().__init__()
        self.input_size = input_size
        self.upscale = upscale
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=input_size, num_layers=num_layers)

    def forward(self, x):
        result_x = []
        for seq in range(x.shape[1]):
            lstm_h, lstm_c = None, None
            data_out = x[:, seq, :]
            data = [data_out]
            for _ in range(self.upscale - 1):
                if lstm_h is None or lstm_c is None:
                    data_out, (lstm_h, lstm_c) = self.lstm(data_out)
                else:
                    data_out, (lstm_h, lstm_c) = self.lstm(data_out, (lstm_h, lstm_c))
                data.append(data_out)
            for _ in range(self.upscale):
                data[_] = data[_].unsqueeze(1)
            result_x.append(torch.cat(data, dim=1))
        return torch.cat(result_x, dim=1)