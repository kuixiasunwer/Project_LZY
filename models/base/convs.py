from torch import nn
import torch
import torch.nn.functional as F

class ResNetConv(nn.Module):
    def __init__(self, input_channel, res_output_channel, output_dims, x_dim=11, y_dim=11):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channel, 64, kernel_size=3),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.GELU(),
            nn.Conv2d(128, 32, kernel_size=3),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(32 * (x_dim - 6) * (y_dim - 6), output_dims) if 16 * (x_dim - 6) * (y_dim - 6) != output_dims else nn.Identity()
        )
        self.resnet_layers = nn.Sequential(
            nn.Conv2d(input_channel, res_output_channel, kernel_size=(x_dim, y_dim)),
            nn.Flatten(),
        )
        self.combine_layer = nn.Linear(output_dims + res_output_channel, output_dims)

    def forward(self, x):
        # (b * sqe_len, channel, x, y)
        conv_x = self.conv_layers(x)
        center_x = self.resnet_layers(x)
        x = torch.cat((center_x, conv_x), dim=-1)
        x = self.combine_layer(x)
        return x


class ResNetSmallConv(nn.Module):
    def __init__(self, input_channel, res_output_channel, output_dims, x_dim=11, y_dim=11):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channel, 8, kernel_size=3),
            nn.GELU(),
            nn.Conv2d(8, 4, kernel_size=3),
            nn.GELU(),
            nn.Conv2d(4, 2, kernel_size=3),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(2 * (x_dim - 6) * (y_dim - 6), output_dims) if 2 * (x_dim - 6) * (y_dim - 6) != output_dims else nn.Identity()
        )
        self.resnet_layers = nn.Sequential(
            nn.Conv2d(input_channel, res_output_channel, kernel_size=(x_dim, y_dim)),
            nn.Flatten(),
        )
        self.combine_layer = nn.Linear(output_dims + res_output_channel, output_dims)

    def forward(self, x):
        # (b * sqe_len, channel, x, y)
        conv_x = self.conv_layers(x)
        center_x = self.resnet_layers(x)
        x = torch.cat((center_x, conv_x), dim=-1)
        x = self.combine_layer(x)
        return x


class MultiResNetConv(nn.Module):
    def __init__(self, input_channel, res_output_channel, output_dims, resnet_conv_list=None):
        super().__init__()
        if resnet_conv_list is None:
            resnet_conv_list = [(11, 11)]
        self.input_channel = input_channel
        self.output_dims = output_dims
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channel, 8, kernel_size=3),
            nn.GELU(),
            nn.Conv2d(8, 4, kernel_size=3),
            nn.GELU(),
            nn.Conv2d(4, 2, kernel_size=3),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(2 * (11 - 6) * (11 - 6), output_dims)
        )

        layers_list = [
            nn.Sequential(
                nn.Conv2d(input_channel, res_output_channel, kernel_size=(x_dim, y_dim)),
                nn.Flatten(),
                nn.Linear(
                    ((11 - (x_dim - 1)) * (11 - (y_dim - 1))) * res_output_channel,
                    res_output_channel
                ),
            ) if (x_dim is not None and y_dim is not None) else None for x_dim, y_dim in resnet_conv_list
        ]
        layers_list = set(layers_list)
        if None in layers_list:
            layers_list.remove(None)
        layers_list = list(layers_list)

        self.resnet_layers_list = nn.ModuleList(
            layers_list
        )

        self.combine_layer = nn.Sequential(
            nn.Linear(len(layers_list) * res_output_channel + output_dims, input_channel // 2 + output_dims),
            nn.GELU(),
            nn.Linear(input_channel // 2 + output_dims, output_dims),
        )

    def forward(self, x):
        # (b * sqe_len, channel, x, y)
        conv_x = self.conv_layers(x)
        resnet_output = [
           model(x) for model in self.resnet_layers_list
        ]
        x = torch.cat((conv_x, *resnet_output), dim=-1)
        x = self.combine_layer(x)
        return x


class SpecialConv(nn.Module):
    def __init__(self, hidden_size, seq_input_dim=24, seq_output_dim=96, resolution=2, shared_layer=False, padding=False, relu_in_end=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.seq_input_dim = seq_input_dim
        self.seq_output_dim = seq_output_dim
        self.resolution = resolution
        self.shared_layer = shared_layer
        self.padding = padding
        if shared_layer:
            self.special_conv_layers = None
            self.special_conv_layers = nn.ModuleList(
                [
                    *[
                        nn.Sequential(
                            nn.Linear(hidden_size * resolution, hidden_size),
                            nn.GELU(),
                            nn.Linear(hidden_size, hidden_size // 2),
                            nn.GELU(),
                            nn.Linear(hidden_size // 2, seq_output_dim // seq_input_dim),
                            nn.ReLU() if relu_in_end else nn.Identity(),
                        )
                    ] * seq_input_dim,
                ]
            )
        else:
            if padding:
                self.special_conv_layers = nn.ModuleList(
                    [
                        *[
                            nn.Sequential(
                                nn.Linear(hidden_size * resolution, hidden_size),
                                nn.GELU(),
                                nn.Linear(hidden_size, hidden_size // 2),
                                nn.GELU(),
                                nn.Linear(hidden_size // 2, seq_output_dim // seq_input_dim),
                                nn.ReLU() if relu_in_end else nn.Identity(),

                            ) for _ in range(seq_input_dim)
                        ]
                    ]
                )
            else:
                self.special_conv_layers = nn.ModuleList(
                    [
                        *[
                            nn.Sequential(
                                nn.Linear(hidden_size * resolution, hidden_size),
                                nn.GELU(),
                                nn.Linear(hidden_size, hidden_size // 2),
                                nn.GELU(),
                                nn.Linear(hidden_size // 2, seq_output_dim // seq_input_dim),
                                nn.ReLU() if relu_in_end else nn.Identity(),
                            ) for _ in range(seq_input_dim - (resolution - 1))
                        ],
                        *[
                            nn.Sequential(
                                nn.Linear(hidden_size * (resolution - (i + 1)), hidden_size * (resolution - 1) // 2),
                                nn.GELU(),
                                nn.Linear(hidden_size * (resolution - 1) // 2, hidden_size * (resolution - 1) // 4),
                                nn.GELU(),
                                nn.Linear(hidden_size * (resolution - 1) // 4, seq_output_dim // seq_input_dim),
                                nn.ReLU() if relu_in_end else nn.Identity(),
                            ) for i in range(resolution - 1)
                        ]
                    ]
                )

    def forward(self, x):
        # x: (batch, seq_input, hidden)
        result = []
        for index, layer in enumerate(self.special_conv_layers):
            target_x = x[:, index: index + self.resolution, :]
            if self.shared_layer or self.padding:
                target_x = F.pad(target_x, (0, 0, 0, self.resolution - target_x.shape[1]), "constant", 0)
            target_x = target_x.flatten(start_dim=1)
            result.append(layer(target_x))
        return torch.cat(result, dim=-1)