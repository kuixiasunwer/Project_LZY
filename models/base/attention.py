import torch
import torch.nn as nn

class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, hidden_size, output_hidden_size=None, num_heads=2, dropout=0.1):
        super(MultiHeadAttentionWrapper, self).__init__()
        self.hidden_size = hidden_size
        self.output_hidden_size = hidden_size if output_hidden_size is None else output_hidden_size

        # 多头注意力模块（输入和输出维度相同）
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, dropout=dropout, batch_first=True)

        # 如果输出维度与输入不同，则添加一个线性变换层
        if hidden_size != output_hidden_size:
            self.out_proj = nn.Linear(hidden_size, output_hidden_size)
        else:
            self.out_proj = nn.Identity()

    def forward(self, x):
        attn_output, _ = self.attn(query=x, key=x, value=x)

        # 线性变换输出
        out = self.out_proj(attn_output)

        return out