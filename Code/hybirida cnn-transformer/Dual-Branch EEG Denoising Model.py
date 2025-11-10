# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------ Branch 1 : CNN Clean ------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=7, stride=1, padding=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, stride, padding, groups=in_ch),  # depthwise
            nn.Conv1d(out_ch, out_ch, 1),                                           # pointwise
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size, 1, padding, groups=out_ch),
            nn.Conv1d(out_ch, out_ch, 1),
            nn.BatchNorm1d(out_ch)
        )
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return F.relu(self.skip(x) + self.conv(x))

class CNNEncoder(nn.Module):
    def __init__(self, channels, layers=[2, 2, 2], base=64):
        super().__init__()
        self.input_proj = nn.Conv1d(channels, base, 7, padding=3)
        self.blocks = nn.ModuleList()
        ch = base
        for i, L in enumerate(layers):
            for _ in range(L):
                self.blocks.append(ResidualBlock(ch, ch*2 if i==len(layers)-1 else ch))
            if i < len(layers)-1:
                self.blocks.append(nn.Conv1d(ch, ch*2, 2, stride=2))  # down
                ch *= 2
        self.out_ch = ch

    def forward(self, x):
        x = self.input_proj(x)
        for layer in self.blocks:
            x = layer(x)
        return x

# ------------------ Branch 2 : Transformer Noise ------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        -(torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0).transpose(1,2))  # (1, d_model, max_len)

    def forward(self, x):
        # x: (B, d_model, T)
        return x + self.pe[:, :, :x.size(2)]

class TransformerEncoder(nn.Module):
    def __init__(self, channels, d_model=128, nhead=8, num_layers=4):
        super().__init__()
        self.input_proj = nn.Conv1d(channels, d_model, 1)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=512,
            dropout=0.1, activation='gelu', batch_first=False)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.d_model = d_model

    def forward(self, x):
        # x: (B, C, T)  -> (T, B, d_model)
        x = self.input_proj(x).transpose(1,2).transpose(0,1)
        x = self.pos_enc(x.transpose(0,2)).transpose(0,2)  # add pos
        x = self.transformer(x)                              # (T, B, d_model)
        return x.transpose(0,1).transpose(1,2)               # (B, d_model, T)

# ------------------ Fusion & Head ------------------
class GatedFusion(nn.Module):
    def __init__(self, cnn_ch, trans_ch, out_ch=64):
        super().__init__()
        self.gate_cnn  = nn.Conv1d(cnn_ch, out_ch, 1)
        self.gate_trans= nn.Conv1d(trans_ch, out_ch, 1)
        self.out_proj  = nn.Conv1d(out_ch, out_ch, 1)

    def forward(self, cnn_feat, trans_feat):
        # cnn_feat: (B, cnn_ch, T)   trans_feat: (B, trans_ch, T)
        g = torch.sigmoid(self.gate_cnn(cnn_feat) + self.gate_trans(trans_feat))
        fused = g * cnn_feat[:, :g.size(1)] + (1-g) * trans_feat[:, :g.size(1)]
        return self.out_proj(fused)

class EEGDenoiser(nn.Module):
    def __init__(self, C=22, T=512):
        super().__init__()
        self.cnn_branch = CNNEncoder(C)
        self.trans_branch= TransformerEncoder(C)
        self.fuse = GatedFusion(self.cnn_branch.out_ch, self.trans_branch.d_model)
        self.head = nn.Conv1d(64, C, 1)

    def forward(self, x):
        # x: (B, C, T)
        cnn_feat = self.cnn_branch(x)
        trans_feat= self.trans_branch(x)
        fused = self.fuse(cnn_feat, trans_feat)
        return self.head(fused)  # residual learning: noise = model(x); clean = x - noise

if __name__ == "__main__":
    from torchinfo import summary
    model = EEGDenoiser(C=22, T=512)
    summary(model, (1, 22, 512))