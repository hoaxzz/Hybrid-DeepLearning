import torch, torch.nn as nn, torch.nn.functional as F
from torch import Tensor

# ---------- utilitas ----------
class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=15, stride=1, padding=7):
        super().__init__()
        self.depth = nn.Conv1d(in_ch, in_ch, kernel, stride, padding, groups=in_ch, bias=False)
        self.point = nn.Conv1d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(self.bn(self.point(self.depth(x))))

class ResBlock(nn.Module):
    def __init__(self, ch, kernel=15):
        super().__init__()
        pad = kernel//2
        self.net = nn.Sequential(
            nn.Conv1d(ch, ch, kernel, 1, pad, bias=False),
            nn.BatchNorm1d(ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(ch, ch, kernel, 1, pad, bias=False),
            nn.BatchNorm1d(ch)
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(x + self.net(x))

# ---------- Transformer ----------
class TransformerBranch(nn.Module):
    def __init__(self, d_model, num_layers, num_heads, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads,
            dim_feedforward=4*d_model, dropout=dropout,
            batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_enc = nn.Parameter(torch.randn(1, 512, d_model)*0.02)  # T max 512
    def forward(self, x):   # x: (B,T,d_model)
        x = x + self.pos_enc[:, :x.size(1), :]
        return self.encoder(x)

# ---------- Fusion Gate ----------
class GatedFusion(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.w = nn.Sequential(
            nn.Conv1d(ch*2, ch, 1, bias=False),
            nn.Sigmoid()
        )
    def forward(self, f1, f2):  # both (B,ch,T)
        merged = torch.cat([f1, f2], dim=1)
        gate = self.w(merged)
        return gate*f1 + (1-gate)*f2

# ---------- Model Utama ----------
class DualBranchDenoiser(nn.Module):
    def __init__(self, C, T,
                 conv_channels=[32,64,128],
                 transformer_dim=128,
                 transformer_layers=4,
                 num_heads=8):
        super().__init__()
        # --- CNN branch (Clean Learning) ---
        self.input_proj = nn.Conv1d(C, conv_channels[0], 1, bias=False)
        cnn = []
        for i in range(len(conv_channels)):
            ch = conv_channels[i]
            cnn += [ResBlock(ch), DepthwiseSeparableConv1d(ch, ch)]
        self.cnn_branch = nn.Sequential(*cnn)

        # --- Transformer branch (Noise Learning) ---
        self.tf_proj = nn.Conv1d(C, transformer_dim, 1)
        self.tf_branch = TransformerBranch(d_model=transformer_dim,
                                           num_layers=transformer_layers,
                                           num_heads=num_heads)

        # --- Fusion & output ---
        self.fusion = GatedFusion(transformer_dim)
        self.out_conv = nn.Conv1d(transformer_dim, C, 1)

    def forward(self, x):  # x: (B,C,T)
        # CNN
        f1 = self.input_proj(x)          # (B,c1,T)
        f1 = self.cnn_branch(f1)         # (B,c_last,T)

        # Transformer
        f2 = self.tf_proj(x).transpose(1,2)  # (B,T,d)
        f2 = self.tf_branch(f2)              # (B,T,d)
        f2 = f2.transpose(1,2)               # (B,d,T)

        # Pastikan dimensi sama
        if f1.size(1) != f2.size(1):
            pad = f2.size(1) - f1.size(1)
            f1 = F.pad(f1, (0,0,0,pad))      # pad channel jika perlu

        fused = self.fusion(f1, f2)
        return self.out_conv(fused)      # (B,C,T)