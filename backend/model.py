import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_output)
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        return x

# U-Net with Transformer
class UNetTransformer(nn.Module):
    def __init__(self):
        super(UNetTransformer, self).__init__()
        base_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)  # Fixed pretrained issue
        self.encoder = nn.Sequential(*list(base_model.children())[:-2])
        self.transformer = TransformerBlock(embed_dim=512, num_heads=8, ff_dim=1024)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.flatten(2).permute(2, 0, 1)
        x = self.transformer(x)
        x = x.permute(1, 2, 0).reshape(-1, 512, 8, 8)
        x = self.decoder(x)
        return x
