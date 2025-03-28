import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, noise_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, output_dim),
            nn.Tanh()  # 假设目标输出在[-1, 1]范围内。如果不是，可以换成其他激活函数，如 nn.Sigmoid() 或 nn.Identity()
        )

    def forward(self, x):
        return self.model(x)

