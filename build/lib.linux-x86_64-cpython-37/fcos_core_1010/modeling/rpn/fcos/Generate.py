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
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Linear(512, output_dim),
            nn.Identity()  # 假设目标输出在[-1, 1]范围内。如果不是，可以换成其他激活函数，如 nn.Sigmoid() 或 nn.Identity()
        )

    def forward(self, x):
        return self.model(x)


class Generator2(nn.Module): #DiffusionPriorUNet(nn.Module):

    def __init__(
            self,
            embed_dim=1024,
            cond_dim=42,
            hidden_dim=[1024, 512, 256, 128, 64],
            act_fn=nn.SiLU,
            dropout=0.0,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim

        # 1. conditional embedding
        # to 3.2, 3,3

        # 3. prior mlp

        # 3.1 input
        self.input_layer = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim[0]),
            nn.LayerNorm(hidden_dim[0]),
            act_fn(),
        )

        # 3.2 hidden encoder
        self.num_layers = len(hidden_dim)
        self.encode_cond_embedding = nn.ModuleList(
            [nn.Linear(cond_dim, hidden_dim[i]) for i in range(self.num_layers - 1)]
        )
        self.encode_layers = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(hidden_dim[i], hidden_dim[i + 1]),
                nn.LayerNorm(hidden_dim[i + 1]),
                act_fn(),
                nn.Dropout(dropout),
            ) for i in range(self.num_layers - 1)]
        )

        # 3.3 hidden decoder
        self.decode_cond_embedding = nn.ModuleList(
            [nn.Linear(cond_dim, hidden_dim[i]) for i in range(self.num_layers - 1, 0, -1)]
        )
        self.decode_layers = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(hidden_dim[i], hidden_dim[i - 1]),
                nn.LayerNorm(hidden_dim[i - 1]),
                act_fn(),
                nn.Dropout(dropout),
            ) for i in range(self.num_layers - 1, 0, -1)]
        )

        # 3.4 output
        self.output_layer = nn.Linear(hidden_dim[0], embed_dim)

    def forward(self, x, c=None):
        # x (batch_size, embed_dim)
        # c (batch_size, cond_dim)

        # 1. conditional embedding
        # to 3.2, 3.3

        # 3. prior mlp

        # 3.1 input
        x = self.input_layer(x)

        # 3.2 hidden encoder
        hidden_activations = []
        for i in range(self.num_layers - 1):
            hidden_activations.append(x)
            if c is not None:
                c_emb = self.encode_cond_embedding[i](c)
            else:
                c_emb = torch.zeros(x.size(0), self.hidden_dim[i], device=x.device)

            x = x + c_emb
            x = self.encode_layers[i](x)

        # 3.3 hidden decoder
        for i in range(self.num_layers - 1):
            if c is not None:
                c_emb = self.decode_cond_embedding[i](c)
            else:
                c_emb = torch.zeros_like(x)

            x = x + c_emb
            x = self.decode_layers[i](x)
            x += hidden_activations[-1 - i]

        # 3.4 output
        x = self.output_layer(x)


        return x