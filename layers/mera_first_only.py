import torch
import torch.nn as nn
import tensornetwork as tn


class MERAFirstOnlyLayer(nn.Module):
    def __init__(self, rank: int = 2):
        super().__init__()
        mean, std = 0.0, 0.5
        input_dim = 9
        output_dim = 4
        self.input_shape = (2,) * input_dim
        self.output_shape = (2,) * output_dim

        self.disentangler_first_layer = [
            nn.Parameter(torch.normal(mean, std, (2, 2, 2, rank, rank), requires_grad=True)),
            nn.Parameter(torch.normal(mean, std, (2, 2, rank, rank), requires_grad=True)),
            nn.Parameter(torch.normal(mean, std, (2, 2, rank, rank), requires_grad=True))
        ]
        self.entangler_first_layer = [
            nn.Parameter(torch.normal(mean, std, (2, rank, rank), requires_grad=True)),
            nn.Parameter(torch.normal(mean, std, (rank, rank, rank), requires_grad=True)),
            nn.Parameter(torch.normal(mean, std, (rank, rank, rank), requires_grad=True)),
            nn.Parameter(torch.normal(mean, std, (rank, 2, rank), requires_grad=True)),
        ]

        self.entangler_second_layer = [
            nn.Parameter(torch.normal(mean, std, (rank, rank, rank, rank), requires_grad=True)) for _ in range(2)
        ]

        self.entangler_third_layer = nn.Parameter(
            torch.normal(mean, std, (rank, rank, rank, rank), requires_grad=True))

        self.bias = nn.Parameter(torch.normal(mean, std, (1, 2 ** output_dim), requires_grad=True))
        self.register_parameter("bias", self.bias)

        for i, core in enumerate(self.disentangler_first_layer):
            self.register_parameter(f"disentangler_first_{i}", core)

        for i, core in enumerate(self.entangler_first_layer):
            self.register_parameter(f"entangler_first_{i}", core)
        for i, core in enumerate(self.entangler_second_layer):
            self.register_parameter(f"entangler_second_{i}", core)
        self.register_parameter("entangler_third_0", self.entangler_third_layer)

    def forward(self, x: torch.tensor) -> torch.tensor:
        # Preparing variables
        dis_first = [tn.Node(core, f"dis_first_{i}") for i, core in enumerate(self.disentangler_first_layer)]
        ent_first = [tn.Node(core, f"ent_first_{i}") for i, core in enumerate(self.entangler_first_layer)]

        ent_second = [tn.Node(core, f"ent_second_{i}") for i, core in enumerate(self.entangler_second_layer)]

        ent_third = [tn.Node(self.entangler_third_layer, "ent_third_0")]

        x_reshaped = tn.Node(x.reshape(x.shape[0], *self.input_shape), "x")

        # Creating TT Network
        # First layer disentangles
        # first entangler
        x_reshaped[2] ^ dis_first[0][0]
        x_reshaped[3] ^ dis_first[0][1]
        x_reshaped[4] ^ dis_first[0][2]

        # second entangler
        x_reshaped[5] ^ dis_first[1][0]
        x_reshaped[6] ^ dis_first[1][1]

        # third entangler
        x_reshaped[7] ^ dis_first[2][0]
        x_reshaped[8] ^ dis_first[2][1]

        # First layer
        ent_first[0][0] ^ x_reshaped[1]
        ent_first[0][1] ^ dis_first[0][3]

        ent_first[1][0] ^ dis_first[0][4]
        ent_first[1][1] ^ dis_first[1][2]

        ent_first[2][0] ^ dis_first[1][3]
        ent_first[2][1] ^ dis_first[2][2]

        ent_first[3][0] ^ dis_first[2][3]
        ent_first[3][1] ^ x_reshaped[9]

        # Second layer
        ent_second[0][0] ^ ent_first[0][2]
        ent_second[0][1] ^ ent_first[1][2]

        ent_second[1][0] ^ ent_first[2][2]
        ent_second[1][1] ^ ent_first[3][2]

        # Third layer
        ent_third[0][0] ^ ent_second[0][3]
        ent_third[0][1] ^ ent_second[1][2]

        output_edges = [ent_second[0][2], ent_second[1][3], ent_third[0][2], ent_third[0][3]]

        # Contraction
        out = tn.contractors.greedy(
            [x_reshaped, *dis_first, *ent_first, *ent_second, *ent_third],
            output_edge_order=[x_reshaped.edges[0], *output_edges]
        )
        out = out.tensor.reshape(x.shape[0], -1)

        return out + self.bias
