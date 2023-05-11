import torch
import torch.nn as nn
import tensornetwork as tn


class TWLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, rank: int = 10):
        super().__init__()
        mean, std = 0.0, 0.5
        self.input_shape = (2,) * input_dim
        self.input_cores = [
            nn.Parameter(torch.normal(mean, std, (rank, 2, 2, rank), requires_grad=True)) for _ in range(input_dim)
        ]
        self.output_cores = [
            nn.Parameter(torch.normal(mean, std, (rank, 2, 2, rank), requires_grad=True)) for _ in range(output_dim)
        ]
        self.central_core = nn.Parameter(torch.normal(mean, std, (2,) * (input_dim + output_dim)))
        self.register_parameter("central_core", self.central_core)

        self.bias = nn.Parameter(torch.normal(mean, std, (1, 2 ** output_dim), requires_grad=True))
        self.register_parameter("bias", self.bias)

        for i, core in enumerate(self.input_cores):
            self.register_parameter(f"input_{i}", core)

        for i, core in enumerate(self.output_cores):
            self.register_parameter(f"output_{i}", core)

    def forward(self, x: torch.Tensor):
        # Preparing variables
        input_net = [tn.Node(core, f"input_{i}") for i, core in enumerate(self.input_cores)]
        output_net = [tn.Node(core, f"output_{i}") for i, core in enumerate(self.output_cores)]
        central_core = tn.Node(self.central_core, "central_core")
        x_reshaped = tn.Node(x.reshape(x.shape[0], *self.input_shape), "x")

        # Creating TT Network
        x_reshaped[1] ^ input_net[0][1]
        input_net[0][2] ^ central_core[0]
        for i in range(1, len(input_net)):
            x_reshaped[i + 1] ^ input_net[i][1]
            input_net[i][2] ^ central_core[i]
            input_net[i - 1][-1] ^ input_net[i][0]

        output_edges = []
        input_net[0][0] ^ output_net[-1][-1]
        input_net[-1][-1] ^ output_net[0][0]

        for i in range(len(output_net) - 1):
            output_edges.append(output_net[i][1])
            output_net[i][2] ^ central_core[len(input_net) + i]
            output_net[i][-1] ^ output_net[i + 1][0]

        output_net[-1][2] ^ central_core[-1]
        output_edges.append(output_net[-1][1])

        out = tn.contractors.greedy(
            [x_reshaped, *input_net, *output_net, central_core],
            output_edge_order=[x_reshaped.edges[0], *output_edges]
        )
        out = out.tensor.reshape(x.shape[0], -1)

        return out + self.bias
