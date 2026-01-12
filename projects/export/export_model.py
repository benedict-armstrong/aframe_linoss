import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        return torch.ones(x.shape[0], 1)  # self.linear(x)


if __name__ == "__main__":
    model = Model()
    dummy_input = torch.randn(1, 10)
    # export model to torchscript
    # traced_script_module = torch.jit.trace(model, dummy_input)
    m = torch.jit.script(model)

    # Save to file
    torch.jit.save(m, "model.pt")
