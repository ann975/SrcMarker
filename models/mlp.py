import torch
import torch.nn as nn

# MLPs - fully connected networks 
# used in watermark encoder 
# watermark decoder, which is fully-connected network that decodes watermark using etrans
# used in transformation selector - contains 2 independent MLPs fvar and ftrans 
# used infeature approximation 


class MLPForMNIST(nn.Module):
    HIDDEN_SIZE = 512
    DROPOUT_RATE = 0.2
    INPUT_SHAPE = 28 * 28
    OUTPUT_CLASSES = 10

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.INPUT_SHAPE, self.HIDDEN_SIZE)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(self.DROPOUT_RATE)
        self.fc2 = nn.Linear(self.HIDDEN_SIZE, self.HIDDEN_SIZE)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(self.DROPOUT_RATE)

        self.output = nn.Linear(self.HIDDEN_SIZE, self.OUTPUT_CLASSES)

    def forward(self, x: torch.Tensor):
        x = self.flatten(x)
        x = self.dropout1(self.act1(self.fc1(x)))
        h = self.act2(self.fc2(x))
        x = self.dropout2(h)
        x = self.output(x)
        return x, h

# 3 layers with dropout at each stage 
# can learn complex functions
# higher risk of overfitting
# fvar and ftrans (transformation selector)
# use when input/ouput relationships highly nonlinear 
# use when not enough data to support deepr networks 
# hidden_dim stays constant 
# dropout after each layer (prevents overfitting)
class MLP3(nn.Module):
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 512,
        output_dim: int = 512,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor):
        x = self.dropout1(self.act1(self.fc1(x)))
        x = self.dropout2(self.act2(self.fc2(x)))
        x = self.out(x)

        return x

# 2 layers with optional batch normalization
# dropout for regularization
# bn flag allows flexibility for batch sizes 
class MLP2(nn.Module):
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 512,
        output_dim: int = 512,
        dropout: float = 0.2,
        bn: bool = True,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        if bn:
            self.bn = nn.BatchNorm1d(hidden_dim)
        else:
            self.bn = None
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        if self.bn is not None:
            x = self.bn(x)

        x = self.dropout1(self.act1(x))
        x = self.output(x)
        return x

# single linear layer + ReLU
# no dropout, no batch norm
# fastest, least expressive 
class MLP1(nn.Module):
    def __init__(self, input_dim: int = 512, output_dim: int = 512) -> None:
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.fc(x))
        return x
