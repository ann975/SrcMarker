import torch
import torch.nn as nn
from .gru_encoder import GRUEncoder
from .transformer_encoder import TransformerEncoderExtractor

# embed tokens and pass through MLP
# output single logit per sample
class MLPDetector(nn.Module):
    def __init__(self, vocab_size: int, embedding_size: int, hidden_size: int):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.fc1 = nn.Linear(embedding_size, hidden_size)
        self.act1 = nn.ReLU()
        self.out = nn.Linear(hidden_size, 1)

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor, src_key_padding_mask: torch.Tensor
    ):
        x = self.embeddings(x)
        x = self.act1(self.fc1(x))

        x = torch.mean(x, dim=1)
        x = self.out(x)
        return x

# uses GRU encoder (recurrent network) instead of simple MLP
# bidirectional 
# linear layer maps to logit
# outputs scalar logit for each sequence
class GRUWMDetector(nn.Module):
    def __init__(
        self, vocab_size: int, embedding_size: int, hidden_size: int, num_layers: int
    ) -> None:
        super().__init__()
        self.encoder = GRUEncoder(
            vocab_size,
            embedding_size,
            hidden_size,
            num_layers,
            bidirectional=True,
            dropout=0.2,
        )
        self.linear = nn.Linear(hidden_size, 1)

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor, src_key_padding_mask: torch.Tensor
    ):
        feature = self.encoder(x, lengths, src_key_padding_mask)
        logits = self.linear(feature)
        return logits

# uses Transformer encoder (like BERT) instead of GRU 
# output scalar logit per sequence 
class TransformerWMDetector(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        hidden_size: int,
        num_layers: int,
        n_heads: int,
    ) -> None:
        super().__init__()
        self.encoder = TransformerEncoderExtractor(
            vocab_size, embedding_size, hidden_size, num_layers, n_heads, dropout=0.1
        )
        self.linear = nn.Linear(hidden_size, 1)

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor, src_key_padding_mask: torch.Tensor
    ):
        feature = self.encoder(x, lengths, src_key_padding_mask)
        logits = self.linear(feature)
        return logits
