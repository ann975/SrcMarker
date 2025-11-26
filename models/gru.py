import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


# uses GRU to process sequences and predict class labels 
class GRUClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embedding_size: int = 512,
        hidden_size: int = 512,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.embedding_dropout = nn.Dropout(dropout)
        self.rnn_unit = nn.GRU(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

        D = 2 if bidirectional else 1
        rnn_out_size = D * num_layers * hidden_size # final hidden tensor has shape

        # classifier head 
        # two-layer classifier
        self.fc1 = nn.Linear(rnn_out_size, hidden_size) # fc1 - condenses large hidden vector down to hidden_size 
        self.act1 = nn.ReLU() # non-linear activation 
        self.fc_dropout = nn.Dropout(dropout) # regularization 
        self.fc2 = nn.Linear(hidden_size, num_classes) # fc2  - projects number of classes (final logits)

    # forward pass 
    # x: (B, T) tensor of token indices (batch of sequences)
    # lengths - actual sequence lengths (used for packing)
    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        # embedding 
        B = x.shape[0]
        x = self.embedding(x)
        x = self.embedding_dropout(x)
        # x is shape (B, T, embedding_size)

        # converts padded seq batch into packed representation so GRU ignores padded tokens 
        # improves efficieny and accuracy 
        x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        
        # GRU
        # output - GRU outputs at all time steps 
        # # hidden - final hidden states from all layers and directions 
        output, hidden = self.rnn_unit(x)
        # flatten hidden states 
        # feature: compressed representation of the sequence (good for embeddings or downstream tasks)
        # output: logits (unnormalized scores) for each class, shape (B, num_classes)
        hidden = hidden.permute(1, 0, 2).reshape(B, -1)

        # classifier head 
        feature = self.act1(self.fc1(hidden))
        output = self.fc2(self.fc_dropout(feature))

        return output, feature
        #output: predictions (logits)
        # feature: the learned intermediate feature vector from the GRU 
        # useful if you want to reuse embeddings (for contrastive loss, feature matching, etc.)
