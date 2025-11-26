import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence # handles variable-length sequences when using RNNs 

# GRU (Gated Recurrent Unit) network 
# encodes sequence into fixed-size feature vector 
# encoder - takes in variable-length sequence and compresses into on representation 
class ExtractGRUEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_size: int = 512, # dimensions of each token's embedding vector 
        # set to 768 for actual experiemnt?
        hidden_size: int = 512, # number of hidden units in each GRU layer 
        num_layers: int = 2, # number of stacked GRU layers 
        bidirectional: bool = True, # whether to process sequences forward and backward 
        dropout: float = 0.2,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        # learns dense vector representation for each token index in vocabulary
        # converts input tensor of token IDs [batch_size, seq_len] -> embeddings [batch_size, seq_len, embedding_size]

        self.embedding_dropout = nn.Dropout(dropout) # randomly zeros out fraction (dropout) of embedding elements during training
        
        # GRU network that reads sequences and outputs hidden states 
        self.rnn_unit = nn.GRU(
            input_size=embedding_size,
            hidden_size=hidden_size, # hidden state = number of neurons in each GRU cell
            num_layers=num_layers, # stacked GRUs (output of one layer feeds into next)
            batch_first=True,
            bidirectional=True, # adds reverse GRU that processes sequence backward, capturing context from both directions 
        )

        # direction factor 
        # if bidirectional, each layer has 2 hidden states (forwards and backwards)r 
        D = 2 if bidirectional else 1
        rnn_out_size = D * num_layers * hidden_size # total size of final hidden vecto

        # MLP Post-Processor 
        # transforms flattened GRU hidden output into final representation
        # runs through TWO nonlinear layers and normalizes in between
        self.mlp = nn.Sequential(
            nn.Linear(rnn_out_size, hidden_size), # reduces dimensionality 
            nn.BatchNorm1d(hidden_size), # normalizes activations per batch for stability 
            nn.ReLU(), # non linear activation 
            nn.Linear(hidden_size, hidden_size), # further transformations 
            nn.ReLU(),
        )

    # defines how input data flows through model 
    # x: token IDs for each example
    # length: true lengths (number of valid tokens before padding)
    # src_key : mask showing which tokens are padding (not used here)
    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor, src_key_padding_mask: torch.Tensor
    ):
        B = x.shape[0] # gets number of sequences (batch size)
        x = self.embedding(x) # CONVERT TOKENS TO VECTORS 
        # converts [B, seq_len] tp [B, seq_len, embedding_size]

        x = self.embedding_dropout(x) # dropout

        #handle functions of different length
        x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        # pack sequences by length
        # Compresses the padded sequences so the GRU doesn’t waste time processing padding tokens
        # enforce_sorted=False allows input batches to be in any order (not sorted by length)

        _, hidden = self.rnn_unit(x) # PROCESS SEQUENCE 
        # GRU returns (outputs, hidden) where
        # output - hidden state at each timestep (not used here)
        # hidden - final hidden states from all layers/ direction 
        hidden = hidden.permute(1, 0, 2).reshape(B, -1) 
        # rearrange and flatten hidden states 
        #  permute(1, 0, 2) changes shape to [batch_size, num_layers * D, hidden_size].
        # .reshape(B, -1) flattens across layers/directions → [B, rnn_out_size].
        # Now each sequence has one long hidden vector combining all layers and directions.

        feature = self.mlp(hidden) # ECODE 

        return feature # returns encoded representation 
        # vector can be used for classification, similarity computation,
        # feeding into decoder (seq2seq tasks), feature approximators, etc.


class GRUEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
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
            bidirectional=bidirectional,
        )

        D = 2 if bidirectional else 1
        rnn_out_size = D * num_layers * hidden_size

        self.fc1 = nn.Linear(rnn_out_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.act1 = nn.ReLU()
        # one linear -> normalizaiton -> ReLU step 

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor, src_key_padding_mask: torch.Tensor
    ):
        B = x.shape[0]
        x = self.embedding(x)
        x = self.embedding_dropout(x)

        x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        outputs, hidden = self.rnn_unit(x)
        hidden = hidden.permute(1, 0, 2).reshape(B, -1)

        feature = self.act1(self.bn1(self.fc1(hidden)))  # B, H

        return feature
