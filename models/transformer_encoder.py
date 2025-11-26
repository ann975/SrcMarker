import math
import torch
import torch.nn as nn

# One of two code encoder variants: Transformer encoder with 3 layers and 4 attention heads
# SrcMarkerTE

# positional encoding added so transformer understands token order 
# References "Attention is All You Need" paper
# Vaswani et al., 2017
# (Google/DeepMind & University of Toronto)


class PositionalEncoding(nn.Module):
    # adds sinusodial positional encodings 
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        # builds encoding table
        #. d_model = embedding dimension
        # max_len = max sequence length to precompute encodigns 
        
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout) # dropout added after adding positional encodings 

        pe = torch.zeros(max_len, d_model) # create empty positional encoding matrix 
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # generate positions 
        # column vector of position indices 

        # compute frequency terms for sine/cosine
        # generate vector of frequencies used for sin/cos waves
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model) 
            # picks every even dimension
        )
        pe[:, 0::2] = torch.sin(position * div_term) # fill even dimension with sin waves 
        pe[:, 1::2] = torch.cos(position * div_term) # odd with cos waves 
        pe = pe.unsqueeze(0).transpose(0, 1) # add batch dimension and reshape 
        self.register_buffer("pe", pe) # store encodings as non-parameter buffer, not trainable 

    # adding positional encoding 
    def forward(self, x):
        x = x + self.pe[: x.size(0), :] # add positional encoding to input embeddings 
        # slices positional encoding to match exact length
        # adds positional encodings elemnt-wise to token embeddings 
        return self.dropout(x)


class TransformerEncoderExtractor(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_size: int = 512,
        hidden_size: int = 512,
        num_layers: int = 3, # 3 layers
        n_heads: int = 4, # 4 attention heads 
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dropout_rate = dropout
        self.embedding_size = embedding_size
        self.n_heads = n_heads

        self.positional_encoding = PositionalEncoding(embedding_size, dropout)
        self.embedding = nn.Embedding(vocab_size, embedding_size) # token embedding 
        # build transformer encoder layer 
        # multi-head self-attention
        # feedforward network
        # layer norm
        # dropout
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_size, nhead=n_heads, dropout=dropout, batch_first=True
        )
        self.code_encoder = nn.TransformerEncoder(encoder_layer, num_layers) # stack multiple layers 

        self.fc1 = nn.Linear(embedding_size, hidden_size) # final fully connected layer 
        self.act1 = nn.ReLU()

    # run ecnoder 
    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor, src_key_padding_mask: torch.Tensor
    ):
        # B, L, H
        embedding = self.embedding(x) * math.sqrt(self.embedding_size) # embed tokens and scale them, (from Transformer paper)
        embedding = self.positional_encoding(embedding)

        # B, L, H
        # pass through transformer encoder 
        feature = self.code_encoder(
            embedding, src_key_padding_mask=src_key_padding_mask
        )
        # mean pooling over sequence length
        pooled = torch.mean(feature, dim=1)  # B, H

        # final linear layer + activation 
        pooled = self.fc1(pooled)
        pooled = self.act1(pooled)

        return pooled # outoput extracted sequence embedding
