import torch
import torch.nn as nn
from .mlp import MLP1


class FeatureApproximator(nn.Module): # inherits from nn.Module (PyTorch model)
    def __init__(self) -> None:
        super().__init__()

    def get_transform_embedding(self, transform_onehots: torch.Tensor):
        raise NotImplementedError()


class TransformerApproximator(FeatureApproximator):
    def __init__(
        self,
        vocab_size: int, # vocabulary size 
        transform_capacity: int, #number of transformations 
        input_dim: int, # size of feature vector entering model 
        output_dim: int,  # size of resulting feature vector after approximation
        dropout: float = 0.2, # prevents model from overfitting, probability that any number of neurons "dropped out"
        n_heads: int = 4, # number of attention heads in the Transformer layer 
        n_layers: int = 1, # number of Transformer layers stacked on top of each other 
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.transform_capacity = transform_capacity

        self.t_embeddings = nn.Embedding(transform_capacity, input_dim)
        # creates embedding table that maps each transformation vector of dimension input_dim
        # for example, if there are 50 possible transformations, creates 50 x input_dim learnable matrix 
        #nn.Embedding(50, 8) creates 50 x 8 matrix of learnable parameters 

        # defines single Transformer encoder layer 
        # Transformer - type of NN that processes sequential data (like sentences/tokenized code)
        # without relying on recurrence (unlike RNN/GRU/LSTM)
        # self-attention (every element can directly influence), positional encoding, parallelization, stacked layers 


        # building block of Transformer model 
        # each layer contains multi-head self-attention, feedforward network, residual connects/layer normalization
        # feedforward - each layer consisted of fully connected neurons where information flows forward only
        # residual connections - add input of layer directly to output (stablizes training, easier learning)
        # layer normalization - normalizs activation of layer across features 

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, # each token/ feature vector has size input_dim
            nhead=n_heads, # how many attention heads to use 
            dropout=dropout,
            batch_first=True, 
            # input tensor has three dimensions (batch_size, seq_length, feature_dim)
            #batch - number of seq processed at once
            # seq length - number of tokens/ time steps in each seq
            # feature dim- size of each token's feature vector

            dim_feedforward=768, # size of internal feedforward layer used inside each Transformer block
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=n_layers)
        # builds full Transformer encoder by stacking multiple identical transformer_layer

        # after Transformer has processed input features, fully connected (linear) layout maps 
        # result to desired output dimension
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.act1 = nn.ReLU() # adds non-linearity (so can learn complex relationships)
        self.dropout1 = nn.Dropout(dropout) # randomly zeros out fraction of outputs to avoid overfitting

    def get_transform_embedding(self, transform_onehots: torch.Tensor):
        return torch.matmul(transform_onehots, self.t_embeddings.weight)
    # given one-hot tensor represnting transformations, computes corresponding dense embedding
    # dense embedding - continuous, low-dim vector representation of discrete item 
    # all elements in vector carry meaningful values 
    # converts discrete items into numeric vectors 
    # allows model to learn similarities between items 

    # explains how input features processed through model to produce output
    # forward pass of network
    def forward(
        self,
        code_feature: torch.Tensor, # feature vector of source code tokens
        # B = batch size, L = sequence length (number of tokens), H = feature dimension
        var_embedding: torch.Tensor,
        # embedding vector of selected variable names, shape (B, H)
        transform_embedding: torch.Tensor,
        # embedding vector selected transformations, shape (B, H)
        src_mask: torch.Tensor,
        # mask for padding tokens, shape (B, L)
        # when training on sequences, often have different lengths
        # but NN need fixed-length input 
        # padding tokens are added to shorter sequences to make them same length as longest seq in batch 
    ):
        # code_feature: B, L, H
        # embeddings: B, H
        var_embedding = var_embedding.unsqueeze(1) 
        transform_embedding = transform_embedding.unsqueeze(1)
        # adds sequence dimension to embeddings, changing shape from (B, H) to (B, 1, H)
        # because Transformer expects input of shape (B, seq_len, H)
        cat_feature = torch.cat(
            [var_embedding, transform_embedding, code_feature], dim=1
        )
        # concatenates along sequence dimension 
        # new shape (B, L + 2, H)
        # sequence now contains var_embedding (first token), transform embedding (2nd), code feature (remaining tokens)
        # allows Transformer to attend to variable and transformation embeddings along with code tokens

        # B, S -> B, (S+2), make up for the newly appended embeddings
        src_mask = torch.cat(
            [torch.zeros(src_mask.shape[0], 2).to(src_mask.device).bool(), src_mask],
            dim=1,
        )
        # adjust src_mask to match new sequence length (L + 2)
        # prepends 2 zeros (False) for new embeddings (not masked)
        # src_mask tells Transformer which tokens are padding and should be ignored

        # B, L', H
        feature = self.transformer(cat_feature, src_key_padding_mask=src_mask)
        # passes concatenated feature thorugh Transformer encoder
        # output shape (B, L + 2, H)
        # each token embedding is now context-aware
        # contains information from all other tokens in the sequence 
        feature = torch.mean(feature, dim=1)  # B, H
        # pools across sequence dimension by taking mean
        # converts (B, L + 2, H) to (B, H)
        # produces single veature vector per example, summarizes code + variable + transformation embeddings

        return self.dropout1(self.act1(self.fc1(feature)))
        # applies fully connected layer (fc1) to reduce/transform feature to desired output size
        # passes through ReLU activation (act1), adds non-linearity
        # applies dropout (dropout1) to prevent overfitting 
        # final output shape (B, output_dim)


# cmobines variable renaming and code transformation
# into one, differentiable, continuous feature vector 
class WeightedSumApproximator(FeatureApproximator):
    def __init__(
        self,
        vocab_size: int,
        transform_capacity: int,
        input_dim: int,
        output_dim: int,
        dropout: float = 0.2,
        bn: bool = False,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.transform_capacity = transform_capacity

        self.t_embeddings = nn.Embedding(transform_capacity, input_dim)
        # creates learnable embedding table for transformations
        # each possible transformation gets input_dim dimensional vector representation

        self.t_warper = MLP1(input_dim * 2, output_dim)
        self.v_warper = MLP1(input_dim * 2, output_dim)
        # two small neural networks (MLP1) that combine code feature with 
        # variable embedding (v_warper) and transformation embedding (t_warper)
        # each MLP takes concatenation of two vectors (code and embedding) which explains input_dim * 2

        self.wt = 0.5
        self.wv = 0.5
        # static weights for combining variable and transformation contributions equally
        # can later be tuned or learned

    def get_transform_embedding(self, transform_onehots: torch.Tensor):
        return torch.matmul(transform_onehots, self.t_embeddings.weight)
    # takes one-hot tensor representing selected transformations
    # multiplies with embedding weight matrix to get actual dense embedding vector 
    

    def forward(
        self,
        code_feature: torch.Tensor,
        var_embedding: torch.Tensor,
        transform_embedding: torch.Tensor,
        src_mask: torch.Tensor,
    ):
        assert code_feature.shape[1] == var_embedding.shape[1]
        assert code_feature.shape[1] == transform_embedding.shape[1]
        # ensures all feature vectors have compatible dimensions 

        ev = self.v_warper(torch.cat([code_feature, var_embedding], dim=1))
        et = self.t_warper(torch.cat([code_feature, transform_embedding], dim=1))
        # concatenates base code feature with variable or transformation embeddings
        # passeed through 2 MLPs to get 
        # ev: variable-adjusted feature
        # et: transformation-adjusted feature 

        return self.wv * ev + self.wt * et # produce final approximated transformation feature 
        # ~e trans = 0.5ev + 0.5et


# predicts how code features change under transformations 
# instead of learning two separate "warpers" and combining results (as in weighted)
# simply adds everything together in feature space and passes through few fully connected layers 
# effects of variable and transformation embeddings can be captured by additive composition 
class AdditionApproximator(FeatureApproximator):
    def __init__(
        self,
        vocab_size: int,
        transform_capacity: int,
        input_dim: int,
        output_dim: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.transform_capacity = transform_capacity
        self.t_embeddings = nn.Embedding(transform_capacity, input_dim) #encode how transformation shifts code's feature vector 

        # three-layer feedforward MLP
        #layer 1 - epands and nonlinearly transforms input
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        # layer 2 - projects from input space to output (feature) space
        self.fc2 = nn.Linear(input_dim, output_dim)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        # output layer - final linear projection to desired feature dimension
        self.out = nn.Linear(output_dim, output_dim)

    def get_transform_embedding(self, transform_onehots: torch.Tensor):
        return torch.matmul(transform_onehots, self.t_embeddings.weight)

    def forward(
        self,
        code_feature: torch.Tensor,
        var_embedding: torch.Tensor,
        transform_embedding: torch.Tensor,
        src_mask: torch.Tensor,
    ):
        assert code_feature.shape[1] == var_embedding.shape[1]
        assert code_feature.shape[1] == transform_embedding.shape[1]

        # combine embeddings additively 
        # meaning of transformed code = original meaning + variable effect + transformation effect 
        x = code_feature + var_embedding + transform_embedding

        # feed through MLP
        x = self.dropout1(self.act1(self.fc1(x)))
        x = self.dropout2(self.act2(self.fc2(x)))
        x = self.out(x)

        return x # returns predicted feature representation of transformed code
        # can be compared (via loss) to true feature representation extracted from
        # actually transformed code
        # allows model to learn how code transformations change representaitons 


# predicts how code's feature representation changes under chosen variable substiution and transformation
# instead of adding or weighting, concatenates embeddings and learns how to combine through neural network
# assumes that best combination of features can be learned by concatenating all inputs and letting neural network find relationships mong them 
class ConcatApproximator(FeatureApproximator):
    def __init__(
        self,
        vocab_size: int,
        transform_capacity: int,
        input_dim: int,
        output_dim: int,
        dropout: float = 0.2,
        bn: bool = False, # whether to use batch normalization (for training stability)
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.transform_capacity = transform_capacity

        # transformation embeddings 
        # each transformation gets learnable embedding vector of size, input_dim
        self.t_embeddings = nn.Embedding(transform_capacity, input_dim)
        # first layer
        # 3 * input_dim (concatenate three things: code feature, variable embedding, transformation embedding)
        # layer then reduces to 2 * input_dim
        self.fc1 = nn.Linear(input_dim * 3, input_dim * 2)

        # batch normalization 
        # if bn= True, normalize activations to stablize training and prevent exploding gradients
        # otherwise skip
        if bn:
            self.bn1 = nn.BatchNorm1d(input_dim * 2)
        else:
            self.bn1 = nn.Identity()
        # nonlinear activation and dropout
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        # second layer 
        # projects intermediate representation down to desired feature dimension
        self.fc2 = nn.Linear(input_dim * 2, output_dim)
        
        # optional batch norm + activation + dropout
        if bn:
            self.bn2 = nn.BatchNorm1d(output_dim)
        else:
            self.bn2 = nn.Identity()
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

    # if transform_onehots is hard one-hot, pick that embedding
    # if soft (probabilistic), compute weighted average (keeps everything differentiable)
    def get_transform_embedding(self, transform_onehots: torch.Tensor):
        return torch.matmul(transform_onehots, self.t_embeddings.weight)

    def forward(
        self,
        code_feature: torch.Tensor,
        var_embedding: torch.Tensor,
        transform_embedding: torch.Tensor,
        src_mask: torch.Tensor,
    ):
        assert code_feature.shape[1] == var_embedding.shape[1]
        assert code_feature.shape[1] == transform_embedding.shape[1]

        # concatenate - makes one long vector per code sample
        # gives model full visibilitiy of all three at once
        # allowing it to jointly learn nonlinera interactions between them
        x = torch.cat([code_feature, var_embedding, transform_embedding], dim=1)

        # pass through MLP
        x = self.act1(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = self.act2(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        return x # predicted transformed feature vector 


# deals only with variable substitutions, not full code transformation
# how changing variable names affects code's feature approximation
class VarApproximator(FeatureApproximator):
    def __init__(
        self,
        vocab_size: int,
        input_dim: int,
        output_dim: int,
        dropout: float = 0.2,
        bn: bool = False,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size

        # concatenates code feature and variable embedding 
        # total dimension = 2 * input_dim
        # passes through 2 fully connected layers (MLPs)
        # first keeps dimension
        # second projects desired output size 
        self.fc1 = nn.Linear(input_dim * 2, input_dim * 2)
        if bn:
            self.bn1 = nn.BatchNorm1d(input_dim * 2)
        else:
            self.bn1 = nn.Identity()
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(input_dim * 2, output_dim)
        if bn:
            self.bn2 = nn.BatchNorm1d(output_dim)
        else:
            self.bn2 = nn.Identity()
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

    def get_transform_embedding(self, transform_onehots: torch.Tensor):
        raise RuntimeError("VarApproximator does not support transform embedding")

    def forward(
        self,
        code_feature: torch.Tensor,
        var_embedding: torch.Tensor,
        src_mask: torch.Tensor,
    ):
        assert code_feature.shape[1] == var_embedding.shape[1]

        # concatenate features 
        x = torch.cat([code_feature, var_embedding], dim=1)
        # pass through neural layers 
        x = self.act1(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = self.act2(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        return x


# for transformations only 
class TransformationApproximator(FeatureApproximator):
    def __init__(
        self,
        transform_capacity: int,
        input_dim: int,
        output_dim: int,
        dropout: float = 0.2,
        bn: bool = False,
    ) -> None:
        super().__init__()
        self.transform_capacity = transform_capacity

        self.t_embeddings = nn.Embedding(transform_capacity, input_dim)
        self.fc1 = nn.Linear(input_dim * 2, input_dim * 2)
        if bn:
            self.bn1 = nn.BatchNorm1d(input_dim * 2)
        else:
            self.bn1 = nn.Identity()
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(input_dim * 2, output_dim)
        if bn:
            self.bn2 = nn.BatchNorm1d(output_dim)
        else:
            self.bn2 = nn.Identity()
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

    def get_transform_embedding(self, transform_onehots: torch.Tensor):
        return torch.matmul(transform_onehots, self.t_embeddings.weight)

    def forward(
        self,
        code_feature: torch.Tensor,
        transform_embedding: torch.Tensor,
        src_mask: torch.Tensor,
    ):
        assert code_feature.shape[1] == transform_embedding.shape[1]

        x = torch.cat([code_feature, transform_embedding], dim=1)
        x = self.act1(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = self.act2(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        return x
