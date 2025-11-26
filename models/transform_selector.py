import torch
import torch.nn as nn
from typing import Optional, List

# implements transformation selector 

#Selector module (MLP structure)
class Selector(nn.Module):
    def __init__(
        self,
        num_warpers: int,
        # warper - neural module that "warps" (modifies) code feature vector based on variable substituion, code transformation, or both
        # warper = nerual network that predicts how code representaiton should change 
        
        input_dim: int = 512,
        dropout: float = 0.2,
        bn: bool = True,
    ) -> None:
        super().__init__()
        self.fc = nn.Linear(input_dim * 2, input_dim) # takes concatenated input 
        if bn:
            self.bn = nn.BatchNorm1d(input_dim) # normalizes activations across batch to stabilize training 
        else:
            self.bn = nn.Identity()
        self.dropout = nn.Dropout(dropout) # randomly zeroes out some activations to prevent overfitting
        self.act = nn.ReLU()

        self.selector = nn.Linear(input_dim, num_warpers) # output layer 
        # converts 512-dim hiddne vector into vector of size num_warpers 

        # ecode + ewm -> Linear(768*2, 768) -> BatchNorm -> ReLU -> Dropout -> Linear (768, num_warpers) -> logits 

    def forward(self, code_feature: torch.Tensor, wm_feature: torch.Tensor):
        assert code_feature.shape[1] == wm_feature.shape[1]

        x = torch.cat([code_feature, wm_feature], dim=1) # creates concatenation of two feature vecotrs along feature dimension
        #alternative:
        # x = code_feature + wm_feature

        x = self.fc(x) # first linear layer
        x = self.bn(x) # normalize vector across batch 
        x = self.dropout(self.act(x)) # applies ReLU (adds non-linearity)
        x = self.selector(x) # convert processed feature fector into vector of logits 

        return x # returns vector scores not probabilities (logits)


class TransformSelector(nn.Module):
    vocab_mask: Optional[torch.Tensor]

    def __init__(
        self,
        vocab_size: int,
        transform_capacity: int,
        input_dim: int = 512,
        random_mask_prob: float = 0.5,
        dropout: float = 0.2,
        bn: bool = True,
        vocab_mask: Optional[List[bool]] = None,
    ) -> None:
        super().__init__()
        self.mask_prob = random_mask_prob

        #fvar 
        self.var_selector = Selector(vocab_size, input_dim, dropout, bn)

        # ftrans
        self.transform_selector = Selector(transform_capacity, input_dim, dropout, bn)
        
        # probability masking mechanism  - validity mask (prevent selecting invalid tokens)
        if vocab_mask is not None:
            self.register_buffer(
                "vocab_mask", torch.tensor(vocab_mask, dtype=torch.bool)
            )
        else:
            self.vocab_mask = None

    # random mask - 50% mask of variable names 
    def _get_random_mask_like(self, like: torch.Tensor):
        return torch.rand_like(like) < self.mask_prob

    # pvar - variable selector 
    def var_selector_forward(
        self,
        code_feature: torch.Tensor, #ecode 
        wm_feature: torch.Tensor, #ewm
        variable_mask: Optional[torch.Tensor] = None,
        random_mask: bool = True, # random mask true 
        return_probs: bool = False,
    ):
        outputs = self.var_selector(code_feature, wm_feature) #fvar(ecode * ewm)

        if variable_mask is None:
            variable_mask = self.vocab_mask

        if variable_mask is not None:
            outputs = torch.masked_fill(outputs, variable_mask, float("-inf"))

       
        if random_mask:
            rand_mask = self._get_random_mask_like(outputs)
            outputs = torch.masked_fill(outputs, rand_mask.bool(), float("-inf"))

        if return_probs:
            return torch.softmax(outputs, dim=-1) # Softmax -> pvar 
        else:
            return outputs # raw logits 

    # transformation selector (ptrans)
    def transform_selector_forward(
        self,
        code_feature: torch.Tensor, # ecode 
        wm_feature: torch.Tensor, # ewm
        transform_mask: Optional[torch.Tensor] = None,
        random_mask: bool = False, # random mask false - feasability space is already limited 
        # given few lines of code
        return_probs: bool = False,
    ):
        outputs = self.transform_selector(code_feature, wm_feature) # ftrans

        if transform_mask is not None:
            outputs = torch.masked_fill(outputs, transform_mask, float("-inf"))

        if random_mask:
            rand_mask = self._get_random_mask_like(outputs)
            outputs = torch.masked_fill(outputs, rand_mask.bool(), float("-inf"))

        if return_probs:
            return torch.softmax(outputs, dim=-1) # Softmax -> p trans 
        else:
            return outputs
        
        # two outputs - logits and probabilities
        # training: logits, gumbel- softmax for differentaible sampling
        # inference: probabilities 
