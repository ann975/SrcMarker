# wraps 2 BlockSwapper variants (NormalBLockSwapper and NegatedBlockSwapper)

from mutable_tree.nodes import Node
from .code_transformer import CodeTransformer
from ..tree_manip.visitors import NormalBlockSwapper, NegatedBlockSwapper

# rewrite if statements by swapping branches 
class IfBlockSwapTransformer(CodeTransformer):
    name = "IfBlockSwapTransformer"
    TRANSFORM_IF_BLOCK_NORMAL = "IfBlockSwapTransformer.normal" # swaps if with logical components 
    TRANSFORM_IF_BLOCK_NEGATED = "IfBlockSwapTransformer.negated" # swaps with negated condition

    def __init__(self) -> None:
        super().__init__()

    def get_available_transforms(self):
        return [self.TRANSFORM_IF_BLOCK_NORMAL, self.TRANSFORM_IF_BLOCK_NEGATED]

    def mutable_tree_transform(self, node: Node, dst_style: str):
        return {
            self.TRANSFORM_IF_BLOCK_NORMAL: NormalBlockSwapper(), 
            # NormalBlockSwapper - converts if to logical complement, then swaps if else blocks
            self.TRANSFORM_IF_BLOCK_NEGATED: NegatedBlockSwapper(),
            # converts if condition to negated form (adding/toggling not) and swaps branches
        }[dst_style].visit(node)
