# transforms if statements 
from mutable_tree.nodes import Node
from .code_transformer import CodeTransformer
from ..tree_manip.visitors import CompoundIfVisitor, NestedIfVisitor


class CompoundIfTransformer(CodeTransformer): # inherits CodeTransformer 
    # identifiers for available transformations transformer can perform 
    name = "CompoundIfTransformer"
    TRANSFORM_IF_COMPOUND = "CompoundIfTransformer.if_compound" # merge nested ifs into single combined condition
    TRANSFORM_IF_NESTED = "CompoundIfTransformer.if_nested" # split compound condition into nested ifs 

    def __init__(self) -> None:
        super().__init__()

    # implements abstract method from base class
    # supports two distinct operations 
    def get_available_transforms(self):
        return [self.TRANSFORM_IF_COMPOUND, self.TRANSFORM_IF_NESTED]

    # looks at dst_style argument (specifies which transformation to perform)
    # picks appropriate visitor (CompoundIf or NestedIf)
    # runs .visit(node) to walk and transform given AST 
    def mutable_tree_transform(self, node: Node, dst_style: str):
        return {
            self.TRANSFORM_IF_COMPOUND: CompoundIfVisitor(),
            self.TRANSFORM_IF_NESTED: NestedIfVisitor(),
        }[dst_style].visit(node)
