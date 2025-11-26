import abc # Abstract Base Class - defines classes that can't be instantiated unless abstract method implemented 
from ..nodes import Node # AST Node 
from ..tree_manip.visitors import TransformingVisitor # base visitor that can transform and modify AST nodes 
from typing import Dict


class CodeTransformer: # defines base interface for all code transformer classes 
    name: str

    @abc.abstractmethod # abstract methods 
    def get_available_transforms(self): # returns list/dict of all transformations class can perform 
        pass

    @abc.abstractmethod
    def code_transform(self, code: str, dst_style: str): # transform raw source code (as string) into another style/format
        pass

    @abc.abstractmethod
    def mutable_tree_transform(self, node: Node, dst_style: str): # perform actual AST manipulators (using visitors/transformers)
        pass

    def throw_invalid_dst_style(self, dst_style: str):
        msg = f"invalid dst_style: {dst_style} for {self.__class__.__name__}" # error msg when transformer receives unsupported dst_style
        raise ValueError(msg)
