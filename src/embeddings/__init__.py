"""
向量化模型模块
提供文本向量化的接口定义
"""

from abc import ABC, abstractmethod
from typing import List, Union
from pydantic import BaseModel


class EmbeddingModel(ABC):
    """向量化模型基类接口"""
    
    @abstractmethod
    def embed(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """
        文本向量化
        
        Args:
            texts: 待向量化的文本，可以是单个字符串或字符串列表
            
        Returns:
            List[List[float]]: 向量化结果，每个文本对应一个向量
        """
        pass


class BGEEmbedder(EmbeddingModel):
    """BGE中文向量化器接口"""
    
    @abstractmethod
    def __init__(self, model_name: str = "BAAI/bge-large-zh-v1.5"):
        """
        初始化BGE向量化器
        
        Args:
            model_name: 模型名称
        """
        pass
