"""
向量化模型模块
提供文本向量化的接口定义
"""

from typing import Protocol, List, Union
from .bge_embedder import BGEEmbedder, default_embedder, embed_text


class Embedder(Protocol):
    """BGE中文向量化器协议接口"""
    
    def embed(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """继承自EmbeddingModel的向量化方法"""
        ...


# 导出公共接口
__all__ = [
    'Embedder',         # 向量化模型协议
    'BGEEmbedder',      # BGE向量化器实现
    'default_embedder', # 默认BGE向量化器实例
    'embed_text',       # 便捷向量化函数
]
