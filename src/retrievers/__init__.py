"""
检索器模块
提供向量存储和检索的接口定义
"""

from typing import Protocol, List, Dict, Any
from ..model import DocumentChunk, SearchResult
from .qdrant_retriever import QdrantRetriever, default_retriever, create_retriever


class VectorStore(Protocol):
    """向量存储协议接口"""
    
    def add_documents(self, chunks: List[DocumentChunk]) -> None:
        """
        批量添加文档块到向量存储
        
        Args:
            chunks: 文档分片列表
        """
        ...
    
    def search(self, query_embedding: List[float], top_k: int) -> List[SearchResult]:
        """
        相似度检索
        
        Args:
            query_embedding: 查询向量
            top_k: 返回结果数量
            
        Returns:
            List[SearchResult]: 检索结果列表
        """
        ...

# 导出公共接口
__all__ = [
    'VectorStore',           # 向量存储协议
    'QdrantRetriever',       # Qdrant向量检索器实现
    'default_retriever',     # 默认检索器实例
    'create_retriever',      # 便捷创建函数
]
