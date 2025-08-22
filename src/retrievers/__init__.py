"""
检索器模块
提供向量存储和检索的接口定义
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from ..model import DocumentChunk, SearchResult


class VectorStore(ABC):
    """向量存储基类接口"""
    
    @abstractmethod
    def add_documents(self, chunks: List[DocumentChunk]) -> None:
        """
        批量添加文档块到向量存储
        
        Args:
            chunks: 文档分片列表
        """
        pass
    
    @abstractmethod
    def search(self, query_embedding: List[float], top_k: int) -> List[SearchResult]:
        """
        相似度检索
        
        Args:
            query_embedding: 查询向量
            top_k: 返回结果数量
            
        Returns:
            List[SearchResult]: 检索结果列表
        """
        pass


class QdrantRetriever(VectorStore):
    """Qdrant向量检索器接口"""
    
    @abstractmethod
    def setup_collection(self, collection_name: str, vector_size: int) -> None:
        """
        创建向量集合
        
        Args:
            collection_name: 集合名称
            vector_size: 向量维度
        """
        pass


class HybridSearchEngine(ABC):
    """混合检索引擎接口"""
    
    @abstractmethod
    def search(self, query: str, top_k: int = 3) -> List[SearchResult]:
        """
        多策略融合检索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            List[SearchResult]: 检索结果列表
        """
        pass
