"""
RAG系统核心模块
提供文档检索增强生成系统的核心接口
"""

from typing import List
from .parsers import DocumentParser, ImageProcessor, TableProcessor
from .chunkers import DocumentChunker, HybridChunker, MetadataGenerator
from .embeddings import EmbeddingModel, BGEEmbedder
from .retrievers import VectorStore, QdrantRetriever, HybridSearchEngine
from .model import ParsedDocument, DocumentChunk, SearchResult, RAGConfig


class RAGService:
    """RAG系统主服务接口"""
    
    def __init__(self, 
                 parser: DocumentParser,
                 chunker: DocumentChunker,
                 embedder: EmbeddingModel,
                 vector_store: VectorStore):
        """
        初始化RAG服务
        
        Args:
            parser: 文档解析器
            chunker: 文档分片器
            embedder: 向量化模型
            vector_store: 向量存储
        """
        self.parser = parser
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
    
    def index_document(self, document_path: str) -> None:
        """
        索引文档
        
        Args:
            document_path: 文档文件路径
        """
        pass
    
    def query(self, question: str, top_k: int = 3) -> List[SearchResult]:
        """
        查询文档
        
        Args:
            question: 查询问题
            top_k: 返回结果数量
            
        Returns:
            List[SearchResult]: 检索结果列表
        """
        pass


# 导出所有核心接口
__all__ = [
    # 解析器相关
    'DocumentParser',
    'ImageProcessor',
    'TableProcessor',
    
    # 分片器相关
    'DocumentChunker',
    'HybridChunker',
    'MetadataGenerator',
    
    # 向量化相关
    'EmbeddingModel',
    'BGEEmbedder',
    
    # 检索器相关
    'VectorStore',
    'QdrantRetriever',
    'HybridSearchEngine',
    
    # 数据模型
    'ParsedDocument',
    'DocumentChunk', 
    'SearchResult',
    'RAGConfig',
    
    # 主服务
    'RAGService'
]
