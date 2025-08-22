"""
RAG系统异常定义
"""


class RAGSystemError(Exception):
    """RAG系统基础异常"""
    pass


class DocumentProcessingError(RAGSystemError):
    """文档处理异常"""
    pass


class QueryError(RAGSystemError):
    """查询异常"""
    pass


class ConfigurationError(RAGSystemError):
    """配置异常"""
    pass


class FileNotFoundError(RAGSystemError):
    """文件未找到异常"""
    pass


class VectorStoreError(RAGSystemError):
    """向量存储异常"""
    pass


class EmbeddingError(RAGSystemError):
    """向量化异常"""
    pass
