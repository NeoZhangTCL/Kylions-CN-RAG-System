"""
配置相关数据模型
"""

from pydantic import BaseModel


class RAGConfig(BaseModel):
    """RAG系统配置模型"""
    output_dir: str = "data/processed"
    chunk_size: int = 500
    chunk_overlap: int = 50
    embedding_model: str = "BAAI/bge-large-zh-v1.5"
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    collection_name: str = "kylinos_docs"
    default_top_k: int = 3
