"""
RAG系统核心模块
提供文档检索增强生成系统的核心接口
"""

from typing import List
# Import available modules only
from .embeddings import Embedder, BGEEmbedder, default_embedder, embed_text


# 导出可用的接口
__all__ = [
    # 向量化相关
    'Embedder',         # 向量化协议
    'BGEEmbedder',      # BGE向量化器实现
    'default_embedder', # 默认向量化器实例
    'embed_text',       # 便捷向量化函数
]
