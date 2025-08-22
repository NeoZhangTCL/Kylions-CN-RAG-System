"""
文档相关数据模型
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel


class ParsedDocument(BaseModel):
    """解析后的文档数据模型"""
    markdown_content: str
    images: List[Dict[str, Any]]
    tables: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class DocumentChunk(BaseModel):
    """文档分片数据模型"""
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
