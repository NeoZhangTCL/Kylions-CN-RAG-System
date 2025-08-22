"""
检索相关数据模型
"""

from typing import Dict, Any
from pydantic import BaseModel


class SearchResult(BaseModel):
    """检索结果数据模型"""
    content: str
    metadata: Dict[str, Any]
    score: float
