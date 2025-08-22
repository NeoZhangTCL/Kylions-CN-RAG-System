"""
文档分片器模块
提供文档分片处理的接口定义
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from ..model import ParsedDocument, DocumentChunk


class DocumentChunker(ABC):
    """文档分片器基类接口"""
    
    @abstractmethod
    def chunk(self, document: ParsedDocument) -> List[DocumentChunk]:
        """
        将文档分片
        
        Args:
            document: 解析后的文档数据
            
        Returns:
            List[DocumentChunk]: 文档分片列表
        """
        pass


class HybridChunker(DocumentChunker):
    """混合分片策略实现接口"""
    
    @abstractmethod
    def chunk_markdown(self, markdown_content: str) -> List[DocumentChunk]:
        """
        混合分片策略：层级 + 递归
        
        Args:
            markdown_content: Markdown内容
            
        Returns:
            List[DocumentChunk]: 分片结果
        """
        pass


class MetadataGenerator(ABC):
    """元数据生成器接口"""
    
    @abstractmethod
    def generate_metadata(self, chunk_content: str, section_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        为文档分片生成元数据
        
        Args:
            chunk_content: 分片内容
            section_info: 章节信息
            
        Returns:
            Dict[str, Any]: 生成的元数据
        """
        pass
