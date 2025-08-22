"""
文档分片器模块
提供文档分片处理的接口定义
"""

from typing import Protocol, List, Dict, Any, Optional, runtime_checkable
from ..model import ParsedDocument, DocumentChunk


@runtime_checkable
class DocumentChunker(Protocol):
    """文档分片器协议接口"""
    
    def chunk(self, document: ParsedDocument) -> List[DocumentChunk]:
        """
        将文档分片
        
        Args:
            document: 解析后的文档数据
            
        Returns:
            List[DocumentChunk]: 文档分片列表
        """
        ...


class MetadataGenerator(Protocol):
    """元数据生成器协议接口"""
    
    def generate_metadata(self, chunk_content: str, section_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        为文档分片生成元数据
        
        Args:
            chunk_content: 分片内容
            section_info: 章节信息
            
        Returns:
            Dict[str, Any]: 生成的元数据
        """
        ...


# 导入具体实现
from .simple_overlap_chunker import SimpleOverlapChunker

# 导出公共接口
__all__ = [
    'DocumentChunker',       # 文档分片器协议
    'HybridChunker',         # 混合分片策略协议
    'MetadataGenerator',     # 元数据生成器协议
    'SimpleOverlapChunker',  # 简单重叠分片器
]
