"""
简单重叠分片器实现
使用固定长度重叠分片策略
"""

from typing import List, Dict, Any
from ..model import ParsedDocument, DocumentChunk
from . import DocumentChunker


class SimpleOverlapChunker(DocumentChunker):
    """简单重叠分片器
    
    使用固定长度和重叠量对文档进行分片
    """
    
    def __init__(self, chunk_size: int = 100, overlap_size: int = 20):
        """
        初始化分片器
        
        Args:
            chunk_size: 每个分片的字符数，默认100
            overlap_size: 重叠字符数，默认20
        """
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        
        if overlap_size >= chunk_size:
            raise ValueError("重叠大小不能大于或等于分片大小")
    
    def chunk(self, document: ParsedDocument) -> List[DocumentChunk]:
        """
        将文档分片
        
        Args:
            document: 解析后的文档数据
            
        Returns:
            List[DocumentChunk]: 文档分片列表
        """
        content = document.markdown_content
        chunks = []
        
        # 计算步长（下一个分片的起始位置）
        step_size = self.chunk_size - self.overlap_size
        
        start = 0
        chunk_index = 0
        
        while start < len(content):
            # 计算当前分片的结束位置
            end = min(start + self.chunk_size, len(content))
            
            # 提取分片内容
            chunk_content = content[start:end].strip()
            
            # 跳过空分片
            if not chunk_content:
                start += step_size
                continue
            
            # 创建分片元数据
            chunk_metadata = {
                "chunk_index": chunk_index,
                "start_position": start,
                "end_position": end,
                "chunk_size": len(chunk_content),
                "overlap_size": self.overlap_size if start > 0 else 0,
                "source_document": document.metadata.get("source", "unknown")
            }
            
            # 合并原文档的元数据
            chunk_metadata.update(document.metadata)
            
            # 创建DocumentChunk对象
            chunk = DocumentChunk(
                content=chunk_content,
                metadata=chunk_metadata
            )
            
            chunks.append(chunk)
            chunk_index += 1
            
            # 移动到下一个分片的起始位置
            start += step_size
        
        return chunks
    
    def get_info(self) -> Dict[str, Any]:
        """
        获取分片器配置信息
        
        Returns:
            Dict[str, Any]: 分片器配置
        """
        return {
            "chunker_type": "SimpleOverlapChunker",
            "chunk_size": self.chunk_size,
            "overlap_size": self.overlap_size,
            "step_size": self.chunk_size - self.overlap_size
        }
