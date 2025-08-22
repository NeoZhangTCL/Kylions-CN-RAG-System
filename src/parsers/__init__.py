"""
文档解析器模块
提供PDF文档解析的接口定义
"""

from typing import Protocol, List, Dict, Any, Union

# 导入数据模型
try:
    from ..model import ParsedDocument
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    from src.model import ParsedDocument


class DocumentParser(Protocol):
    """文档解析器协议接口 - 使用 Protocol 实现结构化子类型"""
    
    def parse(self, document_path: str) -> ParsedDocument:
        """
        解析文档，返回结构化数据
        
        Args:
            document_path: 文档文件路径
            
        Returns:
            ParsedDocument: 解析后的文档数据
        """
        ...


# 导入具体实现
from .simple_pdf_parser import SimplePDFParser


class ImageProcessor(Protocol):
    """图片处理器协议接口"""
    
    def process_image(self, image_data: bytes, page_num: int, image_index: int) -> Dict[str, Any]:
        """
        处理单张图片：保存 + 生成描述
        
        Args:
            image_data: 图片数据
            page_num: 页码
            image_index: 图片索引
            
        Returns:
            Dict[str, Any]: 图片处理结果，包含路径、描述等信息
        """
        ...


class TableProcessor(Protocol):
    """表格处理器协议接口"""
    
    def extract_table_to_markdown(self, table_data: Any, context_before: str, context_after: str) -> str:
        """
        将表格转换为Markdown格式，保留上下文
        
        Args:
            table_data: 表格数据
            context_before: 表格前上下文
            context_after: 表格后上下文
            
        Returns:
            str: Markdown格式的表格
        """
        ...


# 导出公共接口
__all__ = [
    'DocumentParser',      # 文档解析器协议
    'SimplePDFParser',     # 简单PDF解析器实现
    'ImageProcessor',      # 图片处理器协议
    'TableProcessor',      # 表格处理器协议
]
