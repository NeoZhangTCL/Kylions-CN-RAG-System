"""
文档解析器模块
提供PDF文档解析的接口定义
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union
from ..model import ParsedDocument


class DocumentParser(ABC):
    """文档解析器基类接口"""
    
    @abstractmethod
    def parse(self, document_path: str) -> ParsedDocument:
        """
        解析文档，返回结构化数据
        
        Args:
            document_path: 文档文件路径
            
        Returns:
            ParsedDocument: 解析后的文档数据
        """
        pass


class ImageProcessor(ABC):
    """图片处理器接口"""
    
    @abstractmethod
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
        pass


class TableProcessor(ABC):
    """表格处理器接口"""
    
    @abstractmethod
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
        pass
