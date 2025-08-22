"""
简单PDF解析器
只处理文字部分，不处理图片和表格
"""

import os
import re
from typing import List, Dict, Any
from pathlib import Path

import fitz  # PyMuPDF
from loguru import logger
from ..model.document_models import ParsedDocument


class SimplePDFParser:
    """
    简单PDF解析器
    
    功能：
    1. 提取PDF中的纯文本内容
    2. 转换为Markdown格式
    3. 保持基本的文档结构
    4. 不处理图片和表格
    """
    
    def __init__(self, output_dir: str = "data/processed", filter_patterns: List[str] = None):
        """
        初始化解析器
        
        Args:
            output_dir: 输出目录路径
            filter_patterns: 自定义过滤模式列表（可选）
        """
        self.output_dir = Path(output_dir)
        self.markdown_dir = self.output_dir / "markdown"
        
        # 创建输出目录
        self.markdown_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置过滤模式
        self.filter_patterns = filter_patterns or self._get_default_filter_patterns()
        
        logger.info(f"简单PDF解析器初始化完成，输出目录: {self.output_dir}")
        logger.info(f"已配置 {len(self.filter_patterns)} 个过滤模式")
    
    def parse(self, document_path: str) -> ParsedDocument:
        """
        解析PDF文档，提取文字内容
        
        Args:
            document_path: PDF文件路径
            
        Returns:
            ParsedDocument: 解析后的文档数据
        """
        if not os.path.exists(document_path):
            raise FileNotFoundError(f"PDF文件不存在: {document_path}")
        
        logger.info(f"开始解析PDF文件: {document_path}")
        
        try:
            # 打开PDF文档
            doc = fitz.open(document_path)
            
            # 提取文本内容
            markdown_content = self._extract_text_to_markdown(doc)
            
            # 提取元数据
            metadata = self._extract_metadata(doc, document_path)
            
            # 关闭文档
            doc.close()
            
            # 创建解析结果
            parsed_document = ParsedDocument(
                markdown_content=markdown_content,
                images=[],  # 简单解析器不处理图片
                tables=[],  # 简单解析器不处理表格
                metadata=metadata
            )
            
            logger.info(f"PDF解析完成，共处理 {metadata.get('page_count', 0)} 页")
            
            return parsed_document
            
        except Exception as e:
            logger.error(f"PDF解析失败: {str(e)}")
            raise
    
    def _extract_text_to_markdown(self, doc: fitz.Document) -> str:
        """
        提取PDF文本并转换为Markdown格式
        
        Args:
            doc: PyMuPDF文档对象
            
        Returns:
            str: Markdown格式的文本内容
        """
        markdown_lines = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            
            # 提取页面文本
            text = page.get_text()
            
            if not text.strip():
                continue
            
            # 处理页面文本并添加到结果中
            processed_lines = self._process_page_text(text, page_num + 1)
            markdown_lines.extend(processed_lines)
        
        return "\n".join(markdown_lines)
    
    def _process_page_text(self, text: str, page_num: int) -> List[str]:
        """
        处理单页文本内容
        
        Args:
            text: 原始文本
            page_num: 页码
            
        Returns:
            List[str]: 处理后的文本行列表
        """
        lines = []
        
        
        # 按行处理文本
        text_lines = text.split('\n')
        
        for line in text_lines:
            line = line.strip()
            
            if not line:
                lines.append("")
                continue
            
            # 过滤重复的页眉页脚模式
            if self._should_filter_line(line):
                continue
            
            # 简单的标题检测（基于字体大小和位置，这里用简化逻辑）
            if self._is_likely_heading(line):
                lines.append(f"### {line}")
            else:
                lines.append(line)
        
        return lines
    
    def _is_likely_heading(self, line: str) -> bool:
        """
        简单判断是否为标题
        
        Args:
            line: 文本行
            
        Returns:
            bool: 是否为标题
        """
        # 简单的标题判断规则
        if len(line) < 50 and line.isupper():
            return True
        
        # 检查是否以数字开头（如：1. 2. 等）
        if re.match(r'^\d+\.\s+', line):
            return True
        
        # 检查是否以常见标题关键词开头
        heading_keywords = ['第', '章', '节', '条', '款', '项']
        for keyword in heading_keywords:
            if line.startswith(keyword):
                return True
        
        return False
    
    def _get_default_filter_patterns(self) -> List[str]:
        """
        获取默认的过滤模式列表
        
        Returns:
            List[str]: 默认过滤模式列表
        """
        return [
            # 重复的文档标题
            r'^银河麒麟桌面操作系统V10\s*用户手册$',
            r'^银河麒麟桌面操作系统V10\s*1$',
            
            # 页码信息
            r'^第\d+\s*页共\d+\s*页$',
            r'^第\d+\s*页$',
            
            # 版权信息（通常在每页重复出现）
            r'^版权所有©\s*\d{4}-\d{4}\s*麒麟软件有限公司',
            r'^麒麟软件有限公司$',
            
            # 其他重复的页眉页脚
            r'^目\s*录$',
            r'^麒麟软件有限公司（简称"麒麟软件"）',
            
            # 通用模式
            r'^第\s*\d+\s*页\s*共\s*\d+\s*页$',
            r'^第\s*\d+\s*页$',
        ]
    
    def _should_filter_line(self, line: str) -> bool:
        """
        判断是否应该过滤掉这一行
        
        Args:
            line: 文本行
            
        Returns:
            bool: 是否应该过滤
        """
        # 检查是否匹配任何过滤模式
        for pattern in self.filter_patterns:
            if re.match(pattern, line, re.IGNORECASE):
                return True
        
        # 过滤掉过短的无意义行
        if len(line.strip()) < 3:
            return True
        
        return False
    
    def _extract_metadata(self, doc: fitz.Document, file_path: str) -> Dict[str, Any]:
        """
        提取文档元数据
        
        Args:
            doc: PyMuPDF文档对象
            file_path: 文件路径
            
        Returns:
            Dict[str, Any]: 元数据字典
        """
        metadata = {
            'file_path': file_path,
            'file_name': Path(file_path).name,
            'page_count': len(doc),
            'parser_type': 'SimplePDFParser'
        }
        
        # 尝试提取PDF元数据
        try:
            pdf_metadata = doc.metadata
            if pdf_metadata:
                metadata.update({
                    'title': pdf_metadata.get('title', ''),
                    'author': pdf_metadata.get('author', ''),
                    'subject': pdf_metadata.get('subject', ''),
                    'creator': pdf_metadata.get('creator', ''),
                    'producer': pdf_metadata.get('producer', ''),
                    'creation_date': pdf_metadata.get('creationDate', ''),
                    'modification_date': pdf_metadata.get('modDate', '')
                })
        except Exception as e:
            logger.warning(f"提取PDF元数据失败: {str(e)}")
        
        return metadata
    
    def save_markdown(self, parsed_document: ParsedDocument, output_filename: str = None) -> str:
        """
        保存Markdown文件
        
        Args:
            parsed_document: 解析后的文档
            output_filename: 输出文件名（可选）
            
        Returns:
            str: 保存的文件路径
        """
        if output_filename is None:
            file_name = parsed_document.metadata.get('file_name', 'document')
            output_filename = f"{Path(file_name).stem}_simple.md"
        
        output_path = self.markdown_dir / output_filename
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(parsed_document.markdown_content)
            
            logger.info(f"Markdown文件已保存: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"保存Markdown文件失败: {str(e)}")
            raise
