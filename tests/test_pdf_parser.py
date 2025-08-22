#!/usr/bin/env python3
"""
测试简单PDF解析器
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.parsers.simple_pdf_parser import SimplePDFParser
from loguru import logger

def test_simple_pdf_parser():
    """测试简单PDF解析器功能"""
    
    # 设置日志
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    
    # 检查PDF文件是否存在
    pdf_path = "./data/raw/kylions_handle_book.pdf"
    if not os.path.exists(pdf_path):
        logger.error(f"PDF文件不存在: {pdf_path}. 请确保测试从项目根目录运行。")
        return False
    
    try:
        logger.info("=== 开始测试简单PDF解析器 ===")
        
        # 初始化解析器
        parser = SimplePDFParser(output_dir="data/processed")
        
        # 解析PDF
        logger.info(f"开始解析PDF文件: {pdf_path}")
        result = parser.parse(pdf_path)
        
        # 保存markdown
        markdown_path = parser.save_markdown(result)
        
        # 输出结果统计
        logger.success("=== 解析完成 ===")
        logger.info(f"Markdown文件保存路径: {markdown_path}")
        logger.info(f"Markdown长度: {len(result.markdown_content)} 字符")
        
        # 显示前500个字符的内容预览
        if result.markdown_content:
            preview = result.markdown_content[:500]
            logger.info(f"内容预览:\n{preview}...")
        
        # 检查生成的文件
        markdown_file = Path(markdown_path)
        
        if markdown_file.exists():
            logger.success(f"Markdown文件已成功生成: {markdown_file}")
        else:
            logger.error(f"Markdown文件生成失败: {markdown_file}")
            return False

        # 简单解析器不处理图片和表格，确认其为空
        if len(result.images) == 0 and len(result.tables) == 0:
            logger.info("图片和表格数量检查通过 (均为0).")
        else:
            logger.error(f"图片或表格数量不为0: images={len(result.images)}, tables={len(result.tables)}")
            return False

        logger.success("简单PDF解析器测试完成！")
        return True
        
    except Exception as e:
        logger.error(f"测试失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = test_simple_pdf_parser()
    sys.exit(0 if success else 1)
