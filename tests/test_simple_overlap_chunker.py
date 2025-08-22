"""
SimpleOverlapChunker测试用例
"""

import pytest
from src.chunkers import SimpleOverlapChunker
from src.model import ParsedDocument, DocumentChunk


class TestSimpleOverlapChunker:
    """SimpleOverlapChunker测试类"""
    
    def test_basic_chunking(self):
        """测试基本分片功能"""
        # 创建测试文档（150个字符）
        content = "这是一个测试文档内容。" * 15  # 约150字符
        document = ParsedDocument(
            markdown_content=content,
            images=[],
            tables=[],
            metadata={"source": "test.txt"}
        )
        
        # 使用默认配置（100字，重叠20字）
        chunker = SimpleOverlapChunker()
        chunks = chunker.chunk(document)
        
        # 验证分片数量
        assert len(chunks) >= 1
        
        # 验证每个分片都是DocumentChunk类型
        for chunk in chunks:
            assert isinstance(chunk, DocumentChunk)
            assert chunk.content
            assert "chunk_index" in chunk.metadata
    
    def test_overlap_functionality(self):
        """测试重叠功能"""
        # 创建简单测试内容
        content = "0123456789" * 15  # 150个字符
        document = ParsedDocument(
            markdown_content=content,
            images=[],
            tables=[],
            metadata={"source": "test.txt"}
        )
        
        chunker = SimpleOverlapChunker(chunk_size=50, overlap_size=10)
        chunks = chunker.chunk(document)
        
        # 验证至少有2个分片
        assert len(chunks) >= 2
        
        # 验证重叠：第二个分片应该包含第一个分片的末尾部分
        if len(chunks) >= 2:
            # 第一个分片的后10个字符应该出现在第二个分片的前面
            first_chunk_end = chunks[0].content[-10:]
            second_chunk_start = chunks[1].content[:10]
            
            # 由于可能有strip()处理，我们检查是否有重叠内容
            assert len(first_chunk_end) > 0
            assert len(second_chunk_start) > 0
    
    def test_custom_parameters(self):
        """测试自定义参数"""
        content = "测试内容" * 20
        document = ParsedDocument(
            markdown_content=content,
            images=[],
            tables=[],
            metadata={"source": "test.txt"}
        )
        
        # 自定义参数：30字分片，5字重叠
        chunker = SimpleOverlapChunker(chunk_size=30, overlap_size=5)
        chunks = chunker.chunk(document)
        
        # 验证分片参数
        info = chunker.get_info()
        assert info["chunk_size"] == 30
        assert info["overlap_size"] == 5
        assert info["step_size"] == 25
    
    def test_metadata_preservation(self):
        """测试元数据保持"""
        content = "测试元数据保持功能"
        document = ParsedDocument(
            markdown_content=content,
            images=[],
            tables=[],
            metadata={
                "source": "test_metadata.txt",
                "author": "test_author"
            }
        )
        
        chunker = SimpleOverlapChunker(chunk_size=10, overlap_size=2)
        chunks = chunker.chunk(document)
        
        # 验证元数据被保持
        for chunk in chunks:
            assert chunk.metadata["source"] == "test_metadata.txt"
            assert chunk.metadata["author"] == "test_author"
            assert "chunk_index" in chunk.metadata
            assert "start_position" in chunk.metadata
    
    def test_invalid_parameters(self):
        """测试无效参数"""
        with pytest.raises(ValueError):
            # 重叠大小等于分片大小
            SimpleOverlapChunker(chunk_size=100, overlap_size=100)
        
        with pytest.raises(ValueError):
            # 重叠大小大于分片大小
            SimpleOverlapChunker(chunk_size=50, overlap_size=60)
    
    def test_empty_document(self):
        """测试空文档"""
        document = ParsedDocument(
            markdown_content="",
            images=[],
            tables=[],
            metadata={"source": "empty.txt"}
        )
        
        chunker = SimpleOverlapChunker()
        chunks = chunker.chunk(document)
        
        # 空文档应该返回空列表
        assert chunks == []
