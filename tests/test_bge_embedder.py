"""
BGE向量化器测试
验证BGE embedder的功能和性能
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from typing import List

from src.embeddings import BGEEmbedder, default_embedder, embed_text


class TestBGEEmbedder:
    """BGE向量化器测试用例"""

    def test_init_default_parameters(self):
        """测试默认参数初始化"""
        embedder = BGEEmbedder()
        assert embedder.model_name == "BAAI/bge-large-zh-v1.5"
        assert embedder.device is None
        assert embedder._model is None

    def test_init_custom_parameters(self):
        """测试自定义参数初始化"""
        model_name = "custom/model"
        device = "cpu"
        embedder = BGEEmbedder(model_name=model_name, device=device)
        assert embedder.model_name == model_name
        assert embedder.device == device

    @patch('src.embeddings.bge_embedder.SentenceTransformer')
    def test_lazy_model_loading(self, mock_transformer):
        """测试懒加载模型机制"""
        # Mock the SentenceTransformer
        mock_model = Mock()
        mock_model.device = "cpu"
        mock_transformer.return_value = mock_model
        
        embedder = BGEEmbedder()
        
        # 初始时模型未加载
        assert embedder._model is None
        
        # 访问模型属性时才加载
        model = embedder.model
        assert model is mock_model
        mock_transformer.assert_called_once_with("BAAI/bge-large-zh-v1.5", device=None)
        
        # 再次访问使用缓存的模型
        model2 = embedder.model
        assert model2 is mock_model
        assert mock_transformer.call_count == 1  # 只调用一次

    @patch('src.embeddings.bge_embedder.SentenceTransformer')
    def test_embed_single_text(self, mock_transformer):
        """测试单个文本向量化"""
        # Mock the model
        mock_model = Mock()
        mock_model.device = "cpu"
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_transformer.return_value = mock_model
        
        embedder = BGEEmbedder()
        
        # 测试单个字符串输入
        text = "这是测试文本"
        result = embedder.embed(text)
        
        # 验证结果
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], list)
        assert len(result[0]) == 3
        assert result == [[0.1, 0.2, 0.3]]
        
        # 验证模型调用
        mock_model.encode.assert_called_once_with(
            [text],
            normalize_embeddings=True,
            show_progress_bar=False
        )

    @patch('src.embeddings.bge_embedder.SentenceTransformer')
    def test_embed_multiple_texts(self, mock_transformer):
        """测试多个文本向量化"""
        # Mock the model
        mock_model = Mock()
        mock_model.device = "cpu"
        mock_model.encode.return_value = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]
        ])
        mock_transformer.return_value = mock_model
        
        embedder = BGEEmbedder()
        
        # 测试多个字符串输入
        texts = ["第一个文本", "第二个文本"]
        result = embedder.embed(texts)
        
        # 验证结果
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(emb, list) for emb in result)
        assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        
        # 验证模型调用
        mock_model.encode.assert_called_once_with(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False
        )

    @patch('src.embeddings.bge_embedder.SentenceTransformer')
    def test_embed_progress_bar_for_large_batch(self, mock_transformer):
        """测试大批量文本显示进度条"""
        # Mock the model
        mock_model = Mock()
        mock_model.device = "cpu"
        mock_model.encode.return_value = np.array([[0.1] * 15])
        mock_transformer.return_value = mock_model
        
        embedder = BGEEmbedder()
        
        # 测试大批量输入（>10个）
        texts = [f"文本{i}" for i in range(15)]
        embedder.embed(texts)
        
        # 验证显示进度条
        mock_model.encode.assert_called_once_with(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True
        )

    def test_embed_empty_input(self):
        """测试空输入处理"""
        embedder = BGEEmbedder()
        
        # 测试空列表
        with pytest.raises(ValueError, match="Input texts cannot be empty"):
            embedder.embed([])
        
        # 测试空字符串
        with pytest.raises(ValueError, match="Input texts cannot be empty"):
            embedder.embed("")

    def test_embed_invalid_input(self):
        """测试无效输入处理"""
        embedder = BGEEmbedder()
        
        # 测试包含空字符串的列表
        with pytest.raises(ValueError, match="All inputs must be non-empty strings"):
            embedder.embed(["有效文本", "", "另一个有效文本"])
        
        # 测试包含非字符串的列表
        with pytest.raises(ValueError, match="All inputs must be non-empty strings"):
            embedder.embed(["有效文本", 123, "另一个有效文本"])

    @patch('src.embeddings.bge_embedder.SentenceTransformer')
    def test_embed_model_error_handling(self, mock_transformer):
        """测试模型错误处理"""
        # Mock the model to raise an exception
        mock_model = Mock()
        mock_model.device = "cpu"
        mock_model.encode.side_effect = Exception("模型错误")
        mock_transformer.return_value = mock_model
        
        embedder = BGEEmbedder()
        
        with pytest.raises(RuntimeError, match="Failed to generate embeddings"):
            embedder.embed("测试文本")

    @patch('src.embeddings.bge_embedder.SentenceTransformer')
    def test_get_embedding_dimension(self, mock_transformer):
        """测试获取向量维度"""
        mock_model = Mock()
        mock_model.device = "cpu"
        mock_model.get_sentence_embedding_dimension.return_value = 1024
        mock_transformer.return_value = mock_model
        
        embedder = BGEEmbedder()
        dimension = embedder.get_embedding_dimension()
        
        assert dimension == 1024
        mock_model.get_sentence_embedding_dimension.assert_called_once()

    @patch('src.embeddings.bge_embedder.SentenceTransformer')
    def test_get_model_info_loaded(self, mock_transformer):
        """测试获取已加载模型信息"""
        mock_model = Mock()
        mock_model.device = "cuda:0"
        mock_model.get_sentence_embedding_dimension.return_value = 1024
        mock_model.max_seq_length = 512
        mock_transformer.return_value = mock_model
        
        embedder = BGEEmbedder()
        # 触发模型加载
        _ = embedder.model
        
        info = embedder.get_model_info()
        
        expected = {
            "model_name": "BAAI/bge-large-zh-v1.5",
            "device": "cuda:0",
            "embedding_dimension": 1024,
            "max_sequence_length": 512
        }
        assert info == expected

    def test_get_model_info_not_loaded(self):
        """测试获取未加载模型信息"""
        embedder = BGEEmbedder()
        info = embedder.get_model_info()
        
        expected = {
            "model_name": "BAAI/bge-large-zh-v1.5",
            "device": "not_loaded",
            "embedding_dimension": 1024,
            "max_sequence_length": "unknown"
        }
        assert info == expected


class TestDefaultEmbedder:
    """测试默认embedder和便捷函数"""

    @patch('src.embeddings.bge_embedder.default_embedder')
    def test_embed_text_function(self, mock_default_embedder):
        """测试便捷函数"""
        mock_default_embedder.embed.return_value = [[0.1, 0.2, 0.3]]
        
        result = embed_text("测试文本")
        
        assert result == [[0.1, 0.2, 0.3]]
        mock_default_embedder.embed.assert_called_once_with("测试文本")

    def test_default_embedder_exists(self):
        """测试默认embedder存在"""
        assert default_embedder is not None
        assert isinstance(default_embedder, BGEEmbedder)


class TestIntegration:
    """集成测试"""

    @pytest.mark.slow
    def test_real_model_integration(self):
        """
        真实模型集成测试
        注意：这个测试需要真实下载模型，标记为slow test
        """
        # 跳过如果没有网络连接或资源有限
        pytest.skip("Integration test - requires model download")
        
        embedder = BGEEmbedder()
        
        # 测试中文文本
        chinese_texts = [
            "麒麟操作系统是一个优秀的Linux发行版",
            "这个系统适合办公和学习使用",
            "系统具有良好的安全性和稳定性"
        ]
        
        embeddings = embedder.embed(chinese_texts)
        
        # 验证结果格式
        assert len(embeddings) == 3
        assert all(len(emb) == 1024 for emb in embeddings)  # BGE-large维度
        
        # 验证向量归一化（余弦相似度友好）
        for emb in embeddings:
            norm = np.linalg.norm(emb)
            assert abs(norm - 1.0) < 1e-6  # 应该接近1
        
        # 验证语义相似性
        emb1, emb2, emb3 = embeddings
        sim_1_2 = np.dot(emb1, emb2)  # 系统相关
        sim_1_3 = np.dot(emb1, emb3)  # 系统相关
        sim_2_3 = np.dot(emb2, emb3)  # 都是正面描述
        
        # 所有相似度应该为正值（相关内容）
        assert sim_1_2 > 0
        assert sim_1_3 > 0 
        assert sim_2_3 > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
