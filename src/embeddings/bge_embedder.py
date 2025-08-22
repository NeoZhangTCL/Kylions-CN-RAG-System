"""
BGE中文向量化器实现
使用 bge-large-zh-v1.5 模型进行文本向量化
"""

from typing import List, Union
import logging
from sentence_transformers import SentenceTransformer

# BGEEmbedder implicitly implements the Embedder protocol


logger = logging.getLogger(__name__)


class BGEEmbedder:
    """
    BGE中文向量化器
    使用 bge-large-zh-v1.5 模型，专门针对中文文本优化
    
    该实现提供高质量的中文文本向量化，适用于：
    - 中文文档检索
    - 语义相似度计算
    - RAG系统的向量化需求
    """
    
    def __init__(self, model_name: str = "BAAI/bge-large-zh-v1.5", device: str = None):
        """
        初始化BGE向量化器
        
        Args:
            model_name: 模型名称，默认使用 bge-large-zh-v1.5
            device: 运行设备，None表示自动选择（优先GPU）
        """
        self.model_name = model_name
        self.device = device
        self._model = None
        
        logger.info(f"Initializing BGE embedder with model: {model_name}")
    
    @property
    def model(self) -> SentenceTransformer:
        """懒加载模型，首次访问时才加载"""
        if self._model is None:
            logger.info("Loading BGE model...")
            self._model = SentenceTransformer(
                self.model_name,
                device=self.device
            )
            logger.info(f"BGE model loaded successfully on device: {self._model.device}")
        return self._model
    
    def embed(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """
        对文本进行向量化
        
        Args:
            texts: 待向量化的文本，可以是单个字符串或字符串列表
            
        Returns:
            向量列表，每个向量是1024维的浮点数列表
            
        Raises:
            ValueError: 当输入为空或无效时
        """
        if not texts:
            raise ValueError("Input texts cannot be empty")
        
        # 统一处理输入格式
        if isinstance(texts, str):
            input_texts = [texts]
            is_single = True
        else:
            input_texts = texts
            is_single = False
        
        # 验证输入
        if not all(isinstance(text, str) and text.strip() for text in input_texts):
            raise ValueError("All inputs must be non-empty strings")
        
        logger.debug(f"Embedding {len(input_texts)} texts")
        
        try:
            # 使用BGE模型进行向量化
            # normalize_embeddings=True 确保向量归一化，适合余弦相似度计算
            embeddings = self.model.encode(
                input_texts,
                normalize_embeddings=True,
                show_progress_bar=len(input_texts) > 10  # 大批量时显示进度条
            )
            
            # 转换为列表格式
            result = embeddings.tolist()
            
            logger.debug(f"Generated embeddings with shape: {embeddings.shape}")
            return result
            
        except Exception as e:
            logger.error(f"Error during embedding: {str(e)}")
            raise RuntimeError(f"Failed to generate embeddings: {str(e)}") from e
    
    def get_embedding_dimension(self) -> int:
        """获取向量维度"""
        return self.model.get_sentence_embedding_dimension()
    
    def get_model_info(self) -> dict:
        """获取模型信息"""
        return {
            "model_name": self.model_name,
            "device": str(self.model.device) if self._model else "not_loaded",
            "embedding_dimension": self.get_embedding_dimension() if self._model else 1024,
            "max_sequence_length": getattr(self.model, "max_seq_length", "unknown") if self._model else "unknown"
        }


# 创建默认实例，方便直接使用
default_embedder = BGEEmbedder()


def embed_text(texts: Union[str, List[str]]) -> List[List[float]]:
    """
    便捷函数：使用默认BGE模型进行文本向量化
    
    Args:
        texts: 待向量化的文本
        
    Returns:
        向量列表
    """
    return default_embedder.embed(texts)
