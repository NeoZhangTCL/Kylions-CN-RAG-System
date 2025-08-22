"""
Qdrant向量检索器实现
使用Qdrant向量数据库进行文档存储和检索
"""

from typing import List, Dict, Any, Optional
import logging
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct, CollectionInfo,
    OptimizersConfig, ScalarQuantization, ScalarQuantizationConfig, ScalarType
)
from qdrant_client.http.exceptions import ResponseHandlingException

# QdrantRetriever implicitly implements the VectorStore protocol
from ..model import DocumentChunk, SearchResult


logger = logging.getLogger(__name__)


class QdrantRetriever:
    """
    Qdrant向量检索器
    实现基于Qdrant向量数据库的文档存储和检索功能
    
    该实现提供高效的向量相似度搜索，适用于：
    - RAG系统的文档检索
    - 语义搜索
    - 文档相似度匹配
    """
    
    def __init__(self, 
                 collection_name: str = "documents",
                 vector_size: int = 1024,
                 client: Optional[QdrantClient] = None,
                 distance_metric: Distance = Distance.COSINE,
                 on_disk_storage: bool = False):
        """
        初始化Qdrant检索器
        
        Args:
            collection_name: 集合名称，默认为"documents"
            vector_size: 向量维度，默认1024（BGE-large）
            client: Qdrant客户端，None时使用内存模式
            distance_metric: 距离度量，默认余弦距离
            on_disk_storage: 是否使用磁盘存储，内存模式时忽略
        """
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.distance_metric = distance_metric
        self.on_disk_storage = on_disk_storage
        
        # 初始化客户端
        if client is None:
            # 强制使用内存模式，避免文件锁定问题
            self.client = QdrantClient(":memory:")
            logger.info("Using Qdrant in-memory mode")
        else:
            self.client = client
            logger.info("Using provided Qdrant client")
        
        # 初始化集合
        self._setup_collection()
    
    def _setup_collection(self) -> None:
        """创建或配置向量集合"""
        try:
            # 检查集合是否存在
            collections = self.client.get_collections().collections
            collection_exists = any(c.name == self.collection_name for c in collections)
            
            if not collection_exists:
                logger.info(f"Creating collection: {self.collection_name}")
                
                # 配置向量参数
                vector_config = VectorParams(
                    size=self.vector_size,
                    distance=self.distance_metric,
                    on_disk=self.on_disk_storage
                )
                
                # 使用默认优化器配置
                optimizers_config = None
                
                # 使用默认量化配置（简化配置避免版本兼容问题）
                quantization_config = None
                
                # 创建集合
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=vector_config,
                    optimizers_config=optimizers_config,
                    quantization_config=quantization_config
                )
                
                logger.info(f"Collection '{self.collection_name}' created successfully")
            else:
                logger.info(f"Collection '{self.collection_name}' already exists")
                
                # 验证集合配置
                collection_info = self.client.get_collection(self.collection_name)
                actual_size = collection_info.config.params.vectors.size
                if actual_size != self.vector_size:
                    logger.warning(
                        f"Collection vector size ({actual_size}) differs from expected ({self.vector_size})"
                    )
                
        except Exception as e:
            logger.error(f"Failed to setup collection: {e}")
            raise RuntimeError(f"Collection setup failed: {e}") from e
    
    def add_documents(self, chunks: List[DocumentChunk]) -> None:
        """
        批量添加文档块到向量存储
        
        Args:
            chunks: 文档分片列表，每个分片应包含embedding向量
            
        Raises:
            ValueError: 当分片缺少embedding时
            RuntimeError: 当存储操作失败时
        """
        if not chunks:
            logger.warning("No chunks provided for indexing")
            return
        
        logger.info(f"Adding {len(chunks)} document chunks to collection")
        
        # 准备点数据
        points = []
        for i, chunk in enumerate(chunks):
            if chunk.embedding is None or len(chunk.embedding) == 0:
                raise ValueError(f"Chunk {i} is missing embedding vector")
            
            if len(chunk.embedding) != self.vector_size:
                raise ValueError(
                    f"Chunk {i} embedding dimension ({len(chunk.embedding)}) "
                    f"does not match collection dimension ({self.vector_size})"
                )
            
            # 生成唯一ID
            point_id = str(uuid4())
            
            # 准备payload（元数据+内容）
            payload = {
                "content": chunk.content,
                "metadata": chunk.metadata,
                # 添加一些有用的搜索字段
                "content_length": len(chunk.content),
                "has_metadata": bool(chunk.metadata),
            }
            
            # 创建点结构
            point = PointStruct(
                id=point_id,
                vector=chunk.embedding,
                payload=payload
            )
            points.append(point)
        
        try:
            # 批量插入
            operation_info = self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True  # 等待操作完成
            )
            
            if operation_info.status == "completed":
                logger.info(f"Successfully added {len(points)} documents")
            else:
                logger.warning(f"Indexing operation status: {operation_info.status}")
                
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise RuntimeError(f"Document indexing failed: {e}") from e
    
    def search(self, query_embedding: List[float], top_k: int = 3) -> List[SearchResult]:
        """
        相似度检索
        
        Args:
            query_embedding: 查询向量
            top_k: 返回结果数量，默认3
            
        Returns:
            List[SearchResult]: 按相似度降序排列的检索结果
            
        Raises:
            ValueError: 当查询向量维度不匹配时
            RuntimeError: 当检索操作失败时
        """
        if not query_embedding:
            raise ValueError("Query embedding cannot be empty")
        
        if len(query_embedding) != self.vector_size:
            raise ValueError(
                f"Query embedding dimension ({len(query_embedding)}) "
                f"does not match collection dimension ({self.vector_size})"
            )
        
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        
        logger.debug(f"Searching for {top_k} similar documents")
        
        try:
            # 执行向量搜索
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                with_payload=True,  # 包含payload信息
                with_vectors=False,  # 不返回向量（节省带宽）
                score_threshold=None,  # 不设置阈值，返回top_k结果
            )
            
            # 转换为SearchResult格式
            results = []
            for hit in search_results:
                result = SearchResult(
                    content=hit.payload["content"],
                    metadata=hit.payload["metadata"],
                    score=hit.score
                )
                results.append(result)
            
            logger.debug(f"Found {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise RuntimeError(f"Search operation failed: {e}") from e
    
    def search_with_filter(self, 
                          query_embedding: List[float], 
                          top_k: int = 3,
                          metadata_filter: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """
        带过滤条件的相似度检索
        
        Args:
            query_embedding: 查询向量
            top_k: 返回结果数量
            metadata_filter: 元数据过滤条件
            
        Returns:
            List[SearchResult]: 过滤后的检索结果
        """
        # 构建过滤条件
        query_filter = None
        if metadata_filter:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            
            conditions = []
            for key, value in metadata_filter.items():
                condition = FieldCondition(
                    key=f"metadata.{key}",
                    match=MatchValue(value=value)
                )
                conditions.append(condition)
            
            if conditions:
                query_filter = Filter(must=conditions)
        
        try:
            # 执行过滤搜索
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
                with_vectors=False,
            )
            
            # 转换结果
            results = []
            for hit in search_results:
                result = SearchResult(
                    content=hit.payload["content"],
                    metadata=hit.payload["metadata"],
                    score=hit.score
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Filtered search failed: {e}")
            raise RuntimeError(f"Filtered search operation failed: {e}") from e
    
    def get_collection_info(self) -> Dict[str, Any]:
        """获取集合信息"""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "vectors_count": collection_info.vectors_count,
                "indexed_vectors_count": collection_info.indexed_vectors_count,
                "points_count": collection_info.points_count,
                "segments_count": collection_info.segments_count,
                "vector_size": collection_info.config.params.vectors.size,
                "distance_metric": collection_info.config.params.vectors.distance.value,
                "status": collection_info.status.value,
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {"error": str(e)}
    
    def clear_collection(self) -> None:
        """清空集合中的所有数据"""
        try:
            self.client.delete_collection(self.collection_name)
            self._setup_collection()
            logger.info(f"Collection '{self.collection_name}' cleared and recreated")
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            raise RuntimeError(f"Clear collection failed: {e}") from e
    
    def delete_documents(self, ids: List[str]) -> None:
        """
        删除指定ID的文档
        
        Args:
            ids: 要删除的文档ID列表
        """
        if not ids:
            return
        
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=ids,
                wait=True
            )
            logger.info(f"Deleted {len(ids)} documents")
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            raise RuntimeError(f"Delete operation failed: {e}") from e


# 创建默认实例
default_retriever = QdrantRetriever()


def create_retriever(collection_name: str = "documents", 
                    vector_size: int = 1024) -> QdrantRetriever:
    """
    便捷函数：创建Qdrant检索器实例
    
    Args:
        collection_name: 集合名称
        vector_size: 向量维度
        
    Returns:
        QdrantRetriever: 检索器实例
    """
    return QdrantRetriever(
        collection_name=collection_name,
        vector_size=vector_size
    )
