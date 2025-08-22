"""
RAG系统主控制器
封装文档解析、分片、向量化、存储和检索的完整流程
"""

import time
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

from .parsers.simple_pdf_parser import SimplePDFParser
from .embeddings.bge_embedder import BGEEmbedder
from .chunkers.simple_overlap_chunker import SimpleOverlapChunker
from .retrievers.qdrant_retriever import QdrantRetriever
from .model.search_models import SearchResult
from .config import load_config, get_default_config
from .exceptions import (
    RAGSystemError, DocumentProcessingError, QueryError, 
    ConfigurationError, FileNotFoundError as RAGFileNotFoundError
)


logger = logging.getLogger(__name__)


class RAGSystem:
    """
    RAG系统主控制器
    封装所有组件，提供简洁的文档处理和查询接口
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化RAG系统
        
        Args:
            config: 配置字典，如果为None则使用默认配置
        """
        # 加载配置
        if config is None:
            self.config = get_default_config()
        else:
            self.config = config
        
        logger.info("正在初始化RAG系统...")
        
        # 初始化各个组件
        self._init_components()
        
        # 系统状态
        self.is_document_processed = False
        self.document_count = 0
        self.chunk_count = 0
        self.last_processed_document = None
        
        logger.info("RAG系统初始化完成")
    
    def _init_components(self) -> None:
        """初始化所有组件"""
        try:
            # 初始化PDF解析器
            parser_config = self.config['parser']
            self.parser = SimplePDFParser(
                output_dir=parser_config['output_dir'],
                filter_patterns=parser_config.get('filter_patterns')
            )
            
            # 初始化向量化器
            embedder_config = self.config['embedder']
            self.embedder = BGEEmbedder(
                model_name=embedder_config['model_name'],
                device=embedder_config.get('device')
            )
            
            # 初始化分片器
            chunker_config = self.config['chunker']
            self.chunker = SimpleOverlapChunker(
                chunk_size=chunker_config['chunk_size'],
                overlap_size=chunker_config['overlap_size']
            )
            
            # 初始化检索器
            retriever_config = self.config['retriever']
            self.retriever = QdrantRetriever(
                collection_name=retriever_config['collection_name'],
                vector_size=retriever_config['vector_size']
            )
            
            logger.info("所有组件初始化完成")
            
        except Exception as e:
            logger.error(f"组件初始化失败: {e}")
            raise ConfigurationError(f"RAG系统组件初始化失败: {e}") from e
    
    def process_document(self, pdf_path: str) -> Dict[str, Any]:
        """
        处理文档的完整流程：解析 -> 分片 -> 向量化 -> 存储
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            处理结果统计信息
            
        Raises:
            DocumentProcessingError: 文档处理失败时
        """
        start_time = time.time()
        
        try:
            # 1. 文件验证
            if not Path(pdf_path).exists():
                raise RAGFileNotFoundError(f"PDF文件不存在: {pdf_path}")
            
            logger.info(f"开始处理PDF文档: {pdf_path}")
            
            # 2. PDF解析
            logger.info("正在解析PDF文档...")
            parsed_doc = self.parser.parse(pdf_path)
            logger.info(f"PDF解析完成，共 {parsed_doc.metadata.get('page_count', 0)} 页")
            
            # 3. 文档分片
            logger.info("正在进行文档分片...")
            chunks = self.chunker.chunk(parsed_doc)
            logger.info(f"文档分片完成，共生成 {len(chunks)} 个分片")
            
            if not chunks:
                raise DocumentProcessingError("文档分片后没有生成任何内容块")
            
            # 4. 向量化
            logger.info(f"正在向量化 {len(chunks)} 个分片...")
            texts = [chunk.content for chunk in chunks]
            
            # 批量向量化
            embeddings = self.embedder.embed(texts)
            
            # 为每个chunk添加embedding
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding = embedding
            
            logger.info("向量化完成")
            
            # 5. 存储到向量数据库
            logger.info("正在存储到向量数据库...")
            self.retriever.add_documents(chunks)
            logger.info("存储完成")
            
            # 6. 更新系统状态
            self.is_document_processed = True
            self.document_count += 1
            self.chunk_count += len(chunks)
            self.last_processed_document = pdf_path
            
            # 7. 计算处理时间
            processing_time = time.time() - start_time
            
            # 8. 返回统计信息
            result = {
                "success": True,
                "document_path": pdf_path,
                "chunks_created": len(chunks),
                "total_characters": sum(len(chunk.content) for chunk in chunks),
                "processing_time": f"{processing_time:.2f}秒",
                "pages_processed": parsed_doc.metadata.get('page_count', 0)
            }
            
            logger.info(f"文档处理完成: {result}")
            return result
            
        except Exception as e:
            logger.error(f"文档处理失败: {e}")
            if isinstance(e, (RAGFileNotFoundError, DocumentProcessingError)):
                raise
            else:
                raise DocumentProcessingError(f"处理文档 {pdf_path} 时发生错误: {e}") from e
    
    def query(self, question: str, top_k: Optional[int] = None) -> List[SearchResult]:
        """
        查询相关文档片段
        
        Args:
            question: 查询问题
            top_k: 返回结果数量，None时使用配置中的默认值
            
        Returns:
            按相似度降序排列的检索结果
            
        Raises:
            QueryError: 查询失败时
        """
        try:
            # 1. 状态验证
            if not self.is_document_processed:
                raise QueryError("请先处理文档后再进行查询。使用 process_document() 方法处理PDF文档。")
            
            if not question or not question.strip():
                raise QueryError("查询问题不能为空")
            
            # 2. 参数处理
            if top_k is None:
                top_k = self.config['query']['default_top_k']
            
            if top_k <= 0:
                raise QueryError("返回结果数量必须大于0")
            
            logger.info(f"开始查询: {question[:50]}{'...' if len(question) > 50 else ''}")
            
            # 3. 问题向量化
            logger.debug("正在向量化查询问题...")
            query_embedding = self.embedder.embed([question])[0]
            
            # 4. 向量检索
            logger.debug(f"正在检索相关文档片段 (top_k={top_k})...")
            results = self.retriever.search(query_embedding, top_k)
            
            # 5. 结果后处理
            filtered_results = self._post_process_results(results, question)
            
            logger.info(f"查询完成，返回 {len(filtered_results)} 个结果")
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"查询失败: {e}")
            if isinstance(e, QueryError):
                raise
            else:
                raise QueryError(f"查询过程中发生错误: {e}") from e
    
    def _post_process_results(self, results: List[SearchResult], question: str) -> List[SearchResult]:
        """
        结果后处理：相似度过滤、内容截断等
        
        Args:
            results: 原始检索结果
            question: 查询问题（用于日志）
            
        Returns:
            处理后的结果
        """
        if not results:
            return results
        
        # 1. 相似度阈值过滤
        min_score = self.config['query']['min_score_threshold']
        filtered_results = [r for r in results if r.score >= min_score]
        
        if len(filtered_results) < len(results):
            logger.debug(f"相似度过滤: {len(results)} -> {len(filtered_results)} 个结果")
        
        # 2. 内容长度限制
        max_length = self.config['query']['max_content_length']
        for result in filtered_results:
            if len(result.content) > max_length:
                result.content = result.content[:max_length] + "..."
        
        return filtered_results
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        获取系统状态信息
        
        Returns:
            系统信息字典
        """
        try:
            # 获取向量数据库信息
            vector_store_info = self.retriever.get_collection_info()
            
            # 获取向量模型信息
            embedder_info = self.embedder.get_model_info()
            
            return {
                "is_ready": self.is_document_processed,
                "document_count": self.document_count,
                "chunk_count": self.chunk_count,
                "last_processed_document": self.last_processed_document,
                "vector_store_info": vector_store_info,
                "embedder_info": embedder_info,
                "vector_dimension": embedder_info.get('embedding_dimension', 1024),
                "chunker_info": self.chunker.get_info(),
                "config": self.config
            }
        except Exception as e:
            logger.error(f"获取系统信息失败: {e}")
            return {
                "error": f"获取系统信息失败: {e}",
                "is_ready": self.is_document_processed,
                "document_count": self.document_count,
                "chunk_count": self.chunk_count
            }
    
    def clear_database(self) -> None:
        """
        清空向量数据库
        
        Raises:
            RAGSystemError: 清空失败时
        """
        try:
            logger.info("正在清空向量数据库...")
            self.retriever.clear_collection()
            
            # 重置状态
            self.is_document_processed = False
            self.document_count = 0
            self.chunk_count = 0
            self.last_processed_document = None
            
            logger.info("向量数据库已清空")
            
        except Exception as e:
            logger.error(f"清空数据库失败: {e}")
            raise RAGSystemError(f"清空向量数据库失败: {e}") from e
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        更新系统配置（部分配置需要重新初始化组件）
        
        Args:
            new_config: 新的配置字典
            
        Raises:
            ConfigurationError: 配置更新失败时
        """
        try:
            # 验证新配置
            from .config import validate_config, merge_configs
            
            merged_config = merge_configs(self.config, new_config)
            validate_config(merged_config)
            
            # 检查是否需要重新初始化组件
            need_reinit = False
            
            # 检查关键配置是否变化
            for component in ['embedder', 'retriever']:
                if component in new_config and new_config[component] != self.config.get(component):
                    need_reinit = True
                    break
            
            # 更新配置
            self.config = merged_config
            
            # 如果需要，重新初始化组件
            if need_reinit:
                logger.warning("检测到关键配置变化，正在重新初始化组件...")
                self._init_components()
                
                # 如果数据库配置变化，需要清空状态
                if 'retriever' in new_config:
                    self.is_document_processed = False
                    self.document_count = 0
                    self.chunk_count = 0
                    self.last_processed_document = None
                    logger.warning("检索器配置已变化，请重新处理文档")
            
            logger.info("配置更新完成")
            
        except Exception as e:
            logger.error(f"配置更新失败: {e}")
            raise ConfigurationError(f"更新系统配置失败: {e}") from e


# 便捷函数
def create_rag_system(config_path: Optional[str] = None) -> RAGSystem:
    """
    便捷函数：创建RAG系统实例
    
    Args:
        config_path: 配置文件路径，None时使用默认配置
        
    Returns:
        RAGSystem实例
    """
    if config_path:
        config = load_config(config_path)
    else:
        config = None
    
    return RAGSystem(config)
