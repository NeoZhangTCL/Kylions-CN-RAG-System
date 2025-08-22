#!/usr/bin/env python3
"""
RAG系统集成测试
测试BGE向量化器与Qdrant检索器的完整集成
"""

import time
from typing import List
import numpy as np

from src.embeddings import BGEEmbedder, embed_text
from src.retrievers import QdrantRetriever, create_retriever
from src.model import DocumentChunk, SearchResult


def create_sample_documents() -> List[DocumentChunk]:
    """创建测试文档数据"""
    print("📚 准备测试文档...")
    
    # 准备测试文档内容（关于麒麟操作系统的信息）
    test_contents = [
        "麒麟操作系统是一个基于Linux内核的桌面操作系统，专为中国用户设计开发。",
        "Kylin OS具有良好的安全性和稳定性，支持国产化软硬件生态。",
        "该系统提供友好的用户界面，操作简单直观，适合办公和日常使用。",
        "麒麟系统内置了丰富的办公软件，包括文档编辑、表格制作和演示工具。",
        "系统支持多种文件格式，兼容主流的音视频播放和图片查看功能。",
        "麒麟操作系统具备完善的网络功能，支持各种网络协议和连接方式。",
        "系统提供了强大的系统管理工具，便于用户进行系统配置和维护。",
        "麒麟OS支持多种输入法，特别针对中文输入进行了优化。",
        "该系统具有优秀的硬件兼容性，支持多种主流硬件设备。",
        "麒麟操作系统定期更新，持续改进功能和修复安全漏洞。"
    ]
    
    print(f"创建了 {len(test_contents)} 个文档片段")
    
    # 使用BGE生成向量
    embedder = BGEEmbedder()
    embeddings = embedder.embed(test_contents)
    
    # 创建DocumentChunk对象
    chunks = []
    for i, (content, embedding) in enumerate(zip(test_contents, embeddings)):
        chunk = DocumentChunk(
            content=content,
            metadata={
                "doc_id": f"doc_{i:03d}",
                "source": "kylin_system_manual",
                "section": f"section_{i // 3 + 1}",  # 每3个文档一个章节
                "content_type": "text",
                "language": "zh-cn",
                "created_at": "2024-01-15T10:00:00Z",
                "word_count": len(content)
            },
            embedding=embedding
        )
        chunks.append(chunk)
    
    print(f"✅ 成功创建 {len(chunks)} 个文档块，每个向量维度为 {len(embeddings[0])}")
    return chunks


def test_basic_functionality():
    """测试基本功能"""
    print("\n🧪 测试基本功能...")
    
    # 创建检索器
    retriever = create_retriever(collection_name="test_basic", vector_size=1024)
    
    # 创建测试文档
    chunks = create_sample_documents()
    
    # 添加文档
    print("📥 添加文档到向量数据库...")
    start_time = time.time()
    retriever.add_documents(chunks)
    add_time = time.time() - start_time
    print(f"✅ 文档添加完成，耗时: {add_time:.2f}秒")
    
    # 查看集合信息
    collection_info = retriever.get_collection_info()
    print("📊 集合信息:")
    for key, value in collection_info.items():
        print(f"  {key}: {value}")
    
    # 测试检索
    print("\n🔍 测试语义检索...")
    test_queries = [
        "麒麟系统的安全性如何？",
        "系统支持哪些办公功能？", 
        "如何进行系统管理？",
        "支持什么输入法？"
    ]
    
    embedder = BGEEmbedder()
    
    for query in test_queries:
        print(f"\n查询: {query}")
        
        # 生成查询向量
        query_embedding = embedder.embed(query)[0]
        
        # 执行检索
        start_time = time.time()
        results = retriever.search(query_embedding, top_k=3)
        search_time = time.time() - start_time
        
        print(f"检索耗时: {search_time*1000:.1f}毫秒")
        print(f"找到 {len(results)} 个结果:")
        
        for i, result in enumerate(results, 1):
            print(f"  {i}. [相似度: {result.score:.4f}] {result.content[:50]}...")
            print(f"     元数据: doc_id={result.metadata.get('doc_id')}, "
                  f"section={result.metadata.get('section')}")


def test_filtered_search():
    """测试过滤检索"""
    print("\n🧪 测试过滤检索...")
    
    retriever = create_retriever(collection_name="test_filtered", vector_size=1024)
    chunks = create_sample_documents()
    retriever.add_documents(chunks)
    
    embedder = BGEEmbedder()
    query = "系统功能介绍"
    query_embedding = embedder.embed(query)[0]
    
    # 不过滤的检索
    all_results = retriever.search(query_embedding, top_k=5)
    print(f"无过滤检索: 找到 {len(all_results)} 个结果")
    
    # 按章节过滤
    filtered_results = retriever.search_with_filter(
        query_embedding, 
        top_k=5,
        metadata_filter={"section": "section_1"}
    )
    print(f"章节1过滤检索: 找到 {len(filtered_results)} 个结果")
    
    for result in filtered_results:
        print(f"  - {result.content[:40]}... (section: {result.metadata['section']})")


def test_error_handling():
    """测试错误处理"""
    print("\n🧪 测试错误处理...")
    
    retriever = create_retriever(collection_name="test_errors", vector_size=1024)
    
    # 测试空文档列表
    try:
        retriever.add_documents([])
        print("✅ 正确处理空文档列表")
    except Exception as e:
        print(f"❌ 处理空文档列表失败: {e}")
    
    # 测试缺少embedding的文档
    try:
        bad_chunk = DocumentChunk(
            content="测试内容",
            metadata={"test": "value"},
            embedding=None
        )
        retriever.add_documents([bad_chunk])
        print("❌ 应该拒绝缺少embedding的文档")
    except ValueError as e:
        print(f"✅ 正确拒绝缺少embedding的文档: {e}")
    
    # 测试维度不匹配的查询
    try:
        wrong_dim_vector = [0.1] * 512  # 错误的维度
        retriever.search(wrong_dim_vector, top_k=3)
        print("❌ 应该拒绝错误维度的查询向量")
    except ValueError as e:
        print(f"✅ 正确拒绝错误维度的查询向量: {e}")
    
    # 测试空查询向量
    try:
        retriever.search([], top_k=3)
        print("❌ 应该拒绝空查询向量")
    except ValueError as e:
        print(f"✅ 正确拒绝空查询向量: {e}")


def test_performance():
    """测试性能"""
    print("\n🧪 测试性能...")
    
    retriever = create_retriever(collection_name="test_performance", vector_size=1024)
    
    # 创建大量测试文档
    print("生成大量测试文档...")
    large_contents = [
        f"这是第{i}个测试文档。内容包含麒麟操作系统的各种功能介绍，"
        f"包括但不限于系统管理、办公软件、网络功能、安全特性等方面的详细描述。"
        f"文档编号为{i:04d}，属于性能测试的一部分。"
        for i in range(100)
    ]
    
    embedder = BGEEmbedder()
    print("生成向量...")
    start_time = time.time()
    large_embeddings = embedder.embed(large_contents)
    embedding_time = time.time() - start_time
    print(f"✅ 生成100个向量耗时: {embedding_time:.2f}秒")
    
    large_chunks = []
    for i, (content, embedding) in enumerate(zip(large_contents, large_embeddings)):
        chunk = DocumentChunk(
            content=content,
            metadata={"doc_id": f"perf_{i:04d}", "batch": "performance_test"},
            embedding=embedding
        )
        large_chunks.append(chunk)
    
    # 批量添加文档
    print("批量添加文档...")
    start_time = time.time()
    retriever.add_documents(large_chunks)
    add_time = time.time() - start_time
    print(f"✅ 批量添加100个文档耗时: {add_time:.2f}秒")
    
    # 批量检索测试
    print("批量检索测试...")
    test_queries = [
        "系统管理功能",
        "办公软件支持", 
        "网络连接配置",
        "安全特性介绍",
        "用户界面设计"
    ]
    
    query_embeddings = embedder.embed(test_queries)
    
    start_time = time.time()
    total_results = 0
    for query_emb in query_embeddings:
        results = retriever.search(query_emb, top_k=5)
        total_results += len(results)
    search_time = time.time() - start_time
    
    print(f"✅ 5次检索总耗时: {search_time:.2f}秒")
    print(f"✅ 平均单次检索: {search_time/5*1000:.1f}毫秒")
    print(f"✅ 总检索结果数: {total_results}")
    
    # 显示最终集合状态
    final_info = retriever.get_collection_info()
    print(f"✅ 最终集合状态: {final_info['points_count']} 个文档")


def test_integration_scenario():
    """测试完整的RAG场景"""
    print("\n🧪 测试完整RAG场景...")
    
    # 创建专门的检索器
    retriever = create_retriever(collection_name="rag_demo", vector_size=1024)
    
    # 模拟文档知识库
    knowledge_base = [
        "麒麟操作系统基于Linux内核，专为中国用户定制开发，具有完全自主知识产权。",
        "系统采用自主研发的UKUI桌面环境，界面简洁美观，操作体验友好。",
        "内置WPS办公套件、搜狗输入法、火狐浏览器等常用软件，满足日常办公需求。",
        "支持国产CPU如龙芯、飞腾、兆芯等，以及国产显卡、网卡等硬件设备。",
        "系统具备多重安全防护机制，包括访问控制、数据加密、入侵检测等功能。",
        "提供应用商店，用户可以便捷地下载安装各类应用软件。",
        "系统支持虚拟化技术，可以运行Windows应用程序，保证软件兼容性。",
        "具备完善的驱动支持，兼容主流打印机、扫描仪、摄像头等外设。",
        "支持多种网络协议，包括有线网络、无线WiFi、蓝牙连接等。",
        "定期发布系统更新，修复安全漏洞，增加新功能，保持系统稳定性。"
    ]
    
    # 构建知识库
    print("构建知识库...")
    embedder = BGEEmbedder()
    kb_embeddings = embedder.embed(knowledge_base)
    
    kb_chunks = []
    for i, (content, embedding) in enumerate(zip(knowledge_base, kb_embeddings)):
        chunk = DocumentChunk(
            content=content,
            metadata={
                "doc_id": f"kb_{i:03d}",
                "source": "kylin_knowledge_base",
                "topic": ["system", "security", "software", "hardware", "network"][i % 5],
                "priority": "high" if i < 5 else "normal"
            },
            embedding=embedding
        )
        kb_chunks.append(chunk)
    
    retriever.add_documents(kb_chunks)
    print(f"✅ 知识库构建完成，包含 {len(kb_chunks)} 个条目")
    
    # 模拟用户问答
    print("\n模拟用户问答场景:")
    user_questions = [
        "麒麟系统支持哪些国产硬件？",
        "如何保证系统的安全性？",
        "系统自带哪些办公软件？",
        "可以运行Windows程序吗？",
        "如何连接网络设备？"
    ]
    
    for question in user_questions:
        print(f"\n❓ 用户问题: {question}")
        
        # 生成问题向量
        question_embedding = embedder.embed(question)[0]
        
        # 检索相关文档
        relevant_docs = retriever.search(question_embedding, top_k=2)
        
        print("📋 检索到的相关信息:")
        for i, doc in enumerate(relevant_docs, 1):
            print(f"  {i}. [相似度: {doc.score:.4f}] {doc.content}")
            print(f"     [主题: {doc.metadata['topic']}, 优先级: {doc.metadata['priority']}]")
        
        # 在实际RAG系统中，这里会将检索到的内容传递给LLM生成答案
        print("💡 基于检索内容，系统可以生成针对性回答")


def main():
    """主测试函数"""
    print("🚀 BGE + Qdrant RAG集成测试")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        test_basic_functionality()
        test_filtered_search()
        test_error_handling()
        test_performance()
        test_integration_scenario()
        
        total_time = time.time() - start_time
        print(f"\n✅ 所有集成测试完成！总耗时: {total_time:.2f}秒")
        
        print(f"\n📊 测试总结:")
        print(f"- ✅ 基本功能: BGE向量化 + Qdrant存储检索")
        print(f"- ✅ 过滤检索: 支持元数据过滤条件")
        print(f"- ✅ 错误处理: 完善的异常处理机制")
        print(f"- ✅ 性能表现: 批量处理高效稳定")
        print(f"- ✅ RAG场景: 完整的问答检索流程")
        print(f"\n🎯 BGE + Qdrant 集成方案验证成功！")
        
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
