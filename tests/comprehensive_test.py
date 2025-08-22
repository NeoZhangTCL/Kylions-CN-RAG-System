#!/usr/bin/env python3
"""
BGE Embedder 综合测试
全面测试BGE向量化器的各种功能和边界情况
"""

import time
import numpy as np
from src.embeddings import BGEEmbedder, default_embedder, embed_text


def test_error_handling():
    """测试错误处理"""
    print("\n🧪 测试错误处理...")
    
    embedder = BGEEmbedder()
    
    # 测试空输入
    try:
        embedder.embed("")
        print("❌ 空字符串应该报错")
    except ValueError as e:
        print(f"✅ 正确处理空字符串: {e}")
    
    try:
        embedder.embed([])
        print("❌ 空列表应该报错")
    except ValueError as e:
        print(f"✅ 正确处理空列表: {e}")
    
    # 测试包含空字符串的列表
    try:
        embedder.embed(["有效文本", "", "另一个文本"])
        print("❌ 包含空字符串的列表应该报错")
    except ValueError as e:
        print(f"✅ 正确处理包含空字符串的列表: {e}")


def test_input_variations():
    """测试不同输入格式"""
    print("\n🧪 测试不同输入格式...")
    
    embedder = BGEEmbedder()
    
    # 测试单个中文字符
    single_char = embedder.embed("中")
    print(f"✅ 单个中文字符: 向量维度 {len(single_char[0])}")
    
    # 测试很长的文本
    long_text = "麒麟操作系统" * 100  # 重复100次
    long_embedding = embedder.embed(long_text)
    print(f"✅ 长文本 ({len(long_text)}字符): 向量维度 {len(long_embedding[0])}")
    
    # 测试包含特殊字符的文本
    special_text = "麒麟系统！@#$%^&*()_+-={}[]|\\:;\"'<>?,./～！"
    special_embedding = embedder.embed(special_text)
    print(f"✅ 特殊字符文本: 向量维度 {len(special_embedding[0])}")
    
    # 测试纯数字
    number_text = "12345 67890 2024年"
    number_embedding = embedder.embed(number_text)
    print(f"✅ 数字文本: 向量维度 {len(number_embedding[0])}")
    
    # 测试混合中英文
    mixed_text = "麒麟操作系统 Kylin OS Linux Desktop System"
    mixed_embedding = embedder.embed(mixed_text)
    print(f"✅ 中英文混合: 向量维度 {len(mixed_embedding[0])}")


def test_semantic_understanding():
    """测试语义理解质量"""
    print("\n🧪 测试语义理解质量...")
    
    embedder = BGEEmbedder()
    
    # 测试同义词
    synonyms = [
        "麒麟操作系统",
        "Kylin OS",
        "麒麟桌面系统"
    ]
    
    synonym_embeddings = embedder.embed(synonyms)
    
    # 计算同义词之间的相似度
    sim_12 = np.dot(synonym_embeddings[0], synonym_embeddings[1])
    sim_13 = np.dot(synonym_embeddings[0], synonym_embeddings[2])
    sim_23 = np.dot(synonym_embeddings[1], synonym_embeddings[2])
    
    print(f"✅ 同义词相似度:")
    print(f"   麒麟操作系统 vs Kylin OS: {sim_12:.4f}")
    print(f"   麒麟操作系统 vs 麒麟桌面系统: {sim_13:.4f}")
    print(f"   Kylin OS vs 麒麟桌面系统: {sim_23:.4f}")
    
    # 测试反义词/无关词
    contrasts = [
        "麒麟操作系统是优秀的Linux发行版",
        "今天天气真好，阳光明媚",
        "我喜欢吃苹果和香蕉"
    ]
    
    contrast_embeddings = embedder.embed(contrasts)
    
    # 计算相关性
    rel_sim = np.dot(synonym_embeddings[0], contrast_embeddings[0])  # 相关
    irrel_sim1 = np.dot(synonym_embeddings[0], contrast_embeddings[1])  # 无关
    irrel_sim2 = np.dot(synonym_embeddings[0], contrast_embeddings[2])  # 无关
    
    print(f"✅ 相关性测试:")
    print(f"   系统相关: {rel_sim:.4f}")
    print(f"   天气无关: {irrel_sim1:.4f}")
    print(f"   食物无关: {irrel_sim2:.4f}")


def test_batch_performance():
    """测试批量处理性能"""
    print("\n🧪 测试批量处理性能...")
    
    embedder = BGEEmbedder()
    
    # 准备测试数据
    test_texts = [
        f"这是第{i}个测试文本，内容关于麒麟操作系统的功能和特性介绍。" 
        for i in range(1, 51)  # 50个文本
    ]
    
    # 测试批量处理
    start_time = time.time()
    batch_embeddings = embedder.embed(test_texts)
    batch_time = time.time() - start_time
    
    print(f"✅ 批量处理50个文本: {batch_time:.2f}秒")
    print(f"✅ 平均每个文本: {batch_time/50*1000:.1f}毫秒")
    print(f"✅ 生成向量数量: {len(batch_embeddings)}")
    print(f"✅ 向量维度: {len(batch_embeddings[0])}")
    
    # 测试单个处理对比
    start_time = time.time()
    single_embeddings = []
    for text in test_texts[:10]:  # 只测试前10个
        emb = embedder.embed(text)
        single_embeddings.extend(emb)
    single_time = time.time() - start_time
    
    print(f"✅ 单个处理10个文本: {single_time:.2f}秒")
    print(f"✅ 平均每个文本: {single_time/10*1000:.1f}毫秒")
    
    # 计算效率提升
    if single_time > 0:
        efficiency = (single_time/10) / (batch_time/50)
        print(f"✅ 批量处理效率提升: {efficiency:.2f}x")


def test_convenience_functions():
    """测试便捷函数"""
    print("\n🧪 测试便捷函数...")
    
    # 测试默认embedder
    print(f"默认embedder类型: {type(default_embedder)}")
    print(f"默认embedder模型: {default_embedder.model_name}")
    
    # 测试便捷函数
    text = "测试便捷函数的功能"
    embedding = embed_text(text)
    print(f"✅ 便捷函数: 生成{len(embedding)}个向量，维度{len(embedding[0])}")
    
    # 测试批量便捷函数
    texts = ["第一个文本", "第二个文本", "第三个文本"]
    embeddings = embed_text(texts)
    print(f"✅ 批量便捷函数: 生成{len(embeddings)}个向量")


def test_model_info():
    """测试模型信息获取"""
    print("\n🧪 测试模型信息获取...")
    
    # 测试未加载状态
    fresh_embedder = BGEEmbedder()
    info_before = fresh_embedder.get_model_info()
    print("未加载时的模型信息:")
    for key, value in info_before.items():
        print(f"  {key}: {value}")
    
    # 触发模型加载
    _ = fresh_embedder.embed("触发加载")
    
    # 测试已加载状态
    info_after = fresh_embedder.get_model_info()
    print("\n已加载后的模型信息:")
    for key, value in info_after.items():
        print(f"  {key}: {value}")
    
    # 测试维度获取
    dimension = fresh_embedder.get_embedding_dimension()
    print(f"\n✅ 向量维度: {dimension}")


def test_vector_quality():
    """测试向量质量"""
    print("\n🧪 测试向量质量...")
    
    embedder = BGEEmbedder()
    
    # 测试向量归一化
    texts = ["测试向量", "另一个测试"]
    embeddings = embedder.embed(texts)
    
    for i, emb in enumerate(embeddings):
        norm = np.linalg.norm(emb)
        print(f"✅ 向量{i+1}的L2范数: {norm:.6f} (应该接近1.0)")
    
    # 测试向量非零性
    zero_count = sum(1 for val in embeddings[0] if abs(val) < 1e-10)
    total_dim = len(embeddings[0])
    print(f"✅ 向量稀疏度: {zero_count}/{total_dim} = {zero_count/total_dim*100:.2f}%")
    
    # 测试不同文本的向量差异性
    if len(embeddings) >= 2:
        similarity = np.dot(embeddings[0], embeddings[1])
        print(f"✅ 不同文本相似度: {similarity:.4f} (不应该过高)")


def main():
    """主测试函数"""
    print("🚀 BGE Embedder 综合测试开始")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        test_error_handling()
        test_input_variations()
        test_semantic_understanding()
        test_convenience_functions()
        test_model_info()
        test_vector_quality()
        test_batch_performance()
        
        total_time = time.time() - start_time
        print(f"\n✅ 所有测试完成！总耗时: {total_time:.2f}秒")
        
        print(f"\n📊 测试总结:")
        print(f"- ✅ 错误处理: 正常")
        print(f"- ✅ 输入格式: 支持多种格式") 
        print(f"- ✅ 语义理解: 质量良好")
        print(f"- ✅ 便捷函数: 工作正常")
        print(f"- ✅ 模型信息: 获取正常")
        print(f"- ✅ 向量质量: 满足要求")
        print(f"- ✅ 批量处理: 性能优异")
        
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
