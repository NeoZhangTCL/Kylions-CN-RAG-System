# 麒麟操作系统手册RAG检索系统 - 使用说明

## 🎯 系统概述

本系统实现了基于麒麟操作系统手册的RAG（检索增强生成）检索系统，支持中文文档的智能问答。

### ✨ 主要功能

- 📄 **PDF文档解析**：自动解析麒麟操作系统手册PDF文件
- 🧩 **智能分片**：将长文档分割为合适的文本块
- 🔤 **中文向量化**：使用BGE-large-zh-v1.5模型进行向量化
- 🔍 **语义检索**：基于问题内容检索相关文档片段
- 💬 **交互查询**：支持命令行和交互式查询模式

## 🚀 快速开始

### 1. 环境准备

```bash
# 激活虚拟环境
source .venv/bin/activate

# 确保PDF文档在正确位置
ls data/raw/kylions_handle_book.pdf
```

### 2. 智能问答（推荐）

```bash
# 🌟 一步完成：处理文档并查询（推荐方式）
python main.py ask data/raw/kylions_handle_book.pdf "如何安装软件?"

# 查询其他问题
python main.py ask data/raw/kylions_handle_book.pdf "麒麟系统有什么特点?"
```

### 3. 分步处理（可选）

```bash
# 先处理PDF文档
python main.py process data/raw/kylions_handle_book.pdf

# 然后查询（注意：仅在同一会话中有效，重启程序需重新处理）
python main.py query "如何安装软件?"

# 或使用交互模式
python main.py interactive
```

## 📖 详细使用指南

### 命令行选项

```bash
# 查看帮助
python main.py --help

# 使用自定义配置文件
python main.py --config config.yaml process data/raw/kylions_handle_book.pdf

# 设置日志级别
python main.py --log-level DEBUG query "系统升级"
```

### 主要命令

#### 1. process - 处理PDF文档
```bash
# 基本用法
python main.py process data/raw/kylions_handle_book.pdf

# 自定义分片参数
python main.py process data/raw/kylions_handle_book.pdf --chunk-size 800 --overlap 100
```

#### 2. query - 查询文档
```bash
# 基本查询
python main.py query "如何安装软件?"

# 返回更多结果
python main.py query "系统升级" --top-k 5

# 设置相似度阈值
python main.py query "麒麟系统特点" --min-score 0.5
```

#### 3. ask - 智能问答（🌟推荐）
```bash
# 基本用法：一步完成处理和查询
python main.py ask data/raw/kylinos_handle_book.pdf "如何安装软件?"

# 返回更多结果
python main.py ask data/raw/kylinos_handle_book.pdf "系统特点" --top-k 5

# 自定义分片参数
python main.py ask data/raw/kylinos_handle_book.pdf "安装步骤" --chunk-size 800 --overlap 100
```

#### 4. interactive - 交互模式
```bash
# 进入交互模式
python main.py interactive

# 自动加载默认文档
python main.py interactive --auto-load
```

#### 5. info - 查看系统信息
```bash
# 基本信息
python main.py info

# 详细信息
python main.py info --detailed
```

#### 6. clear - 清空数据库
```bash
# 清空向量数据库
python main.py clear --confirm
```

### 交互模式使用

在交互模式下：
- 直接输入问题进行查询
- 输入 `info` 查看系统状态
- 输入 `help` 获取帮助
- 输入 `quit`、`exit` 或 `q` 退出
- 按 `Ctrl+C` 随时退出

## 🔧 配置说明

### 默认配置

系统使用以下默认配置：

```python
{
    "chunker": {
        "chunk_size": 500,    # 分片大小（字符数）
        "overlap_size": 50    # 重叠大小（字符数）
    },
    "embedder": {
        "model_name": "BAAI/bge-large-zh-v1.5",  # 向量化模型
        "device": None        # 自动选择设备
    },
    "query": {
        "default_top_k": 3,          # 默认返回结果数
        "min_score_threshold": 0.1   # 最小相似度阈值
    }
}
```

### 自定义配置

创建 `config.yaml` 文件：

```yaml
chunker:
  chunk_size: 800
  overlap_size: 100

query:
  default_top_k: 5
  min_score_threshold: 0.3
```

使用自定义配置：

```bash
python main.py --config config.yaml process data/raw/kylions_handle_book.pdf
```

## 📊 处理效果

### 处理统计
- **文档页数**：156页
- **生成分片**：约123个
- **总字符数**：约61,000字符
- **处理时间**：约30秒
- **向量维度**：1024维

### 查询效果
- **查询延迟**：通常<1秒
- **相似度范围**：0.0-1.0
- **建议阈值**：0.3以上为相关内容

## ❓ 常见问题

### Q1: 查询提示"请先处理文档"
**A**: 系统使用内存模式的向量数据库，每次重启程序需要重新处理PDF文档。  
**💡 推荐解决方案**: 使用 `ask` 命令一步完成处理和查询：
```bash
python main.py ask data/raw/kylinos_handle_book.pdf "你的问题"
```

### Q2: 找不到相关结果
**A**: 尝试：
- 使用不同的关键词
- 降低相似度阈值 `--min-score 0.2`
- 增加返回结果数 `--top-k 10`

### Q3: 处理速度慢
**A**: 首次加载BGE模型需要时间，后续查询会很快。

### Q4: 内存使用高
**A**: BGE模型约占用1.3GB内存，这是正常的。

## 🛠️ 系统架构

```
PDF文档 → 文本解析 → 智能分片 → 向量化 → Qdrant存储
                                              ↓
用户查询 → 问题向量化 → 相似度检索 → 结果排序 → 返回结果
```

### 核心组件

1. **SimplePDFParser**: PDF文档解析
2. **SimpleOverlapChunker**: 文档分片
3. **BGEEmbedder**: 中文向量化
4. **QdrantRetriever**: 向量存储和检索
5. **RAGSystem**: 系统主控制器

## 📈 扩展可能

- 支持多文档处理
- 添加持久化存储
- 集成生成模型（如ChatGLM）
- 支持图片和表格处理
- Web界面开发

## 🎉 总结

本系统成功实现了：
- ✅ 完整的RAG处理流程
- ✅ 中文文档的智能检索
- ✅ 友好的命令行界面
- ✅ 灵活的配置管理
- ✅ 稳定的错误处理

系统已经可以正常运行，能够对麒麟操作系统手册进行智能问答。
