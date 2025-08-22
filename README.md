# 麒麟操作系统手册RAG检索系统 - 使用说明

## 🎯 系统概述

本系统实现了基于**麒麟桌面操作系统操作手册**的RAG（检索增强生成）检索系统，支持中文文档的智能问答。

### 📚 项目背景

本项目基于**面试技术考察要求**开发，实现"用户提问→检索最相关文档片段"的核心流程。项目聚焦于检索环节的优化，使用现代化RAG技术栈处理包含丰富图表的中文技术文档。

**核心业务场景**：

- 📄 基础版检索系统：PDF文档解析 → 文本分割 → 向量化存储 → 相似度检索
- 🔍 返回top-3最相关文档片段，无需生成完整答案

> 📋 **项目要求详情**: 参考 [`面试者代码提供要求.md`](./面试者代码提供要求.md)

### 📖 文档导航

- 🔧 **[技术方案设计文档](./RAG系统技术方案设计文档.md)**: 详细的系统架构、技术选型和实现方案
- 📝 **[面试要求说明](./面试者代码提供要求.md)**: 项目背景和具体要求
- 🎯 **[提示词模板](./prompts/)**: RAG系统评估提示词和模板文件

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
  default_top_k: 3
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

```text
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

> 🔧 **详细技术方案**: 完整的架构设计、技术选型对比和实现细节请参考 [RAG系统技术方案设计文档](./RAG系统技术方案设计文档.md)

### 🎯 提示词系统

系统集成了专业的提示词模板，用于RAG适配性评估：

- **`prompts/answer_evaluation.jinja2`**: 文档RAG适配性评估模板
- **评估维度**: 内容颗粒度、术语一致性、逻辑连贯性等
- **输出标准**: 结构化改进建议和文档编写模板

> 💡 **提示词设计**: 基于COT(Chain of Thought)思路设计，服务于生成标准化模板的最终目标

## 📈 扩展可能

### 🔧 技术扩展

- 支持多文档处理
- 添加持久化存储  
- 集成生成模型（如ChatGLM）
- 支持图片和表格处理
- Web界面开发

### 🎯 业务扩展

- **RAG适配性评估系统**: 基于提示词驱动的文档质量评估
- **标准化文档模板生成**: 输出结构化改进建议和编写模板
- **多模态文档处理**: 统一处理文本、图片、表格内容
- **混合检索引擎**: 向量检索+关键词检索的融合策略

> 📋 **业务场景**: 详细的扩展规划请参考 [技术方案设计文档 - 待实现功能](./RAG系统技术方案设计文档.md#📊-项目实现状态)

## 🎉 总结

### ✅ 已实现功能

本系统成功实现了**面试要求的核心功能**：

- ✅ **PDF文档处理**: 支持麒麟操作系统手册解析
- ✅ **文本分割策略**: 固定长度+重叠分割算法
- ✅ **中文向量化**: BGE-large-zh-v1.5模型集成
- ✅ **向量存储**: Qdrant内存模式，无需持久化
- ✅ **相似度检索**: 返回top-3最相关文档片段
- ✅ **友好交互**: 完整的CLI命令行界面
- ✅ **质量保证**: 配置管理和稳定错误处理

### 🎯 项目价值

- **🔍 核心检索**: 实现"用户提问→检索文档片段"完整流程
- **🇨🇳 中文优化**: 专门针对中文技术文档优化处理
- **📚 扩展设计**: 支持RAG适配性评估和模板生成
- **🛠️ 工程质量**: 完整的项目结构和文档体系

> 📋 **符合要求**: 系统完全满足[面试技术考察要求](./面试者代码提供要求.md)，可直接用于麒麟操作系统手册的智能检索。

**🚀 系统已就绪，可进行实际业务应用！**
