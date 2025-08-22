# 麒麟操作系统手册RAG检索系统

## 🌟 项目简介

基于麒麟桌面操作系统操作手册构建的中文智能文档检索系统，实现"用户提问→检索最相关文档片段"的核心功能。采用现代化RAG（检索增强生成）技术栈，专门针对中文文档优化。

## ✨ 主要特性

- 📄 **PDF文档解析**：智能解析麒麟操作系统手册PDF文件
- 🧩 **文档分片**：智能分片策略，保持语义完整性
- 🔤 **中文优化**：使用BGE-large-zh-v1.5中文向量模型
- 🔍 **语义检索**：基于向量相似度的智能检索
- 💬 **多种交互**：支持命令行、交互式查询
- 🎯 **一键问答**：ask命令一步完成处理+查询

## 🚀 快速开始

### 环境要求

- Python 3.8+
- 虚拟环境 (.venv)
- 约2GB可用内存（BGE模型）

### 安装使用

```bash
# 1. 激活虚拟环境
source .venv/bin/activate

# 2. 一键智能问答（推荐）
python main.py ask data/raw/kylions_handle_book.pdf "如何安装软件?"

# 3. 查看帮助
python main.py --help
```

### 基本命令

```bash
# 🌟 智能问答（一步完成）
python main.py ask data/raw/kylions_handle_book.pdf "麒麟系统有什么特点?"

# 📄 处理PDF文档
python main.py process data/raw/kylions_handle_book.pdf

# 🔍 查询文档
python main.py query "系统升级怎么操作?"

# 💬 交互模式
python main.py interactive

# ℹ️ 查看系统信息
python main.py info
```

## 🏗️ 技术架构

```
PDF文档 → 文本解析 → 智能分片 → BGE向量化 → Qdrant存储
                                              ↓
用户查询 → 问题向量化 → 相似度检索 → 结果排序 → 返回答案
```

### 核心组件

- **SimplePDFParser**: PDF文档解析器
- **SimpleOverlapChunker**: 文档分片器  
- **BGEEmbedder**: BGE-large-zh-v1.5中文向量化
- **QdrantRetriever**: 向量存储和检索
- **RAGSystem**: 系统主控制器

### 技术栈

- **文档解析**: PyMuPDF
- **向量模型**: BGE-large-zh-v1.5 (1024维)
- **向量数据库**: Qdrant (内存模式)
- **分片策略**: 重叠分片 (500字符/50重叠)

## 📊 性能指标

- **处理速度**: ~30秒处理156页PDF
- **分片效果**: 123个分片，平均500字符
- **查询延迟**: <1秒返回结果
- **相似度**: 0.5-0.7高质量匹配
- **内存使用**: ~1.3GB (BGE模型)

## 💡 使用示例

### 安装相关查询
```bash
python main.py ask data/raw/kylions_handle_book.pdf "如何安装软件?"
```
**结果**: 返回系统安装步骤、硬件兼容性检查等相关内容

### 系统特性查询
```bash
python main.py ask data/raw/kylions_handle_book.pdf "麒麟系统有什么特点?"
```
**结果**: 返回麒麟软件公司介绍、产品特性、技术优势等

## 📁 项目结构

```
rag_kylinos/
├── src/
│   ├── parsers/          # PDF解析器
│   ├── chunkers/         # 文档分片器
│   ├── embeddings/       # 向量化器
│   ├── retrievers/       # 检索器
│   ├── model/           # 数据模型
│   ├── config.py        # 配置管理
│   ├── exceptions.py    # 异常定义
│   └── rag_system.py    # 主控制器
├── data/
│   ├── raw/             # 原始PDF文档
│   └── processed/       # 处理结果
├── tests/               # 测试文件
├── main.py             # CLI入口
├── 使用说明.md          # 详细使用说明
└── requirements.txt     # 依赖包
```

## 🔧 配置选项

系统支持灵活配置，主要参数：

```yaml
chunker:
  chunk_size: 500      # 分片大小
  overlap_size: 50     # 重叠大小

query:
  default_top_k: 3     # 默认返回结果数
  min_score_threshold: 0.1  # 最小相似度阈值

embedder:
  model_name: "BAAI/bge-large-zh-v1.5"
  device: null         # 自动选择设备
```

## ❓ 常见问题

### Q: 查询提示"请先处理文档"？
**A**: 使用`ask`命令一步完成处理+查询：
```bash
python main.py ask data/raw/kylions_handle_book.pdf "你的问题"
```

### Q: 找不到相关结果？
**A**: 
- 尝试不同关键词
- 降低相似度阈值: `--min-score 0.2`
- 增加返回数量: `--top-k 10`

### Q: 处理速度慢？
**A**: 首次加载BGE模型需要时间，后续查询很快

## 🎯 设计亮点

1. **中文优化**: 专用BGE中文向量模型
2. **一键问答**: ask命令解决内存模式数据丢失问题
3. **用户友好**: 完整CLI工具和交互模式  
4. **模块化设计**: 组件独立，易于扩展
5. **错误处理**: 完善的异常处理和用户提示

## 🛣️ 未来规划

- [ ] 支持多文档处理
- [ ] 添加持久化存储
- [ ] 集成生成模型(ChatGLM等)
- [ ] 支持图片和表格处理
- [ ] Web界面开发
- [ ] 性能优化和并发支持

## 📜 开源协议

本项目采用 [AGPL-3.0](LICENSE) 开源协议。

## 🤝 贡献指南

欢迎提交Issue和Pull Request！请遵循以下规范：

1. 代码提交使用中文注释
2. 提交信息使用中文描述
3. 确保代码通过测试
4. 遵循项目代码风格

---

⭐ **如果觉得项目有用，请给个Star支持！**
