#!/usr/bin/env python3
"""
麒麟操作系统手册RAG检索系统
主程序CLI入口
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

# 设置Python路径以导入src模块
sys.path.insert(0, str(Path(__file__).parent))

from src.rag_system import RAGSystem, create_rag_system
from src.exceptions import RAGSystemError, DocumentProcessingError, QueryError
from src.config import load_config, save_config, get_default_config


def setup_logging(level: str = "INFO") -> None:
    """设置日志配置"""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def main():
    """主程序入口"""
    
    parser = argparse.ArgumentParser(
        description="麒麟操作系统手册RAG检索系统",
        epilog="使用示例:\n"
               "  python main.py process data/raw/kylinos_handle_book.pdf\n"
               "  python main.py query '如何安装软件?'\n"
               "  python main.py ask data/raw/kylinos_handle_book.pdf '如何安装软件?'\n"
               "  python main.py interactive",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # 全局参数
    parser.add_argument(
        '--config', '-c',
        help='配置文件路径 (YAML/JSON格式)'
    )
    parser.add_argument(
        '--log-level', 
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='日志级别'
    )
    
    # 添加子命令
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # process命令：处理PDF文档
    process_parser = subparsers.add_parser('process', help='处理PDF文档')
    process_parser.add_argument('pdf_path', help='PDF文件路径')
    process_parser.add_argument('--chunk-size', type=int, help='分片大小 (覆盖配置文件)')
    process_parser.add_argument('--overlap', type=int, help='重叠大小 (覆盖配置文件)')
    
    # query命令：查询文档
    query_parser = subparsers.add_parser('query', help='查询文档内容')
    query_parser.add_argument('question', help='查询问题')
    query_parser.add_argument('--top-k', type=int, default=3, help='返回结果数量')
    query_parser.add_argument('--min-score', type=float, help='最小相似度阈值')
    
    # interactive命令：交互模式
    interactive_parser = subparsers.add_parser('interactive', help='交互查询模式')
    interactive_parser.add_argument('--auto-load', action='store_true', help='自动加载默认文档')
    
    # info命令：系统信息
    info_parser = subparsers.add_parser('info', help='显示系统信息')
    info_parser.add_argument('--detailed', action='store_true', help='显示详细信息')
    
    # clear命令：清空数据库
    clear_parser = subparsers.add_parser('clear', help='清空向量数据库')
    clear_parser.add_argument('--confirm', action='store_true', help='跳过确认提示')
    
    # ask命令：一次性处理和查询
    ask_parser = subparsers.add_parser('ask', help='处理文档并查询（一步完成）')
    ask_parser.add_argument('pdf_path', help='PDF文件路径')
    ask_parser.add_argument('question', help='查询问题')
    ask_parser.add_argument('--top-k', type=int, default=3, help='返回结果数量')
    ask_parser.add_argument('--chunk-size', type=int, help='分片大小')
    ask_parser.add_argument('--overlap', type=int, help='重叠大小')
    
    # config命令：配置管理
    config_parser = subparsers.add_parser('config', help='配置管理')
    config_subparsers = config_parser.add_subparsers(dest='config_action')
    
    show_config_parser = config_subparsers.add_parser('show', help='显示当前配置')
    save_config_parser = config_subparsers.add_parser('save', help='保存当前配置到文件')
    save_config_parser.add_argument('output_path', help='配置文件保存路径')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # 设置日志
    setup_logging(args.log_level)
    
    try:
        # 初始化RAG系统
        print("🚀 正在初始化RAG系统...")
        
        # 加载配置
        config = None
        if args.config:
            try:
                config = load_config(args.config)
                print(f"📋 已加载配置文件: {args.config}")
            except Exception as e:
                print(f"❌ 加载配置文件失败: {e}")
                return
        
        # 应用命令行参数覆盖
        if config is None:
            config = get_default_config()
        
        # 覆盖分片配置
        if hasattr(args, 'chunk_size') and args.chunk_size:
            config['chunker']['chunk_size'] = args.chunk_size
        if hasattr(args, 'overlap') and args.overlap:
            config['chunker']['overlap_size'] = args.overlap
        if hasattr(args, 'min_score') and args.min_score:
            config['query']['min_score_threshold'] = args.min_score
        
        rag_system = RAGSystem(config)
        print("✅ RAG系统初始化完成\n")
        
        # 执行对应命令
        if args.command == 'process':
            handle_process_command(rag_system, args)
        elif args.command == 'query':
            handle_query_command(rag_system, args)
        elif args.command == 'ask':
            handle_ask_command(rag_system, args)
        elif args.command == 'interactive':
            handle_interactive_mode(rag_system, args)
        elif args.command == 'info':
            handle_info_command(rag_system, args)
        elif args.command == 'clear':
            handle_clear_command(rag_system, args)
        elif args.command == 'config':
            handle_config_command(rag_system, args)
            
    except KeyboardInterrupt:
        print("\n\n👋 操作已取消，再见!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 执行失败: {e}")
        if args.log_level == 'DEBUG':
            import traceback
            traceback.print_exc()
        sys.exit(1)


def handle_process_command(rag_system: RAGSystem, args):
    """处理process命令"""
    print(f"📄 开始处理PDF文档: {args.pdf_path}")
    print("⏳ 处理中，请稍候...\n")
    
    try:
        result = rag_system.process_document(args.pdf_path)
        
        print("🎉 文档处理完成!")
        print(f"  📁 文档路径: {result['document_path']}")
        print(f"  📑 处理页数: {result['pages_processed']}")
        print(f"  🧩 分片数量: {result['chunks_created']}")
        print(f"  📝 总字符数: {result['total_characters']:,}")
        print(f"  ⏱️  处理时间: {result['processing_time']}")
        print("  ✅ 已建立向量索引，可以开始查询")
        
    except DocumentProcessingError as e:
        print(f"❌ 文档处理失败: {e}")
        sys.exit(1)


def handle_query_command(rag_system: RAGSystem, args):
    """处理query命令"""
    print(f"🔍 查询问题: {args.question}")
    print("⏳ 搜索中...\n")
    
    try:
        results = rag_system.query(args.question, args.top_k)
        
        if not results:
            print("😔 未找到相关内容")
            print("💡 建议:")
            print("  - 尝试使用不同的关键词")
            print("  - 检查是否已正确处理文档")
            return
        
        print(f"📋 查询结果 (共找到 {len(results)} 条相关内容):\n")
        
        for i, result in enumerate(results, 1):
            print(f"【结果 {i}】(相似度: {result.score:.3f})")
            print(f"{result.content}")
            
            # 显示元数据信息
            if result.metadata:
                chunk_info = result.metadata.get('chunk_index', '未知')
                source = result.metadata.get('source_document', 
                        result.metadata.get('file_name', '未知来源'))
                print(f"  📍 来源: {Path(source).name if source else '未知'} - 分片 {chunk_info}")
            
            if i < len(results):
                print("-" * 60)
        
    except QueryError as e:
        print(f"❌ 查询失败: {e}")
        sys.exit(1)


def handle_ask_command(rag_system: RAGSystem, args):
    """处理ask命令：一次性处理文档并查询"""
    print(f"🤖 智能问答模式")
    print(f"📄 文档: {args.pdf_path}")
    print(f"❓ 问题: {args.question}")
    print("=" * 50)
    
    try:
        # 1. 处理文档
        print("⏳ 正在处理PDF文档...")
        process_result = rag_system.process_document(args.pdf_path)
        
        print(f"✅ 文档处理完成!")
        print(f"  📑 处理页数: {process_result['pages_processed']}")
        print(f"  🧩 分片数量: {process_result['chunks_created']}")
        print(f"  ⏱️  处理时间: {process_result['processing_time']}")
        
        # 2. 查询问题
        print(f"\n🔍 正在查询: {args.question}")
        results = rag_system.query(args.question, args.top_k)
        
        if not results:
            print("😔 未找到相关内容")
            print("💡 建议尝试不同的关键词")
            return
        
        print(f"\n📋 查询结果 (共找到 {len(results)} 条相关内容):")
        print("=" * 50)
        
        for i, result in enumerate(results, 1):
            print(f"\n【结果 {i}】(相似度: {result.score:.3f})")
            print(f"{result.content}")
            
            if result.metadata:
                chunk_info = result.metadata.get('chunk_index', '未知')
                print(f"  📍 来源: 分片 {chunk_info}")
            
            if i < len(results):
                print("-" * 40)
        
        print(f"\n🎉 问答完成!")
        
    except (DocumentProcessingError, QueryError) as e:
        print(f"❌ 操作失败: {e}")
        sys.exit(1)


def handle_interactive_mode(rag_system: RAGSystem, args):
    """交互模式"""
    print("🎮 进入交互查询模式")
    print("💡 使用说明:")
    print("  - 输入问题进行查询")
    print("  - 输入 'info' 查看系统信息")
    print("  - 输入 'quit', 'exit' 或 'q' 退出")
    print("  - 按 Ctrl+C 随时退出")
    print("=" * 50)
    
    # 检查系统状态
    if not rag_system.is_document_processed:
        print("⚠️  尚未处理任何文档")
        
        # 如果启用自动加载
        if args.auto_load:
            default_pdf = "data/raw/kylinos_handle_book.pdf"
            if Path(default_pdf).exists():
                print(f"🔄 正在自动加载: {default_pdf}")
                try:
                    rag_system.process_document(default_pdf)
                    print("✅ 文档加载完成")
                except Exception as e:
                    print(f"❌ 自动加载失败: {e}")
                    print("💡 请先使用 'process' 命令处理PDF文档")
            else:
                print("💡 请先使用 'process' 命令处理PDF文档")
        else:
            print("💡 请先使用 'process' 命令处理PDF文档")
    
    print()  # 空行
    
    while True:
        try:
            question = input("❓ 请输入您的问题: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['quit', 'exit', 'q', '退出']:
                print("👋 再见!")
                break
            elif question.lower() == 'info':
                print_system_info(rag_system, detailed=True)
                continue
            elif question.lower() == 'help':
                print("📖 帮助信息:")
                print("  - 直接输入问题进行查询")
                print("  - 'info' - 显示系统信息")
                print("  - 'help' - 显示此帮助")
                print("  - 'quit'/'exit'/'q' - 退出程序")
                continue
            
            # 执行查询
            try:
                results = rag_system.query(question, 3)
                
                if not results:
                    print("😔 未找到相关内容，请尝试其他关键词\n")
                    continue
                
                print(f"\n📋 找到 {len(results)} 条相关内容:")
                for i, result in enumerate(results, 1):
                    print(f"\n【{i}】{result.content[:200]}...")
                    print(f"    📊 相似度: {result.score:.3f}")
                print()
                
            except QueryError as e:
                print(f"❌ 查询失败: {e}\n")
                
        except KeyboardInterrupt:
            print("\n\n👋 再见!")
            break
        except EOFError:
            print("\n👋 再见!")
            break
        except Exception as e:
            print(f"❌ 发生错误: {e}\n")


def handle_info_command(rag_system: RAGSystem, args):
    """处理info命令"""
    print_system_info(rag_system, args.detailed)


def handle_clear_command(rag_system: RAGSystem, args):
    """处理clear命令"""
    if not args.confirm:
        response = input("⚠️  确定要清空向量数据库吗？这将删除所有已处理的文档数据。[y/N]: ")
        if response.lower() not in ['y', 'yes', '是']:
            print("操作已取消")
            return
    
    try:
        rag_system.clear_database()
        print("✅ 向量数据库已清空")
    except Exception as e:
        print(f"❌ 清空失败: {e}")


def handle_config_command(rag_system: RAGSystem, args):
    """处理config命令"""
    if args.config_action == 'show':
        print("📋 当前配置:")
        import json
        print(json.dumps(rag_system.config, indent=2, ensure_ascii=False))
    elif args.config_action == 'save':
        try:
            save_config(rag_system.config, args.output_path)
            print(f"✅ 配置已保存到: {args.output_path}")
        except Exception as e:
            print(f"❌ 保存配置失败: {e}")


def print_system_info(rag_system: RAGSystem, detailed: bool = False):
    """打印系统信息"""
    info = rag_system.get_system_info()
    
    print("\n📊 RAG系统信息:")
    print(f"  🔧 系统状态: {'✅ 就绪' if info.get('is_ready', False) else '❌ 未就绪'}")
    print(f"  📄 已处理文档: {info.get('document_count', 0)} 个")
    print(f"  🧩 文档分片数: {info.get('chunk_count', 0)} 个")
    
    if info.get('last_processed_document'):
        doc_name = Path(info['last_processed_document']).name
        print(f"  📁 最新文档: {doc_name}")
    
    # 向量存储信息
    vector_info = info.get('vector_store_info', {})
    if 'error' not in vector_info:
        print(f"  🗃️  向量数据库: {vector_info.get('name', 'Qdrant')}")
        print(f"  📊 向量数量: {vector_info.get('vectors_count', 0)}")
    
    # 模型信息
    embedder_info = info.get('embedder_info', {})
    print(f"  🤖 向量模型: {embedder_info.get('model_name', 'BGE-large-zh-v1.5')}")
    print(f"  📐 向量维度: {info.get('vector_dimension', 1024)}")
    
    if detailed and 'error' not in info:
        print("\n🔍 详细信息:")
        
        # 分片器信息
        chunker_info = info.get('chunker_info', {})
        print(f"  📝 分片配置: 大小={chunker_info.get('chunk_size', 'unknown')}, "
              f"重叠={chunker_info.get('overlap_size', 'unknown')}")
        
        # 设备信息
        device = embedder_info.get('device', 'unknown')
        print(f"  💻 运行设备: {device}")
        
        # 向量数据库状态
        if vector_info and 'error' not in vector_info:
            print(f"  📈 索引状态: {vector_info.get('indexed_vectors_count', 0)}/{vector_info.get('vectors_count', 0)} 已索引")
            print(f"  🏷️  数据段数: {vector_info.get('segments_count', 0)}")
    
    print()  # 空行


if __name__ == "__main__":
    main()
