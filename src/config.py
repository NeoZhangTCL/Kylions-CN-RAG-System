"""
RAG系统配置管理
"""

from typing import Dict, Any, Optional
from pathlib import Path
import yaml
import json
from .exceptions import ConfigurationError


DEFAULT_CONFIG = {
    # 文档解析配置
    'parser': {
        'output_dir': 'data/processed',
        'filter_patterns': None,  # 使用默认过滤模式
    },
    
    # 分片配置
    'chunker': {
        'chunk_size': 500,
        'overlap_size': 50,
    },
    
    # 向量化配置  
    'embedder': {
        'model_name': 'BAAI/bge-large-zh-v1.5',
        'device': None,  # 自动选择
    },
    
    # 检索配置
    'retriever': {
        'collection_name': 'kylinos_docs',
        'vector_size': 1024,
        'distance_metric': 'cosine',
    },
    
    # 查询配置
    'query': {
        'default_top_k': 3,
        'min_score_threshold': 0.1,
        'max_content_length': 1000,  # 结果内容最大长度
    },
    
    # 日志配置
    'logging': {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    }
}


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径，支持YAML和JSON格式
        
    Returns:
        配置字典
        
    Raises:
        ConfigurationError: 配置文件加载失败时
    """
    config = DEFAULT_CONFIG.copy()
    
    if config_path:
        config_file = Path(config_path)
        
        if not config_file.exists():
            raise ConfigurationError(f"配置文件不存在: {config_path}")
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                if config_file.suffix.lower() in ['.yaml', '.yml']:
                    user_config = yaml.safe_load(f)
                elif config_file.suffix.lower() == '.json':
                    user_config = json.load(f)
                else:
                    raise ConfigurationError(f"不支持的配置文件格式: {config_file.suffix}")
            
            # 递归合并配置
            config = merge_configs(config, user_config)
            
        except Exception as e:
            raise ConfigurationError(f"加载配置文件失败: {e}") from e
    
    # 验证配置
    validate_config(config)
    
    return config


def merge_configs(base_config: Dict[str, Any], user_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    递归合并配置字典
    
    Args:
        base_config: 基础配置
        user_config: 用户配置
        
    Returns:
        合并后的配置
    """
    result = base_config.copy()
    
    for key, value in user_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


def validate_config(config: Dict[str, Any]) -> None:
    """
    验证配置的合法性
    
    Args:
        config: 配置字典
        
    Raises:
        ConfigurationError: 配置无效时
    """
    try:
        # 验证分片配置
        chunk_size = config['chunker']['chunk_size']
        overlap_size = config['chunker']['overlap_size']
        
        if chunk_size <= 0:
            raise ConfigurationError("分片大小必须大于0")
        
        if overlap_size < 0:
            raise ConfigurationError("重叠大小不能小于0")
        
        if overlap_size >= chunk_size:
            raise ConfigurationError("重叠大小不能大于或等于分片大小")
        
        # 验证向量配置
        vector_size = config['retriever']['vector_size']
        if vector_size <= 0:
            raise ConfigurationError("向量维度必须大于0")
        
        # 验证查询配置
        top_k = config['query']['default_top_k']
        if top_k <= 0:
            raise ConfigurationError("默认返回结果数量必须大于0")
        
        min_score = config['query']['min_score_threshold']
        if min_score < 0 or min_score > 1:
            raise ConfigurationError("最小相似度阈值必须在0-1之间")
            
    except KeyError as e:
        raise ConfigurationError(f"配置缺少必要字段: {e}") from e


def get_default_config() -> Dict[str, Any]:
    """获取默认配置"""
    return DEFAULT_CONFIG.copy()


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    保存配置到文件
    
    Args:
        config: 配置字典
        config_path: 保存路径
        
    Raises:
        ConfigurationError: 保存失败时
    """
    config_file = Path(config_path)
    
    try:
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_file, 'w', encoding='utf-8') as f:
            if config_file.suffix.lower() in ['.yaml', '.yml']:
                yaml.safe_dump(config, f, default_flow_style=False, allow_unicode=True)
            elif config_file.suffix.lower() == '.json':
                json.dump(config, f, ensure_ascii=False, indent=2)
            else:
                raise ConfigurationError(f"不支持的配置文件格式: {config_file.suffix}")
                
    except Exception as e:
        raise ConfigurationError(f"保存配置文件失败: {e}") from e
