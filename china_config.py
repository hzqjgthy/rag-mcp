"""
中国本地化配置文件
支持国内LLM服务和本地向量数据库
"""

import os
from dataclasses import dataclass
from typing import Optional, List, Dict
from pathlib import Path

# 自动加载.env文件
def load_china_env():
    """加载中国本地化环境变量"""
    try:
        from dotenv import load_dotenv
        env_files = ['.env.china', '.env.local', '.env']
        
        for env_file in env_files:
            if Path(env_file).exists():
                load_dotenv(env_file)
                print(f"已加载环境变量文件: {env_file}")
                break
        else:
            print("未找到环境变量文件，使用默认配置")
    except ImportError:
        print("python-dotenv未安装，跳过环境变量加载")
    except Exception as e:
        print(f"加载环境变量时出错: {e}")

# 加载环境变量
load_china_env()

@dataclass
class ChinaLLMConfig:
    """中国LLM服务配置"""
    provider: str = os.getenv('CHINA_LLM_PROVIDER', 'deepseek')  # deepseek, zhipu, qwen, moonshot
    api_key: str = os.getenv('CHINA_LLM_API_KEY', '')
    base_url: str = os.getenv('CHINA_LLM_BASE_URL', '')
    model_name: str = os.getenv('CHINA_LLM_MODEL', '')
    max_tokens: int = int(os.getenv('CHINA_LLM_MAX_TOKENS', '4096'))
    temperature: float = float(os.getenv('CHINA_LLM_TEMPERATURE', '0.7'))
    
    def __post_init__(self):
        """根据提供商设置默认值"""
        if not self.base_url or not self.model_name:
            self._set_provider_defaults()
    
    def _set_provider_defaults(self):
        """设置不同提供商的默认配置"""
        provider_configs = {
            'deepseek': {
                'base_url': 'https://api.deepseek.com',
                'model_name': 'deepseek-chat'
            },
            'zhipu': {
                'base_url': 'https://open.bigmodel.cn/api/paas/v4',
                'model_name': 'glm-4'
            },
            'qwen': {
                'base_url': 'https://dashscope.aliyuncs.com/api/v1',
                'model_name': 'qwen-turbo'
            },
            'moonshot': {
                'base_url': 'https://api.moonshot.cn/v1',
                'model_name': 'moonshot-v1-8k'
            }
        }
        
        config = provider_configs.get(self.provider, provider_configs['deepseek'])
        if not self.base_url:
            self.base_url = config['base_url']
        if not self.model_name:
            self.model_name = config['model_name']

@dataclass
class ChinaVectorDBConfig:
    """中国向量数据库配置"""
    db_type: str = os.getenv('CHINA_VECTOR_DB_TYPE', 'chromadb')  # chromadb, faiss, qdrant
    persist_directory: str = os.getenv('CHINA_VECTOR_DB_PATH', './china_vector_db')
    collection_name: str = os.getenv('CHINA_VECTOR_COLLECTION', 'rag_mcp_tools_cn')
    embedding_model: str = os.getenv('CHINA_EMBEDDING_MODEL', 'shibing624/text2vec-base-chinese')
    max_results: int = int(os.getenv('CHINA_VECTOR_MAX_RESULTS', '10'))
    enable_chinese_optimization: bool = os.getenv('CHINA_ENABLE_CHINESE_OPT', 'true').lower() == 'true'

@dataclass
class ChinaMCPConfig:
    """中国MCP配置"""
    command: str = os.getenv('CHINA_MCP_COMMAND', 'npx')
    args: Optional[List[str]] = None
    env: Optional[Dict[str, str]] = None
    
    def __post_init__(self):
        if self.args is None:
            mcp_args_str = os.getenv('CHINA_MCP_ARGS', '-y,@modelcontextprotocol/server-filesystem,.')
            self.args = [arg.strip() for arg in mcp_args_str.split(',')]
        
        if self.env is None:
            self.env = {}

@dataclass
class ChinaChatConfig:
    """中国聊天配置"""
    llm: ChinaLLMConfig = None
    vector_db: ChinaVectorDBConfig = None
    mcp: ChinaMCPConfig = None
    max_tool_rounds: int = int(os.getenv('CHINA_CHAT_MAX_TOOL_ROUNDS', '5'))
    enable_auto_tool_calling: bool = os.getenv('CHINA_CHAT_ENABLE_AUTO_TOOLS', 'true').lower() == 'true'
    enable_chinese_optimization: bool = os.getenv('CHINA_ENABLE_CHINESE_OPT', 'true').lower() == 'true'
    
    def __post_init__(self):
        if self.llm is None:
            self.llm = ChinaLLMConfig()
        if self.vector_db is None:
            self.vector_db = ChinaVectorDBConfig()
        if self.mcp is None:
            self.mcp = ChinaMCPConfig()
    
    def validate(self) -> None:
        """验证配置"""
        errors = []
        
        # 检查LLM API密钥
        if not self.llm.api_key:
            errors.append(f"缺少{self.llm.provider}的API密钥，请设置CHINA_LLM_API_KEY环境变量")
        
        # 检查向量数据库路径
        if not self.vector_db.persist_directory:
            errors.append("向量数据库路径未设置")
        
        if errors:
            raise ValueError("配置验证失败:\n" + "\n".join(f"- {error}" for error in errors))

def load_china_config() -> ChinaChatConfig:
    """
    加载并验证中国本地化配置
    
    Returns:
        ChinaChatConfig: 验证后的配置对象
        
    Raises:
        ValueError: 如果必需的配置缺失
    """
    config = ChinaChatConfig()
    config.validate()
    return config

# 支持的LLM提供商信息
SUPPORTED_LLM_PROVIDERS = {
    'deepseek': {
        'name': 'DeepSeek',
        'website': 'https://platform.deepseek.com/',
        'description': '深度求索，支持长上下文对话',
        'models': ['deepseek-chat', 'deepseek-coder']
    },
    'zhipu': {
        'name': '智谱AI',
        'website': 'https://open.bigmodel.cn/',
        'description': '清华大学技术，GLM系列模型',
        'models': ['glm-4', 'glm-3-turbo']
    },
    'qwen': {
        'name': '通义千问',
        'website': 'https://dashscope.aliyuncs.com/',
        'description': '阿里云大模型服务',
        'models': ['qwen-turbo', 'qwen-plus', 'qwen-max']
    },
    'moonshot': {
        'name': 'Moonshot AI',
        'website': 'https://platform.moonshot.cn/',
        'description': 'Kimi大模型，支持超长上下文',
        'models': ['moonshot-v1-8k', 'moonshot-v1-32k', 'moonshot-v1-128k']
    }
}

def print_supported_providers():
    """打印支持的LLM提供商信息"""
    print("🇨🇳 支持的中国LLM提供商:")
    print("=" * 50)
    for key, info in SUPPORTED_LLM_PROVIDERS.items():
        print(f"{info['name']} ({key})")
        print(f"  网站: {info['website']}")
        print(f"  描述: {info['description']}")
        print(f"  模型: {', '.join(info['models'])}")
        print()

if __name__ == "__main__":
    print_supported_providers()
    
    try:
        config = load_china_config()
        print("✅ 配置验证通过")
        print(f"LLM提供商: {config.llm.provider}")
        print(f"向量数据库: {config.vector_db.db_type}")
        print(f"嵌入模型: {config.vector_db.embedding_model}")
    except ValueError as e:
        print(f"❌ 配置验证失败: {e}") 