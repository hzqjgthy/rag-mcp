"""
中国本地化适配器
支持国内主流LLM服务和本地向量数据库
"""

import json
import logging
import asyncio
import time
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
import requests
import jieba
from china_config import ChinaLLMConfig, ChinaVectorDBConfig

logger = logging.getLogger(__name__)

# 异常类定义
class ChinaLLMError(Exception):
    """中国LLM服务错误"""
    pass

class ChinaVectorDBError(Exception):
    """中国向量数据库错误"""
    pass

@dataclass
class ChinaLLMResponse:
    """中国LLM响应格式"""
    content: str
    usage: Dict[str, int]
    model: str
    finish_reason: str

class ChinaLLMClient:
    """中国LLM服务客户端"""
    
    def __init__(self, config: ChinaLLMConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'RAG-MCP-China/1.0'
        })
        
        # 设置认证头
        self._setup_auth()
        
    def _setup_auth(self):
        """设置不同提供商的认证方式"""
        if self.config.provider == 'deepseek':
            self.session.headers['Authorization'] = f'Bearer {self.config.api_key}'
        elif self.config.provider == 'zhipu':
            self.session.headers['Authorization'] = f'Bearer {self.config.api_key}'
        elif self.config.provider == 'qwen':
            self.session.headers['Authorization'] = f'Bearer {self.config.api_key}'
        elif self.config.provider == 'moonshot':
            self.session.headers['Authorization'] = f'Bearer {self.config.api_key}'
    
    def converse(self, messages: List[Dict[str, Any]], **kwargs) -> ChinaLLMResponse:
        """
        统一的对话接口
        
        Args:
            messages: 对话消息列表
            **kwargs: 额外参数
            
        Returns:
            ChinaLLMResponse: 统一的响应格式
        """
        try:
            if self.config.provider == 'deepseek':
                return self._deepseek_chat(messages, **kwargs)
            elif self.config.provider == 'zhipu':
                return self._zhipu_chat(messages, **kwargs)
            elif self.config.provider == 'qwen':
                return self._qwen_chat(messages, **kwargs)
            elif self.config.provider == 'moonshot':
                return self._moonshot_chat(messages, **kwargs)
            else:
                raise ChinaLLMError(f"不支持的LLM提供商: {self.config.provider}")
                
        except Exception as e:
            logger.error(f"LLM调用失败: {str(e)}")
            raise ChinaLLMError(f"LLM调用失败: {str(e)}")
    
    def _deepseek_chat(self, messages: List[Dict], **kwargs) -> ChinaLLMResponse:
        """DeepSeek API调用"""
        data = {
            "model": self.config.model_name,
            "messages": self._convert_messages(messages),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "stream": False
        }
        
        response = self.session.post(
            f"{self.config.base_url}/chat/completions",
            json=data,
            timeout=60
        )
        
        if response.status_code != 200:
            raise ChinaLLMError(f"DeepSeek API错误: {response.status_code} - {response.text}")
        
        result = response.json()
        
        return ChinaLLMResponse(
            content=result['choices'][0]['message']['content'],
            usage=result.get('usage', {}),
            model=result.get('model', self.config.model_name),
            finish_reason=result['choices'][0].get('finish_reason', 'stop')
        )
    
    def _zhipu_chat(self, messages: List[Dict], **kwargs) -> ChinaLLMResponse:
        """智谱AI GLM API调用"""
        data = {
            "model": self.config.model_name,
            "messages": self._convert_messages(messages),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "stream": False
        }
        
        response = self.session.post(
            f"{self.config.base_url}/chat/completions",
            json=data,
            timeout=60
        )
        
        if response.status_code != 200:
            raise ChinaLLMError(f"智谱AI API错误: {response.status_code} - {response.text}")
        
        result = response.json()
        
        return ChinaLLMResponse(
            content=result['choices'][0]['message']['content'],
            usage=result.get('usage', {}),
            model=result.get('model', self.config.model_name),
            finish_reason=result['choices'][0].get('finish_reason', 'stop')
        )
    
    def _qwen_chat(self, messages: List[Dict], **kwargs) -> ChinaLLMResponse:
        """通义千问API调用"""
        data = {
            "model": self.config.model_name,
            "input": {
                "messages": self._convert_messages(messages)
            },
            "parameters": {
                "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                "temperature": kwargs.get("temperature", self.config.temperature)
            }
        }
        
        response = self.session.post(
            f"{self.config.base_url}/services/aigc/text-generation/generation",
            json=data,
            timeout=60
        )
        
        if response.status_code != 200:
            raise ChinaLLMError(f"通义千问API错误: {response.status_code} - {response.text}")
        
        result = response.json()
        
        if result.get('code') != '200':
            raise ChinaLLMError(f"通义千问API错误: {result.get('message', '未知错误')}")
        
        output = result['output']
        
        return ChinaLLMResponse(
            content=output['text'],
            usage=result.get('usage', {}),
            model=self.config.model_name,
            finish_reason=output.get('finish_reason', 'stop')
        )
    
    def _moonshot_chat(self, messages: List[Dict], **kwargs) -> ChinaLLMResponse:
        """Moonshot AI API调用"""
        data = {
            "model": self.config.model_name,
            "messages": self._convert_messages(messages),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "stream": False
        }
        
        response = self.session.post(
            f"{self.config.base_url}/chat/completions",
            json=data,
            timeout=60
        )
        
        if response.status_code != 200:
            raise ChinaLLMError(f"Moonshot API错误: {response.status_code} - {response.text}")
        
        result = response.json()
        
        return ChinaLLMResponse(
            content=result['choices'][0]['message']['content'],
            usage=result.get('usage', {}),
            model=result.get('model', self.config.model_name),
            finish_reason=result['choices'][0].get('finish_reason', 'stop')
        )
    
    def _convert_messages(self, messages: List[Dict]) -> List[Dict]:
        """转换消息格式为标准OpenAI格式"""
        converted = []
        for msg in messages:
            role = msg["role"]
            content = msg.get("content", [])
            
            if isinstance(content, list):
                # 提取文本内容
                text_parts = []
                for c in content:
                    if "text" in c:
                        text_parts.append(c["text"])
                    elif "toolUse" in c:
                        # 工具使用转换为文本描述
                        tool_use = c["toolUse"]
                        text_parts.append(f"[使用工具: {tool_use.get('name', '未知工具')}]")
                    elif "toolResult" in c:
                        # 工具结果转换为文本
                        tool_result = c["toolResult"]
                        result_content = tool_result.get("content", [])
                        if result_content and isinstance(result_content, list):
                            for rc in result_content:
                                if "text" in rc:
                                    text_parts.append(f"[工具结果: {rc['text']}]")
                
                text_content = " ".join(text_parts)
            else:
                text_content = str(content)
            
            converted.append({"role": role, "content": text_content})
        
        return converted
    
    def get_usage_info(self, response: ChinaLLMResponse) -> Dict[str, int]:
        """提取使用信息"""
        usage = response.usage
        return {
            'input_tokens': usage.get('prompt_tokens', 0),
            'output_tokens': usage.get('completion_tokens', 0),
            'total_tokens': usage.get('total_tokens', 0)
        }

class ChinaVectorDB:
    """中国本地化向量数据库"""
    
    def __init__(self, config: ChinaVectorDBConfig):
        self.config = config
        self._init_db()
        self._init_embedding_model()
        
    def _init_db(self):
        """初始化向量数据库"""
        if self.config.db_type == 'chromadb':
            self._init_chromadb()
        elif self.config.db_type == 'faiss':
            self._init_faiss()
        else:
            raise ChinaVectorDBError(f"不支持的向量数据库类型: {self.config.db_type}")
    
    def _init_chromadb(self):
        """初始化ChromaDB"""
        try:
            import chromadb
            self.client = chromadb.PersistentClient(path=self.config.persist_directory)
            self.collection = self.client.get_or_create_collection(
                name=self.config.collection_name,
                metadata={"description": "RAG-MCP中国本地化工具集合"}
            )
            logger.info(f"ChromaDB初始化成功: {self.config.persist_directory}")
        except ImportError:
            raise ChinaVectorDBError("ChromaDB未安装，请运行: pip install chromadb")
        except Exception as e:
            raise ChinaVectorDBError(f"ChromaDB初始化失败: {str(e)}")
    
    def _init_faiss(self):
        """初始化FAISS"""
        try:
            import faiss
            import numpy as np
            import pickle
            from pathlib import Path
            
            self.faiss_index = None
            self.faiss_metadata = []
            self.index_path = Path(self.config.persist_directory) / "faiss_index.bin"
            self.metadata_path = Path(self.config.persist_directory) / "faiss_metadata.pkl"
            
            # 创建目录
            Path(self.config.persist_directory).mkdir(parents=True, exist_ok=True)
            
            # 加载现有索引
            if self.index_path.exists() and self.metadata_path.exists():
                self.faiss_index = faiss.read_index(str(self.index_path))
                with open(self.metadata_path, 'rb') as f:
                    self.faiss_metadata = pickle.load(f)
                logger.info("FAISS索引加载成功")
            
        except ImportError:
            raise ChinaVectorDBError("FAISS未安装，请运行: pip install faiss-cpu")
        except Exception as e:
            raise ChinaVectorDBError(f"FAISS初始化失败: {str(e)}")
    
    def _init_embedding_model(self):
        """初始化嵌入模型"""
        try:
            from sentence_transformers import SentenceTransformer
            
            # 中文优化的嵌入模型
            model_name = self.config.embedding_model
            logger.info(f"正在加载嵌入模型: {model_name}")
            
            self.embedding_model = SentenceTransformer(model_name)
            logger.info("嵌入模型加载成功")
            
        except ImportError:
            raise ChinaVectorDBError("sentence-transformers未安装，请运行: pip install sentence-transformers")
        except Exception as e:
            raise ChinaVectorDBError(f"嵌入模型初始化失败: {str(e)}")
    

    def add_tools(self, tools: List[Dict[str, Any]]) -> None:
        """添加工具到向量数据库"""
        try:
            if self.config.db_type == 'chromadb':
                self._add_tools_chromadb(tools)
            elif self.config.db_type == 'faiss':
                self._add_tools_faiss(tools)
                
            logger.info(f"成功添加 {len(tools)} 个工具到向量数据库")
            
        except Exception as e:
            raise ChinaVectorDBError(f"添加工具失败: {str(e)}")
    
    
    def _add_tools_chromadb(self, tools: List[Dict[str, Any]]):
        """添加工具到ChromaDB"""
        documents = []
        metadatas = []
        ids = []
        
        for i, tool in enumerate(tools):
            tool_spec = tool.get("toolSpec", {})
            name = tool_spec.get('name', '')
            desc = tool_spec.get('description', '')
            
            # 中文分词优化
            doc_text = f"{name} {desc}"
            if self.config.enable_chinese_optimization:
                doc_text = " ".join(jieba.cut(doc_text))
            
            documents.append(doc_text)
            
            # 创建简化的metadata，只包含ChromaDB支持的数据类型
            metadata = {
                "name": name,
                "description": desc,
                "tool_data": json.dumps(tool, ensure_ascii=False)  # 将完整工具信息序列化为字符串
            }
            metadatas.append(metadata)
            ids.append(f"tool_{i}_{int(time.time())}")
        
        # 生成嵌入
        embeddings = self.embedding_model.encode(documents).tolist()
        
        # 添加到集合
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
            ids=ids
        )
    
    def _add_tools_faiss(self, tools: List[Dict[str, Any]]):
        """添加工具到FAISS"""
        import faiss
        import numpy as np
        import pickle
        
        documents = []
        
        for tool in tools:
            tool_spec = tool.get("toolSpec", {})
            name = tool_spec.get('name', '')
            desc = tool_spec.get('description', '')
            
            doc_text = f"{name} {desc}"
            if self.config.enable_chinese_optimization:
                doc_text = " ".join(jieba.cut(doc_text))
            
            documents.append(doc_text)
            self.faiss_metadata.append(tool)
        
        # 生成嵌入
        embeddings = self.embedding_model.encode(documents)
        
        # 创建或更新FAISS索引
        if self.faiss_index is None:
            dimension = embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dimension)  # 内积相似度
        
        # 添加向量
        self.faiss_index.add(embeddings.astype('float32'))
        
        # 保存索引和元数据
        faiss.write_index(self.faiss_index, str(self.index_path))
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.faiss_metadata, f)
    
    def query_tools(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """查询相关工具"""
        if top_k is None:
            top_k = self.config.max_results
            
        try:
            if self.config.db_type == 'chromadb':
                return self._query_tools_chromadb(query, top_k)
            elif self.config.db_type == 'faiss':
                return self._query_tools_faiss(query, top_k)
        except Exception as e:
            raise ChinaVectorDBError(f"查询工具失败: {str(e)}")
    
    def _query_tools_chromadb(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """在ChromaDB中查询工具"""
        # 中文查询优化
        if self.config.enable_chinese_optimization:
            query_processed = " ".join(jieba.cut(query))
        else:
            query_processed = query
        
        # 生成查询嵌入
        query_embedding = self.embedding_model.encode([query_processed]).tolist()
        
        # 查询
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )
        
        # 反序列化工具数据
        tools = []
        if results["metadatas"]:
            for metadata in results["metadatas"][0]:
                try:
                    tool_data = json.loads(metadata["tool_data"])
                    tools.append(tool_data)
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"无法解析工具数据: {e}")
                    continue
        
        return tools
    
    def _query_tools_faiss(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """在FAISS中查询工具"""
        import numpy as np
        
        if self.faiss_index is None or len(self.faiss_metadata) == 0:
            return []
        
        # 中文查询优化
        if self.config.enable_chinese_optimization:
            query_processed = " ".join(jieba.cut(query))
        else:
            query_processed = query
        
        # 生成查询嵌入
        query_embedding = self.embedding_model.encode([query_processed])
        
        # 搜索
        scores, indices = self.faiss_index.search(
            query_embedding.astype('float32'), 
            min(top_k, len(self.faiss_metadata))
        )
        
        # 返回结果
        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0 and idx < len(self.faiss_metadata):
                results.append(self.faiss_metadata[idx])
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """获取数据库统计信息"""
        try:
            if self.config.db_type == 'chromadb':
                count = self.collection.count()
                return {
                    "type": "ChromaDB",
                    "tool_count": count,
                    "collection_name": self.config.collection_name,
                    "persist_directory": self.config.persist_directory
                }
            elif self.config.db_type == 'faiss':
                count = len(self.faiss_metadata) if self.faiss_metadata else 0
                return {
                    "type": "FAISS",
                    "tool_count": count,
                    "index_size": self.faiss_index.ntotal if self.faiss_index else 0,
                    "persist_directory": self.config.persist_directory
                }
        except Exception as e:
            logger.error(f"获取统计信息失败: {str(e)}")
            return {"error": str(e)}
    
    

# 工具函数
def test_llm_connection(config: ChinaLLMConfig) -> bool:
    """测试LLM连接"""
    try:
        client = ChinaLLMClient(config)
        response = client.converse([
            {"role": "user", "content": [{"text": "你好，请回复'连接成功'"}]}
        ])
        return "连接成功" in response.content or "成功" in response.content
    except Exception as e:
        logger.error(f"LLM连接测试失败: {str(e)}")
        return False

def test_vector_db(config: ChinaVectorDBConfig) -> bool:
    """测试向量数据库连接"""
    try:
        db = ChinaVectorDB(config)
        # 添加测试工具
        test_tools = [{
            "toolSpec": {
                "name": "test_tool",
                "description": "测试工具",
                "inputSchema": {"type": "object"}
            }
        }]
        db.add_tools(test_tools)
        
        # 查询测试
        results = db.query_tools("测试", top_k=1)
        return len(results) > 0
    except Exception as e:
        logger.error(f"向量数据库测试失败: {str(e)}")
        return False

if __name__ == "__main__":
    # 测试代码
    from china_config import load_china_config
    
    try:
        config = load_china_config()
        
        print("🧪 测试LLM连接...")
        if test_llm_connection(config.llm):
            print("✅ LLM连接成功")
        else:
            print("❌ LLM连接失败")
        
        print("🧪 测试向量数据库...")
        if test_vector_db(config.vector_db):
            print("✅ 向量数据库连接成功")
        else:
            print("❌ 向量数据库连接失败")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}") 