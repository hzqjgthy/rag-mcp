"""
中国本地化聊天管理器
整合国内LLM服务、本地向量数据库和MCP工具
"""

import json
import logging
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import asdict

from china_config import ChinaChatConfig, load_china_config
from china_adapters import ChinaLLMClient, ChinaVectorDB, ChinaLLMError, ChinaVectorDBError
from chat.mcp_client import MCPClient
from chat.chat_session import ChatSession
from chat.exceptions import ChatError, MCPToolError

logger = logging.getLogger(__name__)

class ChinaChatManager:
    """中国本地化聊天管理器"""
    
    def __init__(self, config: Optional[ChinaChatConfig] = None):
        """
        初始化中国聊天管理器
        
        Args:
            config: 中国聊天配置（使用默认配置如果为None）
        """
        self.config = config or load_china_config()
        self.llm_client = ChinaLLMClient(self.config.llm)
        self.vector_db = ChinaVectorDB(self.config.vector_db)
        self.session = ChatSession()
        self._mcp_client = None
        self._tool_config = None
        
    async def initialize(self, init_mcp: bool = True) -> None:
        """初始化所有组件"""
        try:
            if init_mcp:
                # 初始化MCP客户端
                self._mcp_client = MCPClient(self.config.mcp)
                await self._mcp_client.connect()
                
                # 获取工具并转换格式
                tools_response = await self._mcp_client.list_tools()
                logger.info(f"获取到 {len(tools_response.tools)} 个MCP工具")
                logger.info(f"Tools: {tools_response.tools}")
                self._tool_config = self._mcp_client.convert_tools_to_bedrock_format(tools_response.tools)
                
                # 将工具添加到向量数据库
                await self.sync_tools_to_vector_db()
            
            logger.info("中国聊天管理器初始化成功")
            
        except Exception as e:
            logger.error(f"初始化中国聊天管理器失败: {str(e)}")
            raise ChatError(f"初始化失败: {str(e)}")
    
    async def sync_tools_to_vector_db(self) -> None:
        """同步MCP工具到向量数据库"""
        try:
            if not self._tool_config:
                raise ChatError("没有可同步的工具配置")
            
            tools_list = self._tool_config.get('tools', [])
            if not tools_list:
                logger.warning("没有找到工具列表")
                return
            
            
            # 添加工具到向量数据库
            self.vector_db.add_tools(tools_list)

            
            # 获取统计信息
            stats = self.vector_db.get_stats()
            logger.info(f"工具同步完成: {stats}")
            
        except Exception as e:
            logger.error(f"同步工具到向量数据库失败: {str(e)}")
            raise ChatError(f"工具同步失败: {str(e)}")
    
    async def cleanup(self) -> None:
        """清理资源"""
        try:
            if self._mcp_client:
                try:
                    await self._mcp_client.disconnect()
                except Exception as disconnect_error:
                    logger.warning(f"MCP断开连接时出错: {str(disconnect_error)}")
                finally:
                    self._mcp_client = None
            logger.info("中国聊天管理器清理完成")
        except Exception as e:
            logger.warning(f"清理时出错: {str(e)}")
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self.initialize()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.cleanup()
    
    async def process_message(
        self, 
        user_input: str, 
        use_rag: bool = True,
        max_tool_rounds: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        处理用户消息并生成响应
        
        Args:
            user_input: 用户输入消息
            use_rag: 是否使用RAG检索相关工具
            max_tool_rounds: 最大工具调用轮数
            
        Returns:
            响应信息包括文本和使用统计
        """
        try:
            # 添加用户消息到会话
            self.session.add_user_message(user_input)
            
            # 确定工具配置
            if use_rag and self.vector_db:
                # 使用RAG检索相关工具
                relevant_tools = self.vector_db.query_tools(user_input, top_k=3)
                tool_config = {"tools": relevant_tools} if relevant_tools else None
                logger.info(f"RAG检索到 {len(relevant_tools)} 个相关工具")
            else:
                # 使用所有工具
                tool_config = self._tool_config
                logger.info("使用所有可用工具")
            
            # 生成响应
            response = await self._generate_response_with_tools(
                tool_config=tool_config,
                max_tool_rounds=max_tool_rounds or self.config.max_tool_rounds
            )
            
            return response
            
        except Exception as e:
            logger.error(f"处理消息失败: {str(e)}")
            raise ChatError(f"处理消息失败: {str(e)}")
    
    async def _generate_response_with_tools(
        self,
        tool_config: Optional[Dict[str, Any]] = None,
        max_tool_rounds: int = 5
    ) -> Dict[str, Any]:
        """
        使用工具生成响应，支持多轮工具调用
        
        Args:
            tool_config: 工具配置
            max_tool_rounds: 最大工具调用轮数
            
        Returns:
            响应信息
        """
        total_usage = {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}
        tool_rounds = 0
        
        while tool_rounds < max_tool_rounds:
            messages = self.session.get_messages()
            
            # 调用LLM
            try:
                # 构建提示，包含工具信息
                enhanced_messages = self._enhance_messages_with_tools(messages, tool_config)
                
                response = self.llm_client.converse(enhanced_messages)
                
                # 累计使用统计
                current_usage = self.llm_client.get_usage_info(response)
                for key in total_usage:
                    total_usage[key] += current_usage[key]
                
                # 处理响应
                response_text = response.content
                
                # 检查是否需要工具调用
                tool_calls = self._extract_tool_calls(response_text)
                
                if tool_calls and self._mcp_client:
                    tool_rounds += 1
                    logger.info(f"开始工具调用轮次 {tool_rounds}/{max_tool_rounds}")
                    
                    # 添加助手消息（包含工具调用）
                    self.session.add_assistant_message(response_text)
                    
                    # 执行工具调用
                    await self._handle_tool_calls(tool_calls)
                    
                    # 继续循环进行下一轮
                    continue
                else:
                    # 没有工具调用，返回最终响应
                    if response_text:
                        self.session.add_assistant_message(response_text)
                    
                    logger.info(f"对话完成，共进行 {tool_rounds} 轮工具调用")
                    return {
                        'text': response_text,
                        'usage': total_usage,
                        'tool_rounds': tool_rounds,
                        'provider': self.config.llm.provider,
                        'model': self.config.llm.model_name
                    }
                    
            except ChinaLLMError as e:
                logger.error(f"LLM调用失败: {str(e)}")
                return {
                    'text': f"抱歉，LLM服务出现问题: {str(e)}",
                    'usage': total_usage,
                    'tool_rounds': tool_rounds,
                    'error': str(e)
                }
        
        # 达到最大轮数限制
        logger.warning(f"达到最大工具调用轮数限制 ({max_tool_rounds})")
        
        # 进行最后一次调用获取响应
        try:
            messages = self.session.get_messages()
            enhanced_messages = self._enhance_messages_with_tools(messages, tool_config)
            
            final_response = self.llm_client.converse(enhanced_messages)
            final_usage = self.llm_client.get_usage_info(final_response)
            
            for key in total_usage:
                total_usage[key] += final_usage[key]
            
            final_text = final_response.content
            if final_text:
                self.session.add_assistant_message(final_text)
            
            return {
                'text': final_text,
                'usage': total_usage,
                'tool_rounds': tool_rounds,
                'max_rounds_reached': True,
                'provider': self.config.llm.provider,
                'model': self.config.llm.model_name
            }
            
        except Exception as e:
            logger.error(f"最终响应生成失败: {str(e)}")
            return {
                'text': "抱歉，在生成最终响应时出现问题。",
                'usage': total_usage,
                'tool_rounds': tool_rounds,
                'error': str(e)
            }
    
    def _enhance_messages_with_tools(
        self, 
        messages: List[Dict[str, Any]], 
        tool_config: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        增强消息，添加工具信息
        
        Args:
            messages: 原始消息列表
            tool_config: 工具配置
            
        Returns:
            增强后的消息列表
        """
        if not tool_config or not tool_config.get('tools'):
            return self._convert_messages_for_llm(messages)
        
        # 构建工具描述
        tools_description = self._build_tools_description(tool_config['tools'])
        
        # 在第一条用户消息前添加系统提示
        enhanced_messages = []
        
        # 添加系统提示
        system_prompt = f"""你是一个智能助手，可以使用以下工具来帮助用户：

{tools_description}

当你需要使用工具时，请按以下格式回复：
[TOOL_CALL]
工具名称: tool_name
参数: {{"param1": "value1", "param2": "value2"}}
[/TOOL_CALL]

请根据用户的需求选择合适的工具，并提供有用的回答。"""
        
        enhanced_messages.append({"role": "system", "content": system_prompt})
        
        # 添加转换后的消息
        enhanced_messages.extend(self._convert_messages_for_llm(messages))
        
        return enhanced_messages
    
    def _build_tools_description(self, tools: List[Dict[str, Any]]) -> str:
        """构建工具描述文本"""
        descriptions = []
        
        for tool in tools:
            tool_spec = tool.get("toolSpec", {})
            name = tool_spec.get("name", "未知工具")
            desc = tool_spec.get("description", "无描述")
            
            # 获取参数信息
            input_schema = tool_spec.get("inputSchema", {}).get("json", {})
            properties = input_schema.get("properties", {})
            
            param_desc = ""
            if properties:
                param_list = []
                for param_name, param_info in properties.items():
                    param_type = param_info.get("type", "string")
                    param_list.append(f"{param_name}({param_type})")
                param_desc = f" - 参数: {', '.join(param_list)}"
            
            descriptions.append(f"- {name}: {desc}{param_desc}")
        
        return "\n".join(descriptions)
    
    def _convert_messages_for_llm(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """转换消息格式为LLM可理解的格式"""
        converted = []
        
        for msg in messages:
            role = msg["role"]
            content = msg.get("content", [])
            
            if isinstance(content, list):
                text_parts = []
                for c in content:
                    if "text" in c:
                        text_parts.append(c["text"])
                    elif "toolUse" in c:
                        tool_use = c["toolUse"]
                        text_parts.append(f"[使用工具: {tool_use.get('name', '未知工具')}]")
                    elif "toolResult" in c:
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
    
    def _extract_tool_calls(self, response_text: str) -> List[Dict[str, Any]]:
        """从响应文本中提取工具调用"""
        tool_calls = []
        
        # 简单的工具调用解析
        import re
        
        # 匹配 [TOOL_CALL] ... [/TOOL_CALL] 格式
        pattern = r'\[TOOL_CALL\](.*?)\[/TOOL_CALL\]'
        matches = re.findall(pattern, response_text, re.DOTALL)
        
        for match in matches:
            try:
                # 解析工具名称和参数
                lines = match.strip().split('\n')
                tool_name = None
                parameters = {}
                
                for line in lines:
                    line = line.strip()
                    if line.startswith('工具名称:') or line.startswith('tool_name:'):
                        tool_name = line.split(':', 1)[1].strip()
                    elif line.startswith('参数:') or line.startswith('parameters:'):
                        param_str = line.split(':', 1)[1].strip()
                        try:
                            parameters = json.loads(param_str)
                        except json.JSONDecodeError:
                            logger.warning(f"无法解析参数: {param_str}")
                
                if tool_name:
                    tool_calls.append({
                        'name': tool_name,
                        'parameters': parameters
                    })
                    
            except Exception as e:
                logger.warning(f"解析工具调用失败: {str(e)}")
        
        return tool_calls
    
    async def _handle_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> None:
        """处理工具调用"""
        for tool_call in tool_calls:
            tool_name = tool_call['name']
            parameters = tool_call['parameters']
            
            logger.info(f"执行工具: {tool_name}，参数: {parameters}")
            
            try:
                # 调用MCP工具
                result = await self._mcp_client.call_tool(tool_name, parameters)
                
                # 提取文本内容
                text_content = self._mcp_client.extract_text_content(result)
                
                if text_content:
                    # 添加工具结果到会话
                    self.session.add_user_message(f"[工具 {tool_name} 的执行结果]: {text_content}")
                else:
                    error_msg = f"工具 {tool_name} 没有返回文本内容"
                    self.session.add_user_message(f"[工具执行错误]: {error_msg}")
                    
            except MCPToolError as e:
                error_msg = f"工具 {tool_name} 执行失败: {str(e)}"
                logger.error(error_msg)
                self.session.add_user_message(f"[工具执行错误]: {error_msg}")
            except Exception as e:
                error_msg = f"工具 {tool_name} 执行时出现未知错误: {str(e)}"
                logger.error(error_msg)
                self.session.add_user_message(f"[工具执行错误]: {error_msg}")
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """获取对话历史"""
        return self.session.get_messages()
    
    def clear_conversation(self) -> None:
        """清除对话历史"""
        self.session.clear()
    
    def get_message_count(self) -> int:
        """获取消息数量"""
        return self.session.get_message_count()
    
    def get_vector_db_stats(self) -> Dict[str, Any]:
        """获取向量数据库统计信息"""
        try:
            return self.vector_db.get_stats()
        except Exception as e:
            logger.error(f"获取向量数据库统计失败: {str(e)}")
            return {"error": str(e)}
    
    def get_config_info(self) -> Dict[str, Any]:
        """获取配置信息"""
        return {
            "llm_provider": self.config.llm.provider,
            "llm_model": self.config.llm.model_name,
            "vector_db_type": self.config.vector_db.db_type,
            "embedding_model": self.config.vector_db.embedding_model,
            "max_tool_rounds": self.config.max_tool_rounds,
            "enable_auto_tools": self.config.enable_auto_tool_calling,
            "chinese_optimization": self.config.enable_chinese_optimization
        }

# 便捷函数
async def create_china_chat_manager(
    llm_provider: str = "deepseek",
    api_key: str = None,
    vector_db_type: str = "chromadb"
) -> ChinaChatManager:
    """
    创建中国聊天管理器的便捷函数
    
    Args:
        llm_provider: LLM提供商
        api_key: API密钥
        vector_db_type: 向量数据库类型
        
    Returns:
        配置好的中国聊天管理器
    """
    import os
    
    # 设置环境变量
    if api_key:
        os.environ['CHINA_LLM_API_KEY'] = api_key
    if llm_provider:
        os.environ['CHINA_LLM_PROVIDER'] = llm_provider
    if vector_db_type:
        os.environ['CHINA_VECTOR_DB_TYPE'] = vector_db_type
    
    # 创建配置
    config = load_china_config()
    
    # 创建管理器
    manager = ChinaChatManager(config)
    await manager.initialize()
    
    return manager

if __name__ == "__main__":
    # 测试代码
    async def test_china_chat_manager():
        try:
            # 创建管理器
            manager = ChinaChatManager()
            await manager.initialize(init_mcp=False)  # 不初始化MCP进行基础测试
            
            # 测试基本对话
            response = await manager.process_message(
                "你好，请介绍一下你自己", 
                use_rag=False
            )
            
            print("🤖 响应:", response['text'])
            print("📊 统计:", response['usage'])
            print("🔧 配置:", manager.get_config_info())
            
            await manager.cleanup()
            
        except Exception as e:
            print(f"❌ 测试失败: {e}")
    
    # 运行测试
    asyncio.run(test_china_chat_manager()) 