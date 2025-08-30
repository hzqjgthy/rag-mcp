#!/usr/bin/env python3
"""
RAG-MCP 中国本地化演示脚本
展示完整的RAG-MCP功能，包括工具检索、LLM对话和性能对比
"""

import asyncio
import json
import time
import logging
from typing import List, Dict, Any
from pathlib import Path

from china_config import load_china_config, print_supported_providers
from china_chat_manager import ChinaChatManager
from china_adapters import test_llm_connection, test_vector_db

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('china_demo.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class ChinaRAGMCPDemo:
    """中国RAG-MCP演示类"""
    
    def __init__(self):
        self.config = None
        self.chat_manager = None
        
    async def run_complete_demo(self):
        """运行完整演示"""
        print("🇨🇳 RAG-MCP 中国本地化版本演示")
        print("=" * 60)
        
        # 1. 显示支持的提供商
        await self.show_supported_providers()
        
        # 2. 加载配置
        await self.load_and_validate_config()
        
        # 3. 测试连接
        await self.test_connections()
        
        # 4. 初始化系统
        await self.initialize_system()
        
        # 5. 演示基础对话
        await self.demo_basic_chat()
        
        # 6. 演示工具检索
        await self.demo_tool_retrieval()
        
        # 7. 演示RAG vs 全工具对比
        await self.demo_rag_comparison()
        
        # 8. 演示中文优化
        await self.demo_chinese_optimization()
        
        # 9. 显示统计信息
        await self.show_statistics()
        
        # 10. 清理资源
        await self.cleanup()
        
        print("\n🎉 演示完成！")
    
    async def show_supported_providers(self):
        """显示支持的LLM提供商"""
        print("\n📋 支持的LLM提供商:")
        print("-" * 40)
        print_supported_providers()
        
        print("\n💡 提示: 您需要先申请相应提供商的API密钥")
        print("推荐使用DeepSeek，性价比高且支持长上下文")
    
    async def load_and_validate_config(self):
        """加载和验证配置"""
        print("\n⚙️ 加载配置...")
        
        try:
            self.config = load_china_config()
            print(f"✅ 配置加载成功")
            print(f"   LLM提供商: {self.config.llm.provider}")
            print(f"   LLM模型: {self.config.llm.model_name}")
            print(f"   向量数据库: {self.config.vector_db.db_type}")
            print(f"   嵌入模型: {self.config.vector_db.embedding_model}")
            
        except ValueError as e:
            print(f"❌ 配置验证失败: {e}")
            print("\n请检查环境变量配置:")
            print("- CHINA_LLM_API_KEY: LLM API密钥")
            print("- CHINA_LLM_PROVIDER: LLM提供商")
            print("\n或创建 .env.china 配置文件")
            return False
        
        return True
    
    async def test_connections(self):
        """测试各组件连接"""
        print("\n🧪 测试连接...")
        
        # 测试LLM连接
        print("📡 测试LLM连接...")
        if test_llm_connection(self.config.llm):
            print("✅ LLM连接成功")
        else:
            print("❌ LLM连接失败")
            return False
        
        # 测试向量数据库
        print("🗄️ 测试向量数据库...")
        if test_vector_db(self.config.vector_db):
            print("✅ 向量数据库连接成功")
        else:
            print("❌ 向量数据库连接失败")
            return False
        
        print("🎉 所有连接测试通过！")
        return True
    
    async def initialize_system(self):
        """初始化系统"""
        print("\n🚀 初始化系统...")
        
        try:
            self.chat_manager = ChinaChatManager(self.config)
            await self.chat_manager.initialize(init_mcp=True)
            print("✅ 系统初始化完成")
            
            # 显示工具统计
            stats = self.chat_manager.get_vector_db_stats()
            if 'error' not in stats:
                print(f"📊 已加载 {stats.get('tool_count', 0)} 个工具到向量数据库")
            
        except Exception as e:
            print(f"⚠️ MCP初始化失败，将使用基础模式: {e}")
            self.chat_manager = ChinaChatManager(self.config)
            await self.chat_manager.initialize(init_mcp=False)
    
    async def demo_basic_chat(self):
        """演示基础对话功能"""
        print("\n💬 演示基础对话功能")
        print("-" * 30)
        
        test_queries = [
            "你好，请介绍一下你自己",
            "你能做什么？",
            "请解释一下什么是RAG技术"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{i}. 用户: {query}")
            
            start_time = time.time()
            response = await self.chat_manager.process_message(
                query, 
                use_rag=False  # 基础对话不使用RAG
            )
            end_time = time.time()
            
            print(f"🤖 助手: {response['text']}")
            print(f"⏱️ 响应时间: {end_time - start_time:.2f}秒")
            print(f"📊 令牌使用: 输入 {response['usage']['input_tokens']}，输出 {response['usage']['output_tokens']}，总计 **** {response['usage']['total_tokens']} ****")
    
    async def demo_tool_retrieval(self):
        """演示工具检索功能"""
        print("\n🔧 演示工具检索功能")
        print("-" * 30)
        
        # 检查是否有工具可用
        stats = self.chat_manager.get_vector_db_stats()
        if stats.get('tool_count', 0) == 0:
            print("⚠️ 没有可用工具，跳过工具检索演示")
            return
        
        test_queries = [
            "我想读取一个文件",
            "列出目录中的文件",
            "搜索包含特定内容的文件"
        ]
        
        for query in test_queries:
            print(f"\n🔍 查询: {query}")
            
            # 检索相关工具
            relevant_tools = self.chat_manager.vector_db.query_tools(query, top_k=3)
            
            print(f"📋 检索到 {len(relevant_tools)} 个相关工具:")
            for tool in relevant_tools:
                tool_spec = tool.get("toolSpec", {})
                name = tool_spec.get("name", "未知工具")
                desc = tool_spec.get("description", "无描述")
                print(f"  - {name}: {desc}")
    
    async def demo_rag_comparison(self):
        """演示RAG vs 全工具对比"""
        print("\n⚖️ 演示RAG vs 全工具对比")
        print("-" * 30)
        
        # 检查是否有工具可用
        stats = self.chat_manager.get_vector_db_stats()
        if stats.get('tool_count', 0) == 0:
            print("⚠️ 没有可用工具，跳过对比演示")
            return
        
        test_query = "我需要查看当前目录下的所有文件"
        print(f"📝 测试查询: {test_query}")
        
        # 清除对话历史
        self.chat_manager.clear_conversation()
        
        # RAG模式
        print("\n🎯 RAG模式 (仅检索相关工具):")
        start_time = time.time()
        rag_response = await self.chat_manager.process_message(
            test_query, 
            use_rag=True
        )
        rag_time = time.time() - start_time
        
        print(f"📊 RAG结果:")
        print(f"  - 响应时间: {rag_time:.2f}秒")
        # print(f"  - 令牌使用: {rag_response['usage']['total_tokens']}")
        print(f"  - 令牌使用: 输入 {rag_response['usage']['input_tokens']}，输出 {rag_response['usage']['output_tokens']}，总计 **** {rag_response['usage']['total_tokens']} ****")
        print(f"  - 工具轮数: {rag_response.get('tool_rounds', 0)}")
        
        # 清除对话历史
        self.chat_manager.clear_conversation()
        
        # 全工具模式
        print("\n🎯 全工具模式 (使用所有工具):")
        start_time = time.time()
        full_response = await self.chat_manager.process_message(
            test_query, 
            use_rag=False
        )
        full_time = time.time() - start_time
        
        print(f"📊 全工具结果:")
        print(f"  - 响应时间: {full_time:.2f}秒")
        # print(f"  - 令牌使用: {full_response['usage']['total_tokens']}")
        print(f"  - 令牌使用: 输入 {full_response['usage']['input_tokens']}，输出 {full_response['usage']['output_tokens']}，总计 **** {full_response['usage']['total_tokens']} ****")
        print(f"  - 工具轮数: {full_response.get('tool_rounds', 0)}")
        
        # 对比分析
        if rag_response['usage']['total_tokens'] > 0 and full_response['usage']['total_tokens'] > 0:
            token_reduction = (1 - rag_response['usage']['total_tokens'] / full_response['usage']['total_tokens']) * 100
            time_improvement = (1 - rag_time / full_time) * 100 if full_time > 0 else 0
            
            print(f"\n📈 性能对比:")
            print(f"  - 令牌减少: {token_reduction:.1f}%")
            print(f"  - 时间改善: {time_improvement:.1f}%")
    
    async def demo_chinese_optimization(self):
        """演示中文优化功能"""
        print("\n🇨🇳 演示中文优化功能")
        print("-" * 30)
        
        chinese_queries = [
            "帮我读取配置文件",
            "显示项目目录结构",
            "查找包含'测试'的文件"
        ]
        
        for query in chinese_queries:
            print(f"\n🔍 中文查询: {query}")
            
            # 检索相关工具（使用中文优化）
            if hasattr(self.chat_manager, 'vector_db'):
                relevant_tools = self.chat_manager.vector_db.query_tools(query, top_k=2)
                
                if relevant_tools:
                    print(f"✅ 检索到 {len(relevant_tools)} 个相关工具:")
                    for tool in relevant_tools:
                        tool_spec = tool.get("toolSpec", {})
                        name = tool_spec.get("name", "未知工具")
                        print(f"  - {name}")
                else:
                    print("❌ 未检索到相关工具")
    
    async def show_statistics(self):
        """显示统计信息"""
        print("\n📊 系统统计信息")
        print("-" * 30)
        
        # 配置信息
        config_info = self.chat_manager.get_config_info()
        print(f"LLM提供商: {config_info['llm_provider']}")
        print(f"LLM模型: {config_info['llm_model']}")
        print(f"向量数据库: {config_info['vector_db_type']}")
        print(f"嵌入模型: {config_info['embedding_model']}")
        
        # 向量数据库统计
        vector_stats = self.chat_manager.get_vector_db_stats()
        if 'error' not in vector_stats:
            print(f"工具数量: {vector_stats.get('tool_count', 0)}")
            print(f"数据库类型: {vector_stats.get('type', '未知')}")
        
        # 对话统计
        message_count = self.chat_manager.get_message_count()
        print(f"对话消息数: {message_count}")
    
    async def cleanup(self):
        """清理资源"""
        print("\n🧹 清理资源...")
        
        if self.chat_manager:
            await self.chat_manager.cleanup()
        
        print("✅ 清理完成")

async def run_quick_demo():
    """运行快速演示"""
    print("🚀 RAG-MCP 中国版快速演示")
    print("=" * 40)
    
    try:
        # 加载配置
        config = load_china_config()
        
        # 创建聊天管理器
        async with ChinaChatManager(config) as manager:
            await manager.initialize(init_mcp=False)  # 快速演示不启用MCP
            
            print("✅ 初始化完成")
            print(f"使用 {config.llm.provider}/{config.llm.model_name}")
            
            # 简单对话测试
            test_queries = [
                "你好！",
                "你能做什么？",
                "解释一下RAG技术"
            ]
            
            for query in test_queries:
                print(f"\n👤 用户: {query}")
                
                response = await manager.process_message(query, use_rag=False)
                
                print(f"🤖 助手: {response['text']}")
                print(f"📊 令牌: {response['usage']['total_tokens']}")
            
            print("\n🎉 快速演示完成！")
            
    except Exception as e:
        print(f"❌ 演示失败: {e}")
        print("\n请检查:")
        print("1. 是否设置了正确的API密钥")
        print("2. 网络连接是否正常")
        print("3. 依赖包是否安装完整")

async def run_interactive_demo():
    """运行交互式演示"""
    print("🎮 RAG-MCP 中国版交互式演示")
    print("=" * 40)
    print("输入 'quit' 退出演示")
    
    try:
        config = load_china_config()
        
        async with ChinaChatManager(config) as manager:
            await manager.initialize(init_mcp=False)
            
            print("✅ 系统就绪，开始对话...")
            
            while True:
                try:
                    user_input = input("\n👤 您: ")
                    
                    if user_input.lower() in ['quit', 'exit', '退出']:
                        break
                    
                    if not user_input.strip():
                        continue
                    
                    print("🤔 思考中...")
                    
                    response = await manager.process_message(user_input, use_rag=False)
                    
                    print(f"🤖 助手: {response['text']}")
                    
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"❌ 处理出错: {e}")
            
            print("\n👋 再见！")
            
    except Exception as e:
        print(f"❌ 演示失败: {e}")

def main():
    """主函数"""
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == 'quick':
            asyncio.run(run_quick_demo())
        elif mode == 'interactive':
            asyncio.run(run_interactive_demo())
        elif mode == 'full':
            demo = ChinaRAGMCPDemo()
            asyncio.run(demo.run_complete_demo())
        else:
            print("使用方法:")
            print("  python china_demo.py quick      # 快速演示")
            print("  python china_demo.py interactive # 交互式演示")
            print("  python china_demo.py full       # 完整演示")
    else:
        # # 默认运行快速演示
        # asyncio.run(run_quick_demo())
        demo = ChinaRAGMCPDemo()
        asyncio.run(demo.run_complete_demo())

if __name__ == "__main__":
    main() 