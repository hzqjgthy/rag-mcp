#!/usr/bin/env python3
"""
向量数据库清理脚本
清理RAG-MCP中国版向量数据库中的所有工具，解决重复工具问题
"""

import os
import sys
import logging
import shutil
from pathlib import Path
from typing import Optional

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('cleanup.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

def cleanup_chromadb(persist_directory: str = "./china_vector_db") -> bool:
    """清理ChromaDB数据库"""
    try:
        print(f"🧹 正在清理ChromaDB数据库: {persist_directory}")
        
        if os.path.exists(persist_directory):
            # 删除整个目录
            shutil.rmtree(persist_directory)
            print(f"✅ 已删除ChromaDB目录: {persist_directory}")
            logger.info(f"ChromaDB目录已删除: {persist_directory}")
        else:
            print(f"ℹ️ ChromaDB目录不存在: {persist_directory}")
        
        return True
        
    except Exception as e:
        print(f"❌ 清理ChromaDB失败: {e}")
        logger.error(f"清理ChromaDB失败: {e}")
        return False

def cleanup_faiss(persist_directory: str = "./china_vector_db") -> bool:
    """清理FAISS数据库"""
    try:
        print(f"🧹 正在清理FAISS数据库: {persist_directory}")
        
        # FAISS相关文件
        faiss_files = [
            "faiss.index",
            "faiss_metadata.pkl",
            "faiss_metadata.json"
        ]
        
        cleaned_count = 0
        for filename in faiss_files:
            file_path = os.path.join(persist_directory, filename)
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"✅ 已删除文件: {file_path}")
                logger.info(f"FAISS文件已删除: {file_path}")
                cleaned_count += 1
        
        if cleaned_count == 0:
            print(f"ℹ️ 没有找到FAISS文件: {persist_directory}")
        else:
            print(f"✅ 共清理了 {cleaned_count} 个FAISS文件")
        
        return True
        
    except Exception as e:
        print(f"❌ 清理FAISS失败: {e}")
        logger.error(f"清理FAISS失败: {e}")
        return False

def cleanup_all_vector_dbs(persist_directory: str = "./china_vector_db") -> bool:
    """清理所有向量数据库"""
    try:
        print("🚀 开始清理所有向量数据库...")
        print("=" * 50)
        
        success = True
        
        # 清理ChromaDB
        if not cleanup_chromadb(persist_directory):
            success = False
        
        print()  # 空行分隔
        
        # 清理FAISS
        if not cleanup_faiss(persist_directory):
            success = False
        
        print()
        if success:
            print("🎉 所有向量数据库清理完成！")
            logger.info("所有向量数据库清理完成")
        else:
            print("⚠️ 部分清理操作失败，请查看日志")
            logger.warning("部分清理操作失败")
        
        return success
        
    except Exception as e:
        print(f"❌ 清理过程中发生错误: {e}")
        logger.error(f"清理过程中发生错误: {e}")
        return False

def get_vector_db_info(persist_directory: str = "./china_vector_db") -> dict:
    """获取向量数据库信息"""
    info = {
        "chromadb_exists": False,
        "chromadb_size": 0,
        "faiss_files": [],
        "total_size": 0
    }
    
    try:
        if os.path.exists(persist_directory):
            # 检查ChromaDB
            if os.path.isdir(persist_directory):
                info["chromadb_exists"] = True
                # 计算目录大小
                total_size = 0
                for dirpath, dirnames, filenames in os.walk(persist_directory):
                    for filename in filenames:
                        filepath = os.path.join(dirpath, filename)
                        if os.path.exists(filepath):
                            total_size += os.path.getsize(filepath)
                info["chromadb_size"] = total_size
                info["total_size"] += total_size
            
            # 检查FAISS文件
            faiss_files = ["faiss.index", "faiss_metadata.pkl", "faiss_metadata.json"]
            for filename in faiss_files:
                filepath = os.path.join(persist_directory, filename)
                if os.path.exists(filepath):
                    file_size = os.path.getsize(filepath)
                    info["faiss_files"].append({
                        "name": filename,
                        "size": file_size
                    })
                    info["total_size"] += file_size
    
    except Exception as e:
        logger.error(f"获取向量数据库信息失败: {e}")
    
    return info

def format_size(size_bytes: int) -> str:
    """格式化文件大小"""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.2f} {size_names[i]}"

def show_vector_db_status(persist_directory: str = "./china_vector_db"):
    """显示向量数据库状态"""
    print("📊 向量数据库状态")
    print("-" * 30)
    
    info = get_vector_db_info(persist_directory)
    
    # ChromaDB状态
    if info["chromadb_exists"]:
        print(f"ChromaDB: ✅ 存在 ({format_size(info['chromadb_size'])})")
    else:
        print("ChromaDB: ❌ 不存在")
    
    # FAISS状态
    if info["faiss_files"]:
        print("FAISS文件:")
        for file_info in info["faiss_files"]:
            print(f"  - {file_info['name']}: {format_size(file_info['size'])}")
    else:
        print("FAISS文件: ❌ 不存在")
    
    # 总大小
    if info["total_size"] > 0:
        print(f"总大小: {format_size(info['total_size'])}")
    else:
        print("总大小: 0 B")

def interactive_cleanup():
    """交互式清理"""
    print("🇨🇳 RAG-MCP 向量数据库清理工具")
    print("=" * 40)
    
    # 从环境变量或默认值获取数据库路径
    from china_config import load_china_config
    try:
        config = load_china_config()
        persist_directory = config.vector_db.persist_directory
        print(f"📁 数据库路径: {persist_directory}")
    except Exception:
        persist_directory = "./china_vector_db"
        print(f"📁 使用默认路径: {persist_directory}")
    
    print()
    
    # 显示当前状态
    show_vector_db_status(persist_directory)
    
    print()
    
    # 询问用户是否继续
    while True:
        choice = input("是否要清理向量数据库？(y/n): ").lower().strip()
        if choice in ['y', 'yes', '是', '确定']:
            break
        elif choice in ['n', 'no', '否', '取消']:
            print("👋 取消清理操作")
            return
        else:
            print("请输入 y 或 n")
    
    print()
    
    # 执行清理
    success = cleanup_all_vector_dbs(persist_directory)
    
    print()
    
    # 显示清理后状态
    if success:
        print("📊 清理后状态:")
        show_vector_db_status(persist_directory)

def main():
    """主函数"""
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        # 获取数据库路径
        persist_directory = "./china_vector_db"
        if len(sys.argv) > 2:
            persist_directory = sys.argv[2]
        
        if command == "clean":
            # 直接清理
            cleanup_all_vector_dbs(persist_directory)
        elif command == "status":
            # 显示状态
            show_vector_db_status(persist_directory)
        elif command == "chromadb":
            # 只清理ChromaDB
            cleanup_chromadb(persist_directory)
        elif command == "faiss":
            # 只清理FAISS
            cleanup_faiss(persist_directory)
        elif command == "help":
            # 显示帮助
            print("RAG-MCP 向量数据库清理工具")
            print()
            print("用法:")
            print("  python cleanup_vector_db.py                    # 交互式清理")
            print("  python cleanup_vector_db.py clean [路径]       # 直接清理所有")
            print("  python cleanup_vector_db.py chromadb [路径]    # 只清理ChromaDB")
            print("  python cleanup_vector_db.py faiss [路径]       # 只清理FAISS")
            print("  python cleanup_vector_db.py status [路径]      # 显示状态")
            print("  python cleanup_vector_db.py help              # 显示帮助")
            print()
            print("示例:")
            print("  python cleanup_vector_db.py clean")
            print("  python cleanup_vector_db.py status ./my_vector_db")
        else:
            print(f"❌ 未知命令: {command}")
            print("使用 'python cleanup_vector_db.py help' 查看帮助")
    else:
        # 默认交互式模式
        interactive_cleanup()

if __name__ == "__main__":
    main() 