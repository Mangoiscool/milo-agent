"""
长期记忆系统演示

演示内容：
1. 添加记忆到长期存储
2. 跨会话语义检索
3. 重要性评分功能
4. 混合记忆系统使用

运行方式：
    python examples/08_long_term_memory_demo.py
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from datetime import datetime

from core.llm.base import Message, Role
from core.memory.long_term import LongTermMemory, RetrievedMemory
from core.memory.hybrid import HybridMemory
from core.memory.short_term import ShortTermMemory
from core.rag.embeddings import create_embedding, BaseEmbedding
from core.logger import get_logger


def print_separator(title: str = ""):
    """打印分隔线"""
    if title:
        print(f"\n{'='*20} {title} {'='*20}\n")
    else:
        print(f"\n{'='*60}\n")


def test_embedding_connection(embedding: BaseEmbedding) -> bool:
    """测试 Embedding 连接"""
    print("📡 测试 Embedding 连接...")
    try:
        test_vec = embedding.embed("测试文本")
        print(f"   ✅ Embedding 正常，维度: {len(test_vec)}")
        return True
    except Exception as e:
        print(f"   ❌ Embedding 失败: {e}")
        return False


def demo_long_term_memory():
    """演示长期记忆功能"""
    print_separator("长期记忆演示")

    # 1. 初始化 Embedding
    print("📦 初始化 Embedding 模型...")
    try:
        # 尝试使用 Ollama
        embedding = create_embedding("ollama", model="qwen3-embedding:0.6b")
        if not test_embedding_connection(embedding):
            print("   尝试使用备用 Embedding...")
            # 如果 Ollama 不可用，尝试其他选项
            embedding = create_embedding("sentence-transformers")
            if not test_embedding_connection(embedding):
                print("   ❌ 无法初始化 Embedding，请确保 Ollama 运行或安装 sentence-transformers")
                return
    except Exception as e:
        print(f"   ❌ 初始化 Embedding 失败: {e}")
        return

    # 2. 初始化长期记忆
    print("\n🧠 初始化长期记忆...")
    persist_dir = project_root / "demo_memory"
    memory = LongTermMemory(
        embedding_model=embedding,
        session_id="demo_session_001",
        persist_directory=str(persist_dir),
        default_top_k=5
    )
    print(f"   ✅ 长期记忆已初始化")
    print(f"   📁 持久化目录: {persist_dir}")

    # 3. 添加测试记忆
    print("\n📝 添加测试记忆...")

    test_messages = [
        ("user", "我叫 Mango，是一名系统架构师，专注于分布式系统设计"),
        ("user", "我喜欢 Python 编程，特别是 AI 和机器学习领域"),
        ("assistant", "很高兴认识你！我会记住你的背景和兴趣。"),
        ("user", "记住我的邮箱是 mango@example.com"),
        ("user", "【重要】下周三下午 3 点有个重要的项目评审会议"),
        ("user", "我最近在学习 RAG 和向量数据库技术"),
        ("assistant", "RAG 是很有前景的技术方向，结合向量数据库可以实现高效的语义检索。"),
        ("user", "我家的地址是北京市朝阳区xxx路xxx号"),
    ]

    for role_str, content in test_messages:
        role = Role.USER if role_str == "user" else Role.ASSISTANT
        msg = Message(role=role, content=content)
        memory.add(msg)

    print(f"   ✅ 已添加 {len(test_messages)} 条记忆")
    print(f"   📊 当前记忆总数: {memory.count()}")

    # 4. 测试语义检索
    print_separator("语义检索测试")

    queries = [
        ("用户的职业", "职业"),
        ("编程爱好", "编程"),
        ("联系方式", "联系方式"),
        ("会议安排", "会议"),
        ("学习方向", "学习"),
        ("地址信息", "地址"),
    ]

    for query, desc in queries:
        print(f"🔍 查询: \"{query}\" ({desc})")
        results = memory.retrieve(query, top_k=3)

        if results:
            for i, r in enumerate(results, 1):
                print(f"   [{i}] 相似度: {r.similarity:.3f}")
                print(f"       内容: {r.entry.content[:60]}...")
                print(f"       重要性: {r.entry.importance:.2f}")
        else:
            print("   (无结果)")
        print()

    # 5. 测试跨会话检索
    print_separator("跨会话检索测试")

    print("🔄 创建新会话...")
    new_session_id = "demo_session_002"
    memory2 = LongTermMemory(
        embedding_model=embedding,
        session_id=new_session_id,
        persist_directory=str(persist_dir)  # 使用相同的持久化目录
    )

    # 添加一些新会话的记忆
    print("   添加新会话记忆...")
    memory2.add(Message(role=Role.USER, content="今天天气很好，我想去公园散步"))
    memory2.add(Message(role=Role.USER, content="我最近在研究如何优化向量检索性能"))

    print(f"\n📊 新会话记忆数: {memory2.count()}")

    # 跨会话检索
    print("\n🔍 跨会话检索 '向量检索':")
    results = memory2.retrieve("向量检索", top_k=5)

    for i, r in enumerate(results, 1):
        session_info = r.entry.session_id[:8] + "..."
        print(f"   [{i}] 会话: {session_info} | 相似度: {r.similarity:.3f}")
        print(f"       内容: {r.entry.content[:50]}...")

    # 6. 测试重要性评分
    print_separator("重要性评分测试")

    print("📊 查看各记忆的重要性分数:")
    all_results = memory.retrieve(" Mango Python 会议 地址 邮箱", top_k=20)

    # 按重要性排序
    sorted_results = sorted(all_results, key=lambda x: x.entry.importance, reverse=True)

    for i, r in enumerate(sorted_results[:5], 1):
        print(f"   [{i}] 重要性: {r.entry.importance:.2f}")
        print(f"       内容: {r.entry.content[:50]}...")
        print()

    # 7. 清理（可选）
    print_separator()
    print("🧹 清理演示数据...")
    # memory.clear()  # 取消注释以清空演示数据
    print("   (保留数据用于后续测试)")


def demo_hybrid_memory():
    """演示混合记忆功能"""
    print_separator("混合记忆演示")

    # 初始化
    print("📦 初始化组件...")
    try:
        embedding = create_embedding("ollama", model="qwen3-embedding:0.6b")
        # 测试连接
        embedding.embed("test")
    except Exception as e:
        print(f"   ❌ Embedding 初始化失败: {e}")
        print("   请确保 Ollama 服务正在运行")
        return

    persist_dir = project_root / "demo_memory"

    # 创建混合记忆
    print("\n🧠 创建混合记忆系统...")
    hybrid = HybridMemory(
        short_term=ShortTermMemory(max_messages=10),
        long_term=LongTermMemory(
            embedding_model=embedding,
            session_id="hybrid_demo_001",
            persist_directory=str(persist_dir)
        )
    )

    # 添加对话
    print("\n📝 模拟对话...")
    conversations = [
        (Role.USER, "你好，我是 Mango"),
        (Role.ASSISTANT, "你好 Mango！很高兴认识你。"),
        (Role.USER, "我喜欢 Python 和 AI"),
        (Role.ASSISTANT, "Python 是 AI 开发的首选语言！"),
        (Role.USER, "我正在做一个 RAG 项目"),
        (Role.ASSISTANT, "RAG 项目很有趣，结合检索和生成能力。"),
    ]

    for role, content in conversations:
        hybrid.add(Message(role=role, content=content))

    print(f"   短期记忆: {hybrid.count()} 条")
    print(f"   长期记忆: {hybrid.count_long_term()} 条")

    # 构建上下文
    print_separator("构建上下文")

    query = "我的爱好是什么？"
    print(f"🔍 用户提问: \"{query}\"")
    print("\n📚 构建的上下文:")

    context = hybrid.build_context(query, long_term_limit=3)

    for i, msg in enumerate(context, 1):
        role = msg.role.value
        content_preview = msg.content[:80] if msg.content else "(空)"
        if len(msg.content or "") > 80:
            content_preview += "..."
        print(f"   [{i}] {role}: {content_preview}")

    # 记忆统计
    print_separator("记忆统计")
    stats = hybrid.get_memory_stats()
    print(f"📊 短期记忆: {stats['short_term']['count']}/{stats['short_term']['max_messages']}")
    print(f"📊 长期记忆: {stats['long_term']['count']} 条")


def main():
    """主函数"""
    print("\n" + "="*60)
    print("   长期记忆系统演示 - LongTermMemory Demo")
    print("="*60)

    # 演示 1: 长期记忆
    demo_long_term_memory()

    # 演示 2: 混合记忆
    demo_hybrid_memory()

    print_separator()
    print("✅ 演示完成！")
    print("\n💡 提示:")
    print("   - 长期记忆会自动持久化到 demo_memory/ 目录")
    print("   - 跨会话检索需要使用相同的 persist_directory")
    print("   - 确保 Ollama 服务运行中 (ollama serve)")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()