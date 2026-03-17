"""RAG Agent 演示

展示 RAG Agent 的基本用法。
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import asyncio

from agents.rag import RAGAgent
from core.llm.factory import create_llm
from core.rag.embeddings import create_embedding
from core.rag.text_splitter import SplitConfig


def demo_rag_agent():
    """RAG Agent 演示"""
    print("=" * 60)
    print("RAG Agent 演示")
    print("=" * 60)

    # 1. 创建 LLM 和 Embedding
    print("\n[1] 初始化 LLM 和 Embedding 模型...")
    llm = create_llm("ollama", model="qwen3.5:4b", think=False)
    embedding = create_embedding("ollama")  # 默认使用 qwen3-embedding:0.6b

    # 2. 创建 RAG Agent
    print("[2] 创建 RAG Agent...")
    agent = RAGAgent(
        llm=llm,
        embedding_model=embedding,
        persist_directory="./demo_knowledge_base",
        knowledge_base_name="demo",
        splitter_config=SplitConfig(chunk_size=500, chunk_overlap=50),
        top_k=3
    )

    # 3. 添加知识
    print("[3] 添加知识到知识库...")

    # 示例文本
    sample_text = """
# RAG 技术简介

RAG（Retrieval-Augmented Generation）是一种结合检索和生成的AI技术。

## 核心组件

RAG 系统包含以下核心组件：

1. **文档加载器（Document Loader）**
   - 负责解析各种格式的文档
   - 支持 PDF、Markdown、Word、Excel 等格式
   - 将文档转换为统一的 Document 对象

2. **文本切分器（Text Splitter）**
   - 将长文档切分成适当大小的片段
   - 支持递归切分、Markdown 切分、代码切分
   - 保持语义完整性

3. **Embedding 模型**
   - 将文本转换为向量表示
   - 语义相似的文本在向量空间中距离更近
   - 支持本地模型和 API 服务

4. **向量存储（Vector Store）**
   - 存储文本向量和元数据
   - 支持相似度检索
   - 常用 ChromaDB、FAISS 等

5. **检索器（Retriever）**
   - 根据查询检索相关文档
   - 支持相似度检索、MMR 多样性检索
   - 可以添加过滤条件

## 工作流程

RAG 的工作流程如下：

1. 文档入库：加载 → 切分 → 嵌入 → 存储
2. 用户提问：检索相关文档 → 构建上下文 → LLM 生成回答

## 优势

- 可以处理大量知识，突破 LLM 上下文限制
- 知识可更新，无需重新训练模型
- 可以引用来源，提高可信度
- 数据隐私，敏感数据可以本地处理

## 应用场景

- 企业知识库问答
- 技术文档助手
- 客户服务机器人
- 法律文档分析
- 医疗知识问答
"""

    chunk_count = agent.add_text(sample_text, source="rag_intro.md")
    print(f"   添加了 {chunk_count} 个文本块")

    # 4. 查看知识库信息
    print("\n[4] 知识库信息：")
    stats = agent.get_stats()
    print(f"   知识库名称: {stats['knowledge_base_name']}")
    print(f"   文档数量: {stats['document_count']}")
    print(f"   文档来源: {stats['sources']}")
    print(f"   向量维度: {stats['embedding_dimension']}")

    # 5. 问答演示
    print("\n[5] 问答演示：")
    print("-" * 40)

    questions = [
        "RAG 是什么？",
        "RAG 的核心组件有哪些？",
        "RAG 有什么优势？",
    ]

    for question in questions:
        print(f"\n问: {question}")
        answer = agent.chat(question)
        print(f"答: {answer}")
        print("-" * 40)

    # 6. 清理
    print("\n[6] 清理知识库...")
    agent.clear_knowledge_base()
    print("   已清空")

    print("\n演示完成！")


async def demo_async():
    """异步演示"""
    print("=" * 60)
    print("RAG Agent 异步演示")
    print("=" * 60)

    llm = create_llm("ollama", model="qwen3.5:4b", think=False)
    embedding = create_embedding("ollama")  # 默认使用 qwen3-embedding:0.6b

    agent = RAGAgent(
        llm=llm,
        embedding_model=embedding,
        persist_directory="./demo_knowledge_base_async"
    )

    # 添加知识
    agent.add_text("Python 是一种流行的编程语言，以其简洁的语法著称。", source="python_intro.txt")

    # 异步问答
    answer = await agent.achat("Python 是什么？")
    print(f"回答: {answer}")

    # 清理
    agent.clear_knowledge_base()


if __name__ == "__main__":
    # 同步演示
    demo_rag_agent()

    # 异步演示（可选）
    # asyncio.run(demo_async())