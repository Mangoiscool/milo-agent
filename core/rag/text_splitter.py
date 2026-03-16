"""文本切分器"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .base import Chunk, Document


@dataclass
class SplitConfig:
    """切分配置"""
    chunk_size: int = 500  # 每个切片的最大字符数
    chunk_overlap: int = 50  # 切片之间的重叠字符数
    separators: list[str] | None = None  # 分隔符优先级

    def __post_init__(self):
        if self.separators is None:
            self.separators = ["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]


class BaseTextSplitter(ABC):
    """文本切分器基类"""

    def __init__(self, config: SplitConfig | None = None):
        self.config = config or SplitConfig()

    @abstractmethod
    def split_text(self, text: str) -> list[str]:
        """将文本切分成多个片段"""
        pass

    def split_document(self, document: Document) -> list[Chunk]:
        """将文档切分成多个 Chunk"""
        texts = self.split_text(document.content)
        chunks = []
        for i, text in enumerate(texts):
            chunk = Chunk.from_document(document, text, i)
            chunks.append(chunk)
        return chunks

    def split_documents(self, documents: list[Document]) -> list[Chunk]:
        """批量切分文档"""
        all_chunks = []
        for doc in documents:
            chunks = self.split_document(doc)
            all_chunks.extend(chunks)
        return all_chunks


class RecursiveCharacterTextSplitter(BaseTextSplitter):
    """
    递归字符文本切分器

    按优先级尝试不同的分隔符，直到找到合适的切分点。
    这是 LangChain 中最常用的切分策略。
    """

    def split_text(self, text: str) -> list[str]:
        """递归切分文本"""
        return self._split_text_recursive(
            text,
            self.config.separators or ["\n\n", "\n", " ", ""]
        )

    def _split_text_recursive(self, text: str, separators: list[str]) -> list[str]:
        """递归切分实现"""
        if not text:
            return []

        # 如果文本已经足够小，直接返回
        if len(text) <= self.config.chunk_size:
            return [text.strip()] if text.strip() else []

        # 尝试按分隔符切分
        for i, separator in enumerate(separators):
            if separator == "":
                # 最后的手段：按字符数切分
                return self._split_by_size(text)

            if separator in text:
                # 按当前分隔符切分
                splits = self._split_by_separator(text, separator)

                # 如果切分后的片段都太大，尝试下一个分隔符
                if all(len(s) > self.config.chunk_size for s in splits):
                    continue

                # 合并小片段，确保不超过 chunk_size
                return self._merge_splits(splits, separators[i + 1:])

        # 所有分隔符都失败，按大小切分
        return self._split_by_size(text)

    def _split_by_separator(self, text: str, separator: str) -> list[str]:
        """按分隔符切分"""
        if separator == "\n\n":
            # 按空行切分（保留段落结构）
            return [s for s in text.split(separator) if s.strip()]
        elif separator == "\n":
            # 按行切分
            return [s for s in text.split(separator) if s.strip()]
        else:
            # 按其他分隔符切分
            parts = text.split(separator)
            # 保留分隔符（对于句子边界很重要）
            result = []
            for i, part in enumerate(parts):
                if part.strip():
                    result.append(part.strip())
            return result

    def _merge_splits(self, splits: list[str], remaining_separators: list[str]) -> list[str]:
        """合并切分片段，确保不超过 chunk_size"""
        merged = []
        current_chunk = ""
        overlap_text = ""

        for split in splits:
            # 考虑重叠
            test_chunk = overlap_text + split if not current_chunk else current_chunk + "\n" + split

            if len(test_chunk) <= self.config.chunk_size:
                current_chunk = test_chunk
            else:
                # 当前片段已满，保存并开始新片段
                if current_chunk:
                    merged.append(current_chunk.strip())
                    # 计算重叠部分
                    overlap_text = self._get_overlap(current_chunk)

                # 处理超大的单个片段
                if len(split) > self.config.chunk_size:
                    # 递归处理
                    sub_splits = self._split_text_recursive(split, remaining_separators)
                    merged.extend(sub_splits)
                    current_chunk = ""
                    overlap_text = ""
                else:
                    current_chunk = overlap_text + split if overlap_text else split

        # 添加最后一个片段
        if current_chunk:
            merged.append(current_chunk.strip())

        return [m for m in merged if m]

    def _get_overlap(self, text: str) -> str:
        """获取重叠部分"""
        if self.config.chunk_overlap <= 0:
            return ""

        # 从文本末尾取 overlap 长度的内容
        # 尝试在句子边界截断
        overlap = text[-self.config.chunk_overlap:]

        # 找到第一个完整句子/词的开始
        for sep in ["。", "！", "？", "\n", " "]:
            idx = overlap.find(sep)
            if idx != -1 and idx < len(overlap) - 1:
                overlap = overlap[idx + 1:]
                break

        return overlap

    def _split_by_size(self, text: str) -> list[str]:
        """按字符数强制切分"""
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.config.chunk_size

            # 尝试在句子边界切分
            if end < len(text):
                # 向后查找句子边界
                for sep in ["。", "！", "？", "\n", ".", "!", "?", " "]:
                    idx = text.rfind(sep, start, end + 50)  # 最多向后找 50 字符
                    if idx > start:
                        end = idx + 1
                        break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # 下一片考虑重叠
            start = end - self.config.chunk_overlap if end < len(text) else end

        return chunks


class MarkdownTextSplitter(RecursiveCharacterTextSplitter):
    """
    Markdown 专用切分器

    优先按标题层级切分，保持 Markdown 结构完整。
    """

    def __init__(self, config: SplitConfig | None = None):
        super().__init__(config)
        # Markdown 专用分隔符
        self.config.separators = [
            "\n## ",  # 二级标题
            "\n### ",  # 三级标题
            "\n#### ",  # 四级标题
            "\n##### ",  # 五级标题
            "\n###### ",  # 六级标题
            "\n\n",  # 段落
            "\n- ",  # 列表项
            "\n* ",  # 列表项
            "\n1. ",  # 有序列表
            "\n",  # 行
            " ",
            ""
        ]


class CodeTextSplitter(RecursiveCharacterTextSplitter):
    """
    代码专用切分器

    按代码结构（函数、类等）切分，保持代码完整性。
    """

    def __init__(self, config: SplitConfig | None = None, language: str = "python"):
        super().__init__(config)

        # 根据语言设置分隔符
        language_separators = {
            "python": ["\nclass ", "\ndef ", "\n\tdef ", "\n\n", "\n", " ", ""],
            "javascript": ["\nfunction ", "\nconst ", "\nlet ", "\nvar ", "\nclass ", "\n\n", "\n", " ", ""],
            "java": ["\npublic ", "\nprivate ", "\nclass ", "\n\n", "\n", " ", ""],
            "go": ["\nfunc ", "\ntype ", "\nvar ", "\nconst ", "\n\n", "\n", " ", ""],
        }

        self.config.separators = language_separators.get(
            language,
            ["\n\n", "\n", " ", ""]
        )


def create_splitter(
    splitter_type: str = "recursive",
    config: SplitConfig | None = None
) -> BaseTextSplitter:
    """
    工厂方法：创建文本切分器

    Args:
        splitter_type: 切分器类型 (recursive, markdown, code)
        config: 切分配置

    Returns:
        文本切分器实例
    """
    splitters = {
        "recursive": RecursiveCharacterTextSplitter,
        "markdown": MarkdownTextSplitter,
        "code": CodeTextSplitter,
    }

    splitter_class = splitters.get(splitter_type, RecursiveCharacterTextSplitter)
    return splitter_class(config)