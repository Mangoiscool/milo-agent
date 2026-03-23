"""Text Formatting Utilities for TUI"""

import re
from typing import List, Tuple
from rich.text import Text
from rich.syntax import Syntax
from rich.console import Group
from rich.panel import Panel
from rich.markdown import Markdown


def extract_code_blocks(text: str) -> List[Tuple[str, str, str]]:
    """
    提取代码块
    返回: [(完整匹配, 语言, 代码内容), ...]
    """
    pattern = r'```(\w+)?\n(.*?)```'
    matches = []
    for match in re.finditer(pattern, text, re.DOTALL):
        full = match.group(0)
        lang = match.group(1) or "text"
        code = match.group(2)
        matches.append((full, lang, code))
    return matches


def format_message_content(text: str) -> Group:
    """
    格式化消息内容，支持 Markdown 和代码块
    """
    if not text:
        return Group(Text(""))

    elements = []
    remaining = text

    # 处理代码块
    code_blocks = extract_code_blocks(text)
    for full_match, lang, code in code_blocks:
        # 代码块前的文本
        before = remaining.split(full_match, 1)[0]
        if before.strip():
            elements.append(Markdown(before.strip()))

        # 代码块
        if code.strip():
            syntax = Syntax(
                code.strip(),
                lang,
                theme="monokai",
                line_numbers=True,
                word_wrap=False
            )
            elements.append(Panel(
                syntax,
                border_style="dim",
                padding=(1, 1)
            ))

        # 更新剩余文本
        remaining = remaining.split(full_match, 1)[1] if full_match in remaining else ""

    # 处理剩余文本
    if remaining.strip():
        elements.append(Markdown(remaining.strip()))

    return Group(*elements) if elements else Group(Text(text))


def truncate_text(text: str, max_length: int = 50) -> str:
    """截断文本"""
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


def format_timestamp(dt) -> str:
    """格式化时间戳"""
    if dt is None:
        return ""
    return dt.strftime("%H:%M")


def format_tokens(count: int) -> str:
    """格式化 Token 数"""
    if count < 1000:
        return f"{count}"
    return f"{count/1000:.1f}k"


def estimate_tokens(text: str) -> int:
    """估算 Token 数（粗略）"""
    # 中文按字符，英文按单词
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    english_words = len(re.findall(r'[a-zA-Z]+', text))
    return chinese_chars + english_words
