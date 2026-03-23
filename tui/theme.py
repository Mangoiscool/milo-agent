"""Modern TUI Theme System - Claude Code inspired"""

from textual.theme import Theme

# Claude Code 风格主题
CLAUDE_THEME = Theme(
    name="claude",
    primary="#D97757",        # Claude 橙色
    secondary="#6366F1",      # 靛蓝色
    warning="#F59E0B",
    error="#EF4444",
    success="#10B981",
    background="#1E1E2E",     # 深色背景
    surface="#2D2D44",        # 表面色
    panel="#363653",          # 面板色
    dark=True,
    variables={
        "user-bubble": "#2563EB",      # 用户消息蓝色
        "assistant-bubble": "#374151",  # 助手消息灰色
        "text-primary": "#F3F4F6",
        "text-secondary": "#9CA3AF",
        "text-muted": "#6B7280",
        "border-subtle": "#4B5563",
        "border-focus": "#D97757",
    }
)

# 浅色主题
LIGHT_THEME = Theme(
    name="light",
    primary="#D97757",
    secondary="#6366F1",
    warning="#F59E0B",
    error="#EF4444",
    success="#10B981",
    background="#FAFAFA",
    surface="#FFFFFF",
    panel="#F3F4F6",
    dark=False,
    variables={
        "user-bubble": "#3B82F6",
        "assistant-bubble": "#F3F4F6",
        "text-primary": "#1F2937",
        "text-secondary": "#4B5563",
        "text-muted": "#9CA3AF",
        "border-subtle": "#E5E7EB",
        "border-focus": "#D97757",
    }
)

# 主题注册表
THEMES = {
    "claude": CLAUDE_THEME,
    "light": LIGHT_THEME,
}

def get_theme(name: str) -> Theme:
    """获取主题"""
    return THEMES.get(name, CLAUDE_THEME)
