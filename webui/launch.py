#!/usr/bin/env python3
"""
Milo Agent Web UI Launcher
启动 Web UI 服务器
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="启动 Milo Agent Web UI")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址 (默认: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="监听端口 (默认: 8000)")
    parser.add_argument("--reload", action="store_true", help="启用热重载 (开发模式)")

    args = parser.parse_args()

    # 检查是否安装了 FastAPI
    try:
        import uvicorn
        from webui.server import app
    except ImportError as e:
        print("错误: 缺少必要的依赖包")
        print("请运行: pip install 'milo-agent[webui]'")
        print(f"详细信息: {e}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  Milo Agent Web UI")
    print("=" * 60)
    print(f"  访问地址: http://{args.host if args.host != '0.0.0.0' else 'localhost'}:{args.port}")
    print("=" * 60)
    print("  按 Ctrl+C 停止服务器")
    print("=" * 60 + "\n")

    try:
        uvicorn.run(
            "webui.server:app",
            host=args.host,
            port=args.port,
            reload=args.reload
        )
    except KeyboardInterrupt:
        print("\n\n服务器已停止")


if __name__ == "__main__":
    main()
