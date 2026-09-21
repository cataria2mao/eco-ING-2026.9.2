"""Windows 友好的服务启动入口。

psycopg 的异步模式无法在 Windows 默认的 ProactorEventLoop 上运行。uvicorn 0.47 会在
Windows 上强制使用 ProactorEventLoop（通过 loop_factory，而非 event loop policy），
因此这里在启动前把 uvicorn 的 asyncio loop factory 改写成 SelectorEventLoop。

用法：
    python run_server.py                # 默认 http://127.0.0.1:8000
    PORT=9000 python run_server.py
"""
import asyncio
import os
import sys


def _force_selector_loop_on_windows():
    if sys.platform != "win32":
        return
    import uvicorn.loops.asyncio as loops_asyncio

    # uvicorn 的 Config.get_loop_factory() 会以字符串导入该属性并按需调用，
    # 这里替换为始终返回 SelectorEventLoop 的工厂。
    loops_asyncio.asyncio_loop_factory = (
        lambda use_subprocess=False: asyncio.SelectorEventLoop
    )


if __name__ == "__main__":
    _force_selector_loop_on_windows()

    import uvicorn
    from server.web_server import app

    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    print(f"[SERVER] http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, loop="asyncio")
