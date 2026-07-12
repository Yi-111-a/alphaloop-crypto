# AlphaLoop-Crypto 只读监控面板

严格只读:只读取 `LOG/` 目录下的文件和 `state/portfolio_*.db`(sqlite,只读连接打开),
不 import 任何 `LOCKED/`/`ASSET/` 业务模块,没有任何写入/控制接口。
免责声明固定显示在页面顶部,不可折叠。

## 启动

```bash
cd alphaloop
python -m uvicorn webui.app:app --host 127.0.0.1 --port 8080
```

浏览器打开 `http://127.0.0.1:8080/`。

面板可以在交易系统(`main.py`)启动之前就先跑起来——`LOG/` 目录还没有任何数据时,
每个卡片会显示"等待系统启动"而不是报错。

## 依赖

`fastapi`、`uvicorn`(已在项目 `requirements.txt` 里)。前端不需要 npm/构建流程,
单页 HTML 直接从 CDN 加载 Chart.js。
