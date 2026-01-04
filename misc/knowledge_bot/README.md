# Knowledge Bot

这是一个基于 RAG (Retrieval-Augmented Generation) 的多模态智能问答系统。它不仅支持网页知识抓取和问答，还针对国内网络环境进行了深度优化，集成了通义千问大模型和高质量的免费语音服务。

## 🌟 核心功能

1.  **智能爬虫**: 输入任意 URL，自动清洗并提取网页核心内容，去除广告干扰。
2.  **RAG 知识库**:
    *   **本地向量化**: 使用 `all-MiniLM-L6-v2` 模型进行本地 Embedding，保护隐私且零成本。
    *   **语义检索**: 基于 ChromaDB 实现高效的语义搜索。
    *   **大模型问答**: 集成 **阿里云 DashScope (通义千问)**，提供强大的中文理解与生成能力。
3.  **多模态交互**:
    *   **语音转文字 (ASR)**: 使用 **通义听悟 (Paraformer)** 模型，实现高精度中文语音识别。
    *   **文字转语音 (TTS)**: 使用 **Edge-TTS** (微软 Edge 引擎)，提供自然流畅的语音合成，无需 API Key。
4.  **可视化界面**: 基于 Gradio 构建的 Web UI，开箱即用。

## 🛠️ 技术栈

*   **UI 框架**: Gradio
*   **LLM 编排**: LangChain
*   **大模型**: Alibaba Qwen-Turbo (via DashScope)
*   **向量数据库**: ChromaDB
*   **语音识别 (ASR)**: DashScope Paraformer
*   **语音合成 (TTS)**: Edge-TTS
*   **Embedding**: SentenceTransformers

## 🚀 快速开始

### 1. 安装依赖

确保 Python 版本 >= 3.9，然后运行：

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

本项目主要依赖阿里云 DashScope 服务。

1.  前往 [阿里云 DashScope 控制台](https://dashscope.console.aliyun.com/) 开通 **通义千问** 和 **语音识别** 服务。
2.  获取 API Key。
3.  配置环境变量 (推荐) 或创建 `.env` 文件：

```bash
# macOS / Linux
export DASHSCOPE_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxx"

# Windows (PowerShell)
$env:DASHSCOPE_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxx"
```

*(注：系统保留了 OpenAI 接口兼容性，如果配置了 `OPENAI_API_KEY` 且未配置 `DASHSCOPE_API_KEY`，会自动降级使用 OpenAI 服务)*

### 3. 运行应用

```bash
python3 app.py
```

启动成功后，浏览器访问地址通常为：`http://localhost:7860`

## 📂 项目结构

```
knowledge_bot/
├── app.py              # Gradio Web 应用入口
├── crawler.py          # 网页爬虫与内容清洗
├── rag.py              # RAG 核心逻辑 (向量库管理 + 问答链)
├── voice.py            # 语音处理 (ASR: DashScope, TTS: Edge-TTS)
├── config.py           # 配置管理
├── requirements.txt    # 项目依赖
├── data/               # 爬取数据的本地存储目录
└── chroma_db/          # 向量数据库持久化目录
```

## 💡 使用指南

1.  **构建知识库**: 在 "Knowledge Base" 标签页输入目标网页 URL (例如一篇技术博客)，点击 "Crawl & Index"。
2.  **文字对话**: 切换到 "Chatbot" 标签页，就刚才爬取的内容进行提问。
3.  **语音对话**: 切换到 "Voice Bot" 标签页，点击麦克风说话，系统将以语音形式回答你。
