# AromaKiss

AromaKissRAG is a Retrieval-Augmented Generation (RAG) system designed to help a premium handmade candle brand maintain consistent voice and style when creating new content. The system works by loading existing Telegram channel posts, converting them into semantic embeddings using a multilingual sentence transformer model, and then using these embeddings to retrieve the most relevant past posts when generating new content. When a user requests a new post, post ideas, or research on a topic, the system finds similar examples from the brand's existing content library and uses them as context for a fine-tuned LLM model, which generates new content that matches the brand's authentic tone, style, and personality. The system is available both as an interactive command-line bot and as a FastAPI server with REST endpoints.

---
