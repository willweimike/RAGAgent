# RAGAgent
# Agentic PDF RAG with LangGraph & Ollama

This project implements an **Agentic Retrieval-Augmented Generation (RAG)** system using **LangGraph**, **LangChain**, and **Ollama**. Unlike standard RAG pipelines, this implementation uses a ReAct agent that can "decide" when to search the document to provide more accurate, context-aware answers.

# Demo

https://github.com/user-attachments/assets/ba6da54f-8433-4f71-bcce-b0668dd4439d




## Features

* **Local LLM Integration**: Uses `Ollama` for both embeddings (`nomic-embed-text`) and reasoning (`qwen3:8b`), ensuring data privacy.
* **Agentic Reasoning**: Leverages `LangGraph`'s `create_react_agent` to handle complex queries and tool calling.
* **Persistent Vector Store**: Uses `Chroma` to store and retrieve document chunks efficiently.
* **Stateful Workflow**: Built on a `StateGraph` architecture for modular and scalable AI logic.
