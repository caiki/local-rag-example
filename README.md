# API RAG

LOCAL RAG

## Setup

Before you get started with Local RAG, ensure you have:

- A local [Ollama](https://github.com/ollama/ollama/) instance
- At least one model available within Ollama
    - `gemma3:2b` or `gemma3:latest` are good starter models
- Python

**WARNING:** This application is `untested` on Windows Subsystem for Linux. For best results, please utilize a Linux host if possible.

### Local
- `pip install pipenv && pipenv install`
- `pipenv shell && streamlit run main.py`

### Docker
- `docker compose up -d`

# Using Local RAG

## Quick Start

1. Set your Ollama endpoint and model under Settings
2. Upload your documents for processing
3. Once complete, ask questions based on your documents!

## Settings

All options within the RAG pipeline are exposed to users after toggling `Settings > Show Advanced Options`.

### Ollama

| Setting           | Description                                                            | Default                       |
|-------------------|------------------------------------------------------------------------|-------------------------------|
| Ollama Endpoint   | The location of your locally hosted Ollama API                         | http://localhost:11434        |
| Model             | Large language model to use what generating chat completions           |                               |
| System Prompt     | Initial system prompt used when initializing the LLM                   | (Please see source code)      |
| Top K             | Number of most similar documents to retrieve in response to a query    | 3                             |
| Chat Mode         | [Llama Index](#) chat mode to utilize during retrievals                | Best                          |

### Embeddings

| Setting           | Description                                                             | Default               |
|-------------------|-------------------------------------------------------------------------|-----------------------|
| Embedding Model   | Embedding model to be used for vectorize your files                     | bge-large-en-v1.5     |
| Chunk Size        | Improves embedding precision by focusing on smaller text portions       | 1024                  |


### Edit Prompt
Is possible adapt the prompt in components\page_state.py file. Into the "system_prompt" variable.