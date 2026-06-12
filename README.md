# Municipal Bylaw RAG Chatbot

An AI-powered chatbot that helps citizens understand local government
bylaws and regulations in Chennai, Tamil Nadu using RAG (Retrieval-Augmented Generation).

## What it does
- Answers plain-language questions about Chennai municipal laws
- Cites the exact PDF document and page number
- Supports Tamil, Hindi, and English queries
- Runs 100% offline using local LLM (Mistral 7B)

## Tech Stack
| Tool | Purpose |
|---|---|
| Python 3.11 | Core language |
| LlamaIndex | RAG orchestration |
| ChromaDB | Vector storage |
| Sentence-BERT (multilingual) | Text embeddings |
| Mistral 7B (GGUF) | Local LLM |
| Gradio | Web UI |

## Setup

### 1. Install dependencies
pip install llama-index llama-index-core
pip install llama-index-embeddings-huggingface
pip install llama-index-vector-stores-chroma
pip install llama-index-llms-llama-cpp
pip install llama-index-readers-file
pip install chromadb sentence-transformers gradio python-dotenv pypdf2

### 2. Download the LLM model
python -c "from huggingface_hub import hf_hub_download; \
hf_hub_download(repo_id='TheBloke/Mistral-7B-Instruct-v0.2-GGUF', \
filename='mistral-7b-instruct-v0.2.Q2_K.gguf', local_dir='models')"

### 3. Add PDF documents
Place government PDF files into the `data/` folder.
Recommended: TNCDBR 2019, CMDA bylaws, RTI Act 2005

### 4. Build the vector index
python ingest.py

### 5. Run the chatbot
python chatbot.py
# Opens at http://127.0.0.1:7860

## Data Sources
- [CMDA Chennai](https://cmdachennai.gov.in)
- [Greater Chennai Corporation](https://chennaicorporation.gov.in)
- [TNCDBR 2019](https://www.cmdachennai.gov.in/TNCDBR2019.html)
- [RTI Act](https://rti.gov.in)

## Project Structure
law-and-bylaw-rag/
├── data/          ← Add PDF documents here
├── models/        ← LLM model downloaded here (not in git)
├── vectorstore/   ← Auto-created by ingest.py (not in git)
├── ingest.py      ← Builds the vector index
├── chatbot.py     ← Runs the chatbot UI
└── .env           ← API keys (not in git)

## Built by
Sasikumar Baskar — Civic Tech / AI Developer
