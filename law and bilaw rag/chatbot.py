import os
import gradio as gr
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.llama_cpp import LlamaCPP
import chromadb

print("Loading vectorstore...")
chroma_client = chromadb.PersistentClient(path="vectorstore")
chroma_collection = chroma_client.get_or_create_collection("bylaws")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# ── Embedding model ──────────────────────────────────────────────────────────
embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# ── Local LLM (runs 100% offline) ────────────────────────────────────────────
MODEL_PATH = "models/mistral-7b-instruct-v0.2.Q2_K.gguf"

if not os.path.exists(MODEL_PATH):
    print(f"ERROR: Model not found at {MODEL_PATH}")
    print("Run this command first:")
    print('python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id=\'TheBloke/Mistral-7B-Instruct-v0.2-GGUF\', filename=\'mistral-7b-instruct-v0.2.Q2_K.gguf\', local_dir=\'models\')"')
    exit(1)

print("Loading local LLM model (takes ~20 seconds first time)...")
llm = LlamaCPP(
    model_path=MODEL_PATH,
    temperature=0.1,
    max_new_tokens=512,
    context_window=3900,
    generate_kwargs={},
    verbose=False,
)
print("LLM loaded!")

# ── Wire everything into LlamaIndex Settings ─────────────────────────────────
Settings.llm = llm
Settings.embed_model = embed_model

# ── Build index and query engine ─────────────────────────────────────────────
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_vector_store(
    vector_store,
    embed_model=embed_model
)
print("Index loaded successfully!")

query_engine = index.as_query_engine(
    similarity_top_k=4,
    response_mode="compact",
)

SYSTEM_PROMPT = """You are a civic legal assistant helping citizens understand
local government bylaws and regulations in Chennai, Tamil Nadu.
Always answer based on the documents provided.
Always cite which document and section your answer comes from.
If the answer is not in the documents, say:
'I could not find this in the uploaded bylaws — please contact CMDA or GCC directly.'
Never give personal legal advice."""

def ask_question(question: str, history: list) -> str:
    if not question.strip():
        return "Please enter a question."
    try:
        full_query = f"{SYSTEM_PROMPT}\n\nCitizen question: {question}"
        response = query_engine.query(full_query)

        answer = str(response).strip()

        # If response is empty, return a helpful message
        if not answer or answer.lower() in ["none", "empty", ""]:
            return ("I found relevant sections in the documents but could not "
                    "generate an answer. Please try rephrasing your question.")

        # Attach source citations
        sources = set()
        for node in response.source_nodes:
            fname = node.metadata.get("file_name", "Unknown document")
            page  = node.metadata.get("page_label", "")
            sources.add(f"{fname}{' p.' + page if page else ''}")
        if sources:
            answer += f"\n\n**Sources:** {', '.join(sources)}"

        return answer

    except Exception as e:
        return f"Error generating answer: {str(e)}"

# ── Gradio UI ─────────────────────────────────────────────────────────────────
demo = gr.ChatInterface(
    fn=ask_question,
    title="Municipal Bylaw Assistant",
    description="Ask questions about Chennai local laws, zoning rules, and civic regulations.",
    examples=[
        "What is the minimum plot size to build a house in Chennai?",
        "How many floors can I build on a residential plot?",
        "What is the setback rule for a house?",
        "Is rainwater harvesting mandatory?",
        "What documents do I need for a building permit?",
        "How do I file an RTI request?",
    ],
)

if __name__ == "__main__":
    demo.launch(share=False)