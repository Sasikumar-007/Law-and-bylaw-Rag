import os
import sys

print("Checking data folder...")
if not os.path.exists("data"):
    os.makedirs("data")
    print("ERROR: data/ folder was missing. Add PDFs to data/ and run again.")
    sys.exit(1)

pdf_files = [f for f in os.listdir("data") if f.lower().endswith(".pdf")]
if not pdf_files:
    print("ERROR: No PDF files found in data/ folder.")
    sys.exit(1)

print(f"Found {len(pdf_files)} PDF(s): {pdf_files}")

print("\nImporting libraries...")
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
print("Imports OK.")

print("\nLoading PDFs...")
documents = SimpleDirectoryReader("data").load_data()
print(f"Loaded {len(documents)} pages.")

print("\nChunking text...")
splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
nodes = splitter.get_nodes_from_documents(documents)
print(f"Created {len(nodes)} chunks.")

print("\nLoading embedding model (downloads ~90MB first time)...")
embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
print("Embedding model ready.")

print("\nBuilding ChromaDB vector store...")
chroma_client = chromadb.PersistentClient(path="vectorstore")
chroma_collection = chroma_client.get_or_create_collection("bylaws")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex(
    nodes,
    storage_context=storage_context,
    embed_model=embed_model,
    show_progress=True
)

print("\nSaving index...")
index.storage_context.persist(persist_dir="vectorstore")

print("\nFiles saved:")
for f in os.listdir("vectorstore"):
    print(f"  {f}")

print("\nDone! Now run: python chatbot.py")

