import os
import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.settings import Settings

# --- Configuration for Scalability ---
PERSIST_DIR = "./chroma_db"
DATA_DIR = "./data"
COLLECTION_NAME = "company_policy_rag"
LLM_MODEL = "gemini-2.5-flash"
EMBED_MODEL = "models/embedding-001"

# OPTIMIZATION: Fine-tuned chunking for better RAG quality and retrieval speed
CHUNK_SIZE = 512    # Smaller chunks retrieve context more precisely
CHUNK_OVERLAP = 20  # Overlap ensures context isn't lost at the chunk boundary

def initialize_settings():
    """Sets the LLM, Embedding Model, and Chunking Strategy globally."""
    if "GOOGLE_API_KEY" not in os.environ:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")

    # 1. Configure LLM and Embeddings
    Settings.llm = GoogleGenAI(model=LLM_MODEL)
    Settings.embed_model = GoogleGenAIEmbedding(model=EMBED_MODEL)
    
    # 2. Configure Chunking Strategy
    Settings.text_splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    
    print(f"LlamaIndex configured with LLM: {LLM_MODEL}, Embeddings: {EMBED_MODEL}")
    print(f"Optimal Chunking set: Size={CHUNK_SIZE}, Overlap={CHUNK_OVERLAP}")

def create_knowledge_base():
    """Loads documents, applies metadata, indexes, and stores to ChromaDB."""
    initialize_settings()

    # 1. Load Documents & Attach Metadata
    print(f"Loading documents from {DATA_DIR}...")
    # Use SimpleDirectoryReader with full_path=True to ensure metadata includes the filename/path
    documents = SimpleDirectoryReader(
        input_dir=DATA_DIR, 
        recursive=True
    ).load_data()
    print(f"Loaded {len(documents)} documents.")

    # 2. Setup ChromaDB Client and Collection
    db = chromadb.PersistentClient(path=PERSIST_DIR)
    chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 3. Create the Index (uses the optimized Settings)
    print("Starting optimized document indexing (chunking and embedding)...")
    
    # Using the IngestionPipeline for better control over the node processing flow
    pipeline = IngestionPipeline(
        transformations=[
            Settings.text_splitter,
            Settings.embed_model,
        ],
        vector_store=vector_store,
    )
    
    # Run the pipeline to ingest the documents and save to the Vector Store
    pipeline.run(documents=documents)
    
    print("\n Optimized Indexing complete!")
    print(f"Knowledge Base stored persistently in the '{PERSIST_DIR}' directory.")

if __name__ == "__main__":
    create_knowledge_base()
