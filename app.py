import os
import streamlit as st
import chromadb
from llama_index.core import VectorStoreIndex, StorageContext, get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.settings import Settings

# --- Configuration (Must match ingest.py) ---
PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "company_policy_rag"
LLM_MODEL = "gemini-2.5-flash"
EMBED_MODEL = "models/embedding-001"

# --- Initialization and Loading ---

@st.cache_resource(show_spinner=True)
def get_query_engine():
    """Loads the stored index and sets up the RAG query engine."""
    
    # 1. Initialize Settings (LLM and Embeddings)
    if "GOOGLE_API_KEY" not in os.environ:
        st.error("FATAL: GOOGLE_API_KEY environment variable not set. Please set it in your terminal.")
        st.stop()
        
    # Configure the LLM and Embeddings
    Settings.llm = GoogleGenAI(model=LLM_MODEL)
    Settings.embed_model = GoogleGenAIEmbedding(model=EMBED_MODEL)

    # 2. Load ChromaDB Client and Collection
    try:
        db = chromadb.PersistentClient(path=PERSIST_DIR)
        chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
    except Exception as e:
        st.error(f"FATAL: Could not load the ChromaDB index. Did you run python ingest.py? Error: {e}")
        st.stop()

    # 3. Create Storage Context and Load Index
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Load the index from the persisted ChromaDB
    with st.spinner("Loading Knowledge Base..."):
        index = VectorStoreIndex.from_vector_store(
            vector_store,
            storage_context=storage_context,
        )
    
    # OPTIMIZATION: Configure query engine for streaming
    # Set streaming=True when building the query engine
    query_engine = index.as_query_engine(
        streaming=True, # <--- ENABLE STREAMING
        similarity_top_k=3,
    )
    
    return query_engine

# --- Streamlit UI ---

st.set_page_config(page_title="Knowledge Base Agent", layout="wide")

st.title("📚 Business Operations Knowledge Base Agent")
st.subheader("Ask me anything about company policies (HR, IT, Finance)!", divider='blue')

# Initialize the chat history if it doesn't exist
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am ready to answer your questions based on the internal documents."}
    ]

# Load the query engine (cached to run only once)
query_engine = get_query_engine()

# Main chat logic
if prompt := st.chat_input("Your question:"):
    st.session_state.messages.append({"role": "user", "content": prompt})

# Display all messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If the last message was from the user, generate a response
if st.session_state.messages[-1]["role"] == "user":
    with st.chat_message("assistant"):
        with st.spinner("Searching and retrieving context..."): # Change spinner text to reflect RAG
            
            # --- RAG QUERY EXECUTION with Streaming ---
            # Calling query now returns a StreamingResponse object
            response_stream = query_engine.query(prompt)
            
            # Use st.write_stream to display tokens as they arrive
            full_response = ""
            # Iterate over the response generator to display tokens
            for token in response_stream.response_gen:
                full_response += token
                st.write(full_response) # Display progressively
                
            # --- Display Sources (Great for Jury Review!) ---
            if response_stream.source_nodes:
                 with st.expander("🔍 Sources Used (for Audit/Verification)"):
                     for i, node in enumerate(response_stream.source_nodes):
                         st.markdown(f"**Source {i+1}** (Relevance Score: {node.score:.2f}, File: {node.metadata.get('file_name', 'N/A')}):")
                         st.caption(node.get_content()[:300] + "...")
            
            # Add the final, full response text to the session state
            st.session_state.messages.append({"role": "assistant", "content": full_response})
