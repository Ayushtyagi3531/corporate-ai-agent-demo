from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

_db = None

def get_vector_db():
    global _db
    if _db is None:
        print("📥 Loading embedding model and FAISS index into memory...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        # Ensure the path points correctly to your committed index directory
        _db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    return _db

def retrieve_context(query: str):
    db = get_vector_db() # Get index instance dynamically
    docs = db.similarity_search(query, k=3)
    return "\n".join([doc.page_content for doc in docs])
