from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load FAISS vector database
vectorstore = FAISS.load_local(
    "vector_db",
    embeddings,
    allow_dangerous_deserialization=True
)
def retrieve_context(question):
    # Retrieve top 3 relevant text chunks along with metadata tracking
    docs = vectorstore.similarity_search(question, k=3)

    formatted_chunks = []
    for doc in docs:
        # Pull filename metadata to help Gemini context-map the text rules
        source = doc.metadata.get("source", "Unknown Document")
        chunk_text = f"[Source File: {source}]\n{doc.page_content}"
        formatted_chunks.append(chunk_text)

    context = "\n\n---\n\n".join(formatted_chunks)
    return context