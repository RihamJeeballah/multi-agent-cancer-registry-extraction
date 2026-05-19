# === memory/vector_store.py ===

from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings  # ✅ Updated import
from langchain_core.documents import Document

def create_vector_store(texts, metadatas):
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")

    documents = [
        Document(page_content=text, metadata={**metadata, "position": idx})
        for idx, (text, metadata) in enumerate(zip(texts, metadatas))
    ]

    print(f"✅ Vectorstore created with {len(documents)} chunks")  # ✅ Added debug print

    return FAISS.from_documents(documents, embedding_model)

def retrieve_chunks(vectorstore, query, top_k=5, filter_focus=None):
    if filter_focus:
        results = vectorstore.similarity_search(query, k=top_k, filter={"focus": filter_focus})
    else:
        results = vectorstore.similarity_search(query, k=top_k)

    sorted_results = sorted(results, key=lambda d: d.metadata.get("position", 0), reverse=True)
    return sorted_results