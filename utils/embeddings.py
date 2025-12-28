from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever

# Simplified implementation without ContextualCompressionRetriever
# which is not available in current langchain version

def chunk_and_retrieve(
    ref_text: str,
    documents: list,
    top_k: int,
    ranker: str = "compression",
    chunk_size: int = 512,
    chunk_overlap: int = 64,
):
    if len(documents) > 0:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        texts = text_splitter.split_documents(documents)
        if ranker == "compression":
            # Use FAISS retriever directly without compression
            embeddings_model = HuggingFaceEmbeddings(model_name="thenlper/gte-small")
            retriever = FAISS.from_documents(texts, embeddings_model).as_retriever(
                search_kwargs={"k": top_k}
            )
            return retriever.invoke(ref_text)
        elif ranker == "bm25":
            retriever = BM25Retriever.from_documents(texts, k=top_k)
            return retriever.invoke(ref_text)
    else:
        return []
