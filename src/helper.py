from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List
from langchain.schema import Document

# Extract text from pdf file
def load_pdf_files(data):
    loader = DirectoryLoader(
        data, # path
        glob="*.pdf", # loads all the file with .pdf extension
        loader_cls=PyPDFLoader
    )

    documents = loader.load()
    return documents

# Just taking source and page content
def filter_to_docs_minimal(docs: List[Document]) -> List[Document]:
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content = doc.page_content,
                metadata = {"source": src}
            )
        )
    return minimal_docs

# Split the dcocuments into chunks
def text_split(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 150,
        chunk_overlap = 50
    )
    texts_chunk = text_splitter.split_documents(minimal_docs)
    return texts_chunk

# Create Embeddings
def download_embeddings():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name = model_name
    )
    return embeddings

embedding = download_embeddings()