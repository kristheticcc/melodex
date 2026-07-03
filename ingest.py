# Imports
import glob
import os.path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import embedding

# Function to check the knowledge base
def knowledge_base_check():
    full_content = ""
    files = glob.glob("knowledge_base/*/*.md")

    for file in files:
        with open(file, "r", encoding = "utf-8") as f:
            full_content+=f.read()
    return f"Number of files = {len(files)} | Number of characters = {len(full_content)}"

# Function to load documents from knowledge base
def load_knowledge_base():
    documents = []

    # Getting all the folders contained inside knowledge base
    folders = glob.glob("knowledge_base/*")

    for folder in folders:
        doc_type = os.path.basename(folder)
        loader = DirectoryLoader(
            folder,
            loader_cls=TextLoader,
            loader_kwargs={"encoding" : "utf-8"},
        )
        folder_docs = loader.load()

        for doc in folder_docs:
            doc.metadata["doc_type"] = doc_type
            documents.append(doc)

    return documents

# Function for chunking documents
def chunk_documents(docs):

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    return chunks, f"Number of chunks: {len(chunks)}"

# Function to create chroma vector store
def create_vector_store():
    # Checking the knowledge base
    check = knowledge_base_check()
    print(check)

    # Loading the data into documents
    docs = load_knowledge_base()

    # Splitting the document into multiple chunks
    chunks, chunk_len = chunk_documents(docs)
    print(chunk_len)

    # Vector database name
    db_name = "vector_db"

    # If database already exists, delete it before creating a new one
    if os.path.exists(db_name):
        Chroma(persist_directory=db_name, embedding_function=embedding).delete_collection()

    # Creating a vector store
    vector_store = Chroma.from_documents(
        chunks,
        embedding = embedding,
        persist_directory = db_name
    )

    print(f"Vector database created with {vector_store._collection.count()} documents")

create_vector_store()

