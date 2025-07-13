import os
from typing import TypedDict, Callable
from langchain_chroma import Chroma
from langchain.schema import Document
# To get the embedding function
from langchain_openai import OpenAIEmbeddings


class VectorstoreInputs(TypedDict):
    chunks: list[Document]
    embedding_function: Callable
    video_id: str


class VectorstoreOutputs(TypedDict):
    vectorstore: Chroma
    video_id: str

# CONFIGURATION for the vector store
COLLECTION_NAME = "yt_store"
PERSIST_DIRECTORY = "./db"

if not os.path.isdir(PERSIST_DIRECTORY):
    os.makedirs(PERSIST_DIRECTORY)

def get_vector_store():
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=OpenAIEmbeddings(
            model=os.getenv("OPENAI_EMBEDDINGS_MODEL", "text-embedding-3-large")
        ),
    )
