import os
from typing import TypedDict, Callable
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain_core.runnables import RunnableLambda


class VectorstoreInputs(TypedDict):
    chunks: list[Document]
    embedding_function: Callable


class VectorstoreOutputs(TypedDict):
    vectorstore: Chroma

# CONFIGURATION for the vector store
COLLECTION_NAME = "yt_store"
# PERSIST_DIRECTORY = "./db"

# if not os.path.isdir(PERSIST_DIRECTORY):
#     os.makedirs(PERSIST_DIRECTORY)

def get_vector_store():
    return Chroma(
        collection_name=COLLECTION_NAME,
        # persist_directory=PERSIST_DIRECTORY
    )

def __add_docs_to_vector_store(inputs: VectorstoreInputs) -> VectorstoreOutputs:
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        # persist_directory=PERSIST_DIRECTORY,
        embedding_function=inputs["embedding_function"],
    )
    # Add the given documents to the store
    vectorstore.add_documents(inputs["chunks"])

    inputs['vectorstore'] = vectorstore
    return inputs


# Export this function as runnable
add_docs_to_vector_store = RunnableLambda(__add_docs_to_vector_store)
