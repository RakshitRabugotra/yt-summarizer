import os
from typing import Iterable, TypedDict, Callable
from langchain.schema import Document

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnableLambda

# To get the embedding function
from langchain_openai import OpenAIEmbeddings

CHUNK_SIZE=1000
CHUNK_OVERLAP = 0.20 * CHUNK_SIZE

# Perform a check if we don't have OpenAI Key, then trow error
if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError("`OPENAI_API_KEY` is not set, it is required!")

# The required inputs for this chain is `documents`
class TextSplitterInputs(TypedDict):
    documents: Iterable[Document]

# The outputs for this class would be `chunks` and `embedding function`
class TextSplitterOutput(TypedDict):
    chunks: list[Document]
    embedding_function: Callable

"""
Splits the documents and returns embedding vectors
"""

def __split_and_embed(inputs: TextSplitterInputs) -> TextSplitterOutput:
    """
    Splits the documents and returns embedding vectors
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    # Split the documents
    chunks = splitter.split_documents(inputs['documents'])

    inputs['chunks'] = chunks
    inputs['embedding_function'] = OpenAIEmbeddings(
            model=os.getenv("OPENAI_EMBEDDINGS_MODEL", "text-embedding-3-large")
        )
    
    return inputs

split_and_embed = RunnableLambda(__split_and_embed)
