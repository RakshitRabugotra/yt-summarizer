import os
from typing import Iterator, TypedDict
from langchain.schema import Document

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnableLambda

# Custom modules
from src.indexing.vectorstore import get_vector_store

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 0.20 * CHUNK_SIZE

# Perform a check if we don't have OpenAI Key, then trow error
if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError("`OPENAI_API_KEY` is not set, it is required!")

"""
Formats the documents into strings
"""


class FormatDocumentsInputs(TypedDict):
    chunks: Iterator[Document]
    query: str


class FormatDocumentOutputs(TypedDict):
    context: str
    query: str


def __format_documents(inputs: FormatDocumentsInputs) -> FormatDocumentOutputs:
    """
    Format the chunks of documents into a string with paragraph formatting
    Args:
        inputs: { chunks: Iterator[Document], query: str }
    Returns:
        outputs: { context: str, query: str }
    """
    # Print for the debug
    print(f"[DEBUG]: Formatting chunks into context...")

    formatted_text = "\n\n".join(chunk.page_content for chunk in inputs["chunks"])
    inputs["context"] = formatted_text
    return inputs


runnable_format_documents = RunnableLambda(__format_documents)

"""
Splits the documents and returns embedding vectors
"""


# The required inputs for this chain is `documents`
class SplitEmbedAndStoreInputs(TypedDict):
    query: str
    docs: Iterator[Document]
    video_url: str


# The outputs for this class would be `chunks` and `embedding function`
class SplitEmbedAndStoreOutput(TypedDict):
    query: str
    video_url: str


def __split_embed_and_store(
    inputs: SplitEmbedAndStoreInputs,
) -> SplitEmbedAndStoreOutput:
    """
    Splits the documents, embeds them and stores them to the vector store
    Args:
        inputs: { query: str, docs: Iterator[Document], video_url: str }
    Returns:
        output: { query: str, video_url: str }
    """
    # Print for a debug statement
    print(f"[DEBUG]: Creating chunks and storing...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    # Split the documents
    chunks = splitter.split_documents(inputs["docs"])
    # Get the vector store
    vectorstore = get_vector_store()
    # Add the chunks to the vector store
    vectorstore.add_documents(chunks)
    return inputs


runnable_split_embed_and_store = RunnableLambda(__split_embed_and_store)
