from typing import TypedDict

# Langchain imports
from langchain_core.runnables import RunnableLambda
from langchain.schema import Document

# Custom imports
from src.indexing.vectorstore import get_vector_store
from src.indexing.document_loader import YouTubeTranscriptsLoader


class RetrievalInputs(TypedDict):
    query: str
    video_url: str


class RetrievalOutputs(TypedDict):
    chunks: list[Document]
    query: str
    video_url: str


def __retrieve_docs(inputs: RetrievalInputs) -> RetrievalOutputs:
    """
    Retrieves the documents for given input
    Args:
        inputs: { query: str, video_url: str }

    Returns:
        outputs: { chunks: list[Document], query: str, video_url: str}
    """
    print(f"[DEBUG]: Retrieving documents for video='{inputs['video_url']}'")
    # Get the video_id
    video_id = YouTubeTranscriptsLoader.get_video_id(inputs["video_url"])

    # Configure the retriever
    retriever_kwargs = dict(
        search_type="similarity",
        search_kwargs={"k": 4, "filter": {"video_id": video_id}},
    )
    # Get the retriever
    retriever = get_vector_store().as_retriever(**retriever_kwargs)

    try:
        # Append matching chunks to the output
        inputs["chunks"] = retriever.invoke(inputs["query"])
    except Exception as e:
        # Check if the exception is result of an unexpected vector size
        if not str(e).startswith(
            "Collection expecting embedding with dimension of"
        ) or "got" not in str(e):
            raise e
        # We're dealing with a shift in embedding vector type, clear the store and reset the db
        get_vector_store().delete_collection()
        # Initialize a new store, and a retriever
        retriever = get_vector_store().as_retriever(**retriever_kwargs)
        # Append matching chunks to the output
        inputs["chunks"] = retriever.invoke(inputs["query"])
    
    # Return the inputs, converted to output
    return inputs


# The main export from this module
runnable_retrieve_docs = RunnableLambda(__retrieve_docs)
