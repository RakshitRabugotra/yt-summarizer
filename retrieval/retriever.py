from typing import TypedDict
from langchain_core.runnables import RunnableLambda

from indexing.vectorstore import get_vector_store
from indexing.document_loader import YouTubeTranscriptsLoader


class RetrievalInputs(TypedDict):
    query: str
    video_url: str


class RetrievalOutputs(TypedDict):
    chunks: str
    query: str
    video_url: str


def __retrieve_docs(inputs: RetrievalInputs):

    # Get the video_id
    video_id = YouTubeTranscriptsLoader.get_video_id(inputs['video_url'])

    # Get the retriever
    retriever = get_vector_store().as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4, "filter": {"video_id": video_id}},
    )

    # Append matching chunks to the output
    inputs['chunks'] = retriever.invoke(inputs['query'])

    return inputs



# The main export from this module
runnable_retrieve_docs = RunnableLambda(__retrieve_docs)
