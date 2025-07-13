import os
from typing import TypedDict, Callable
from langchain_chroma import Chroma
from langchain.schema import Document

# To get the embedding function
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEndpointEmbeddings


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

# The API keys for our models
OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
HUGGINGFACEHUB_API_TOKEN: str | None = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# On initialization of the model, if we have hugging face token, then login
if HUGGINGFACEHUB_API_TOKEN:
    from huggingface_hub import login

    login(HUGGINGFACEHUB_API_TOKEN)


# Get the embedding model
def get_embedding_function():
    # The precedence goes like
    # OpenAI, HuggingFace
    OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
    HUGGINGFACEHUB_API_TOKEN: str | None = os.getenv("HUGGINGFACEHUB_API_TOKEN")

    # First preference
    if OPENAI_API_KEY:
        return OpenAIEmbeddings(
            model=os.getenv("OPENAI_EMBEDDINGS_MODEL", "text-embedding-3-large")
        )

    # Second preference
    if HUGGINGFACEHUB_API_TOKEN:
        return HuggingFaceEndpointEmbeddings(
            model=os.getenv(
                "HUGGINGFACE_EMBEDDINGS_MODEL", "intfloat/e5-mistral-7b-instruct"
            ),
            task="feature-extraction",
            huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
        )

    raise EnvironmentError(
        "Neither API Key set for OPENAI nor HUGGINGFACE\nSet `OPENAI_API_KEY` or `HUGGINGFACEHUB_API_TOKEN` in your environment"
    )


# Returns the currently used store
def get_vector_store():
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=get_embedding_function(),
    )
