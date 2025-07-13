"""
YouTube Video QA Summarizer Chain

This module creates a LangChain-based retriever and summarizer pipeline designed to:
- Load video transcripts or associated documents
- Chunk, embed, and store them if not already processed
- Retrieve relevant document chunks based on a user query
- Augment the query with additional context
- Generate a response using an LLM

Dependencies:
- LangChain
- dotenv
- Custom indexing, retrieval, and generation modules

To run as a script, the user must input:
1. One or more YouTube video URLs
2. A query to ask about the video(s)

Outputs are saved as markdown in `./out/response.md`.
"""

from typing import TypedDict

# Load all the env variables
from dotenv import load_dotenv

load_dotenv()

# Langchain imports
from langchain_core.runnables import RunnablePassthrough, RunnableBranch, Runnable

# Then load all the modules
from src.indexing.document_loader import runnable_load_documents
from src.indexing.text_splitter import (
    runnable_format_documents,
    runnable_split_embed_and_store,
)
from src.retrieval.retriever import runnable_retrieve_docs
from src.augmentation.augment_query import runnable_qa_augment_prompt
from src.generation.llm import runnable_generate


class RetrieverChainInputs(TypedDict):
    """
    Typed dictionary representing input to the retriever chain.

    Attributes:
        query (str): The user's query to ask about the video.
        video_url (str): The YouTube video URL or ID to process.
    """

    query: str
    video_url: str


def get_retriever_chain() -> Runnable:
    """
    Constructs and returns a LangChain Runnable representing the full retriever pipeline.

    The pipeline performs the following:
    - Attempts to retrieve relevant chunks for a video
    - If chunks are not found:
        - Logs the missing state
        - Loads documents from the video
        - Splits, embeds, and stores them
        - Retrieves relevant chunks
    - Formats retrieved chunks
    - Augments the user query
    - Generates a final response using an LLM

    Returns:
        Runnable: A composable LangChain pipeline to process video queries.
    """
    # The chain which will run, if the video document is not present in the database
    runnable_fetch_docs_if_not_exists = (
        RunnablePassthrough(
            lambda inputs: print(f"[DEBUG]: Video '{inputs['video_url']}' not found in database")
        )
        | runnable_load_documents
        | runnable_split_embed_and_store
        | runnable_retrieve_docs
    )

    # The chain which runs when we're given chunks and query
    runnable_format_chunks_and_generate = (
        runnable_format_documents | runnable_qa_augment_prompt | runnable_generate
    )

    # If we have the chunks, then this chain will run
    runnable_chunks_found = RunnablePassthrough(
        lambda _: print("[DEBUG]: Found chunks for the video")
    )

    # This is our main chain now (with all the workflow)
    return (
        runnable_retrieve_docs
        | RunnableBranch(
            (
                lambda inputs: len(inputs["chunks"]) == 0,
                runnable_fetch_docs_if_not_exists,
            ),
            runnable_chunks_found,
        )
        | runnable_format_chunks_and_generate
    )


def get_summary_results(inputs: RetrieverChainInputs) -> tuple[bool, str]:
    """
    Executes the summarization and retrieval chain for a given input.

    Args:
        inputs (RetrieverChainInputs): A dictionary containing the query and video URL.

    Returns:
        (success, response) (bool, str): A tuple where:
            - First element is a boolean indicating if an error occurred
            - Second element is the result (response string or error message)
    """
    # Get the retriever chain
    retriever_chain = get_retriever_chain()
    # Return the documents
    # try:
    response = retriever_chain.invoke(inputs)
    return True, response
    # except Exception as e:
        # print("[ERROR]: " + str(e))
        # return False, str(e)


if __name__ == "__main__":
    """
    CLI interface to interactively run the summarization pipeline.

    Prompts the user to:
    - Enter one or more YouTube video URLs
    - Enter a query

    The result is processed and saved to `./out/response.md`.
    """
    # Ask the user for youtube urls or ids
    vids = input(
        "Enter the URLs of YT Video you want to summarize (Separate by comma, if multiple)\n: "
    )

    if vids.count(",") >= 1:
        vids = [vid.strip() for vid in vids.split(",")]
    else:
        vids = [vids.strip()]

    # Also ask the question that the user want to ask
    query = input("\nGreat, now enter your query!\n: ").strip()

    # We will load the documents, chunk them, embed them, and store them to vector
    # After that, we can safely generate a retriever and retrieve the documents
    output = get_summary_results(
        {"query": query, "video_url": vids if isinstance(vids, str) else vids[0]}
    )

    if not output:
        exit(-1)

    filename = "./out/response.md"
    with open(filename, mode="w+", encoding="utf-16") as file:
        file.write(output)
    print("Output is written to the file: " + filename)
