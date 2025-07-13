from typing import TypedDict

# Load all the env variables
from dotenv import load_dotenv

load_dotenv()

# Langchain imports
from langchain_core.runnables import RunnablePassthrough, RunnableBranch, Runnable

# Then load all the modules
from indexing.document_loader import runnable_load_documents
from indexing.text_splitter import (
    runnable_format_documents,
    runnable_split_embed_and_store,
)
from retrieval.retriever import runnable_retrieve_docs
from augmentation.augment_query import runnable_augment_prompt
from generation.llm import runnable_generate


class RetrieverChainInputs(TypedDict):
    query: str
    video_url: str


def get_retriever_chain() -> Runnable:
    # Let's create all the necessary chains first
    runnable_fetch_docs_if_not_exists = (
        runnable_load_documents
        | runnable_split_embed_and_store
        | runnable_retrieve_docs
    )
    runnable_format_chunks_and_generate = (
        runnable_format_documents | runnable_augment_prompt | runnable_generate
    )

    # This is our main chain now (with all the workflow)
    return (
        RunnablePassthrough(lambda x: print("Received inputs: " + str(x)))
        | runnable_retrieve_docs
        | RunnableBranch(
            (
                lambda inputs: len(inputs["chunks"]) == 0,
                RunnablePassthrough(lambda inputs: print(f"Video '{inputs['video_url']}' not found in database")) | runnable_fetch_docs_if_not_exists,
            ),
            RunnablePassthrough(),
        )
        | runnable_format_chunks_and_generate
    )


def get_summary_results(inputs: RetrieverChainInputs) -> str:
    # Get the retriever chain
    retriever_chain = get_retriever_chain()
    # Return the documents
    try:
        return retriever_chain.invoke(inputs)
    except Exception as e:
        print("[ERROR]: " + str(e))
        return None


if __name__ == "__main__":
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

    filename = "./out/response.md"
    with open(filename, mode="w+", encoding="utf-16") as file:
        file.write(output)
    print("Output is written to the file: " + filename)
