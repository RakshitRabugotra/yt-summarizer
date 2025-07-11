# Load all the env variables
from dotenv import load_dotenv
load_dotenv()

# Then load all the modules
from indexing.document_loader import load_documents
from indexing.text_splitter import split_and_embed
from indexing.vectorstore import add_docs_to_vector_store

from retrieval.retriever import get_retrieved_docs

from augmentation.augment_query import augmented_prompt

from generation.llm import generate


def main():
    # Ask the user for youtube urls or ids
    vids = input("Enter the URLs of YT Video you want to summarize (Separate by comma, if multiple)\n: ")

    if vids.count(",") >= 1:
        vids = [vid.strip() for vid in vids.split(",")]
    else:
        vids = [vids.strip()]

    # Also ask the question that the user want to ask
    query = input("\nGreat, now enter your query!\n: ").strip()

    # We will load the documents, chunk them, embed them, and store them to vector
    # After that, we can safely generate a retriever and retrieve the documents 
    rag_chain = load_documents | split_and_embed | add_docs_to_vector_store | get_retrieved_docs | augmented_prompt | generate

    # Invoking this we will get the resultant vectorstore
    output = rag_chain.invoke({
        'query': query,
        'yt_video_urls': vids,
    })

    filename = './out/response.md'
    with open(filename, mode='w+', encoding='utf-16') as file:
        file.write(output)
    print("Output is written to the file: " + filename)


if __name__ == '__main__':
    main()