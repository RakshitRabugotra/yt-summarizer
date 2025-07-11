from typing import TypedDict
from langchain_core.runnables import RunnableLambda
from indexing.vectorstore import Chroma

class RetrievalInputs(TypedDict):
    vectorstore: Chroma
    query: str

class RetrievalOutputs(TypedDict):
    query: str
    context: str

def __get_retrieved_docs(inputs: RetrievalInputs):
    # Get the retriever
    retriever = inputs['vectorstore'].as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    matching_docs = retriever.invoke(inputs['query'])
    formatted_text = "\n\n".join(doc.page_content for doc in matching_docs)
    # Return the retrieved context
    inputs["context"] = formatted_text
    return inputs

# The main export from this module
get_retrieved_docs = RunnableLambda(__get_retrieved_docs)