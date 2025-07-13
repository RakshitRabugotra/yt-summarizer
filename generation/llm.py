import os
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Also parse the output
str_parser = StrOutputParser()

# Create an llm
llm = init_chat_model(
    model=os.getenv("GOOGLE_GENERATIVE_MODEL", "gemini-2.5-flash"),
    model_provider="google_genai",
    temperature=0.5,
)

# The chain to translate to english if not already
runnable_generate_translation = (
    RunnablePassthrough(lambda _: print(f"[DEBUG]: Generating response for query"))
    | llm
    | str_parser
    | RunnablePassthrough(lambda _: print(f"[DEBUG]: Translation completed"))
)

# The main exportable here
runnable_generate = (
    RunnablePassthrough(lambda _: print(f"[DEBUG]: Generating response for query"))
    | llm
    | str_parser
    | RunnablePassthrough(lambda _: print(f"[DEBUG]: Response generation complete"))
)
