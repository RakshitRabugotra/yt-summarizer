import os
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser

# Also parse the output
str_parser = StrOutputParser()

# Create an llm
llm = init_chat_model(
    model=os.getenv("GOOGLE_GENERATIVE_MODEL", "gemini-2.5-flash"),
    model_provider="google_genai",
    temperature=0.5
)

# The main exportable here
runnable_generate = (llm | str_parser)