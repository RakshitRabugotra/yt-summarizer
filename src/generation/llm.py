import os
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Also parse the output
str_parser = StrOutputParser()

# Get the environment variables for all api keys
GOOGLE_API_KEY: str | None = os.getenv("GOOGLE_API_KEY")
HUGGINGFACEHUB_ACCESS_TOKEN: str | None = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")

# On initialization of the model, if we have hugging face token, then login
if HUGGINGFACEHUB_ACCESS_TOKEN:
    from huggingface_hub import login
    login(HUGGINGFACEHUB_ACCESS_TOKEN)

def get_model_config():
    # The precedence goes like
    # Gemini,
    if GOOGLE_API_KEY:
        return dict(
            model=os.getenv("GOOGLE_GENERATIVE_MODEL", "gemini-2.5-flash"),
            model_provider="google_genai",
        )

    # Huggingface
    if HUGGINGFACEHUB_ACCESS_TOKEN:
        return dict(
            model=os.getenv("HUGGINGFACE_MODEL", "deepseek-ai/DeepSeek-R1-0528"),
            model_provider="huggingface",
        )

    # OpenAI
    if OPENAI_API_KEY:
        return dict(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"), model_provider="openai"
        )

    raise EnvironmentError(
        "None of `GOOGLE_API_KEY`, `HUGGINGFACEHUB_ACCESS_TOKEN`, `OPENAI_API_KEY` are set. Set at least one of these for LLM model usage"
    )


# Get the model configurations
model_config = get_model_config()

# Create an llm
llm = init_chat_model(
    model=model_config['model'],
    model_provider=model_config['model_provider'],
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
