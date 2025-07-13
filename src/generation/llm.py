import os
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace


# This is a function to clean <think>...</think> content from the response 
# If we choose to opt for a reasoning model
def __omit_think_tags(text: str) -> str:
    # Find the start of the think process
    think_start_index = text.find("<think>")

    # It doesn't have any think tags
    if think_start_index == -1: return text
    
    # Else, get the ending tag too
    think_end_index = text.rfind("</think>")
    think_close_length = len("</think>")

    # Return all the part excluding think tags
    return text[think_end_index+think_close_length:].strip()

# The runnable which can be used after str_parser
omit_think_output_parser = RunnableLambda(__omit_think_tags)

# Also parse the output
str_parser = StrOutputParser() | omit_think_output_parser

# Get the environment variables for all api keys
GOOGLE_API_KEY: str | None = os.getenv("GOOGLE_API_KEY")
HUGGINGFACEHUB_API_TOKEN: str | None = os.getenv("HUGGINGFACEHUB_API_TOKEN")
OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")

# On initialization of the model, if we have hugging face token, then login
if HUGGINGFACEHUB_API_TOKEN:
    from huggingface_hub import login
    login(HUGGINGFACEHUB_API_TOKEN)

def get_model_config():
    # The precedence goes like
    # Gemini,
    if GOOGLE_API_KEY:
        return dict(
            model=os.getenv("GOOGLE_GENERATIVE_MODEL", "gemini-2.5-flash"),
            model_provider="google_genai",
        )

    # Huggingface
    if HUGGINGFACEHUB_API_TOKEN:
        return dict(
            model=os.getenv("HUGGINGFACE_MODEL", "deepseek-ai/DeepSeek-R1-0528"),
            model_provider="huggingface",
            huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN
        )

    # OpenAI
    if OPENAI_API_KEY:
        return dict(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"), model_provider="openai"
        )

    raise EnvironmentError(
        "None of `GOOGLE_API_KEY`, `HUGGINGFACEHUB_API_TOKEN`, `OPENAI_API_KEY` are set. Set at least one of these for LLM model usage"
    )


# Get the model configurations
model_config = get_model_config()

# Create an llm
llm = None

if 'huggingfacehub_api_token' in model_config:
    llm_endpoint = HuggingFaceEndpoint(
        repo_id=model_config['model'],
        task="text-generation",
        do_sample=False,
        repetition_penalty=1.03,
        provider="auto",  # let Hugging Face choose the best provider for you
    )
    llm = ChatHuggingFace(llm=llm_endpoint)
else:
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
