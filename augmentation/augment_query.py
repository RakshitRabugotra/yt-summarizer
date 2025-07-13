from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# The prompt to translate text to english if it isn't
runnable_convert_to_english_prompt = RunnablePassthrough(
    lambda _: print(f"[DEBUG]: Detecting language, and converting if necessary")
) | ChatPromptTemplate(
    [
        (
            "system",
            """
You are a skilled linguist capable to detect the language from the given text. Your task is to deeply analyze the text input provided within <transcript>...</transcript> and TRANSLATE IT TO ENGLISH IF IT IS NOT IN ENGLISH. This content represent the full or partial transcription or description of a video.
Your goal is to translate the provided transcript to english if it is not already in english.
When generating your response, follow these strict principles:
Be accurate and DO NOT CHANGE THE MEANING OF THE transcript.
DO NOT CHANGE THE transcript IF IT IS ALREADY IN ENGLISH AND RETURN AS IT IS.
Start your response only after fully analyzing the transcript content inside <transcript>...</transcript>
     """,
        ),
        ("human", "<transcript>{transcript}</transcript>"),
    ]
)

# Get the qa prompt
runnable_qa_augment_prompt = RunnablePassthrough(
    lambda _: print(f"[DEBUG]: Augmenting prompt from given query and context")
) | ChatPromptTemplate(
    [
        (
            "system",
            """
You are a highly capable AI agent specialized in understanding and summarizing video content with precision and clarity. Your task is to deeply analyze the transcript, metadata, or descriptive input provided within <context>{context}</context>. This content represents the full or partial transcription or description of a video, and may include visual cues, speaker turns, timestamps, and topics.
Your goal is to respond to the user's query.
When generating your response, follow these strict principles:
Be accurate and grounded in the given context. Do not invent or assume details that are not present within <context>.
Be concise yet informative. Summarize key insights, events, or points relevant to the query in a way that is digestible but complete.
Use structured formatting if needed — bullet points, timelines, key takeaways — but only if it aids clarity.
If the context does not contain enough information to answer the query, state so clearly.
Start your response only after fully analyzing the video content inside <context>...</context>
            """,
        ),
        ("human", "{query}"),
    ]
)
