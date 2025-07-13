"""
The Streamlit client for our interaction
"""
import streamlit as st

# Custom Imports
from src.rag import get_summary_results
from src.indexing.document_loader import YouTubeTranscriptsLoader

#
# Utility function
#
def invoke_retrieval_chain():
    # Get the input from input box
    video_url = st.session_state.video_url
    query = st.session_state.query

    # Check if the given video_url is a valid YT one
    is_valid, _ = YouTubeTranscriptsLoader.is_valid_youtube_url(
        video_url, return_video_id=False
    )

    # If it is not valid, then don't proceed and show an error message
    if not is_valid:
        return st.error("Invalid YouTube video URL. Cannot proceed, please check it")

    # Call the backend chain
    success, response = get_summary_results({
        "query": query,
        "video_url": video_url
    })

    # Set session state accordingly
    if success:
        st.session_state.model_output = response
        if st.session_state.get('model_error'): del st.session_state['model_error']
    else:
        st.session_state.model_error = response
        if st.session_state.get('model_output'): del st.session_state['model_output']

#
# UI Elements
#

# Input for video URL
st.text_input(
    label="Video URL",
    placeholder="https://www.youtube.com/watch?v=4g-fPNjizrw",
    key="video_url",
)

# Input for user query
st.text_input(
    label="Additional Requests",
    placeholder="You can add additional requests here...",
    key="query"
)

# Summarize button
st.button(
    label="Let's summarize",
    key="generate_button",
    on_click=invoke_retrieval_chain,
    disabled=not bool(st.session_state.video_url),
)

# Error Display
if "model_error" in st.session_state:
    st.error("### Error while generating response")
    st.error(st.session_state.model_error)

# Output Display
if "model_output" in st.session_state:
    st.markdown("### Here is what the video says")
    st.markdown(st.session_state.model_output)
