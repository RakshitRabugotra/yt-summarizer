from typing import Union, Tuple, TypedDict, Iterator
from urllib.parse import urlparse, parse_qs
import re

# Langchain imports
from langchain_core.document_loaders import BaseLoader
from langchain_core.runnables import RunnableLambda
from langchain.schema import Document

# Youtube api
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

# Custom imports
from indexing.vectorstore import get_vector_store

class InvalidYouTubeURLException(Exception):
    """Raised when the provided URL is not a valid YouTube URL."""
    pass


class YouTubeTranscriptsLoader(BaseLoader):

    class YouTubeTranscriptsLoaderInitArgs(TypedDict):
        yt_video_urls: list[str] = (None,)
        yt_video_ids: list[str] = (None,)
        transcript_languages: list[str] = ["en"]

    def __init__(
        self,
        yt_video_urls: list[str] = None,
        yt_video_ids: list[str] = None,
        transcript_languages: list[str] = ["en"],
    ):
        """
        Args:
            yt_video_urls: The full length URLs to Youtube video in a list
            yt_video_ids: The "video IDs" to YouTube video is a list
            transcript_languages: The list containing desired languages for transcripts `['en', 'hi']`

        Raises:
            ValueError: If the provided URLs or IDs are invalid

        Example:
            >>> loader = YouTubeTranscriptsLoader(
            >>>     yt_video_urls=["https://www.youtube.com/watch?v=dQw4w9WgXcQ"]
            >>> )
            >>>
            >>> # Get the transcripts of this video
            >>> try:
            >>>     transcripts = loader.load()
            >>>
            >>>     for chunk in transcripts:
            >>>         print(chunk)
            >>> except Exception as e:
            >>>     print("[Error]: ", str(e))
        """
        _vid_ids = None
        # First let's check if the ids are provided
        if yt_video_ids and isinstance(yt_video_ids, list):
            _vid_ids = [
                self.__get_video_id(yt_video_id=vid_id) for vid_id in yt_video_ids
            ]

        # If not, then let's check if the urls are provided
        if yt_video_urls and isinstance(yt_video_urls, list):
            _vid_ids = [
                self.__get_video_id(yt_video_url=vid_url) for vid_url in yt_video_urls
            ]

        # if _vid_ids is still none, raise an exception
        if not _vid_ids:
            raise ValueError("Provided urls/ids are invalid")

        # These are all valid so store them in
        self.yt_video_urls = yt_video_urls
        self.yt_video_ids = yt_video_ids
        # Get the desired transcripts
        self.transcript_languages = transcript_languages
        # Validate the video_urls and ids
        self.video_ids = _vid_ids

    # Another method to initialize the class
    @classmethod
    def from_config(cls, config: YouTubeTranscriptsLoaderInitArgs):
        return cls(**config)

    # Create the generator
    def lazy_load(self) -> Iterator[Document]:
        # Iterate over all the video ids, and return the transcripts
        for vid_id in self.video_ids:
            transcript_list = self.__get_video_transcripts(vid_id)
            # Flatten the transcripts to a plain text
            transcript = " ".join(chunk["text"] for chunk in transcript_list)
            # Now we need to create a document object from this
            yield Document(
                page_content=transcript,
                metadata={"video_id": vid_id, "length": len(transcript)},
            )

    def is_valid_youtube_url(
        self, url: str, return_video_id: bool = True
    ) -> Tuple[bool, Union[str, None]]:
        """
        Validates a YouTube video URL and optionally extracts the video ID.

        Args:
            url (str): The URL to validate.
            return_video_id (bool): Whether to return the extracted video ID.

        Returns:
            Tuple[bool, Union[str, None]]: A tuple where the first element indicates
                whether the URL is valid, and the second is the video ID if requested
                and valid, otherwise None.

        Raises:
            InvalidYouTubeURLException: If the URL is malformed or does not match
                YouTube video URL format.
        """

        if not isinstance(url, str):
            raise TypeError("URL must be a string.")

        # Regex for matching YouTube video URLs
        youtube_regex = re.compile(
            r"""^(https?://)?          # http or https protocol (optional)
            (www\.)?                  # www. (optional)
            (youtube\.com|youtu\.be)  # youtube domain
            (/watch\?v=|/embed/|/v/|/shorts/|/)? # valid paths to video ID
            ([\w-]{11})               # video ID is always 11 characters
            (&.+)?$                   # optional query params
            """,
            re.VERBOSE,
        )

        match = youtube_regex.match(url)
        if not match:
            if return_video_id:
                return False, None
            return False, None

        try:
            # Try parsing to get the video ID in a more robust way
            parsed_url = urlparse(url)

            video_id = None

            # Handle different URL styles
            if "youtube.com" in parsed_url.netloc:
                query_params = parse_qs(parsed_url.query)
                video_id_list = query_params.get("v")
                if video_id_list:
                    video_id = video_id_list[0]
                elif "/shorts/" in parsed_url.path:
                    video_id = parsed_url.path.split("/shorts/")[-1].split("/")[0]
            elif "youtu.be" in parsed_url.netloc:
                video_id = parsed_url.path.lstrip("/")

            # Final check: video_id must be 11 characters
            if video_id and len(video_id) == 11:
                return True, video_id if return_video_id else None
            else:
                return False, None
        except Exception as e:
            raise InvalidYouTubeURLException(f"Error parsing URL: {e}")

    def __get_video_id(self, yt_video_url: str = None, yt_video_id: str = None):
        # The video id takes precedence
        url = yt_video_id or yt_video_url or None

        # If both the id and url are none, raise error
        if not (yt_video_id or yt_video_url):
            raise ValueError(
                "Both `yt_video_url` and `yt_video_id` cannot be None"
            ) from (
                yt_video_url,
                yt_video_id,
            )

        video_id = None
        # If url parameter not is None, means we have the video url
        if yt_video_url is not None:
            # Check if the url is a valid one
            is_valid, video_id = self.is_valid_youtube_url(url, return_video_id=True)
            # If the url is not valid, raise
            if not is_valid:
                raise ValueError(
                    f"Youtube video regex failed... invalid url format: '{url}'"
                )
        else:
            # Else, we have the video if
            video_id = url

        if not video_id:
            raise ValueError(f"Couldn't get video ID from: '{url}'") from url

        return video_id

    def __get_video_transcripts(self, video_id: str):
        """ """
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(
                video_id, languages=self.transcript_languages
            )
        except TranscriptsDisabled as tde:
            raise Exception("No captions available for this video.") from tde
        except Exception as e:
            raise Exception(
                "Unexpected exception happened during fetching transcripts\n[ERROR]: "
                + str(e)
            ) from e

        # Return the transcript list
        return transcript_list


# Utility function which directly returns the document_loader
def __load_documents(inputs: dict[str]):
    # Extract the query
    config = inputs.copy()
    del config['query']
    # Load all the documents
    inputs['documents'] = YouTubeTranscriptsLoader.from_config(config).lazy_load()
    return inputs

# Main exportable from this module
load_documents = RunnableLambda(__load_documents)

if __name__ == "__main__":

    urls = [
        "https://www.youtube.com/watch?v=4g-fPNjizrw"
    ]

    # Create a loader runnable
    loader = YouTubeTranscriptsLoader(yt_video_urls=urls)

    # Get the transcripts of this video
    try:
        transcripts = loader.load()
        for chunk in transcripts:
            print(chunk)
    except Exception as e:
        print("[Error]: ", str(e))

