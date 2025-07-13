import re
from typing import Optional

def get_video_id(youtube_url: str) -> Optional[str]:
    """
    Extract the YouTube video ID from a given URL.

    Args:
        youtube_url (str): A YouTube URL in various supported formats.

    Returns:
        Optional[str]: The 11-character YouTube video ID if found, else None.
    """
    pattern = (
        r'(?:https?:\/\/)?'                         # optional scheme
        r'(?:[0-9A-Z-]+\.)?'                        # optional subdomain
        r'(?:youtube|youtu|youtube-nocookie)\.(?:com|be)\/'  # domain
        r'(?:watch\?v=|watch\?.+&v=|embed\/|v\/|.+\?v=)?'     # path or query
        r'([^&=\n%\?]{11})'                         # capture group for video ID
    )

    match = re.search(pattern, youtube_url, re.IGNORECASE)
    return match.group(1) if match else None


if __name__ == '__main__':
    test_urls = [
        'http://www.youtube.com/watch?v=5Y6HSHwhVlY',
        'http://youtu.be/5Y6HSHwhVlY',
        'http://www.youtube.com/embed/5Y6HSHwhVlY?rel=0" frameborder="0"',
        'https://www.youtube-nocookie.com/v/5Y6HSHwhVlY?version=3&amp;hl=en_US',
        'http://www.youtube.com/',
        'http://example.com/watch?v=5Y6HSHwhVlY',
        'https://www.youtube.com/watch?v=2USUfv7klr8'
    ]

    for url in test_urls:
        video_id = get_video_id(url)
        print(f"{url} => {video_id}")
