"""Web article scraping and YouTube subtitle transcript extraction pipeline.

Implements:
- FEAT-ING-01: Web article extraction via trafilatura into clean markdown/text.
- FEAT-ING-02: YouTube transcript extraction via youtube-transcript-api.
- Latency tracking via @latency_tracker for ingestion pipelines.
- Robust boundary handling: URL scheme validation, HTML entity unescaping,
  error resilience, and non-blocking compatibility with GenericWorker.
"""

import html
import logging
import re
import urllib.parse
from typing import Any

from lang_attack.telemetry import latency_tracker

logger = logging.getLogger("lang_attack.app")

# Lazy / conditional imports for offline and headless resilience
try:
    import trafilatura
except ImportError:
    trafilatura = None  # type: ignore[assignment]

try:
    from youtube_transcript_api import YouTubeTranscriptApi
except ImportError:
    YouTubeTranscriptApi = None  # type: ignore[assignment]


def extract_video_id(url_or_video_id: str) -> str:
    """Extract standard YouTube video ID from various URL formats or raw ID string.

    Supported URL patterns:
    - Standard watch: https://www.youtube.com/watch?v=VIDEO_ID
    - Shortened: https://youtu.be/VIDEO_ID
    - YouTube Shorts: https://www.youtube.com/shorts/VIDEO_ID
    - Embed: https://www.youtube.com/embed/VIDEO_ID
    - Live: https://www.youtube.com/live/VIDEO_ID
    - Raw Video ID: Alphanumeric string with optional '-' or '_'
    """
    if not url_or_video_id or not isinstance(url_or_video_id, str):
        return ""

    clean_input = url_or_video_id.strip()

    # If already a raw video ID (e.g. 11 characters or test strings without path delimiters)
    if (
        re.fullmatch(r"[\w-]{3,32}", clean_input)
        and "://" not in clean_input
        and "/" not in clean_input
    ):
        return clean_input

    try:
        parsed = urllib.parse.urlparse(clean_input)
    except Exception:
        return ""

    # Parse youtube.com URLs
    if "youtube.com" in parsed.netloc or "youtube" in parsed.netloc:
        query_params = urllib.parse.parse_qs(parsed.query)
        if "v" in query_params and query_params["v"]:
            return query_params["v"][0]

        path_parts = [p for p in parsed.path.split("/") if p]
        if path_parts:
            if path_parts[0] in ("shorts", "embed", "v", "live") and len(path_parts) > 1:
                return path_parts[1]
            return path_parts[-1]

    # Parse youtu.be short URLs
    if "youtu.be" in parsed.netloc:
        path_parts = [p for p in parsed.path.split("/") if p]
        if path_parts:
            return path_parts[0]

    # Fallback regex search for embedded ID in query or path
    match = re.search(
        r"(?:v=|\/v\/|\/embed\/|\/shorts\/|\/live\/|youtu\.be\/)([\w-]{3,32})",
        clean_input,
    )
    if match:
        return match.group(1)

    return clean_input


@latency_tracker("extract_youtube_transcript")
def extract_youtube_transcript(
    url_or_video_id: str,
    languages: tuple[str, ...] | list[str] = ("en", "pt"),
) -> list[dict[str, Any]]:
    """Extract subtitle transcript fragments with start times and durations.

    Args:
        url_or_video_id: YouTube video URL or raw video ID.
        languages: Preferred language codes in order of priority (default: ("en", "pt")).

    Returns:
        List of timed fragments: [{"text": str, "start": float, "duration": float}]
        Returns empty list on missing subtitles, invalid ID, or network failures.
    """
    if not url_or_video_id or not isinstance(url_or_video_id, str):
        return []

    video_id = extract_video_id(url_or_video_id)
    if not video_id:
        logger.warning("Failed to extract valid YouTube video ID from input: %s", url_or_video_id)
        return []

    target_languages = list(languages) if languages else ["en", "pt"]

    # Re-import or resolve YouTubeTranscriptApi dynamically for test monkeypatching
    try:
        import youtube_transcript_api

        api_cls = getattr(youtube_transcript_api, "YouTubeTranscriptApi", None)
    except ImportError:
        api_cls = YouTubeTranscriptApi

    if api_cls is None:
        logger.warning("youtube-transcript-api is not installed; cannot fetch transcript.")
        return []

    try:
        raw_transcript = api_cls.get_transcript(
            video_id,
            languages=target_languages,
        )
    except Exception as exc:
        logger.warning(
            "Could not retrieve transcript for video '%s' (languages=%s): %s",
            video_id,
            target_languages,
            exc,
        )
        return []

    if not isinstance(raw_transcript, list):
        return []

    formatted: list[dict[str, Any]] = []
    for item in raw_transcript:
        if isinstance(item, dict):
            text = str(item.get("text", "")).strip()
            # Unescape any HTML entities in subtitles
            text = html.unescape(text)
            try:
                start = float(item.get("start", 0.0))
            except (ValueError, TypeError):
                start = 0.0
            try:
                duration = float(item.get("duration", 0.0))
            except (ValueError, TypeError):
                duration = 0.0

            formatted.append(
                {
                    "text": text,
                    "start": start,
                    "duration": duration,
                }
            )

    return formatted


@latency_tracker("extract_article")
def extract_article(url: str) -> dict[str, Any]:
    """Scrape web page and extract clean text in Markdown format with metadata.

    Args:
        url: Full HTTP or HTTPS web URL to scrape.

    Returns:
        Dictionary containing:
        - title (str): Page title or first header.
        - text (str): Clean extracted text in Markdown format.
        - content (str): Alias for text for compatibility across consumers.
        - author (str): Article author if identified.
        - date (str): Publication date if identified.
        - language (str): Detected ISO language code (default 'en').
        - source_url (str): The requested URL.
        - error (str | None): Error description if fetch or extraction failed.
    """
    if not url or not isinstance(url, str):
        return {
            "title": "",
            "text": "",
            "content": "",
            "author": "",
            "date": "",
            "language": "en",
            "source_url": str(url) if url else "",
            "error": "URL must be a non-empty string",
        }

    clean_url = url.strip()
    try:
        parsed = urllib.parse.urlparse(clean_url)
    except Exception as exc:
        return {
            "title": "",
            "text": "",
            "content": "",
            "author": "",
            "date": "",
            "language": "en",
            "source_url": clean_url,
            "error": f"Malformed URL: {exc}",
        }

    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return {
            "title": "",
            "text": "",
            "content": "",
            "author": "",
            "date": "",
            "language": "en",
            "source_url": clean_url,
            "error": f"Invalid or unsupported URL scheme: '{clean_url}'",
        }

    # Re-import or resolve trafilatura dynamically for test monkeypatching
    try:
        import trafilatura as resolved_trafilatura
    except ImportError:
        resolved_trafilatura = trafilatura

    if resolved_trafilatura is None:
        logger.warning("trafilatura is not installed; cannot scrape web article.")
        return {
            "title": "",
            "text": "",
            "content": "",
            "author": "",
            "date": "",
            "language": "en",
            "source_url": clean_url,
            "error": "trafilatura package is not installed",
        }

    try:
        downloaded = resolved_trafilatura.fetch_url(clean_url)
    except Exception as exc:
        logger.warning("Failed to fetch URL '%s': %s", clean_url, exc)
        return {
            "title": "",
            "text": "",
            "content": "",
            "author": "",
            "date": "",
            "language": "en",
            "source_url": clean_url,
            "error": f"Network error fetching URL: {exc}",
        }

    if downloaded is None:
        return {
            "title": "",
            "text": "",
            "content": "",
            "author": "",
            "date": "",
            "language": "en",
            "source_url": clean_url,
            "error": f"HTTP error or empty response from {clean_url}",
        }

    try:
        extracted = resolved_trafilatura.extract(
            downloaded,
            output_format="markdown",
            include_comments=False,
            include_tables=True,
        )
    except Exception as exc:
        logger.warning("Failed to extract markdown from '%s': %s", clean_url, exc)
        extracted = None

    if not extracted:
        return {
            "title": "",
            "text": "",
            "content": "",
            "author": "",
            "date": "",
            "language": "en",
            "source_url": clean_url,
            "error": "No extractable text content found on page",
        }

    extracted = html.unescape(extracted)

    title = ""
    author = ""
    date = ""
    language = "en"

    # Attempt metadata extraction if available
    if hasattr(resolved_trafilatura, "extract_metadata"):
        try:
            meta = resolved_trafilatura.extract_metadata(downloaded)
            if meta:
                if getattr(meta, "title", None):
                    title = str(meta.title).strip()
                if getattr(meta, "author", None):
                    author = str(meta.author).strip()
                if getattr(meta, "date", None):
                    date = str(meta.date).strip()
                if getattr(meta, "language", None):
                    language = str(meta.language).strip().lower()
        except Exception:
            pass

    # Fallback title: parse first level-1 markdown heading
    if not title:
        for line in extracted.splitlines():
            stripped = line.strip()
            if stripped.startswith("# "):
                title = stripped.lstrip("# ").strip()
                break

    # Fallback title: inspect raw HTML if downloaded is string
    if not title and isinstance(downloaded, str):
        h1_match = re.search(r"<h1[^>]*>(.*?)</h1>", downloaded, re.IGNORECASE | re.DOTALL)
        if h1_match:
            title = re.sub(r"<[^>]+>", "", h1_match.group(1)).strip()

    if not title and isinstance(downloaded, str):
        title_match = re.search(r"<title[^>]*>(.*?)</title>", downloaded, re.IGNORECASE | re.DOTALL)
        if title_match:
            title = re.sub(r"<[^>]+>", "", title_match.group(1)).strip()

    title = html.unescape(title)

    # Fallback language: inspect <html lang="..."> attribute
    if (not language or language == "en") and isinstance(downloaded, str):
        lang_match = re.search(
            r'<html[^>]*\blang=["\']([a-zA-Z]{2}(?:-[a-zA-Z]{2})?)["\']',
            downloaded,
            re.IGNORECASE,
        )
        if lang_match:
            language = lang_match.group(1).lower()

    return {
        "title": title,
        "text": extracted,
        "content": extracted,
        "author": author,
        "date": date,
        "language": language or "en",
        "source_url": clean_url,
        "error": None,
    }


def transcript_to_text(fragments: list[dict[str, Any]]) -> str:
    """Concatenate subtitle transcript fragments into a continuous plain text block."""
    if not fragments:
        return ""
    return " ".join(frag.get("text", "").strip() for frag in fragments if frag.get("text"))


def transcript_to_markdown(
    fragments: list[dict[str, Any]],
    title: str = "YouTube Video Transcript",
) -> str:
    """Format subtitle transcript fragments into a structured Markdown note with timestamps."""
    if not fragments:
        return f"# {title}\n\n*No subtitles available.*"

    lines: list[str] = [f"# {title}\n"]
    for frag in fragments:
        start_secs = int(frag.get("start", 0.0))
        minutes = start_secs // 60
        seconds = start_secs % 60
        timestamp = f"[{minutes:02d}:{seconds:02d}]"
        text = frag.get("text", "").strip()
        if text:
            lines.append(f"- **{timestamp}** {text}")

    return "\n".join(lines)


def chunk_article_text(text: str, max_chunk_size: int = 4000) -> list[str]:
    """Split article text into smaller paragraphs chunks adhering to max size limits."""
    if not text:
        return []
    if len(text) <= max_chunk_size:
        return [text]

    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current_chunk: list[str] = []
    current_length = 0

    for para in paragraphs:
        para_len = len(para) + 2
        if current_length + para_len > max_chunk_size and current_chunk:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = [para]
            current_length = len(para)
        else:
            current_chunk.append(para)
            current_length += para_len

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks


__all__ = [
    "chunk_article_text",
    "extract_article",
    "extract_video_id",
    "extract_youtube_transcript",
    "transcript_to_markdown",
    "transcript_to_text",
]
