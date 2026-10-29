"""Asynchronous worker tasks for audio, scraping, speech evaluation, and AI."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PySide6.QtCore import QThreadPool

from lang_attack.core.asr import transcribe_audio
from lang_attack.core.ingestion import (
    extract_article,
    extract_youtube_transcript,
    transcript_to_markdown,
)
from lang_attack.core.llm_provider import get_llm_provider
from lang_attack.core.phonetics import evaluate_speech
from lang_attack.core.tts import synthesize_speech
from lang_attack.workers.generic import GenericWorker

logger = logging.getLogger("lang_attack.workers.tasks")


def dispatch_task(
    fn: Callable[..., Any],
    *args: Any,
    on_result: Callable[[Any], None] | None = None,
    on_error: Callable[[str, str], None] | None = None,
    on_progress: Callable[[int, str], None] | None = None,
    pool: QThreadPool | None = None,
    **kwargs: Any,
) -> GenericWorker:
    """Convenience helper to wrap any callable in a GenericWorker and dispatch it to QThreadPool."""
    worker = GenericWorker(fn, *args, **kwargs)

    if on_result:
        worker.signals.result.connect(on_result)
        worker.signals.finished.connect(on_result)
    if on_error:
        worker.signals.error.connect(on_error)
    if on_progress:
        worker.signals.progress.connect(on_progress)

    target_pool = pool or QThreadPool.globalInstance()
    target_pool.start(worker)
    return worker


def ingest_url_task(url: str, output_dir: Path | str) -> dict[str, Any]:
    """Ingest web article or YouTube video and save Markdown in output_dir."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if "youtube.com" in url.lower() or "youtu.be" in url.lower():
        data = extract_youtube_transcript(url)
        md_content = transcript_to_markdown(data)
        safe_title = (data.get("title") or "youtube_transcript").replace(" ", "_")
        target_file = out_path / f"{safe_title[:40]}.md"
        target_file.write_text(md_content, encoding="utf-8")
        return {"type": "youtube", "path": str(target_file), "title": data.get("title"), "text": md_content}
    else:
        article = extract_article(url)
        safe_title = (article.get("title") or "web_article").replace(" ", "_")
        target_file = out_path / f"{safe_title[:40]}.md"
        content = f"# {article.get('title')}\n\n{article.get('text', '')}"
        target_file.write_text(content, encoding="utf-8")
        return {"type": "article", "path": str(target_file), "title": article.get("title"), "text": content}


def tts_synthesis_task(text: str, output_path: Path | str, voice: str = "af_heart", speed: float = 1.0) -> Path:
    """Synthesize speech using Kokoro-82M on CPU."""
    return synthesize_speech(text=text, output_path=output_path, voice=voice, speed=speed)


def asr_transcription_task(audio_path: Path | str, language: str | None = None) -> dict[str, Any]:
    """Transcribe audio using faster-whisper."""
    return transcribe_audio(audio_path=audio_path, language=language)


def speech_evaluation_task(target_text: str, audio_path: Path | str, language: str = "en") -> Any:
    """Evaluate speech pronunciation using Wav2Vec 2.0 GOP."""
    return evaluate_speech(target_text=target_text, audio_path=audio_path, language=language)


def exercise_generation_task(
    topic: str,
    note_content: str,
    language: str = "en",
    num_exercises: int = 5,
) -> Any:
    """Generate structured exercises via configured LLM provider."""
    provider = get_llm_provider()
    return provider.generate_exercises(
        topic=topic,
        note_content=note_content,
        language=language,
        num_exercises=num_exercises,
    )


__all__ = [
    "asr_transcription_task",
    "dispatch_task",
    "exercise_generation_task",
    "ingest_url_task",
    "speech_evaluation_task",
    "tts_synthesis_task",
]
