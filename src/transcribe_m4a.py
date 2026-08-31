#!/usr/bin/env python3
"""Transcribe a local M4A audio file with mlx-whisper on Apple GPU."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path
from typing import Optional, Sequence, TextIO

from whisper_mlx import AudioTranscriber as MlxAudioTranscriber
from whisper_mlx import MODEL_ID, SAMPLE_RATE


class TranscriptionError(RuntimeError):
    """Raised when an audio file cannot be decoded or transcribed."""


class AudioTranscriber(MlxAudioTranscriber):
    """Load mlx-whisper once and transcribe 16 kHz mono audio on Metal."""

    def __init__(self, model_id: str = MODEL_ID, language: str = "en") -> None:
        try:
            super().__init__(model_id=model_id, language=language)
        except ImportError as exc:
            dependency = getattr(exc, "name", None) or "mlx-whisper"
            raise TranscriptionError(
                f"Missing dependency '{dependency}'. Install the project dependencies "
                "with: pip install -r requirements.txt"
            ) from exc
        except Exception:
            self.cleanup()
            raise


def validate_audio_path(value: str) -> Path:
    """Return an absolute path after validating a local M4A input."""
    audio_path = Path(value).expanduser()
    if not audio_path.exists():
        raise ValueError(f"Audio file not found: {audio_path}")
    if not audio_path.is_file():
        raise ValueError(f"Audio path is not a file: {audio_path}")
    if audio_path.suffix.lower() != ".m4a":
        raise ValueError(f"Expected an .m4a audio file: {audio_path}")
    return audio_path.resolve()


def convert_m4a_to_wav(audio_path: Path, wav_path: Path) -> None:
    """Decode M4A audio to 16 kHz, mono, 16-bit PCM WAV with FFmpeg."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise TranscriptionError(
            "FFmpeg was not found. Install it and ensure 'ffmpeg' is on PATH."
        )

    command = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(audio_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-c:a",
        "pcm_s16le",
        str(wav_path),
    ]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise TranscriptionError(f"Could not start FFmpeg: {exc}") from exc

    if result.returncode != 0:
        details = result.stderr.strip() or "unknown FFmpeg error"
        raise TranscriptionError(f"FFmpeg could not decode '{audio_path}': {details}")
    if not wav_path.is_file() or wav_path.stat().st_size == 0:
        raise TranscriptionError("FFmpeg completed without producing decoded audio.")


def _clean_chunk_transcript(transcript: str) -> str:
    transcript = transcript.strip().strip('".')
    return " ".join(transcript.split())


def transcribe_wav(
    wav_path: Path,
    transcriber: AudioTranscriber,
    progress_stream: Optional[TextIO] = None,
) -> str:
    """Transcribe a normalized 16 kHz mono WAV in one mlx-whisper pass."""
    if progress_stream is None:
        progress_stream = sys.stderr

    try:
        import numpy as np
    except ImportError as exc:
        raise TranscriptionError(
            "Missing dependency 'numpy'. Install the project dependencies with: "
            "pip install -r requirements.txt"
        ) from exc

    try:
        wav_file = wave.open(str(wav_path), "rb")
    except (OSError, wave.Error) as exc:
        raise TranscriptionError(f"Could not read decoded audio: {exc}") from exc

    with wav_file:
        if wav_file.getnchannels() != 1:
            raise TranscriptionError("Decoded audio is not mono.")
        if wav_file.getframerate() != SAMPLE_RATE:
            raise TranscriptionError(
                f"Decoded audio sample rate is {wav_file.getframerate()} Hz; "
                f"expected {SAMPLE_RATE} Hz."
            )
        if wav_file.getsampwidth() != 2 or wav_file.getcomptype() != "NONE":
            raise TranscriptionError("Decoded audio is not 16-bit PCM WAV.")

        frame_count = wav_file.getnframes()
        if frame_count == 0:
            raise TranscriptionError("The audio file contains no decodable audio frames.")
        duration = frame_count / SAMPLE_RATE
        audio = np.frombuffer(wav_file.readframes(frame_count), dtype="<i2")
        audio = audio.astype(np.float32)
        audio /= 32_768.0

    print(
        f"Transcribing {duration:.1f}s with mlx-whisper (Metal)...",
        file=progress_stream,
    )
    transcript = _clean_chunk_transcript(
        transcriber.transcribe(audio, verbose=False)
    )
    if transcript and not transcript.endswith((".", "!", "?")):
        transcript += "."
    return transcript


def transcribe_m4a(audio_path: Path, language: str = "en") -> str:
    """Decode and transcribe an M4A file, cleaning up all temporary resources."""
    transcriber: Optional[AudioTranscriber] = None
    with tempfile.TemporaryDirectory(prefix="transcribe-m4a-") as temp_dir:
        wav_path = Path(temp_dir) / "audio.wav"
        convert_m4a_to_wav(audio_path, wav_path)

        try:
            transcriber = AudioTranscriber(language=language)
            transcript = transcribe_wav(wav_path, transcriber)
        except TranscriptionError:
            raise
        except Exception as exc:
            raise TranscriptionError(f"Whisper transcription failed: {exc}") from exc
        finally:
            if transcriber is not None:
                transcriber.cleanup()

    if not transcript:
        raise TranscriptionError("Whisper returned an empty transcription.")
    return transcript


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Transcribe a local .m4a audio file with mlx-whisper "
            f"({MODEL_ID}) on Apple GPU. The transcript is printed to stdout "
            "unless --output is used."
        ),
    )
    parser.add_argument("audio_file", help="Path to the input .m4a file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Write the transcription to this UTF-8 text file",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Whisper language name or code (default: en, matching main_langgraph.py)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        audio_path = validate_audio_path(args.audio_file)
    except ValueError as exc:
        parser.error(str(exc))

    language = args.language.strip()
    if not language:
        parser.error("--language cannot be empty.")

    output_path = args.output.expanduser() if args.output is not None else None
    if output_path is not None:
        try:
            same_file = output_path.exists() and output_path.samefile(audio_path)
        except OSError:
            same_file = False
        if same_file or output_path.resolve() == audio_path:
            parser.error("The output path cannot overwrite the input audio file.")

    try:
        transcript = transcribe_m4a(audio_path, language=language)
        if output_path is None:
            print(transcript)
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(f"{transcript}\n", encoding="utf-8")
            print(f"Transcription saved to: {output_path}", file=sys.stderr)
    except (OSError, TranscriptionError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
