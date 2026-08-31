#!/usr/bin/env python3
"""Drop-folder pipeline: .m4a → Whisper transcript → oMLX MoM markdown.

Default usage (no arguments):
  1. Drop .m4a files into ~/Desktop/AudioToMoM
  2. Run this script
  3. Find *.transcript.txt and *.mom.md under AudioToMoM/output/

Already-processed audio is skipped by content SHA-256 (not filename), so
renames or re-drops of the same file do not re-run Whisper or the LLM.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import transcribe_m4a


DEFAULT_INBOX = Path.home() / "Desktop" / "AudioToMoM"
DEFAULT_OMLX_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_OMLX_MODEL = "DeepSeek-V4-Flash-0731-MXFP4-MLX"
DEFAULT_OMLX_SETTINGS = Path.home() / ".omlx" / "settings.json"
REGISTRY_NAME = ".processed.json"
PROMPT_RELATIVE = Path("prompts") / "mom_system.md"
HASH_CHUNK_SIZE = 1024 * 1024


class MomPipelineError(RuntimeError):
    """Raised when the MoM pipeline cannot complete a step."""


@dataclass
class AudioJob:
    """One inbox audio file and the artifacts derived from it."""

    path: Path
    content_hash: str
    stem: str
    transcript_path: Path
    mom_path: Path
    status: str  # pending | transcribed | complete


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_prompt_path() -> Path:
    return project_root() / PROMPT_RELATIVE


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(HASH_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_registry(registry_path: Path) -> Dict[str, Any]:
    if not registry_path.is_file():
        return {"version": 1, "items": {}}
    try:
        data = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MomPipelineError(
            f"Could not read processing registry '{registry_path}': {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise MomPipelineError(f"Invalid registry format in '{registry_path}'.")
    items = data.get("items")
    if items is None:
        data["items"] = {}
    elif not isinstance(items, dict):
        raise MomPipelineError(f"Invalid registry items in '{registry_path}'.")
    data.setdefault("version", 1)
    return data


def save_registry(registry_path: Path, registry: Dict[str, Any]) -> None:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = registry_path.with_suffix(registry_path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(registry, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(registry_path)


def load_system_prompt(prompt_path: Path) -> str:
    try:
        text = prompt_path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise MomPipelineError(f"Could not read MoM prompt '{prompt_path}': {exc}") from exc
    if not text:
        raise MomPipelineError(f"MoM prompt is empty: {prompt_path}")
    return text


def list_inbox_m4a(inbox_dir: Path) -> List[Path]:
    if not inbox_dir.is_dir():
        return []
    files = [
        path
        for path in inbox_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".m4a" and not path.name.startswith(".")
    ]
    return sorted(files, key=lambda p: p.name.lower())


def unique_stem(stem: str, content_hash: str, claimed: Dict[str, str]) -> str:
    """Return a filesystem stem; disambiguate when two files share a name."""
    if stem not in claimed:
        claimed[stem] = content_hash
        return stem
    if claimed[stem] == content_hash:
        return stem
    alt = f"{stem}-{content_hash[:8]}"
    claimed[alt] = content_hash
    return alt


def artifact_paths(output_dir: Path, stem: str) -> tuple[Path, Path]:
    return (
        output_dir / f"{stem}.transcript.txt",
        output_dir / f"{stem}.mom.md",
    )


def job_status_from_registry(
    entry: Optional[Dict[str, Any]],
    transcript_path: Path,
    mom_path: Path,
    force: bool,
) -> str:
    if force:
        return "pending"
    if entry:
        status = str(entry.get("status") or "")
        recorded_mom = Path(entry["mom_path"]) if entry.get("mom_path") else mom_path
        recorded_transcript = (
            Path(entry["transcript_path"])
            if entry.get("transcript_path")
            else transcript_path
        )
        if status == "complete" and recorded_mom.is_file() and recorded_mom.stat().st_size > 0:
            return "complete"
        if recorded_transcript.is_file() and recorded_transcript.stat().st_size > 0:
            # Keep MoM path from registry when resuming so we do not fork artifacts.
            return "transcribed"
    if mom_path.is_file() and mom_path.stat().st_size > 0:
        return "complete"
    if transcript_path.is_file() and transcript_path.stat().st_size > 0:
        return "transcribed"
    return "pending"


def build_jobs(
    audio_files: Sequence[Path],
    output_dir: Path,
    registry: Dict[str, Any],
    force: bool = False,
) -> List[AudioJob]:
    items = registry.get("items", {})
    claimed_stems: Dict[str, str] = {}
    # Prefer stems already recorded so renames keep stable output names.
    for content_hash, entry in items.items():
        if not isinstance(entry, dict):
            continue
        recorded_stem = entry.get("stem")
        if isinstance(recorded_stem, str) and recorded_stem:
            claimed_stems.setdefault(recorded_stem, content_hash)

    jobs: List[AudioJob] = []
    for audio_path in audio_files:
        content_hash = sha256_file(audio_path)
        entry = items.get(content_hash) if isinstance(items.get(content_hash), dict) else None

        if entry and isinstance(entry.get("stem"), str) and entry["stem"]:
            stem = entry["stem"]
            claimed_stems.setdefault(stem, content_hash)
        else:
            stem = unique_stem(audio_path.stem, content_hash, claimed_stems)

        if entry and entry.get("transcript_path") and entry.get("mom_path"):
            transcript_path = Path(entry["transcript_path"])
            mom_path = Path(entry["mom_path"])
        else:
            transcript_path, mom_path = artifact_paths(output_dir, stem)

        status = job_status_from_registry(entry, transcript_path, mom_path, force=force)
        jobs.append(
            AudioJob(
                path=audio_path,
                content_hash=content_hash,
                stem=stem,
                transcript_path=transcript_path,
                mom_path=mom_path,
                status=status,
            )
        )
    return jobs


def update_registry_item(
    registry: Dict[str, Any],
    job: AudioJob,
    status: str,
) -> None:
    items = registry.setdefault("items", {})
    previous = items.get(job.content_hash, {}) if isinstance(items.get(job.content_hash), dict) else {}
    items[job.content_hash] = {
        "status": status,
        "source_name": job.path.name,
        "source_path": str(job.path.resolve()),
        "stem": job.stem,
        "transcript_path": str(job.transcript_path),
        "mom_path": str(job.mom_path),
        "processed_at": utc_now_iso(),
        "source_size": job.path.stat().st_size,
        "first_seen_at": previous.get("first_seen_at") or utc_now_iso(),
    }


def normalize_omlx_base_url(base_url: str) -> str:
    url = base_url.strip().rstrip("/")
    if not url:
        raise MomPipelineError("OMLX base URL cannot be empty.")
    return url


def chat_completions_url(base_url: str) -> str:
    url = normalize_omlx_base_url(base_url)
    if url.endswith("/chat/completions"):
        return url
    return f"{url}/chat/completions"


def resolve_omlx_api_key(
    explicit: Optional[str] = None,
    settings_path: Optional[Path] = None,
) -> Optional[str]:
    """Resolve oMLX API key from arg, env, or ~/.omlx/settings.json."""
    if explicit is not None and explicit.strip():
        return explicit.strip()
    env_key = os.environ.get("OMLX_API_KEY", "").strip()
    if env_key:
        return env_key
    path = (settings_path or DEFAULT_OMLX_SETTINGS).expanduser()
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    auth = data.get("auth") if isinstance(data, dict) else None
    if not isinstance(auth, dict):
        return None
    key = auth.get("api_key")
    if isinstance(key, str) and key.strip():
        return key.strip()
    return None


def generate_mom_markdown(
    transcript: str,
    source_name: str,
    system_prompt: str,
    *,
    base_url: str,
    model: str,
    api_key: Optional[str] = None,
    timeout_seconds: float = 600.0,
) -> str:
    """Call an OpenAI-compatible chat completions endpoint (oMLX)."""
    if not transcript.strip():
        raise MomPipelineError("Cannot generate MoM from an empty transcript.")

    user_message = (
        f"Source audio filename: {source_name}\n\n"
        f"Transcript:\n\n{transcript.strip()}\n"
    )
    payload = {
        "model": model,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    }
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    resolved_key = resolve_omlx_api_key(api_key)
    if resolved_key:
        headers["Authorization"] = f"Bearer {resolved_key}"
    request = urllib.request.Request(
        chat_completions_url(base_url),
        data=body,
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise MomPipelineError(
            f"oMLX HTTP {exc.code} from {chat_completions_url(base_url)}: {detail[:500]}"
        ) from exc
    except urllib.error.URLError as exc:
        raise MomPipelineError(
            f"oMLX not reachable at {normalize_omlx_base_url(base_url)} ({exc.reason}). "
            "Start oMLX and ensure the model is loaded, or set OMLX_BASE_URL."
        ) from exc
    except TimeoutError as exc:
        raise MomPipelineError(
            f"oMLX timed out after {timeout_seconds:.0f}s while generating MoM."
        ) from exc

    try:
        data = json.loads(raw)
        content = data["choices"][0]["message"]["content"]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
        raise MomPipelineError(
            f"Unexpected oMLX response shape: {raw[:500]}"
        ) from exc

    if not isinstance(content, str) or not content.strip():
        raise MomPipelineError("oMLX returned an empty MoM.")
    return content.strip() + "\n"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")


def transcribe_one(
    audio_path: Path,
    transcriber: transcribe_m4a.AudioTranscriber,
) -> str:
    with tempfile.TemporaryDirectory(prefix="mom-from-audio-") as temp_dir:
        wav_path = Path(temp_dir) / "audio.wav"
        transcribe_m4a.convert_m4a_to_wav(audio_path, wav_path)
        transcript = transcribe_m4a.transcribe_wav(wav_path, transcriber)
    if not transcript:
        raise transcribe_m4a.TranscriptionError(
            f"Whisper returned an empty transcription for '{audio_path.name}'."
        )
    return transcript


def run_pipeline(
    *,
    inbox_dir: Path,
    output_dir: Path,
    registry_path: Path,
    audio_files: Optional[Sequence[Path]] = None,
    language: str = "en",
    omlx_base_url: str = DEFAULT_OMLX_BASE_URL,
    omlx_model: str = DEFAULT_OMLX_MODEL,
    omlx_api_key: Optional[str] = None,
    prompt_path: Optional[Path] = None,
    force: bool = False,
    skip_mom: bool = False,
) -> int:
    inbox_dir = inbox_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    registry_path = registry_path.expanduser().resolve()
    prompt_path = (prompt_path or default_prompt_path()).expanduser().resolve()

    if audio_files is None:
        if not inbox_dir.is_dir():
            inbox_dir.mkdir(parents=True, exist_ok=True)
            print(
                f"Created inbox folder: {inbox_dir}\n"
                "Drop .m4a files there and run again.",
                file=sys.stderr,
            )
            return 0
        audio_files = list_inbox_m4a(inbox_dir)
    else:
        audio_files = [path.expanduser().resolve() for path in audio_files]
        for path in audio_files:
            if not path.is_file():
                raise MomPipelineError(f"Audio file not found: {path}")
            if path.suffix.lower() != ".m4a":
                raise MomPipelineError(f"Expected an .m4a audio file: {path}")

    if not audio_files:
        print(f"No .m4a files found in {inbox_dir}", file=sys.stderr)
        return 0

    registry = load_registry(registry_path)
    jobs = build_jobs(audio_files, output_dir, registry, force=force)

    complete = [job for job in jobs if job.status == "complete"]
    need_transcript = [job for job in jobs if job.status == "pending"]
    need_mom = [job for job in jobs if job.status in ("pending", "transcribed")]

    for job in complete:
        print(
            f"SKIP (already processed): {job.path.name} → {job.mom_path.name}",
            file=sys.stderr,
        )

    if not need_transcript and (skip_mom or not need_mom):
        print("Nothing new to process.", file=sys.stderr)
        return 0

    # Phase 1: Whisper for all pending audio (model loaded once).
    if need_transcript:
        print(
            f"Transcribing {len(need_transcript)} file(s) with mlx-whisper (Metal)...",
            file=sys.stderr,
        )
        transcriber: Optional[transcribe_m4a.AudioTranscriber] = None
        try:
            transcriber = transcribe_m4a.AudioTranscriber(language=language)
            for job in need_transcript:
                print(f"  Transcribe: {job.path.name}", file=sys.stderr)
                try:
                    transcript = transcribe_one(job.path, transcriber)
                except transcribe_m4a.TranscriptionError as exc:
                    print(f"  ERROR: {exc}", file=sys.stderr)
                    continue
                write_text(job.transcript_path, transcript)
                job.status = "transcribed"
                update_registry_item(registry, job, status="transcribed")
                save_registry(registry_path, registry)
                print(f"  Wrote {job.transcript_path}", file=sys.stderr)
        finally:
            if transcriber is not None:
                print("Releasing Whisper model from memory...", file=sys.stderr)
                transcriber.cleanup()

    if skip_mom:
        print("Skipping MoM generation (--skip-mom).", file=sys.stderr)
        return 0

    # Phase 2: MoM for every job that has a transcript but no complete MoM.
    ordered_mom_jobs: List[AudioJob] = []
    seen_hashes = set()
    for job in jobs:
        if job.content_hash in seen_hashes:
            continue
        if not job.transcript_path.is_file() or job.transcript_path.stat().st_size == 0:
            continue
        mom_ready = job.mom_path.is_file() and job.mom_path.stat().st_size > 0
        if not force and job.status == "complete" and mom_ready:
            continue
        if not force and mom_ready and job.status != "transcribed":
            # Artifact present even if registry was incomplete — treat as done.
            job.status = "complete"
            update_registry_item(registry, job, status="complete")
            save_registry(registry_path, registry)
            continue
        seen_hashes.add(job.content_hash)
        ordered_mom_jobs.append(job)

    if not ordered_mom_jobs:
        print("No MoMs to generate.", file=sys.stderr)
        return 0

    system_prompt = load_system_prompt(prompt_path)
    resolved_key = resolve_omlx_api_key(omlx_api_key)
    if not resolved_key:
        print(
            "Warning: no oMLX API key found (OMLX_API_KEY or ~/.omlx/settings.json). "
            "Requests may fail with 401.",
            file=sys.stderr,
        )
    print(
        f"Generating {len(ordered_mom_jobs)} MoM(s) via oMLX "
        f"({omlx_model} @ {normalize_omlx_base_url(omlx_base_url)})...",
        file=sys.stderr,
    )

    failures = 0
    for job in ordered_mom_jobs:
        print(f"  MoM: {job.path.name}", file=sys.stderr)
        try:
            transcript = job.transcript_path.read_text(encoding="utf-8")
            mom = generate_mom_markdown(
                transcript,
                source_name=job.path.name,
                system_prompt=system_prompt,
                base_url=omlx_base_url,
                model=omlx_model,
                api_key=resolved_key,
            )
            write_text(job.mom_path, mom)
            job.status = "complete"
            update_registry_item(registry, job, status="complete")
            save_registry(registry_path, registry)
            print(f"  Wrote {job.mom_path}", file=sys.stderr)
        except (OSError, MomPipelineError) as exc:
            failures += 1
            print(f"  ERROR: {exc}", file=sys.stderr)

    if failures:
        print(f"Finished with {failures} MoM failure(s).", file=sys.stderr)
        return 1
    print("Done.", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Process .m4a files from a Desktop drop folder: transcribe with Whisper, "
            "then generate Minutes of Meeting via oMLX. With no arguments, scans "
            f"{DEFAULT_INBOX}."
        )
    )
    parser.add_argument(
        "audio_files",
        nargs="*",
        type=Path,
        help="Optional explicit .m4a paths (skips inbox scan when provided)",
    )
    parser.add_argument(
        "--inbox",
        type=Path,
        default=Path(os.environ.get("MOM_INBOX_DIR", str(DEFAULT_INBOX))),
        help=f"Drop folder for .m4a files (default: {DEFAULT_INBOX})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write .transcript.txt and .mom.md (default: <inbox>/output)",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=None,
        help=f"Processing registry path (default: <inbox>/{REGISTRY_NAME})",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Whisper language name or code (default: en)",
    )
    parser.add_argument(
        "--omlx-base-url",
        default=os.environ.get("OMLX_BASE_URL", DEFAULT_OMLX_BASE_URL),
        help=f"OpenAI-compatible base URL (default: {DEFAULT_OMLX_BASE_URL})",
    )
    parser.add_argument(
        "--omlx-model",
        default=os.environ.get("OMLX_MODEL", DEFAULT_OMLX_MODEL),
        help=f"Model id served by oMLX (default: {DEFAULT_OMLX_MODEL})",
    )
    parser.add_argument(
        "--omlx-api-key",
        default=None,
        help=(
            "oMLX API key (default: OMLX_API_KEY env, else auth.api_key from "
            "~/.omlx/settings.json)"
        ),
    )
    parser.add_argument(
        "--prompt",
        type=Path,
        default=None,
        help=f"System prompt file (default: {PROMPT_RELATIVE})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-process even if content hash is already complete",
    )
    parser.add_argument(
        "--skip-mom",
        action="store_true",
        help="Only transcribe; do not call oMLX",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    language = args.language.strip()
    if not language:
        parser.error("--language cannot be empty.")

    inbox_dir = args.inbox
    output_dir = args.output_dir or (inbox_dir / "output")
    registry_path = args.registry or (inbox_dir / REGISTRY_NAME)
    explicit = list(args.audio_files) if args.audio_files else None

    try:
        return run_pipeline(
            inbox_dir=inbox_dir,
            output_dir=output_dir,
            registry_path=registry_path,
            audio_files=explicit,
            language=language,
            omlx_base_url=args.omlx_base_url,
            omlx_model=args.omlx_model,
            omlx_api_key=args.omlx_api_key,
            prompt_path=args.prompt,
            force=args.force,
            skip_mom=args.skip_mom,
        )
    except (MomPipelineError, transcribe_m4a.TranscriptionError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
