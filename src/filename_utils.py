#!/usr/bin/env python3
"""Filename helpers that stay in lockstep with yt-dlp.

yt-dlp 2024+ maps `/` to U+29F8 (BIG SOLIDUS ⧸), not fullwidth U+FF0F (／).
A previous local sanitizer used the old fullwidth slash, so expected paths
missed the file on disk and a fallback picked an unrelated WAV/MP4.
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

_VIDEO_ID_RE = re.compile(
    r"(?:youtube\.com/shorts/|youtube\.com/watch\?v=|youtu\.be/"
    r"|youtube\.com/embed/|youtube\.com/v/)([a-zA-Z0-9_-]{11})"
)
_BARE_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{11}$")

# output_dir -> {video_id or abspath: json_path}
_transcript_index_cache: Dict[str, Dict[str, str]] = {}

# Current yt-dlp mapping for `/` and `\` (see yt_dlp.utils.sanitize_filename).
YTDLP_SLASH = "\u29F8"  # ⧸
YTDLP_BACKSLASH = "\u29f9"  # ⧹
# Mapping used by this repo before the yt-dlp change.
LEGACY_SLASH = "\uff0f"  # ／
LEGACY_BACKSLASH = "\uff3c"  # ＼

# Full whisper+diarize+save path is 9 node visits per video.
# Extra headroom covers start/finalize and future nodes.
GRAPH_STEPS_PER_VIDEO = 15
GRAPH_STEP_OVERHEAD = 50
GRAPH_MIN_RECURSION_LIMIT = 1000


def sanitize_title(title: str) -> str:
    """Sanitize a title the same way yt-dlp sanitizes filenames."""
    from yt_dlp.utils import sanitize_filename

    return sanitize_filename(title or "")


def _normalize_stem(stem: str) -> str:
    return (
        stem.replace(YTDLP_SLASH, "/")
        .replace(LEGACY_SLASH, "/")
        .replace(YTDLP_BACKSLASH, "\\")
        .replace(LEGACY_BACKSLASH, "\\")
    )


def stems_equivalent(left: str, right: str) -> bool:
    """True if two filename stems differ only by slash encoding."""
    return _normalize_stem(left) == _normalize_stem(right)


def filename_stem_equivalents(stem: str) -> List[str]:
    """Return current and legacy slash-encoded stems, current first."""
    variants = [
        stem,
        stem.replace(YTDLP_SLASH, LEGACY_SLASH).replace(YTDLP_BACKSLASH, LEGACY_BACKSLASH),
        stem.replace(LEGACY_SLASH, YTDLP_SLASH).replace(LEGACY_BACKSLASH, YTDLP_BACKSLASH),
    ]
    seen: List[str] = []
    for variant in variants:
        if variant not in seen:
            seen.append(variant)
    return seen


def _ytdlp_reported_paths(info: Optional[dict]) -> List[str]:
    if not info:
        return []
    paths: List[str] = []
    for key in ("filepath", "_filename", "filename"):
        value = info.get(key)
        if isinstance(value, str):
            paths.append(value)
    for item in info.get("requested_downloads") or []:
        if isinstance(item, dict):
            for key in ("filepath", "filename"):
                value = item.get(key)
                if isinstance(value, str):
                    paths.append(value)
    return paths


def resolve_downloaded_media(
    output_dir: str,
    info: Optional[dict],
    ext: str,
    ydl=None,
) -> Optional[str]:
    """Locate a media file written by yt-dlp. Never pick an unrelated file."""
    ext = ext.lstrip(".")
    suffix = f".{ext}"

    for path in _ytdlp_reported_paths(info):
        candidate = path
        if not candidate.endswith(suffix):
            candidate = os.path.splitext(candidate)[0] + suffix
        if os.path.isfile(candidate):
            return candidate

    if ydl is not None and info is not None:
        try:
            prepared = ydl.prepare_filename(info)
            candidate = os.path.splitext(prepared)[0] + suffix
            if os.path.isfile(candidate):
                return candidate
        except Exception:
            pass

    title = (info or {}).get("title") or ""
    if title:
        for stem in filename_stem_equivalents(sanitize_title(title)):
            candidate = os.path.join(output_dir, f"{stem}{suffix}")
            if os.path.isfile(candidate):
                return candidate

    return None


def youtube_video_id(url_or_id: Optional[str]) -> Optional[str]:
    """Extract an 11-char YouTube video id from a URL or bare id."""
    if not url_or_id:
        return None
    value = url_or_id.strip()
    if _BARE_ID_RE.fullmatch(value):
        return value
    match = _VIDEO_ID_RE.search(value)
    return match.group(1) if match else None


def existing_output_paths(output_dir: str, title: str) -> Tuple[Optional[str], Optional[str]]:
    """Find json/txt written under current or legacy slash encoding."""
    json_path = None
    txt_path = None
    for stem in filename_stem_equivalents(sanitize_title(title)):
        jp = os.path.join(output_dir, f"{stem}.json")
        tp = os.path.join(output_dir, f"{stem}.txt")
        if json_path is None and os.path.isfile(jp):
            json_path = jp
        if txt_path is None and os.path.isfile(tp):
            txt_path = tp
        if json_path and txt_path:
            break
    return json_path, txt_path


def invalidate_transcript_index(output_dir: str) -> None:
    _transcript_index_cache.pop(os.path.abspath(output_dir), None)


def index_transcripts_by_video_id(output_dir: str) -> Dict[str, str]:
    """Map YouTube video id (or local source path) -> transcript json path."""
    key = os.path.abspath(output_dir)
    cached = _transcript_index_cache.get(key)
    if cached is not None:
        return cached
    index: Dict[str, str] = {}
    if os.path.isdir(output_dir):
        for name in os.listdir(output_dir):
            if not name.endswith(".json"):
                continue
            path = os.path.join(output_dir, name)
            try:
                with open(path, encoding="utf-8") as handle:
                    data = json.load(handle)
            except (OSError, json.JSONDecodeError):
                continue
            info = data.get("video_info") or {}
            vid = youtube_video_id(info.get("url"))
            if vid:
                index[vid] = path
            source = info.get("source")
            if source:
                index[os.path.abspath(source)] = path
    _transcript_index_cache[key] = index
    return index


def _companion_txt(json_path: Optional[str]) -> Optional[str]:
    if not json_path:
        return None
    txt_path = os.path.splitext(json_path)[0] + ".txt"
    return txt_path if os.path.isfile(txt_path) else None


def find_existing_transcript(
    output_dir: str,
    title: Optional[str] = None,
    url: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """Find a transcript by title (slash-tolerant) or by video id / local path."""
    if title:
        json_path, txt_path = existing_output_paths(output_dir, title)
        if json_path or txt_path:
            return json_path, txt_path

    if url:
        index = index_transcripts_by_video_id(output_dir)
        vid = youtube_video_id(url)
        lookup_keys = [k for k in (vid, os.path.abspath(url) if os.path.exists(url) else None) if k]
        for lookup in lookup_keys:
            json_path = index.get(lookup)
            if json_path and os.path.isfile(json_path):
                return json_path, _companion_txt(json_path)
    return None, None


def transcript_audio_matches(json_path: str) -> bool:
    """False when a transcript JSON points at a different video's audio file."""
    try:
        with open(json_path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    audio_file = data.get("audio_file")
    if not audio_file:
        return True
    json_stem = os.path.splitext(os.path.basename(json_path))[0]
    audio_stem = os.path.splitext(os.path.basename(audio_file))[0]
    return stems_equivalent(json_stem, audio_stem)


def quarantine_output_files(output_dir: str, paths: Sequence[str], reason: str) -> None:
    dest_dir = os.path.join(output_dir, "_corrupt_transcripts")
    os.makedirs(dest_dir, exist_ok=True)
    for path in paths:
        if not path or not os.path.isfile(path):
            continue
        dest = os.path.join(dest_dir, os.path.basename(path))
        if os.path.exists(dest):
            base, ext = os.path.splitext(dest)
            dest = f"{base}.dup{ext}"
        print(f"  ⚠️  Quarantining {os.path.basename(path)} ({reason})")
        os.rename(path, dest)
    invalidate_transcript_index(output_dir)


def recursion_limit_for(n_videos: int) -> int:
    """LangGraph recursion_limit large enough for n playlist items."""
    n = max(int(n_videos), 1)
    return max(GRAPH_MIN_RECURSION_LIMIT, n * GRAPH_STEPS_PER_VIDEO + GRAPH_STEP_OVERHEAD)
