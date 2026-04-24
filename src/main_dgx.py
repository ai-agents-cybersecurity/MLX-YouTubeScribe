#!/usr/bin/env python3
# =========================================================================
# MLX-YouTubeScribe — NVIDIA DGX Spark edition (transcription only)
# Backend: transformers Whisper native long-form via model.generate (CUDA)
# =========================================================================

import argparse
import gc
import json
import operator
import os
import re
from typing import Annotated, List, Optional, TypedDict

import yt_dlp
from langgraph.graph import END, StateGraph

DEFAULT_MODEL = "openai/whisper-large-v3"
DEFAULT_DTYPE = "bfloat16"  # Blackwell-native; override with --dtype
TARGET_SR = 16000


class WhisperTranscriber:
    _instance = None

    def __init__(self, model_id: str, dtype: str, language: Optional[str]):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable — the DGX path requires a CUDA GPU")

        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[dtype]

        print(f"Loading '{model_id}' on {torch.cuda.get_device_name(0)} ({dtype})...")
        self.model_id = model_id
        self.torch_dtype = torch_dtype
        self.language = language
        self.processor = WhisperProcessor.from_pretrained(model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_id, dtype=torch_dtype
        ).to("cuda")
        self.model.eval()
        print("Model ready")

    @classmethod
    def get_instance(cls, model_id: str = DEFAULT_MODEL,
                     dtype: str = DEFAULT_DTYPE,
                     language: Optional[str] = None):
        if cls._instance is None:
            cls._instance = cls(model_id, dtype, language)
        return cls._instance

    @classmethod
    def cleanup(cls):
        if cls._instance is not None:
            del cls._instance.model
            del cls._instance.processor
            cls._instance = None
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    def transcribe(self, audio_path: str) -> dict:
        import librosa
        import torch

        audio, _ = librosa.load(audio_path, sr=TARGET_SR, mono=True)

        inputs = self.processor(
            audio,
            sampling_rate=TARGET_SR,
            return_tensors="pt",
            truncation=False,
            padding="longest",
            return_attention_mask=True,
        )
        input_features = inputs.input_features.to("cuda", dtype=self.torch_dtype)
        attention_mask = inputs.attention_mask.to("cuda")

        gen_kwargs = {
            "return_timestamps": True,
            "task": "transcribe",
            "condition_on_prev_tokens": False,
            "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            "compression_ratio_threshold": 1.35,
            "logprob_threshold": -1.0,
            "no_speech_threshold": 0.6,
            "no_repeat_ngram_size": 3,
        }
        if self.language:
            gen_kwargs["language"] = self.language

        with torch.inference_mode():
            output = self.model.generate(
                input_features,
                attention_mask=attention_mask,
                **gen_kwargs,
            )

        # output is (1, seq_len); decode the single sequence so `text` is a
        # string and `offsets` is a flat list. Passing the 2D tensor to
        # batch_decode in transformers >=5 returns a dict with list-valued text.
        sequences = output[0] if output.ndim > 1 else output
        result = self.processor.tokenizer.decode(
            sequences, skip_special_tokens=True, output_offsets=True
        )

        text = (result.get("text") or "").strip()
        segments: List[dict] = []
        for off in result.get("offsets") or []:
            ts = off.get("timestamp") or (None, None)
            segments.append({
                "text": off.get("text", ""),
                "start": ts[0],
                "end": ts[1],
            })

        return {"text": text, "segments": segments}


def clean_video_url(url: str) -> str:
    patterns = [
        r"(?:youtube\.com/shorts/)([a-zA-Z0-9_-]{11})",
        r"(?:youtube\.com/watch\?v=)([a-zA-Z0-9_-]{11})",
        r"(?:youtu\.be/)([a-zA-Z0-9_-]{11})",
        r"(?:youtube\.com/embed/)([a-zA-Z0-9_-]{11})",
        r"(?:youtube\.com/v/)([a-zA-Z0-9_-]{11})",
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return f"https://www.youtube.com/watch?v={m.group(1)}"
    return url


def sanitize_title(title: str) -> str:
    # Match yt-dlp's fullwidth replacements so local paths line up with downloaded files.
    fullwidth = {
        '"': "\uff02", "*": "\uff0a", "/": "\uff0f", ":": "\uff1a",
        "<": "\uff1c", ">": "\uff1e", "?": "\uff1f", "\\": "\uff3c", "|": "\uff5c",
    }
    for c, r in fullwidth.items():
        title = title.replace(c, r)
    return title


def resolve_videos(url: str, cookies_from_browser: Optional[str]) -> dict:
    """Expand a URL into {playlist_title, videos: [{url, title, id}]}."""
    ydl_opts = {"quiet": True, "extract_flat": "in_playlist"}
    if cookies_from_browser:
        ydl_opts["cookiesfrombrowser"] = (cookies_from_browser,)

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    if info.get("_type") == "playlist":
        videos = []
        for entry in info.get("entries", []) or []:
            if not entry:
                continue
            vid_id = entry.get("id")
            videos.append({
                "url": entry.get("url") or f"https://www.youtube.com/watch?v={vid_id}",
                "title": entry.get("title") or vid_id,
                "id": vid_id,
            })
        return {
            "playlist_title": info.get("title") or "Unknown Playlist",
            "playlist_id": info.get("id") or "",
            "videos": videos,
        }

    return {
        "playlist_title": info.get("title") or info.get("id") or "video",
        "playlist_id": "",
        "videos": [{
            "url": clean_video_url(url),
            "title": info.get("title") or info.get("id"),
            "id": info.get("id"),
        }],
    }


def download_audio(url: str, output_dir: str,
                   cookies_from_browser: Optional[str]) -> Optional[str]:
    """Download best audio, convert to 16 kHz mono WAV in output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(output_dir, "%(title)s.%(ext)s"),
        "quiet": False,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
            "preferredquality": "0",
        }],
        "postprocessor_args": ["-ar", str(TARGET_SR), "-ac", "1"],
    }
    if cookies_from_browser:
        ydl_opts["cookiesfrombrowser"] = (cookies_from_browser,)

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
    if not info:
        return None

    title = sanitize_title(info.get("title") or info.get("id") or "")
    expected = os.path.join(output_dir, f"{title}.wav")
    if os.path.exists(expected):
        return expected
    for f in os.listdir(output_dir):
        if f.endswith(".wav"):
            return os.path.join(output_dir, f)
    return None


class State(TypedDict):
    url: str
    output_dir: str
    cookies_from_browser: Optional[str]
    model_id: str
    dtype: str
    language: Optional[str]
    playlist_title: str
    videos: List[dict]
    current_idx: int
    results: Annotated[List[dict], operator.add]


def resolve_node(state: State) -> dict:
    info = resolve_videos(state["url"], state.get("cookies_from_browser"))
    print(f"Found {len(info['videos'])} video(s) in '{info['playlist_title']}'")
    return {
        "playlist_title": info["playlist_title"],
        "videos": info["videos"],
        "current_idx": 0,
    }


def process_video_node(state: State) -> dict:
    idx = state["current_idx"]
    total = len(state["videos"])
    vid = state["videos"][idx]

    print(f"\n[{idx + 1}/{total}] {vid['title']}")

    playlist_dir = os.path.join(state["output_dir"], sanitize_title(state["playlist_title"]))
    audio_dir = os.path.join(playlist_dir, "audio")
    title = sanitize_title(vid["title"])
    wav_path = os.path.join(audio_dir, f"{title}.wav")
    txt_path = os.path.join(playlist_dir, f"{title}.txt")
    json_path = os.path.join(playlist_dir, f"{title}.json")

    if os.path.exists(txt_path) and os.path.exists(json_path):
        print("  Already processed — skipping")
        return {
            "current_idx": idx + 1,
            "results": [{"title": vid["title"], "status": "skipped"}],
        }

    if not os.path.exists(wav_path):
        print("  Downloading audio...")
        wav_path = download_audio(vid["url"], audio_dir, state.get("cookies_from_browser"))
        if not wav_path or not os.path.exists(wav_path):
            print("  ✗ Download failed")
            return {
                "current_idx": idx + 1,
                "results": [{"title": vid["title"], "status": "download_failed"}],
            }

    print(f"  Transcribing with '{state['model_id']}'...")
    transcriber = WhisperTranscriber.get_instance(
        state["model_id"], state["dtype"], state.get("language"),
    )
    result = transcriber.transcribe(wav_path)

    os.makedirs(playlist_dir, exist_ok=True)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(result["text"])
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "title": vid["title"],
            "url": vid["url"],
            "id": vid.get("id"),
            "model": state["model_id"],
            "text": result["text"],
            "segments": result["segments"],
        }, f, indent=2, default=str, ensure_ascii=False)

    print(f"  ✓ {len(result['text'])} chars, {len(result['segments'])} segments")
    return {
        "current_idx": idx + 1,
        "results": [{"title": vid["title"], "status": "ok",
                     "chars": len(result["text"])}],
    }


def should_continue(state: State) -> str:
    return "process" if state["current_idx"] < len(state["videos"]) else "done"


def finalize_node(state: State) -> dict:
    ok = sum(1 for r in state["results"] if r["status"] == "ok")
    skipped = sum(1 for r in state["results"] if r["status"] == "skipped")
    failed = len(state["results"]) - ok - skipped
    print(f"\n=== Done: {ok} transcribed, {skipped} skipped, {failed} failed ===")
    WhisperTranscriber.cleanup()
    return {}


def build_graph():
    g = StateGraph(State)
    g.add_node("resolve", resolve_node)
    g.add_node("process", process_video_node)
    g.add_node("finalize", finalize_node)
    g.set_entry_point("resolve")
    g.add_conditional_edges("resolve", should_continue,
                            {"process": "process", "done": "finalize"})
    g.add_conditional_edges("process", should_continue,
                            {"process": "process", "done": "finalize"})
    g.add_edge("finalize", END)
    return g.compile()


def main():
    ap = argparse.ArgumentParser(
        description="YouTube transcription on NVIDIA DGX Spark (transformers Whisper)"
    )
    ap.add_argument("url", help="YouTube video or playlist URL")
    ap.add_argument("-o", "--output-dir", default="output", help="Output directory")
    ap.add_argument("--cookies-from-browser", default=None,
                    help="Browser to source cookies from (chrome, firefox, ...) "
                         "for Premium or age-restricted content")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"HF model id (default: {DEFAULT_MODEL}). "
                         "e.g. openai/whisper-large-v3, openai/whisper-large-v3-turbo, "
                         "distil-whisper/distil-large-v3")
    ap.add_argument("--dtype", default=DEFAULT_DTYPE,
                    choices=["bfloat16", "float16", "float32"],
                    help=f"Torch dtype (default: {DEFAULT_DTYPE})")
    ap.add_argument("--language", default=None,
                    help="Force language (e.g. 'english'). Default: auto-detect.")
    args = ap.parse_args()

    graph = build_graph()
    graph.invoke(
        {
            "url": args.url,
            "output_dir": args.output_dir,
            "cookies_from_browser": args.cookies_from_browser,
            "model_id": args.model,
            "dtype": args.dtype,
            "language": args.language,
            "playlist_title": "",
            "videos": [],
            "current_idx": 0,
            "results": [],
        },
        {"recursion_limit": 2000},
    )


if __name__ == "__main__":
    main()
