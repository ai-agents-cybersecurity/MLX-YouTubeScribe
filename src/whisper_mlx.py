#!/usr/bin/env python3
"""mlx-whisper backend: Whisper on Apple GPU (Metal), not PyTorch CPU/MPS."""

from __future__ import annotations

import gc
import os
import sys
from pathlib import Path
from typing import Optional, Union

import numpy as np


MODEL_ID = "mlx-community/whisper-large-v3-turbo"
SAMPLE_RATE = 16_000

AudioInput = Union[str, os.PathLike, np.ndarray]


def _import_mlx_whisper():
    try:
        import mlx.core as mx
        import mlx_whisper
        from mlx_whisper.transcribe import ModelHolder
    except ImportError as exc:
        missing = getattr(exc, "name", None) or "mlx-whisper"
        raise ImportError(
            f"Missing dependency '{missing}'. Install local STT deps with: "
            "pip install mlx-whisper"
        ) from exc
    return mx, mlx_whisper, ModelHolder


def _clear_mlx_cache() -> None:
    try:
        import mlx.core as mx
    except ImportError:
        return
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()
    elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
        mx.metal.clear_cache()


def unload_whisper_model() -> None:
    """Drop the cached mlx-whisper weights so a later LLM can use the GPU."""
    try:
        from mlx_whisper.transcribe import ModelHolder
    except ImportError:
        return
    ModelHolder.model = None
    ModelHolder.model_path = None
    gc.collect()
    _clear_mlx_cache()


def as_whisper_audio(audio: AudioInput) -> Union[str, np.ndarray]:
    """Prepare audio for mlx-whisper without going through Python lists."""
    if isinstance(audio, (str, os.PathLike, Path)):
        return os.fspath(audio)

    if isinstance(audio, np.ndarray):
        array = np.ascontiguousarray(audio, dtype=np.float32)
    else:
        # mx.array implements __array__; do not use .tolist() (copies every sample).
        array = np.asarray(audio, dtype=np.float32)

    if array.ndim > 1:
        array = np.mean(array, axis=1 if array.shape[-1] <= 8 else 0)
    return np.ascontiguousarray(array, dtype=np.float32)


class AudioTranscriber:
    """Load mlx-whisper once and transcribe files or 16 kHz float32 waveforms."""

    _instance: Optional["AudioTranscriber"] = None

    def __init__(self, model_id: str = MODEL_ID, language: str = "en") -> None:
        self.model_id = model_id
        self.language = language
        self._load_model()

    def _load_model(self) -> None:
        mx, _, ModelHolder = _import_mlx_whisper()
        print(
            f"Loading mlx-whisper '{self.model_id}' on Apple GPU (Metal)...",
            file=sys.stderr,
        )
        ModelHolder.get_model(self.model_id, mx.float16)
        print("mlx-whisper ready (Metal).", file=sys.stderr)

    @classmethod
    def get_instance(cls, language: str = "en") -> "AudioTranscriber":
        if cls._instance is None:
            cls._instance = cls(language=language)
        return cls._instance

    def transcribe(
        self,
        audio: AudioInput,
        *,
        verbose: Optional[bool] = None,
    ) -> str:
        """Transcribe a wav path or a mono float32 waveform at 16 kHz."""
        _, mlx_whisper, _ = _import_mlx_whisper()
        prepared = as_whisper_audio(audio)
        result = mlx_whisper.transcribe(
            prepared,
            path_or_hf_repo=self.model_id,
            language=self.language,
            task="transcribe",
            fp16=True,
            verbose=verbose,
        )
        return (result.get("text") or "").strip()

    def cleanup(self) -> None:
        unload_whisper_model()
        if AudioTranscriber._instance is self:
            AudioTranscriber._instance = None

    @classmethod
    def cleanup_singleton(cls) -> None:
        if cls._instance is not None:
            cls._instance.cleanup()
        else:
            unload_whisper_model()
