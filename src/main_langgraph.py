4#!/usr/bin/env python3

# =========================================================================

# MLX-YouTubeScribe - LangGraph Version

# Objective: Generate transcripts from YouTube videos using local Whisper models with MLX acceleration

# Ported to LangGraph for modular, stateful execution

# =========================================================================

# Author: spidernic (original), ported with AI assistance

# Created: May 2025, Ported: July 2025

# =========================================================================

# Copyright 2025 spidernic

# [Apache License 2.0 details]

import os
import re
import json
import argparse
import yt_dlp
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from typing import Any, List, Tuple, Dict, Optional, TypedDict, Annotated
from scipy.io import wavfile
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from dataclasses import dataclass
import torch
from langgraph.graph import StateGraph, END
import operator
import gc

modelo1 = "openai/whisper-large-v3-turbo"
modelo2 = "openai/whisper-tiny.en"

@dataclass
class AudioTranscriber:
    """Audio transcription using Whisper (Singleton Pattern)"""
    processor: WhisperProcessor
    model: WhisperForConditionalGeneration
    _instance = None

    def __init__(self):
        # Initialize with English-only model
        self.processor = WhisperProcessor.from_pretrained(modelo1)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            modelo1,
            torch_dtype=torch.float32  # Use float32 to match input dtype
        )
    
    @classmethod
    def get_instance(cls):
        """Get or create singleton instance"""
        if cls._instance is None:
            cls._instance = AudioTranscriber()
        return cls._instance
    
    @classmethod
    def cleanup(cls):
        """Clean up the singleton instance"""
        if cls._instance is not None:
            # Clear model from memory
            del cls._instance.model
            del cls._instance.processor
            cls._instance = None
            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def transcribe(self, audio_features: mx.array) -> str:
        """Transcribe audio using Whisper"""
        try:
            # Convert MLX array to numpy for processing
            audio_np = audio_features.tolist()
            if isinstance(audio_np, list):
                audio_np = np.array(audio_np)

            # Process audio features
            inputs = self.processor(
                audio_np,
                return_tensors="pt",
                sampling_rate=16000,
                return_attention_mask=True,
                language="en"  # Explicitly set English as target language
            )

            # Generate transcription using PyTorch
            with torch.no_grad():
                generated_ids = self.model.generate(
                    inputs.input_features,
                    attention_mask=inputs.attention_mask,
                    return_timestamps=False,
                    max_length=448,  # Limit output length
                    language='en',  # Force English language output
                    task='transcribe'
                )

            # Decode the output
            transcription = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0].strip()

            # Clean up intermediate tensors
            del audio_np, inputs, generated_ids
            gc.collect()

            return transcription
        except Exception as e:
            print(f"Error in transcription: {str(e)}")
            return ""


class VoxtralTranscriber:
    """Audio transcription using Voxtral-Mini-3B via mlx-voxtral (Singleton Pattern)"""
    _instance = None

    def __init__(self):
        from mlx_voxtral import VoxtralForConditionalGeneration, VoxtralProcessor
        model_id = "mistralai/Voxtral-Mini-3B-2507"
        print(f"Loading Voxtral model: {model_id}...")
        self.model = VoxtralForConditionalGeneration.from_pretrained(model_id)
        self.processor = VoxtralProcessor.from_pretrained(model_id)
        print("Voxtral model loaded!")

    @classmethod
    def get_instance(cls):
        """Get or create singleton instance"""
        if cls._instance is None:
            cls._instance = VoxtralTranscriber()
        return cls._instance

    @classmethod
    def cleanup(cls):
        """Clean up the singleton instance"""
        if cls._instance is not None:
            del cls._instance.model
            del cls._instance.processor
            cls._instance = None
            gc.collect()

    def transcribe(self, audio_path: str) -> str:
        """Transcribe audio file using Voxtral"""
        try:
            inputs = self.processor.apply_transcrition_request(
                audio=audio_path,
                language="en"
            )
            outputs = self.model.generate(
                input_ids=inputs.input_ids,
                input_features=inputs.input_features,
                max_new_tokens=4096,
                temperature=0.0
            )
            transcript = self.processor.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            ).strip()
            return transcript
        except Exception as e:
            print(f"Error in Voxtral transcription: {str(e)}")
            return ""


@dataclass
class DemucsManager:
    """Audio source separation using Demucs (Singleton Pattern)"""
    model: Any
    _instance = None

    def __init__(self):
        from demucs.pretrained import get_model
        from demucs.apply import apply_model
        import torchaudio

        # Load htdemucs_ft (fine-tuned) - best quality for vocal separation
        # Other options: htdemucs, htdemucs_6s, mdx_extra, mdx_extra_q
        print("Loading Demucs model (htdemucs_ft) for vocal isolation...")
        self.model = get_model("htdemucs_ft")

        # Determine device - prefer MPS on Apple Silicon
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using Apple Silicon GPU (MPS) for Demucs")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Using CUDA GPU for Demucs")
        else:
            self.device = torch.device("cpu")
            print("Using CPU for Demucs")

        self.model.to(self.device)
        self.model.eval()

        # Store the sources order for reference
        # htdemucs_ft sources: ['drums', 'bass', 'other', 'vocals']
        self.sources = self.model.sources
        print(f"Demucs model loaded! Sources: {self.sources}")

    @classmethod
    def get_instance(cls):
        """Get or create singleton instance"""
        if cls._instance is None:
            cls._instance = DemucsManager()
        return cls._instance

    @classmethod
    def cleanup(cls):
        """Clean up the singleton instance"""
        if cls._instance is not None:
            del cls._instance.model
            cls._instance = None
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

    def isolate_vocals(self, audio_path: str, output_path: str = None) -> str:
        """Isolate vocals from audio file using Demucs

        Args:
            audio_path: Path to input audio file
            output_path: Path for output file (optional, will create alongside input if not specified)

        Returns:
            Path to the vocals-only audio file
        """
        from demucs.apply import apply_model
        import torchaudio

        print(f"  Isolating vocals with Demucs...")

        # Determine output path
        if output_path is None:
            base, ext = os.path.splitext(audio_path)
            output_path = f"{base}_vocals{ext}"

        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)

        # Resample to model's sample rate if needed (Demucs expects 44100 Hz)
        if sample_rate != self.model.samplerate:
            print(f"  Resampling from {sample_rate} to {self.model.samplerate} Hz...")
            resampler = torchaudio.transforms.Resample(sample_rate, self.model.samplerate)
            waveform = resampler(waveform)

        # Ensure stereo (Demucs expects stereo input)
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
        elif waveform.shape[0] > 2:
            waveform = waveform[:2]

        # Add batch dimension and move to device
        waveform = waveform.unsqueeze(0).to(self.device)

        # Apply the model
        print(f"  Running source separation (this may take a while)...")
        with torch.no_grad():
            sources = apply_model(
                self.model,
                waveform,
                device=self.device,
                shifts=1,  # Number of random shifts for better quality
                split=True,  # Split audio into segments for memory efficiency
                overlap=0.25,  # Overlap between segments
                progress=True
            )

        # Get the vocals index
        vocals_idx = self.sources.index("vocals")
        vocals = sources[0, vocals_idx]  # Remove batch dimension

        # Move to CPU and save
        vocals = vocals.cpu()
        torchaudio.save(output_path, vocals, self.model.samplerate)
        print(f"  Vocals saved to: {output_path}")

        # Clean up
        del waveform, sources
        gc.collect()

        return output_path


@dataclass
class DiarizerManager:
    """Speaker diarization using pyannote.audio (Singleton Pattern)"""
    pipeline: Any
    embedding_model: Any
    _instance = None

    def __init__(self):
        from pyannote.audio import Pipeline, Inference
        from huggingface_hub import HfFolder

        # Try to get token from environment or huggingface-cli login
        hf_token = os.environ.get("HF_TOKEN") or HfFolder.get_token()

        if not hf_token:
            raise ValueError(
                "No HuggingFace token found. Either:\n"
                "  1. Set HF_TOKEN environment variable: export HF_TOKEN='hf_...'\n"
                "  2. Or run: huggingface-cli login"
            )

        print(f"Using HuggingFace token: {hf_token[:10]}...")

        # Determine device - use MPS on Apple Silicon for faster processing
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using Apple Silicon GPU (MPS) for acceleration")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            print("Using CUDA GPU for acceleration")
        else:
            device = torch.device("cpu")
            print("Using CPU (this will be slow)")

        # Load speaker diarization pipeline
        print("Loading speaker diarization pipeline...")
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=hf_token
        )
        # Move pipeline to GPU if available
        self.pipeline.to(device)

        # Load speaker embedding model for voice matching
        print("Loading speaker embedding model...")
        from pyannote.audio import Model
        embedding_model = Model.from_pretrained(
            "pyannote/wespeaker-voxceleb-resnet34-LM",
            token=hf_token
        )
        embedding_model.to(device)
        self.embedding_model = Inference(embedding_model, window="whole")
        print("Diarization models loaded successfully!")

    @classmethod
    def get_instance(cls):
        """Get or create singleton instance"""
        if cls._instance is None:
            cls._instance = DiarizerManager()
        return cls._instance

    @classmethod
    def cleanup(cls):
        """Clean up the singleton instance"""
        if cls._instance is not None:
            del cls._instance.pipeline
            del cls._instance.embedding_model
            cls._instance = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _load_audio(self, audio_path: str) -> dict:
        """Load audio file as waveform dict for pyannote (workaround for torchcodec issues)"""
        import torchaudio
        waveform, sample_rate = torchaudio.load(audio_path)
        # Resample to 16kHz if needed (pyannote expects 16kHz)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
            sample_rate = 16000
        return {"waveform": waveform, "sample_rate": sample_rate}

    def diarize(self, audio_path: str) -> list:
        """Run speaker diarization on audio file

        Returns list of segments: [{start, end, speaker}, ...]
        """
        print("Running speaker diarization...")
        # Load audio as waveform dict (workaround for torchcodec issues)
        print("  Loading audio...")
        audio = self._load_audio(audio_path)
        duration = audio["waveform"].shape[1] / audio["sample_rate"]
        print(f"  Audio duration: {duration:.1f} seconds")
        print("  Processing (this may take a few minutes)...")
        diarization_output = self.pipeline(audio)
        print("  Diarization complete!")

        segments = []
        # Handle both pyannote 3.x (Annotation) and 4.x (DiarizeOutput) formats
        if hasattr(diarization_output, 'itertracks'):
            # pyannote 3.x format - direct Annotation object
            diarization = diarization_output
        elif hasattr(diarization_output, 'speaker_diarization'):
            # pyannote 4.x format - DiarizeOutput with speaker_diarization attribute
            diarization = diarization_output.speaker_diarization
        else:
            raise ValueError(f"Unknown diarization output format: {type(diarization_output)}")

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker
            })

        # Sort by start time
        segments.sort(key=lambda x: x["start"])
        print(f"Found {len(segments)} speaker segments")
        return segments

    def get_speaker_embedding(self, audio_path: str, start: float = None, end: float = None) -> np.ndarray:
        """Get speaker embedding for audio (or a segment of it)"""
        from pyannote.core import Segment

        # Load audio as waveform
        audio = self._load_audio(audio_path)

        if start is not None and end is not None:
            excerpt = Segment(start, end)
            embedding = self.embedding_model.crop(audio, excerpt)
        else:
            embedding = self.embedding_model(audio)

        return np.array(embedding)

    def identify_speaker_by_reference(self, audio_path: str, reference_path: str, segments: list) -> str:
        """Identify which speaker in the diarization matches the reference audio

        Args:
            audio_path: Path to the main audio file
            reference_path: Path to reference audio clip of target speaker
            segments: Diarization segments from diarize()

        Returns:
            Speaker ID that best matches the reference
        """
        from scipy.spatial.distance import cosine

        print("Matching speaker to reference audio...")

        # Get embedding of reference audio
        reference_embedding = self.get_speaker_embedding(reference_path)

        # Get unique speakers
        speakers = list(set(seg["speaker"] for seg in segments))
        print(f"Found {len(speakers)} unique speakers: {speakers}")

        # Get embedding for each speaker by using their longest segment
        speaker_embeddings = {}
        for speaker in speakers:
            # Find segments for this speaker
            speaker_segments = [s for s in segments if s["speaker"] == speaker]
            # Use longest segment for more reliable embedding
            longest = max(speaker_segments, key=lambda s: s["end"] - s["start"])

            # Get embedding for this speaker
            embedding = self.get_speaker_embedding(
                audio_path,
                start=longest["start"],
                end=longest["end"]
            )
            speaker_embeddings[speaker] = embedding

        # Find closest match using cosine similarity
        best_match = None
        best_similarity = -1

        for speaker, embedding in speaker_embeddings.items():
            # Cosine distance, convert to similarity
            similarity = 1 - cosine(reference_embedding.flatten(), embedding.flatten())
            print(f"  Speaker {speaker}: similarity = {similarity:.4f}")
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = speaker

        print(f"Best match: {best_match} (similarity: {best_similarity:.4f})")
        return best_match


def download_video(youtube_url: str, output_dir: str) -> Optional[str]:
    """Download video in mp4 format using yt-dlp"""
    try:
        youtube_url = clean_video_url(youtube_url)
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'merge_output_format': 'mp4',
            'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
            'quiet': False,
            'no_warnings': False,
        }
        os.makedirs(output_dir, exist_ok=True)
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=True)
        if not info:
            print("Failed to download video.")
            return None
        title = sanitize_title(info.get('title', ''))
        expected_mp4 = os.path.join(output_dir, f"{title}.mp4")
        if os.path.exists(expected_mp4):
            return expected_mp4
        # Fallback: search for any mp4 file
        for file in os.listdir(output_dir):
            if file.endswith('.mp4'):
                mp4_file = os.path.join(output_dir, file)
                print(f"Found video file: {mp4_file}")
                return mp4_file
        print(f"No mp4 file found in {output_dir}.")
        return None
    except Exception as e:
        print(f"Error downloading video: {str(e)}")
        return None


def clean_video_url(url: str) -> str:
    """Normalize a YouTube URL to a clean watch?v= format.
    Handles Shorts URLs, mobile URLs, and removes playlist parameters."""
    # Extract video ID from various YouTube URL formats
    patterns = [
        r'(?:youtube\.com/shorts/)([a-zA-Z0-9_-]{11})',       # Shorts
        r'(?:youtube\.com/watch\?v=)([a-zA-Z0-9_-]{11})',     # Standard watch
        r'(?:youtu\.be/)([a-zA-Z0-9_-]{11})',                 # Short share URL
        r'(?:youtube\.com/embed/)([a-zA-Z0-9_-]{11})',        # Embed
        r'(?:youtube\.com/v/)([a-zA-Z0-9_-]{11})',            # Old embed
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return f'https://www.youtube.com/watch?v={match.group(1)}'
    return url


def sanitize_title(title: str) -> str:
    """Sanitize a video title to match yt-dlp's filename sanitization.
    yt-dlp replaces certain characters with fullwidth Unicode equivalents."""
    # yt-dlp fullwidth character replacements (from yt_dlp.utils.sanitize_filename)
    _FULLWIDTH_MAP = {
        '"': '\uff02',   # ＂
        '*': '\uff0a',   # ＊
        '/': '\uff0f',   # ／
        ':': '\uff1a',   # ：
        '<': '\uff1c',   # ＜
        '>': '\uff1e',   # ＞
        '?': '\uff1f',   # ？
        '\\': '\uff3c',  # ＼
        '|': '\uff5c',   # ｜
    }
    sanitized = title
    for char, replacement in _FULLWIDTH_MAP.items():
        sanitized = sanitized.replace(char, replacement)
    return sanitized

def get_video_info(youtube_url: str, output_dir: str = None) -> Tuple[Optional[dict], Optional[str]]:
    """Get video information and download audio using yt-dlp"""
    try:
        # Clean URL to remove playlist parameters for single video downloads
        youtube_url = clean_video_url(youtube_url)
        # Configure yt-dlp options for highest quality audio
        ydl_opts = {
            'format': 'bestaudio[acodec=opus]/bestaudio[acodec=aac]/bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '0',  # 0 = best quality (lossless for WAV)
            }],
            'quiet': False,
            'no_warnings': False,
            'prefer_free_formats': True,  # Prefer open formats like opus
        }

        # If output_dir is provided, save WAV there, otherwise use temp file
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            # Use sanitized title for output filename
            ydl_opts['outtmpl'] = os.path.join(output_dir, '%(title)s.%(ext)s')

            # Download once and return metadata from the same request
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=True)

            if not info:
                print("Failed to extract video info.")
                return None, None

            # Sanitize title the same way yt-dlp does for filenames
            title = sanitize_title(info.get('title', ''))
            expected_wav = os.path.join(output_dir, f"{title}.wav")

            if os.path.exists(expected_wav):
                return info, expected_wav

            # Fallback: look for any WAV file if the expected one isn't found
            print(f"Expected WAV file {expected_wav} not found. Searching for any WAV file in {output_dir}...")
            for file in os.listdir(output_dir):
                if file.endswith('.wav'):
                    wav_file = os.path.join(output_dir, file)
                    print(f"Found WAV file: {wav_file}")
                    return info, wav_file
            print(f"No WAV file found in {output_dir}.")
            return None, None
        else:
            # Use temporary file
            ydl_opts['outtmpl'] = 'temp_audio'
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=True)
            return info, 'temp_audio.wav'
    except Exception as e:
        print(f"Error downloading video: {str(e)}")
        return None, None

def process_audio_features(audio_data: np.ndarray, sample_rate: int) -> list[mx.array]:
    """Process audio data into features using MLX"""
    # Handle different audio channel configurations
    if len(audio_data.shape) == 1:
        # Already mono
        mono_audio = audio_data
    elif len(audio_data.shape) == 2:
        # Convert stereo to mono by averaging channels
        if audio_data.shape[1] == 2:  # Channels in second dimension
            mono_audio = np.mean(audio_data, axis=1)
        else:  # Channels in first dimension
            mono_audio = np.mean(audio_data, axis=0)
    else:
        raise ValueError(f"Unsupported audio shape: {audio_data.shape}")

    # Convert to float32 and normalize
    if mono_audio.dtype == np.int16:
        mono_audio = mono_audio.astype(np.float32) / 32768.0
    elif mono_audio.dtype == np.int32:
        mono_audio = mono_audio.astype(np.float32) / 2147483648.0
    elif mono_audio.dtype == np.float64:
        # Scale to [-1, 1] range
        max_val = np.max(np.abs(mono_audio))
        if max_val > 0:
            mono_audio = mono_audio.astype(np.float32) / max_val
        else:
            mono_audio = mono_audio.astype(np.float32)
    # Ensure we're working with float32
    mono_audio = mono_audio.astype(np.float32)

    # Resample to 16kHz if needed
    if sample_rate != 16000:
        # Calculate ratio for resampling
        ratio = 16000 / sample_rate
        new_length = int(len(mono_audio) * ratio)
        indices = np.linspace(0, len(mono_audio) - 1, new_length)
        mono_audio = np.interp(indices, np.arange(len(mono_audio)), mono_audio)

    # Split audio into 30-second chunks (16000 samples/sec * 30 sec = 480000 samples)
    chunk_size = 480000
    audio_chunks = []
    for i in range(0, len(mono_audio), chunk_size):
        chunk = mono_audio[i:i + chunk_size]
        # Pad last chunk if needed
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
        audio_chunks.append(mx.array(chunk))
    return audio_chunks

def process_audio(audio_path: str) -> Tuple[str, dict]:
    """Process audio file using MLX and return a summary of its characteristics and detailed metrics"""
    try:
        # Load audio file
        sample_rate, audio_data = wavfile.read(audio_path)
        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        # Convert to float32 and normalize
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype == np.int32:
            audio_data = audio_data.astype(np.float32) / 2147483648.0
        elif audio_data.dtype == np.float64:
            # Scale to [-1, 1] range
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                audio_data = audio_data.astype(np.float32) / max_val
            else:
                audio_data = audio_data.astype(np.float32)

        # Calculate basic audio characteristics
        duration = len(audio_data) / sample_rate
        num_samples = len(audio_data)
        peak_amplitude = float(np.max(np.abs(audio_data)))
        rms = float(np.sqrt(np.mean(np.square(audio_data))))

        # Create detailed metrics dictionary
        metrics = {
            "duration": duration,
            "num_samples": num_samples,
            "sample_rate": sample_rate,
            "peak_amplitude": peak_amplitude,
            "rms": rms
        }

        # Create detailed audio summary
        audio_info = f"""Audio Analysis:
- Duration: {duration:.2f} seconds
- Number of samples: {num_samples}
- Sample rate: {sample_rate} Hz
- Peak Amplitude: {peak_amplitude:.4f}
- RMS Energy: {rms:.4f}
- Channel Processing: Converted to mono for analysis
- Normalization: Applied based on bit depth"""

        return audio_info, metrics
    except Exception as e:
        print(f"Error processing audio: {str(e)}")
        return "", {}

def transcribe_audio(audio_features: list[mx.array]) -> str:
    """Transcribe audio using Whisper MLX"""
    transcriber = AudioTranscriber.get_instance()
    transcriptions = []
    total_chunks = len(audio_features)
    print(f"\nProcessing {total_chunks} audio chunks...")
    print("-" * 50)
    for i, chunk in enumerate(audio_features, 1):
        print(f"\rTranscribing chunk {i}/{total_chunks}... ", end="")
        trans = transcriber.transcribe(chunk)
        if trans:
            # Clean up the transcription
            trans = trans.strip()
            # Remove leading/trailing quotes and periods
            trans = trans.strip('".')
            # Remove any duplicate spaces
            trans = ' '.join(trans.split())
            if trans:
                transcriptions.append(trans)
        # Show progress percentage
        progress = (i / total_chunks) * 100
        print(f"[{progress:3.0f}%]", end="")
        
        # Clean up chunk after processing to free memory
        del chunk
        if i % 5 == 0:  # Periodic garbage collection every 5 chunks
            gc.collect()
    
    print("\n" + "-" * 50)
    # Join transcriptions with proper spacing and punctuation
    full_transcript = ". ".join(t for t in transcriptions if t)
    if full_transcript:
        full_transcript += "."
    return full_transcript

def get_chunk_speaker(chunk_index: int, chunk_duration: float, segments: list) -> str:
    """Determine which speaker has the most time in a given chunk

    Args:
        chunk_index: Index of the 30-second chunk
        chunk_duration: Duration of each chunk in seconds (default 30)
        segments: Diarization segments [{start, end, speaker}, ...]

    Returns:
        Speaker ID with most overlap in this chunk, or None if no overlap
    """
    chunk_start = chunk_index * chunk_duration
    chunk_end = chunk_start + chunk_duration

    # Calculate overlap duration for each speaker
    speaker_durations = {}
    for seg in segments:
        overlap_start = max(seg["start"], chunk_start)
        overlap_end = min(seg["end"], chunk_end)
        overlap = overlap_end - overlap_start

        if overlap > 0:
            speaker = seg["speaker"]
            speaker_durations[speaker] = speaker_durations.get(speaker, 0) + overlap

    if not speaker_durations:
        return None

    return max(speaker_durations, key=speaker_durations.get)


def extract_speaker_segments(audio_path: str, segments: list, speaker_id: str,
                             output_dir: str, min_duration: float = 5.0,
                             max_duration: float = 15.0) -> list:
    """Extract audio segments for a specific speaker and save as individual files

    Args:
        audio_path: Path to the full audio file
        segments: Diarization segments [{start, end, speaker}, ...]
        speaker_id: ID of the target speaker
        output_dir: Directory to save segment files
        min_duration: Minimum segment duration in seconds
        max_duration: Maximum segment duration in seconds

    Returns:
        List of dicts: [{file_path, start, end, duration}, ...]
    """
    from scipy.io import wavfile
    import soundfile as sf

    # Create segments directory
    segments_dir = os.path.join(output_dir, "segments")
    os.makedirs(segments_dir, exist_ok=True)

    # Load audio
    sample_rate, audio_data = wavfile.read(audio_path)

    # Convert to mono if stereo
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)

    # Normalize to float32
    if audio_data.dtype == np.int16:
        audio_data = audio_data.astype(np.float32) / 32768.0
    elif audio_data.dtype == np.int32:
        audio_data = audio_data.astype(np.float32) / 2147483648.0

    # Filter segments for target speaker
    speaker_segments = [s for s in segments if s["speaker"] == speaker_id]

    # Merge adjacent segments that are close together (< 0.5 sec gap)
    merged_segments = []
    for seg in speaker_segments:
        if merged_segments and seg["start"] - merged_segments[-1]["end"] < 0.5:
            # Merge with previous
            merged_segments[-1]["end"] = seg["end"]
        else:
            merged_segments.append({"start": seg["start"], "end": seg["end"]})

    # Extract and save segments
    extracted = []
    segment_idx = 0

    for seg in merged_segments:
        duration = seg["end"] - seg["start"]

        # Skip segments that are too short
        if duration < min_duration:
            continue

        # Split long segments into smaller pieces
        if duration > max_duration:
            # Split into ~10 second chunks
            num_splits = int(np.ceil(duration / 10.0))
            split_duration = duration / num_splits

            for i in range(num_splits):
                split_start = seg["start"] + i * split_duration
                split_end = seg["start"] + (i + 1) * split_duration
                split_dur = split_end - split_start

                if split_dur >= min_duration:
                    start_sample = int(split_start * sample_rate)
                    end_sample = int(split_end * sample_rate)
                    segment_audio = audio_data[start_sample:end_sample]

                    # Save segment
                    filename = f"segment_{segment_idx:04d}.wav"
                    filepath = os.path.join(segments_dir, filename)
                    sf.write(filepath, segment_audio, sample_rate)

                    extracted.append({
                        "file_path": filepath,
                        "filename": filename,
                        "start": split_start,
                        "end": split_end,
                        "duration": split_dur,
                        "transcript": None  # Will be filled in during transcription
                    })
                    segment_idx += 1
        else:
            # Extract as single segment
            start_sample = int(seg["start"] * sample_rate)
            end_sample = int(seg["end"] * sample_rate)
            segment_audio = audio_data[start_sample:end_sample]

            # Save segment
            filename = f"segment_{segment_idx:04d}.wav"
            filepath = os.path.join(segments_dir, filename)
            sf.write(filepath, segment_audio, sample_rate)

            extracted.append({
                "file_path": filepath,
                "filename": filename,
                "start": seg["start"],
                "end": seg["end"],
                "duration": duration,
                "transcript": None
            })
            segment_idx += 1

    print(f"Extracted {len(extracted)} segments for speaker {speaker_id}")
    return extracted


def is_radio_mix_url(url: str) -> bool:
    """Check if URL is a YouTube Radio/Mix (list ID starts with RD)"""
    if 'list=' in url:
        list_id = url.split('list=')[1].split('&')[0]
        return list_id.startswith('RD')
    return False

def is_playlist_url(url: str, force_playlist: bool = False) -> bool:
    """Check if a URL is a playlist URL"""
    if 'list=' in url:
        # Radio/Mix playlists need explicit opt-in
        if is_radio_mix_url(url) and not force_playlist:
            return False
        return True
    return 'playlist' in url

def extract_playlist_info(playlist_url: str) -> dict:
    ydl_opts = {
        'quiet': True,
        'extract_flat': True,
        'no_warnings': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        return ydl.extract_info(playlist_url, download=False)

class State(TypedDict):
    url: str
    is_playlist: bool
    audio_only: bool  # If True, skip transcription
    save_video: bool  # If True, also download video as mp4
    playlist_info: Optional[dict]
    output_dir: str
    video_urls: List[Tuple[str, str]]  # (url, title)
    current_index: int
    current_video_url: Optional[str]
    current_video_title: Optional[str]
    audio_dir: Optional[str]
    video_dir: Optional[str]
    video_info: Optional[dict]
    audio_path: Optional[str]
    video_path: Optional[str]
    audio_analysis: Optional[str]
    audio_metrics: Optional[dict]
    audio_features: Optional[List[mx.array]]
    transcript: Optional[str]
    video_transcripts: Annotated[List[dict], operator.add]
    # Diarization fields
    diarization_segments: Optional[List[dict]]  # [{start, end, speaker}, ...]
    pascal_speaker_id: Optional[str]  # Which speaker ID is Pascal
    reference_audio_path: Optional[str]  # Path to Pascal reference clip
    pascal_segments: Optional[List[dict]]  # Pascal's segments with transcripts for TTS
    transcription_backend: Optional[str]  # "whisper" or "voxtral"

def extract_radio_mix_videos(url: str) -> list:
    """Extract videos from a Radio/Mix by using yt-dlp with the watch URL"""
    ydl_opts = {
        'quiet': True,
        'extract_flat': 'in_playlist',
        'no_warnings': True,
        'noplaylist': False,  # Allow playlist extraction
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        if info and 'entries' in info:
            return [(entry.get('url') or f"https://www.youtube.com/watch?v={entry.get('id')}", 
                     entry.get('title', 'unknown')) for entry in info.get('entries', []) if entry]
        elif info:
            # Single video fallback
            return [(url, info.get('title', 'unknown'))]
    return []

def start_node(state: State) -> dict:
    url = state["url"]
    # Check for force_playlist marker
    force_playlist = '__force_playlist=1' in url
    url = url.replace('&__force_playlist=1', '')  # Clean the marker
    
    is_playlist = is_playlist_url(url, force_playlist=force_playlist)
    if is_playlist:
        # Handle Radio/Mix playlists differently - they need the watch URL format
        if is_radio_mix_url(url) and force_playlist:
            print(f"\nFetching Radio/Mix: {url}")
            print("-" * 50)
            video_urls = extract_radio_mix_videos(url)
            if not video_urls:
                raise ValueError("Could not fetch Radio/Mix information")
            output_dir = os.path.join('output', 'RadioMix')
            os.makedirs(output_dir, exist_ok=True)
            print(f"Found {len(video_urls)} videos in Radio/Mix")
            print("-" * 50 + "\n")
            return {
                "is_playlist": True,
                "playlist_info": None,
                "video_urls": video_urls,
                "output_dir": output_dir,
                "current_index": 0
            }
        else:
            # Regular playlist
            if '&list=' in url:
                playlist_id = url.split('&list=')[1].split('&')[0]
                url = f'https://www.youtube.com/playlist?list={playlist_id}'
            print(f"\nFetching playlist: {url}")
            print("-" * 50)
            playlist_info = extract_playlist_info(url)
            if not playlist_info:
                raise ValueError("Could not fetch playlist information")
            video_urls = [(entry.get('url'), entry.get('title', 'unknown')) for entry in playlist_info.get('entries', [])]
            playlist_title = playlist_info.get('title', 'playlist').replace('/', '_')
            output_dir = os.path.join('output', playlist_title)
            os.makedirs(output_dir, exist_ok=True)
            print(f"Found {len(video_urls)} videos in playlist: {playlist_title}")
            print("-" * 50 + "\n")
            return {
                "is_playlist": True,
                "playlist_info": playlist_info,
                "video_urls": video_urls,
                "output_dir": output_dir,
                "current_index": 0
            }
    else:
        video_urls = [(url, 'unknown')]
        output_dir = 'output'
        os.makedirs(output_dir, exist_ok=True)
        return {
            "is_playlist": False,
            "video_urls": video_urls,
            "output_dir": output_dir,
            "current_index": 0
        }

def set_current(state: State) -> dict:
    index = state["current_index"]
    urls = state["video_urls"]
    if index >= len(urls):
        return {}
    
    # Clean up previous video's data if this isn't the first video
    if index > 0:
        # Explicitly delete large objects from previous iteration
        if state.get("audio_features"):
            for feature in state["audio_features"]:
                del feature
        # Force garbage collection between videos
        gc.collect()
    
    url, title = urls[index]
    audio_dir = os.path.join(state["output_dir"], 'audio')
    os.makedirs(audio_dir, exist_ok=True)
    video_dir = os.path.join(state["output_dir"], 'video') if state.get("save_video") else None
    if video_dir:
        os.makedirs(video_dir, exist_ok=True)
    return {
        "current_video_url": url,
        "current_video_title": title,
        "audio_dir": audio_dir,
        "video_dir": video_dir,
        "video_info": None,
        "audio_path": None,
        "video_path": None,
        "audio_analysis": None,
        "audio_metrics": None,
        "audio_features": None,
        "transcript": None
    }

def print_progress(state: State) -> dict:
    if state["is_playlist"]:
        index = state["current_index"] + 1
        total = len(state["video_urls"])
        title = state["current_video_title"]
        print(f"Processing video {index}/{total}: {title}")
    return {}

def check_already_processed(state: State) -> dict:
    """Check if video has already been processed by looking for output files"""
    if not state["current_video_title"] or state["current_video_title"] == 'unknown':
        # Can't check without title, proceed with processing
        return {}
    
    # Sanitize title the same way yt-dlp and save_node do
    title = sanitize_title(state["current_video_title"])
    json_path = os.path.join(state["output_dir"], f"{title}.json")
    txt_path = os.path.join(state["output_dir"], f"{title}.txt")
    
    # Check if either output file exists
    if os.path.exists(json_path) or os.path.exists(txt_path):
        print("  ⏭️  Already processed - skipping")
        print()
        # Mark as processed by setting transcript to a marker
        # This will cause save_node to skip file writing but still increment counter
        return {"video_info": {"title": title}, "transcript": "ALREADY_PROCESSED"}
    
    return {}

def get_video_info_node(state: State) -> dict:
    info, audio_path = get_video_info(state["current_video_url"], state["audio_dir"])
    if not info or not audio_path:
        print("✗ Error processing video: Could not fetch video information and audio")
        return {"video_info": None, "audio_path": None, "video_path": None, "transcript": None}
    # Download video mp4 if requested
    video_path = None
    if state.get("save_video") and state.get("video_dir"):
        print("\nDownloading video (mp4)...")
        video_path = download_video(state["current_video_url"], state["video_dir"])
        if video_path:
            print(f"✓ Video saved: {video_path}")
        else:
            print("⚠ Video download failed, continuing with audio only")
    if not state.get("audio_only"):
        print("\nGenerating transcript...")
    else:
        print("\n✓ Audio downloaded successfully")
    return {"video_info": info, "audio_path": audio_path, "video_path": video_path}

def process_audio_analysis_node(state: State) -> dict:
    if not state["audio_path"]:
        return {"audio_analysis": None, "audio_metrics": None}
    audio_analysis, audio_metrics = process_audio(state["audio_path"])
    return {"audio_analysis": audio_analysis, "audio_metrics": audio_metrics}

def process_features_node(state: State) -> dict:
    if not state["audio_path"]:
        return {"audio_features": None}
    sample_rate, audio_data = wavfile.read(state["audio_path"])
    features = process_audio_features(audio_data, sample_rate)
    return {"audio_features": features}


def diarize_node(state: State) -> dict:
    """Run speaker diarization and identify Pascal's voice"""
    if not state["audio_path"]:
        return {
            "diarization_segments": None,
            "pascal_speaker_id": None,
            "pascal_segments": None
        }

    reference_path = state.get("reference_audio_path")
    if not reference_path:
        # No reference audio, skip diarization
        print("No reference audio provided, skipping diarization")
        return {
            "diarization_segments": None,
            "pascal_speaker_id": None,
            "pascal_segments": None
        }

    try:
        # Step 1: Isolate vocals using Demucs (removes background music/noise)
        print("Step 1: Isolating vocals with Demucs...")
        demucs = DemucsManager.get_instance()
        vocals_path = demucs.isolate_vocals(state["audio_path"])

        # Step 2: Run diarization on the cleaned audio
        print("Step 2: Running speaker diarization on cleaned audio...")
        diarizer = DiarizerManager.get_instance()

        # Run diarization on vocals-only audio
        segments = diarizer.diarize(vocals_path)

        if not segments:
            print("No speaker segments found")
            return {
                "diarization_segments": [],
                "pascal_speaker_id": None,
                "pascal_segments": None
            }

        # Step 3: Identify Pascal by matching voice to reference
        print("Step 3: Identifying Pascal's voice...")
        pascal_id = diarizer.identify_speaker_by_reference(
            vocals_path,  # Use cleaned audio for better matching
            reference_path,
            segments
        )

        # Step 4: Extract Pascal's segments as individual audio files
        print("Step 4: Extracting Pascal's segments...")
        pascal_segments = extract_speaker_segments(
            vocals_path,  # Use cleaned audio for cleaner TTS training data
            segments,
            pascal_id,
            state["output_dir"],
            min_duration=5.0,
            max_duration=15.0
        )

        return {
            "diarization_segments": segments,
            "pascal_speaker_id": pascal_id,
            "pascal_segments": pascal_segments
        }

    except Exception as e:
        print(f"Error during diarization: {str(e)}")
        return {
            "diarization_segments": None,
            "pascal_speaker_id": None,
            "pascal_segments": None
        }


def transcribe_node(state: State) -> dict:
    """Transcribe audio - either full or Pascal-only segments"""
    backend = state.get("transcription_backend") or "whisper"

    # Voxtral backend: takes audio file path directly, no chunking needed
    if backend == "voxtral":
        if not state.get("audio_path"):
            return {"transcript": None}
        print("\nTranscribing with Voxtral-Mini-3B...")
        transcript = VoxtralTranscriber.get_instance().transcribe(state["audio_path"])
        return {"transcript": transcript}

    # Whisper backend (default)
    if not state.get("audio_features"):
        return {"transcript": None, "pascal_segments": state.get("pascal_segments")}

    # Check if we have Pascal segments to transcribe individually
    pascal_segments = state.get("pascal_segments")
    diarization_segments = state.get("diarization_segments")
    pascal_id = state.get("pascal_speaker_id")

    if pascal_segments and diarization_segments and pascal_id:
        # Transcribe Pascal's segments individually for TTS training
        print(f"\nTranscribing {len(pascal_segments)} Pascal segments...")
        print("-" * 50)

        transcriber = AudioTranscriber.get_instance()
        transcribed_segments = []

        for i, seg in enumerate(pascal_segments, 1):
            print(f"\rTranscribing segment {i}/{len(pascal_segments)}... ", end="")

            # Load segment audio
            sample_rate, audio_data = wavfile.read(seg["file_path"])

            # Process to mono float32 at 16kHz
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype == np.int32:
                audio_data = audio_data.astype(np.float32) / 2147483648.0

            # Resample to 16kHz if needed
            if sample_rate != 16000:
                ratio = 16000 / sample_rate
                new_length = int(len(audio_data) * ratio)
                indices = np.linspace(0, len(audio_data) - 1, new_length)
                audio_data = np.interp(indices, np.arange(len(audio_data)), audio_data)

            # Pad to 30 seconds for Whisper (it needs fixed input size)
            chunk_size = 480000
            if len(audio_data) < chunk_size:
                audio_data = np.pad(audio_data, (0, chunk_size - len(audio_data)))

            # Convert to MLX array and transcribe
            chunk = mx.array(audio_data[:chunk_size].astype(np.float32))
            transcript = transcriber.transcribe(chunk)

            # Clean up transcription
            if transcript:
                transcript = transcript.strip().strip('".').strip()
                transcript = ' '.join(transcript.split())

            seg["transcript"] = transcript
            transcribed_segments.append(seg)

            progress = (i / len(pascal_segments)) * 100
            print(f"[{progress:3.0f}%]", end="")

            if i % 5 == 0:
                gc.collect()

        print("\n" + "-" * 50)

        # Also create full Pascal-only transcript
        pascal_transcripts = [s["transcript"] for s in transcribed_segments if s["transcript"]]
        full_transcript = ". ".join(pascal_transcripts)
        if full_transcript:
            full_transcript += "."

        return {
            "transcript": full_transcript,
            "pascal_segments": transcribed_segments
        }
    else:
        # Standard full transcription (no diarization)
        transcript = transcribe_audio(state["audio_features"])
    return {"transcript": transcript}

def save_node(state: State) -> dict:
    # Check if already processed
    if state.get("transcript") == "ALREADY_PROCESSED":
        # Skip to next video
        return {"current_index": state["current_index"] + 1, "video_transcripts": []}

    # In audio-only mode, we don't need a transcript
    is_audio_only = state.get("audio_only", False)

    if not state["video_info"]:
        # Error case - video processing failed
        error_msg = "✗ Error processing video: No video info"
        print(error_msg)

        # Clean up even on error
        if state.get("audio_features"):
            for feature in state["audio_features"]:
                del feature
        gc.collect()

        return {"current_index": state["current_index"] + 1, "video_transcripts": []}

    # In non-audio-only modes, transcript is required
    if not is_audio_only and not state.get("transcript"):
        # Error case - transcription failed
        error_msg = "✗ Error processing video: Transcription failed"
        print(error_msg)

        # Clean up even on error
        if state.get("audio_features"):
            for feature in state["audio_features"]:
                del feature
        gc.collect()

        return {"current_index": state["current_index"] + 1, "video_transcripts": []}

    title = sanitize_title(state["video_info"].get('title', ''))
    description = state["video_info"].get('description', '')
    duration = state["video_info"].get('duration', 0)
    view_count = state["video_info"].get('view_count', 0)

    # Check if we have diarization/TTS data
    pascal_segments = state.get("pascal_segments")
    diarization_segments = state.get("diarization_segments")
    pascal_id = state.get("pascal_speaker_id")
    has_diarization = pascal_segments and diarization_segments and pascal_id

    output_data = {
        'video_info': {
            'title': title,
            'description': description,
            'duration': duration,
            'view_count': view_count,
            'url': state["current_video_url"]
        },
        'audio_analysis': state.get("audio_analysis"),
        'transcript': state.get("transcript"),
        'audio_file': os.path.basename(state["audio_path"]) if state.get("audio_path") else None,
        'video_file': os.path.basename(state["video_path"]) if state.get("video_path") else None,
        'mode': 'audio_only' if state.get("audio_only") else ('voxtral' if state.get("transcription_backend") == "voxtral" else 'full')
    }

    # Add diarization info if available
    if has_diarization:
        pascal_segments_count = len([s for s in diarization_segments if s["speaker"] == pascal_id])
        output_data['diarization'] = {
            'pascal_speaker_id': pascal_id,
            'total_segments': len(diarization_segments),
            'pascal_raw_segments': pascal_segments_count,
            'pascal_tts_segments': len(pascal_segments),
            'segments': pascal_segments  # Include full segment data with transcripts
        }

    output_path = os.path.join(state["output_dir"], f"{title}.json")
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    txt_path = os.path.join(state["output_dir"], f"{title}.txt")
    with open(txt_path, 'w') as f:
        if has_diarization:
            f.write(f"Pascal Bornet - Voice Segments: {title}\n")
            f.write("=" * 80 + "\n")
            f.write("(Speaker-isolated transcript for TTS training)\n\n")
        else:
            f.write(f"Video Analysis: {title}\n")
            f.write("=" * 80 + "\n\n")

        f.write("Video Information:\n")
        f.write("-" * 20 + "\n")
        f.write(f"URL: {state['current_video_url']}\n")
        f.write(f"Title: {title}\n")
        f.write(f"Duration: {duration} seconds\n")
        f.write(f"Views: {view_count}\n")
        f.write(f"Description: {description}\n")
        f.write(f"Audio File: {os.path.basename(state['audio_path'])}\n")
        if state.get("video_path"):
            f.write(f"Video File: {os.path.basename(state['video_path'])}\n")
        f.write("\n")

        if has_diarization:
            f.write("Diarization Info:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Pascal Speaker ID: {pascal_id}\n")
            f.write(f"Total speaker segments: {len(diarization_segments)}\n")
            f.write(f"Pascal TTS segments: {len(pascal_segments)}\n\n")

        if state.get("transcript"):
            f.write("Transcript:\n")
            f.write("-" * 20 + "\n")
            # Format transcript into sentences
            sentences = state["transcript"].replace('...', '.').split('.')
            sentences = [s.strip() for s in sentences if s.strip()]
            for sentence in sentences:
                if sentence:
                    f.write(f"{sentence}.\n")
            f.write("\n\n")
        elif state.get("audio_only"):
            f.write("\n(Audio-only mode - no transcript generated)\n\n")

    # Save TTS training manifest if we have Pascal segments
    if has_diarization and pascal_segments:
        # Create a manifest file for TTS training
        manifest_path = os.path.join(state["output_dir"], "segments", "manifest.json")
        manifest_data = {
            "speaker": "pascal_bornet",
            "source_video": title,
            "total_segments": len(pascal_segments),
            "total_duration": sum(s["duration"] for s in pascal_segments),
            "segments": [
                {
                    "audio_file": s["filename"],
                    "transcript": s["transcript"],
                    "duration": s["duration"],
                    "start": s["start"],
                    "end": s["end"]
                }
                for s in pascal_segments if s["transcript"]
            ]
        }
        with open(manifest_path, 'w') as f:
            json.dump(manifest_data, f, indent=2)

        # Also save individual transcript files for each segment
        for seg in pascal_segments:
            if seg["transcript"]:
                txt_filename = seg["filename"].replace('.wav', '.txt')
                txt_filepath = os.path.join(state["output_dir"], "segments", txt_filename)
                with open(txt_filepath, 'w') as f:
                    f.write(seg["transcript"])

        print(f"✓ Saved {len(pascal_segments)} TTS training segments")

    transcript_dict = {"url": state["current_video_url"], "title": title, "transcript": state["transcript"]}

    if state["is_playlist"]:
        if state["transcript"]:
            print("✓ Transcript generated successfully")
        else:
            print("⚠ No transcript generated")
        print()

    # Clean up large objects after saving
    if state.get("audio_features"):
        for feature in state["audio_features"]:
            del feature
    gc.collect()

    return {"video_transcripts": [transcript_dict], "current_index": state["current_index"] + 1}

def finalize_node(state: State) -> dict:
    if state["is_playlist"]:
        print("-" * 50)
        print(f"Playlist processing complete! Results saved in: {state['output_dir']}")

    # Print TTS training summary if diarization was used
    pascal_segments = state.get("pascal_segments")
    if pascal_segments:
        total_duration = sum(s["duration"] for s in pascal_segments if s.get("transcript"))
        segments_with_transcripts = len([s for s in pascal_segments if s.get("transcript")])
        print(f"\nTTS Training Data Summary:")
        print(f"  Total segments: {segments_with_transcripts}")
        print(f"  Total duration: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
        print(f"  Output: {state['output_dir']}/segments/")

    # Clean up transcriber singletons at the end
    AudioTranscriber.cleanup()
    VoxtralTranscriber.cleanup()
    # Clean up diarizer singleton
    DiarizerManager.cleanup()
    DemucsManager.cleanup()
    # Final garbage collection
    gc.collect()

    return {}

def condition(state: State):
    if state["current_index"] >= len(state["video_urls"]):
        return "end"
    return "continue"

# Build the graph
graph = StateGraph(State)

graph.add_node("start", start_node)
graph.add_node("set_current", set_current)
graph.add_node("print_progress", print_progress)
graph.add_node("check_processed", check_already_processed)
graph.add_node("get_video_info", get_video_info_node)
graph.add_node("process_analysis", process_audio_analysis_node)
graph.add_node("process_features", process_features_node)
graph.add_node("diarize", diarize_node)
graph.add_node("transcribe", transcribe_node)
graph.add_node("save", save_node)
graph.add_node("finalize", finalize_node)

graph.set_entry_point("start")

graph.add_edge("start", "set_current")
graph.add_conditional_edges("set_current", condition, {"continue": "print_progress", "end": "finalize"})
graph.add_edge("print_progress", "check_processed")

# Conditional edge: skip if already processed, otherwise continue
def skip_condition(state: State):
    if state.get("transcript") == "ALREADY_PROCESSED":
        return "skip"
    return "process"

graph.add_conditional_edges("check_processed", skip_condition, {"process": "get_video_info", "skip": "save"})

# Conditional edge: skip transcription if audio_only mode
def audio_only_condition(state: State):
    if state.get("audio_only"):
        return "audio_only"
    return "transcribe"

graph.add_conditional_edges("get_video_info", audio_only_condition, {"transcribe": "process_analysis", "audio_only": "save"})

# Conditional edge: voxtral skips feature extraction and diarization
def backend_condition(state: State):
    if state.get("transcription_backend") == "voxtral":
        return "voxtral"
    return "whisper"

graph.add_conditional_edges("process_analysis", backend_condition, {"whisper": "process_features", "voxtral": "transcribe"})
graph.add_edge("process_features", "diarize")
graph.add_edge("diarize", "transcribe")
graph.add_edge("transcribe", "save")
graph.add_edge("save", "set_current")
graph.add_edge("finalize", END)

# Compile with high recursion limit for large playlists
# Each video takes ~8 steps through the graph, so 72 videos = 576+ steps
app = graph.compile()

def main():
    parser = argparse.ArgumentParser(
        description='Generate transcripts from YouTube videos using local Whisper models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python main_langgraph.py                              # Interactive mode
  python main_langgraph.py -u "https://youtube.com/..." # URL with interactive mode selection
  python main_langgraph.py -u "https://youtube.com/..." -m 1  # Audio only
  python main_langgraph.py -u "https://youtube.com/..." -m 2  # Audio + Transcription
  python main_langgraph.py -u "https://youtube.com/..." -m 3  # Speaker diarization + TTS
  python main_langgraph.py -u "https://youtube.com/..." -m 4  # Audio + Voxtral transcription
  python main_langgraph.py -u "https://youtube.com/..." -m 2 --save-video  # Transcription + save mp4
        '''
    )
    parser.add_argument('-u', '--url', type=str, help='YouTube URL or Playlist URL')
    parser.add_argument('-m', '--mode', type=str, choices=['1', '2', '3', '4'],
                        help='Mode: 1=Audio only, 2=Audio+Transcription, 3=Speaker diarization+TTS, 4=Audio+Voxtral')
    parser.add_argument('--save-video', action='store_true',
                        help='Also save the video file in mp4 format')
    args = parser.parse_args()

    # Get URL from args or prompt
    if args.url:
        url = args.url
    else:
        url = input('Enter YouTube URL or Playlist URL: ')

    # Get mode from args or prompt
    if args.mode:
        mode = args.mode
    else:
        mode = input('Mode - (1) Audio only  (2) Audio + Transcription  (3) Speaker diarization + TTS  (4) Audio + Voxtral [default: 2]: ').strip()

    audio_only = mode == '1'
    diarization_mode = mode == '3'
    voxtral_mode = mode == '4'
    transcription_backend = "voxtral" if voxtral_mode else "whisper"

    # Determine whether to save video
    if args.save_video:
        save_video = True
    else:
        save_video_input = input('Save video file (.mp4)? (y/n) [default: n]: ').strip().lower()
        save_video = save_video_input in ('y', 'yes')

    if audio_only:
        print("Audio-only mode: will download audio without transcription")
    elif voxtral_mode:
        print("Voxtral mode: will transcribe using Voxtral-Mini-3B-2507")
    if save_video:
        print("Video save enabled: will also download video as mp4")

    # Ask for reference audio if in diarization mode
    reference_audio_path = None
    if diarization_mode:
        print("\nSpeaker diarization mode enabled")
        print("This mode will identify Pascal's voice and extract his speech as separate segments")
        print("for TTS training (5-15 second clips with transcripts).\n")
        reference_audio_path = input('Enter path to Pascal reference audio (10-15 sec clip): ').strip()
        if reference_audio_path:
            if not os.path.exists(reference_audio_path):
                print(f"Warning: Reference file not found at {reference_audio_path}")
                reference_audio_path = None
            else:
                print(f"Will use reference audio: {reference_audio_path}")
        else:
            print("No reference audio provided - will run standard transcription")

    # Check if it's a Radio/Mix URL and ask user preference
    force_playlist = False
    if is_radio_mix_url(url):
        choice = input('Radio/Mix playlist detected. Download all videos? (y/n): ').strip().lower()
        force_playlist = choice in ('y', 'yes')
        if force_playlist:
            print(f"Will download entire Radio/Mix playlist...")
        else:
            print(f"Will download single video only...")

    # Store force_playlist in URL by appending a marker (will be parsed in start_node)
    if force_playlist:
        url = url + '&__force_playlist=1'

    initial_state = {
        "url": url,
        "is_playlist": False,
        "audio_only": audio_only,
        "save_video": save_video,
        "playlist_info": None,
        "output_dir": "",
        "video_urls": [],
        "current_index": 0,
        "current_video_url": None,
        "current_video_title": None,
        "audio_dir": None,
        "video_dir": None,
        "video_info": None,
        "audio_path": None,
        "video_path": None,
        "audio_analysis": None,
        "audio_metrics": None,
        "audio_features": None,
        "transcript": None,
        "video_transcripts": [],
        # Diarization fields
        "diarization_segments": None,
        "pascal_speaker_id": None,
        "reference_audio_path": reference_audio_path,
        "pascal_segments": None,
        "transcription_backend": transcription_backend
    }
    # Set recursion limit high enough for large playlists (72 videos × ~8 steps = ~600)
    app.invoke(initial_state, {"recursion_limit": 1000})

if __name__ == '__main__':
    main()
