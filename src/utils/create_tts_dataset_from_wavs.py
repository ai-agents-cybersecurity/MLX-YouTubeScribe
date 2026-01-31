#!/usr/bin/env python3
"""
Create TTS Training Dataset from WAV Files

This script:
1. Reads WAV files from a source directory
2. Transcribes each file using Whisper
3. Creates a JSONL file in the format required by Qwen3-TTS
4. Copies all audio files to the dataset folder

Required JSONL format:
{"audio":"./audio/utt0001.wav","text":"transcript text","ref_audio":"./audio/ref.wav"}
"""

import os
import json
import shutil
import gc
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import torch
import numpy as np
from scipy.io import wavfile
from transformers import WhisperProcessor, WhisperForConditionalGeneration


# Whisper model to use
WHISPER_MODEL = "openai/whisper-large-v3-turbo"


@dataclass
class Transcriber:
    """Audio transcription using Whisper (Singleton Pattern)"""
    processor: WhisperProcessor
    model: WhisperForConditionalGeneration
    _instance = None

    def __init__(self):
        print(f"Loading Whisper model: {WHISPER_MODEL}...")
        self.processor = WhisperProcessor.from_pretrained(WHISPER_MODEL)
        self.model = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL)

        # Move to MPS if available
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.model.to(self.device)
            print("Using Apple Silicon GPU (MPS) for transcription")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.model.to(self.device)
            print("Using CUDA GPU for transcription")
        else:
            self.device = torch.device("cpu")
            print("Using CPU for transcription")

        print("Whisper model loaded!")

    @classmethod
    def get_instance(cls):
        """Get or create singleton instance"""
        if cls._instance is None:
            cls._instance = Transcriber()
        return cls._instance

    @classmethod
    def cleanup(cls):
        """Clean up the singleton instance"""
        if cls._instance is not None:
            del cls._instance.model
            del cls._instance.processor
            cls._instance = None
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

    def transcribe(self, audio_path: str) -> str:
        """Transcribe a single audio file"""
        try:
            # Read audio file
            sample_rate, audio_data = wavfile.read(audio_path)

            # Convert to float32 and normalize
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype == np.int32:
                audio_data = audio_data.astype(np.float32) / 2147483648.0
            elif audio_data.dtype == np.float64:
                audio_data = audio_data.astype(np.float32)

            # Convert stereo to mono if needed
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)

            # Resample to 16kHz if needed
            if sample_rate != 16000:
                import torchaudio
                waveform = torch.from_numpy(audio_data).unsqueeze(0)
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
                audio_data = waveform.squeeze().numpy()

            # Process audio features
            inputs = self.processor(
                audio_data,
                return_tensors="pt",
                sampling_rate=16000,
                return_attention_mask=True,
                language="en"
            )

            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate transcription
            with torch.no_grad():
                generated_ids = self.model.generate(
                    inputs["input_features"],
                    attention_mask=inputs.get("attention_mask"),
                    return_timestamps=False,
                    max_length=448,
                    language='en'
                )

            # Decode the output
            transcription = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0].strip()

            # Clean up
            del inputs, generated_ids

            return transcription

        except Exception as e:
            print(f"Error transcribing {audio_path}: {str(e)}")
            return ""


def get_audio_duration(audio_path: str) -> float:
    """Get duration of audio file in seconds"""
    try:
        sample_rate, audio_data = wavfile.read(audio_path)
        if len(audio_data.shape) > 1:
            duration = len(audio_data) / sample_rate
        else:
            duration = len(audio_data) / sample_rate
        return duration
    except Exception as e:
        print(f"Error getting duration for {audio_path}: {e}")
        return 0.0


def find_best_reference(segments: List[Dict], min_duration: float = 8.0, max_duration: float = 15.0) -> Optional[Dict]:
    """Find best segment to use as reference audio"""
    # Filter by duration
    candidates = [s for s in segments if min_duration <= s.get("duration", 0) <= max_duration]

    if not candidates:
        candidates = [s for s in segments if s.get("duration", 0) >= 5.0]

    if not candidates:
        candidates = [s for s in segments if s.get("transcript")]

    if not candidates:
        return None

    # Sort by duration (prefer closer to 10 seconds)
    candidates.sort(key=lambda s: abs(s.get("duration", 0) - 10.0))
    return candidates[0]


def create_tts_dataset(
    source_dir: str,
    output_dir: str,
    speaker_name: str = "nic",
    min_duration: float = 2.0,
    max_duration: float = 30.0
):
    """
    Create TTS training dataset from WAV files.

    Args:
        source_dir: Directory containing WAV files
        output_dir: Output directory for dataset
        speaker_name: Speaker name for the dataset
        min_duration: Minimum segment duration to include (seconds)
        max_duration: Maximum segment duration to include (seconds)
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    audio_dir = output_path / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    # Find all WAV files
    wav_files = sorted(source_path.glob("*.wav"))
    print(f"\nFound {len(wav_files)} WAV files in {source_dir}")

    if not wav_files:
        print("No WAV files found!")
        return

    # Initialize transcriber
    transcriber = Transcriber.get_instance()

    # Process each file
    segments = []
    print(f"\nTranscribing {len(wav_files)} files...")
    print("-" * 60)

    for i, wav_file in enumerate(wav_files, 1):
        # Get duration
        duration = get_audio_duration(str(wav_file))

        # Skip files outside duration range
        if duration < min_duration:
            print(f"[{i:3d}/{len(wav_files)}] Skipping {wav_file.name} (too short: {duration:.1f}s)")
            continue
        if duration > max_duration:
            print(f"[{i:3d}/{len(wav_files)}] Skipping {wav_file.name} (too long: {duration:.1f}s)")
            continue

        # Transcribe
        print(f"[{i:3d}/{len(wav_files)}] Transcribing {wav_file.name} ({duration:.1f}s)...", end=" ")
        transcript = transcriber.transcribe(str(wav_file))

        if transcript:
            print(f"OK - \"{transcript[:50]}{'...' if len(transcript) > 50 else ''}\"")
            segments.append({
                "source_file": str(wav_file),
                "filename": wav_file.name,
                "duration": duration,
                "transcript": transcript
            })
        else:
            print("FAILED (no transcript)")

        # Periodic garbage collection
        if i % 10 == 0:
            gc.collect()

    print("-" * 60)
    print(f"Successfully transcribed {len(segments)} files")

    if not segments:
        print("No segments with transcripts!")
        Transcriber.cleanup()
        return

    # Find best reference segment
    ref_segment = find_best_reference(segments)
    if not ref_segment:
        print("Could not find suitable reference segment!")
        Transcriber.cleanup()
        return

    # Copy reference audio
    ref_src = Path(ref_segment["source_file"])
    ref_dst = audio_dir / "ref.wav"
    shutil.copy2(ref_src, ref_dst)

    print(f"\nReference audio selected:")
    print(f"  File: {ref_segment['filename']}")
    print(f"  Duration: {ref_segment['duration']:.2f}s")
    print(f"  Transcript: \"{ref_segment['transcript'][:80]}{'...' if len(ref_segment['transcript']) > 80 else ''}\"")

    # Copy all segments and create JSONL entries
    print(f"\nCopying audio files...")
    jsonl_entries = []

    for i, seg in enumerate(segments):
        src_path = Path(seg["source_file"])
        new_filename = f"utt_{i:04d}.wav"
        dst_path = audio_dir / new_filename

        shutil.copy2(src_path, dst_path)

        entry = {
            "audio": f"./audio/{new_filename}",
            "text": seg["transcript"],
            "ref_audio": "./audio/ref.wav"
        }
        jsonl_entries.append(entry)

    # Write JSONL file
    jsonl_path = output_path / "train_raw.jsonl"
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for entry in jsonl_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    # Calculate total duration
    total_duration = sum(s["duration"] for s in segments)

    # Write dataset info
    info = {
        "speaker": speaker_name,
        "total_segments": len(segments),
        "total_duration_seconds": total_duration,
        "total_duration_minutes": round(total_duration / 60, 2),
        "reference_file": "audio/ref.wav",
        "reference_duration": ref_segment["duration"],
        "reference_transcript": ref_segment["transcript"],
        "source_directory": source_dir,
        "min_duration_filter": min_duration,
        "max_duration_filter": max_duration
    }

    info_path = output_path / "dataset_info.json"
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    # Clean up transcriber
    Transcriber.cleanup()

    # Print summary
    print(f"\n{'='*60}")
    print(f"Dataset created successfully!")
    print(f"{'='*60}")
    print(f"Output directory: {output_path}")
    print(f"Audio files: {len(segments) + 1} (including reference)")
    print(f"JSONL file: {jsonl_path}")
    print(f"Dataset info: {info_path}")
    print(f"\nTotal training duration: {total_duration/60:.2f} minutes ({total_duration:.0f} seconds)")

    # Print next steps
    print(f"\n{'='*60}")
    print("Next steps for Qwen3-TTS fine-tuning:")
    print(f"{'='*60}")
    print(f"""
1. Navigate to Qwen3-TTS finetuning directory:
   cd /path/to/Qwen3-TTS/finetuning

2. Prepare data (extract audio_codes):
   python prepare_data.py \\
     --device cuda:0 \\
     --tokenizer_model_path Qwen/Qwen3-TTS-Tokenizer-12Hz \\
     --input_jsonl {jsonl_path} \\
     --output_jsonl {output_path}/train_with_codes.jsonl

3. Fine-tune the model:
   python sft_12hz.py \\
     --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \\
     --output_model_path {output_path}/model_output \\
     --train_jsonl {output_path}/train_with_codes.jsonl \\
     --batch_size 2 \\
     --lr 2e-5 \\
     --num_epochs 3 \\
     --speaker_name {speaker_name}
""")

    # Create test inference script
    test_script = f'''#!/usr/bin/env python3
"""
Test inference with fine-tuned Qwen3-TTS model
"""
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

device = "cuda:0"  # or "mps" for Apple Silicon

# Load fine-tuned model
tts = Qwen3TTSModel.from_pretrained(
    "{output_path}/model_output/checkpoint-epoch-2",
    device_map=device,
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

# Generate speech
text = "Hello, this is a test of the fine-tuned voice model."
wavs, sr = tts.generate_custom_voice(
    text=text,
    speaker="{speaker_name}",
)

# Save output
sf.write("test_output.wav", wavs[0], sr)
print("Generated test_output.wav")
'''

    test_script_path = output_path / "test_inference.py"
    with open(test_script_path, 'w') as f:
        f.write(test_script)
    print(f"Test inference script: {test_script_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create TTS training dataset from WAV files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python create_tts_dataset_from_wavs.py --source /path/to/wavs --output ./dataset

  # With custom speaker name
  python create_tts_dataset_from_wavs.py --source /path/to/wavs --output ./dataset --speaker my_voice
        """
    )
    parser.add_argument(
        "--source", type=str,
        default="/Users/spider/Desktop/nicsvoice",
        help="Source directory containing WAV files"
    )
    parser.add_argument(
        "--output", type=str,
        default="/Users/spider/Documents/GITHUB/projects/MLX-YouTubeScribe/dataset2",
        help="Output directory for the dataset"
    )
    parser.add_argument(
        "--speaker", type=str, default="nic",
        help="Speaker name for the dataset"
    )
    parser.add_argument(
        "--min-duration", type=float, default=2.0,
        help="Minimum segment duration in seconds (default: 2.0)"
    )
    parser.add_argument(
        "--max-duration", type=float, default=30.0,
        help="Maximum segment duration in seconds (default: 30.0)"
    )

    args = parser.parse_args()

    create_tts_dataset(
        source_dir=args.source,
        output_dir=args.output,
        speaker_name=args.speaker,
        min_duration=args.min_duration,
        max_duration=args.max_duration
    )
