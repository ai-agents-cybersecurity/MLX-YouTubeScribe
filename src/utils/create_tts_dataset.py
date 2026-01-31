#!/usr/bin/env python3
"""
Create TTS Training Dataset for Qwen3-TTS Fine-tuning

This script:
1. Reads diarization segments from video JSON files
2. Copies audio segments to the dataset folder
3. Creates a JSONL file in the format required by Qwen3-TTS

Required JSONL format:
{"audio":"./data/utt0001.wav","text":"transcript text","ref_audio":"./data/ref.wav"}
"""

import os
import json
import shutil
from pathlib import Path
from typing import List, Dict, Optional


def find_best_reference_segment(segments: List[Dict], min_duration: float = 8.0, max_duration: float = 15.0) -> Optional[Dict]:
    """
    Find the best segment to use as reference audio.
    Prefers segments that are:
    - Between 8-15 seconds (ideal for reference)
    - Have clean transcripts (no partial words)
    """
    # Filter segments by duration
    candidates = [s for s in segments if min_duration <= s.get("duration", 0) <= max_duration]

    if not candidates:
        # Fallback to any segment over 5 seconds
        candidates = [s for s in segments if s.get("duration", 0) >= 5.0]

    if not candidates:
        # Last resort: use any segment with transcript
        candidates = [s for s in segments if s.get("transcript")]

    if not candidates:
        return None

    # Sort by duration (prefer closer to 10 seconds)
    candidates.sort(key=lambda s: abs(s.get("duration", 0) - 10.0))

    return candidates[0]


def collect_segments_from_video_jsons(source_dir: str) -> List[Dict]:
    """
    Collect all diarization segments from video JSON files in the source directory.

    Args:
        source_dir: Directory containing video JSON files and segments folder

    Returns:
        List of all segments with their file paths resolved
    """
    source_path = Path(source_dir)
    all_segments = []

    # Find all JSON files (excluding manifest.json and dataset files)
    json_files = [
        f for f in source_path.glob("*.json")
        if f.name not in ("manifest.json", "dataset_info.json")
    ]

    print(f"Found {len(json_files)} video JSON files")

    for json_file in sorted(json_files):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Check if this file has diarization data
            diarization = data.get("diarization", {})
            segments = diarization.get("segments", [])

            if segments:
                video_title = data.get("video_info", {}).get("title", json_file.stem)
                print(f"  {video_title}: {len(segments)} segments")

                # Add source info to each segment
                for seg in segments:
                    seg["_source_video"] = video_title
                    seg["_source_file"] = str(json_file)

                    # Resolve the file path
                    file_path = seg.get("file_path", "")
                    if file_path:
                        # Handle relative paths
                        if not Path(file_path).is_absolute():
                            # Try to resolve relative to source_dir
                            resolved = source_path / file_path
                            if not resolved.exists():
                                # Try segments subfolder
                                resolved = source_path / "segments" / seg.get("filename", "")
                            seg["_resolved_path"] = str(resolved)
                        else:
                            seg["_resolved_path"] = file_path
                    else:
                        # Try to construct path from filename
                        seg["_resolved_path"] = str(source_path / "segments" / seg.get("filename", ""))

                all_segments.extend(segments)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  Warning: Could not parse {json_file.name}: {e}")
            continue

    return all_segments


def create_tts_dataset(
    source_dir: str,
    output_dir: str,
    speaker_name: str = "pascal_bornet",
    min_duration: float = 2.0
):
    """
    Create TTS training dataset from diarization segments.

    Args:
        source_dir: Directory containing video JSON files and segments
        output_dir: Output directory for dataset
        speaker_name: Speaker name for the dataset
        min_duration: Minimum segment duration to include (seconds)
    """
    # Create output directories
    output_path = Path(output_dir)
    audio_dir = output_path / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    # Collect all segments from video JSON files
    print(f"\nCollecting segments from {source_dir}...")
    all_segments = collect_segments_from_video_jsons(source_dir)

    if not all_segments:
        print("No diarization segments found!")
        print("Make sure you have processed videos with mode 3 (speaker diarization)")
        return

    print(f"\nTotal segments collected: {len(all_segments)}")

    # Find best reference segment
    ref_segment = find_best_reference_segment(all_segments)
    if not ref_segment:
        print("Could not find suitable reference segment!")
        return

    # Copy reference audio
    ref_src = Path(ref_segment.get("_resolved_path", ""))
    ref_dst = audio_dir / "ref.wav"

    if ref_src.exists():
        shutil.copy2(ref_src, ref_dst)
        print(f"\nReference audio selected:")
        print(f"  File: {ref_segment.get('filename', 'unknown')}")
        print(f"  Duration: {ref_segment['duration']:.2f}s")
        print(f"  Transcript: \"{ref_segment['transcript'][:80]}{'...' if len(ref_segment['transcript']) > 80 else ''}\"")
        print(f"  From video: {ref_segment.get('_source_video', 'unknown')}")
    else:
        print(f"Warning: Reference file not found: {ref_src}")
        return

    # Process all segments
    print(f"\nProcessing segments...")
    jsonl_entries = []
    copied_count = 0
    skipped_count = 0
    missing_count = 0

    for i, seg in enumerate(all_segments):
        transcript = seg.get("transcript", "").strip()
        duration = seg.get("duration", 0)
        src_path = Path(seg.get("_resolved_path", ""))

        # Skip segments without transcript
        if not transcript:
            skipped_count += 1
            continue

        # Skip very short segments
        if duration < min_duration:
            skipped_count += 1
            continue

        # Check if source file exists
        if not src_path.exists():
            missing_count += 1
            continue

        # Copy audio file with sequential naming
        new_filename = f"utt_{copied_count:04d}.wav"
        dst_path = audio_dir / new_filename

        shutil.copy2(src_path, dst_path)
        copied_count += 1

        # Create JSONL entry with relative paths
        entry = {
            "audio": f"./audio/{new_filename}",
            "text": transcript,
            "ref_audio": "./audio/ref.wav"
        }
        jsonl_entries.append(entry)

    # Write JSONL file
    jsonl_path = output_path / "train_raw.jsonl"
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for entry in jsonl_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    # Calculate total duration
    total_duration = sum(
        s.get("duration", 0) for s in all_segments
        if s.get("transcript") and s.get("duration", 0) >= min_duration
    )

    # Write dataset info
    info = {
        "speaker": speaker_name,
        "total_segments": len(jsonl_entries),
        "total_duration_seconds": total_duration,
        "total_duration_minutes": round(total_duration / 60, 2),
        "reference_file": "audio/ref.wav",
        "reference_duration": ref_segment["duration"],
        "reference_transcript": ref_segment["transcript"],
        "source_directory": source_dir,
        "min_duration_filter": min_duration
    }

    info_path = output_path / "dataset_info.json"
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Dataset created successfully!")
    print(f"{'='*60}")
    print(f"Output directory: {output_path}")
    print(f"Audio files copied: {copied_count}")
    print(f"Segments skipped (short/no transcript): {skipped_count}")
    print(f"Segments missing (file not found): {missing_count}")
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

4. Test the fine-tuned model:
   See {output_path}/test_inference.py for example code
""")

    # Create a test inference script
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
    "{output_path}/model_output/checkpoint-epoch-2",  # Adjust epoch as needed
    device_map=device,
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

# Generate speech
text = "Hello, this is a test of the fine-tuned Pascal Bornet voice model."
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
        description="Create TTS training dataset from diarization segments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with defaults
  python create_tts_dataset.py

  # Custom source and output directories
  python create_tts_dataset.py --source ./output/pascalvoice --output ./dataset1

  # Change minimum segment duration
  python create_tts_dataset.py --min-duration 3.0
        """
    )
    parser.add_argument(
        "--source", type=str,
        default="/Users/spider/Documents/GITHUB/projects/MLX-YouTubeScribe/output/pascalvoice",
        help="Source directory containing video JSON files and segments folder"
    )
    parser.add_argument(
        "--output", type=str,
        default="/Users/spider/Documents/GITHUB/projects/MLX-YouTubeScribe/dataset1",
        help="Output directory for the dataset"
    )
    parser.add_argument(
        "--speaker", type=str, default="pascal_bornet",
        help="Speaker name for the dataset (used in fine-tuning)"
    )
    parser.add_argument(
        "--min-duration", type=float, default=2.0,
        help="Minimum segment duration in seconds (default: 2.0)"
    )

    args = parser.parse_args()

    create_tts_dataset(
        source_dir=args.source,
        output_dir=args.output,
        speaker_name=args.speaker,
        min_duration=args.min_duration
    )
