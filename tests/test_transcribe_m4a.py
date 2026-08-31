import contextlib
import io
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import transcribe_m4a


class ValidateAudioPathTests(unittest.TestCase):
    def test_accepts_existing_m4a_case_insensitively(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "recording.M4A"
            audio_path.touch()

            self.assertEqual(
                transcribe_m4a.validate_audio_path(str(audio_path)),
                audio_path.resolve(),
            )

    def test_rejects_missing_file(self):
        with self.assertRaisesRegex(ValueError, "not found"):
            transcribe_m4a.validate_audio_path("missing.m4a")

    def test_rejects_non_m4a_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "recording.wav"
            audio_path.touch()

            with self.assertRaisesRegex(ValueError, "Expected an .m4a"):
                transcribe_m4a.validate_audio_path(str(audio_path))


class FfmpegConversionTests(unittest.TestCase):
    @mock.patch("transcribe_m4a.subprocess.run")
    @mock.patch("transcribe_m4a.shutil.which", return_value="/usr/bin/ffmpeg")
    def test_uses_argument_list_for_paths_with_spaces(self, _which, run):
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "input with spaces.m4a"
            wav_path = Path(temp_dir) / "output with spaces.wav"
            audio_path.touch()

            def create_output(command, **_kwargs):
                Path(command[-1]).write_bytes(b"wav")
                return subprocess.CompletedProcess(command, 0, "", "")

            run.side_effect = create_output
            transcribe_m4a.convert_m4a_to_wav(audio_path, wav_path)

            command = run.call_args.args[0]
            self.assertIsInstance(command, list)
            self.assertIn(str(audio_path), command)
            self.assertIn(str(wav_path), command)
            self.assertIn("16000", command)
            self.assertIn("pcm_s16le", command)

    @mock.patch("transcribe_m4a.shutil.which", return_value=None)
    def test_reports_missing_ffmpeg(self, _which):
        with self.assertRaisesRegex(transcribe_m4a.TranscriptionError, "not found"):
            transcribe_m4a.convert_m4a_to_wav(Path("audio.m4a"), Path("audio.wav"))

    @mock.patch("transcribe_m4a.subprocess.run")
    @mock.patch("transcribe_m4a.shutil.which", return_value="/usr/bin/ffmpeg")
    def test_reports_ffmpeg_decode_error(self, _which, run):
        run.return_value = subprocess.CompletedProcess([], 1, "", "invalid data")

        with self.assertRaisesRegex(transcribe_m4a.TranscriptionError, "invalid data"):
            transcribe_m4a.convert_m4a_to_wav(Path("audio.m4a"), Path("audio.wav"))


class TranscriptionLifecycleTests(unittest.TestCase):
    @mock.patch(
        "whisper_mlx._import_mlx_whisper",
        side_effect=ImportError("mlx-whisper"),
    )
    def test_missing_mlx_whisper_is_transcription_error(self, _import_mlx):
        with self.assertRaisesRegex(transcribe_m4a.TranscriptionError, "mlx-whisper"):
            transcribe_m4a.AudioTranscriber()


    @mock.patch("transcribe_m4a.transcribe_wav", side_effect=RuntimeError("model failed"))
    @mock.patch("transcribe_m4a.convert_m4a_to_wav")
    @mock.patch("transcribe_m4a.AudioTranscriber")
    def test_model_is_cleaned_up_after_transcription_error(
        self, transcriber_class, convert, _transcribe_wav
    ):
        transcriber = transcriber_class.return_value

        def create_wav(_audio_path, wav_path):
            wav_path.touch()

        convert.side_effect = create_wav

        with self.assertRaisesRegex(
            transcribe_m4a.TranscriptionError,
            "Whisper transcription failed: model failed",
        ):
            transcribe_m4a.transcribe_m4a(Path("audio.m4a"))

        transcriber.cleanup.assert_called_once_with()


class WavTranscriptionTests(unittest.TestCase):
    def test_transcribes_full_wav_and_cleans_text(self):
        transcriber = mock.Mock()
        transcriber.transcribe.return_value = '  "Hello there."  '

        with tempfile.TemporaryDirectory() as temp_dir:
            wav_path = Path(temp_dir) / "audio.wav"
            with wave.open(str(wav_path), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(transcribe_m4a.SAMPLE_RATE)
                wav_file.writeframes(b"\x00\x00" * 5)

            transcript = transcribe_m4a.transcribe_wav(
                wav_path,
                transcriber,
                progress_stream=io.StringIO(),
            )

        self.assertEqual(transcript, "Hello there.")
        transcriber.transcribe.assert_called_once()
        audio = transcriber.transcribe.call_args.args[0]
        self.assertEqual(audio.shape, (5,))
        self.assertEqual(audio.dtype, np.float32)
        self.assertEqual(transcriber.transcribe.call_args.kwargs.get("verbose"), False)

    def test_rejects_empty_decoded_audio(self):
        transcriber = mock.Mock()

        with tempfile.TemporaryDirectory() as temp_dir:
            wav_path = Path(temp_dir) / "empty.wav"
            with wave.open(str(wav_path), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(transcribe_m4a.SAMPLE_RATE)

            with self.assertRaisesRegex(
                transcribe_m4a.TranscriptionError,
                "contains no decodable audio frames",
            ):
                transcribe_m4a.transcribe_wav(
                    wav_path,
                    transcriber,
                    progress_stream=io.StringIO(),
                )


class CliTests(unittest.TestCase):
    @mock.patch("transcribe_m4a.transcribe_m4a", return_value="Hello world.")
    def test_prints_only_transcript_to_stdout(self, transcribe):
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "audio.m4a"
            audio_path.touch()
            stdout = io.StringIO()

            with contextlib.redirect_stdout(stdout):
                exit_code = transcribe_m4a.main([str(audio_path)])

            self.assertEqual(exit_code, 0)
            self.assertEqual(stdout.getvalue(), "Hello world.\n")
            transcribe.assert_called_once_with(audio_path.resolve(), language="en")

    @mock.patch("transcribe_m4a.transcribe_m4a", return_value="Saved text.")
    def test_writes_requested_output_file(self, _transcribe):
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "audio.m4a"
            output_path = Path(temp_dir) / "nested" / "transcript.txt"
            audio_path.touch()

            exit_code = transcribe_m4a.main(
                [str(audio_path), "--output", str(output_path)]
            )

            self.assertEqual(exit_code, 0)
            self.assertEqual(output_path.read_text(encoding="utf-8"), "Saved text.\n")

    @mock.patch(
        "transcribe_m4a.transcribe_m4a",
        side_effect=transcribe_m4a.TranscriptionError("failed"),
    )
    def test_returns_one_when_transcription_fails(self, _transcribe):
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "audio.m4a"
            audio_path.touch()
            stderr = io.StringIO()

            with contextlib.redirect_stderr(stderr):
                exit_code = transcribe_m4a.main([str(audio_path)])

            self.assertEqual(exit_code, 1)
            self.assertIn("Error: failed", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
