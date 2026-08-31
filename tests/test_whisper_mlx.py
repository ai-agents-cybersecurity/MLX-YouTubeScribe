import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import whisper_mlx


class PoisonListArray:
    """Array-like that must not be converted via Python lists."""

    def __array__(self, dtype=None):
        return np.array([0.25, -0.5], dtype=dtype or np.float32)

    def tolist(self):
        raise AssertionError("as_whisper_audio must not call .tolist()")


class AsWhisperAudioTests(unittest.TestCase):
    def test_keeps_file_paths(self):
        self.assertEqual(whisper_mlx.as_whisper_audio("/tmp/a.wav"), "/tmp/a.wav")

    def test_uses_numpy_buffer_not_python_lists(self):
        audio = whisper_mlx.as_whisper_audio(PoisonListArray())
        np.testing.assert_array_almost_equal(audio, np.array([0.25, -0.5], dtype=np.float32))
        self.assertEqual(audio.dtype, np.float32)


class AudioTranscriberTests(unittest.TestCase):
    def tearDown(self):
        whisper_mlx.AudioTranscriber._instance = None

    def test_transcribe_calls_mlx_whisper_with_metal_repo(self):
        fake_mx = types.SimpleNamespace(float16="float16", clear_cache=mock.Mock())
        fake_holder = mock.Mock()
        fake_mlx_whisper = types.SimpleNamespace(
            transcribe=mock.Mock(return_value={"text": "  hello from metal  "})
        )

        with mock.patch.object(
            whisper_mlx,
            "_import_mlx_whisper",
            return_value=(fake_mx, fake_mlx_whisper, fake_holder),
        ):
            transcriber = whisper_mlx.AudioTranscriber(language="en")
            text = transcriber.transcribe(np.zeros(4, dtype=np.float32), verbose=False)

        fake_holder.get_model.assert_called_once_with(
            "mlx-community/whisper-large-v3-turbo",
            "float16",
        )
        kwargs = fake_mlx_whisper.transcribe.call_args.kwargs
        self.assertEqual(kwargs["path_or_hf_repo"], "mlx-community/whisper-large-v3-turbo")
        self.assertEqual(kwargs["language"], "en")
        self.assertTrue(kwargs["fp16"])
        self.assertEqual(text, "hello from metal")

    def test_cleanup_drops_cached_model(self):
        holder = types.SimpleNamespace(model=object(), model_path="repo")
        fake_module = types.ModuleType("mlx_whisper.transcribe")
        fake_module.ModelHolder = holder

        with mock.patch.dict(sys.modules, {"mlx_whisper.transcribe": fake_module}):
            whisper_mlx.unload_whisper_model()

        self.assertIsNone(holder.model)
        self.assertIsNone(holder.model_path)
