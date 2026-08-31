import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import mom_from_audio


class HashAndRegistryTests(unittest.TestCase):
    def test_sha256_is_stable_for_same_bytes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "a.m4a"
            path.write_bytes(b"same-bytes")
            self.assertEqual(
                mom_from_audio.sha256_file(path),
                mom_from_audio.sha256_file(path),
            )

    def test_registry_roundtrip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_path = Path(temp_dir) / ".processed.json"
            registry = {
                "version": 1,
                "items": {
                    "abc": {
                        "status": "complete",
                        "stem": "meeting",
                        "mom_path": str(Path(temp_dir) / "meeting.mom.md"),
                    }
                },
            }
            mom_from_audio.save_registry(registry_path, registry)
            loaded = mom_from_audio.load_registry(registry_path)
            self.assertEqual(loaded["items"]["abc"]["status"], "complete")


class JobDiscoveryTests(unittest.TestCase):
    def test_list_inbox_only_top_level_m4a(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            inbox = Path(temp_dir)
            (inbox / "one.m4a").write_bytes(b"1")
            (inbox / "two.M4A").write_bytes(b"2")
            (inbox / "notes.txt").write_text("nope", encoding="utf-8")
            nested = inbox / "nested"
            nested.mkdir()
            (nested / "hidden.m4a").write_bytes(b"3")

            names = [p.name.lower() for p in mom_from_audio.list_inbox_m4a(inbox)]
            self.assertEqual(names, ["one.m4a", "two.m4a"])

    def test_skips_complete_hash_even_if_renamed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            inbox = Path(temp_dir)
            output = inbox / "output"
            output.mkdir()
            audio = inbox / "renamed.m4a"
            audio.write_bytes(b"unique-audio-bytes")
            content_hash = mom_from_audio.sha256_file(audio)
            mom_path = output / "original.mom.md"
            transcript_path = output / "original.transcript.txt"
            mom_path.write_text("# Minutes\n", encoding="utf-8")
            transcript_path.write_text("hello\n", encoding="utf-8")

            registry = {
                "version": 1,
                "items": {
                    content_hash: {
                        "status": "complete",
                        "stem": "original",
                        "transcript_path": str(transcript_path),
                        "mom_path": str(mom_path),
                        "source_name": "original.m4a",
                    }
                },
            }
            jobs = mom_from_audio.build_jobs([audio], output, registry, force=False)
            self.assertEqual(len(jobs), 1)
            self.assertEqual(jobs[0].status, "complete")
            self.assertEqual(jobs[0].stem, "original")

    def test_force_marks_pending(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            inbox = Path(temp_dir)
            output = inbox / "output"
            output.mkdir()
            audio = inbox / "a.m4a"
            audio.write_bytes(b"bytes")
            content_hash = mom_from_audio.sha256_file(audio)
            mom_path = output / "a.mom.md"
            mom_path.write_text("# Minutes\n", encoding="utf-8")
            registry = {
                "version": 1,
                "items": {
                    content_hash: {
                        "status": "complete",
                        "stem": "a",
                        "transcript_path": str(output / "a.transcript.txt"),
                        "mom_path": str(mom_path),
                    }
                },
            }
            jobs = mom_from_audio.build_jobs([audio], output, registry, force=True)
            self.assertEqual(jobs[0].status, "pending")

    def test_resumes_from_transcript_only(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            inbox = Path(temp_dir)
            output = inbox / "output"
            output.mkdir()
            audio = inbox / "a.m4a"
            audio.write_bytes(b"bytes")
            content_hash = mom_from_audio.sha256_file(audio)
            transcript_path = output / "a.transcript.txt"
            transcript_path.write_text("partial transcript\n", encoding="utf-8")
            registry = {
                "version": 1,
                "items": {
                    content_hash: {
                        "status": "transcribed",
                        "stem": "a",
                        "transcript_path": str(transcript_path),
                        "mom_path": str(output / "a.mom.md"),
                    }
                },
            }
            jobs = mom_from_audio.build_jobs([audio], output, registry)
            self.assertEqual(jobs[0].status, "transcribed")


class OmlxClientTests(unittest.TestCase):
    def test_chat_completions_url_normalization(self):
        self.assertEqual(
            mom_from_audio.chat_completions_url("http://127.0.0.1:8000/v1/"),
            "http://127.0.0.1:8000/v1/chat/completions",
        )
        self.assertEqual(
            mom_from_audio.chat_completions_url(
                "http://127.0.0.1:8000/v1/chat/completions"
            ),
            "http://127.0.0.1:8000/v1/chat/completions",
        )

    def test_generate_mom_parses_openai_shape(self):
        payload = {
            "choices": [
                {"message": {"content": "# Minutes of Meeting\n\n## Meta\n- ok\n"}}
            ]
        }

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps(payload).encode("utf-8")

        with mock.patch(
            "mom_from_audio.urllib.request.urlopen", return_value=FakeResponse()
        ) as urlopen:
            text = mom_from_audio.generate_mom_markdown(
                "We decided to ship on Friday.",
                source_name="sync.m4a",
                system_prompt="Be a secretary.",
                base_url="http://127.0.0.1:8000/v1",
                model="DeepSeek-V4-Flash-0731-MXFP4-MLX",
                api_key="test-key",
            )
        self.assertIn("# Minutes of Meeting", text)
        request = urlopen.call_args.args[0]
        self.assertEqual(request.get_header("Authorization"), "Bearer test-key")

    def test_resolve_api_key_from_settings_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = Path(temp_dir) / "settings.json"
            settings.write_text(
                json.dumps({"auth": {"api_key": "from-file"}}),
                encoding="utf-8",
            )
            self.assertEqual(
                mom_from_audio.resolve_omlx_api_key(settings_path=settings),
                "from-file",
            )


class PipelineIntegrationTests(unittest.TestCase):
    def test_empty_inbox_is_zero(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            inbox = Path(temp_dir)
            code = mom_from_audio.run_pipeline(
                inbox_dir=inbox,
                output_dir=inbox / "output",
                registry_path=inbox / ".processed.json",
                skip_mom=True,
            )
            self.assertEqual(code, 0)

    def test_skip_complete_without_calling_whisper(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            inbox = Path(temp_dir)
            output = inbox / "output"
            output.mkdir()
            audio = inbox / "done.m4a"
            audio.write_bytes(b"already-done")
            content_hash = mom_from_audio.sha256_file(audio)
            mom_path = output / "done.mom.md"
            transcript_path = output / "done.transcript.txt"
            mom_path.write_text("# Minutes\n", encoding="utf-8")
            transcript_path.write_text("text\n", encoding="utf-8")
            registry = {
                "version": 1,
                "items": {
                    content_hash: {
                        "status": "complete",
                        "stem": "done",
                        "transcript_path": str(transcript_path),
                        "mom_path": str(mom_path),
                        "source_name": "done.m4a",
                    }
                },
            }
            registry_path = inbox / ".processed.json"
            mom_from_audio.save_registry(registry_path, registry)

            with mock.patch.object(mom_from_audio.transcribe_m4a, "AudioTranscriber") as tr:
                stderr = io.StringIO()
                with mock.patch("sys.stderr", stderr):
                    code = mom_from_audio.run_pipeline(
                        inbox_dir=inbox,
                        output_dir=output,
                        registry_path=registry_path,
                    )
                self.assertEqual(code, 0)
                tr.assert_not_called()
                self.assertIn("already processed", stderr.getvalue())

    def test_transcribe_then_mom_and_second_run_skips(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            inbox = Path(temp_dir)
            output = inbox / "output"
            audio = inbox / "meeting.m4a"
            audio.write_bytes(b"audio-bytes")
            registry_path = inbox / ".processed.json"
            prompt_path = Path(temp_dir) / "prompt.md"
            prompt_path.write_text("System prompt", encoding="utf-8")

            fake_transcriber = mock.Mock()

            with mock.patch.object(
                mom_from_audio.transcribe_m4a,
                "AudioTranscriber",
                return_value=fake_transcriber,
            ), mock.patch.object(
                mom_from_audio,
                "transcribe_one",
                return_value="Alice will ship the patch by Friday.",
            ), mock.patch.object(
                mom_from_audio,
                "generate_mom_markdown",
                return_value="# Minutes of Meeting\n\nDone.\n",
            ) as gen:
                code = mom_from_audio.run_pipeline(
                    inbox_dir=inbox,
                    output_dir=output,
                    registry_path=registry_path,
                    prompt_path=prompt_path,
                    omlx_model="DeepSeek-V4-Flash-0731-MXFP4-MLX",
                )
                self.assertEqual(code, 0)
                self.assertTrue((output / "meeting.transcript.txt").is_file())
                self.assertTrue((output / "meeting.mom.md").is_file())
                self.assertEqual(gen.call_count, 1)
                fake_transcriber.cleanup.assert_called()

                # Second run: identical content must not re-enter Whisper/LLM.
                fake_transcriber.reset_mock()
                gen.reset_mock()
                code = mom_from_audio.run_pipeline(
                    inbox_dir=inbox,
                    output_dir=output,
                    registry_path=registry_path,
                    prompt_path=prompt_path,
                )
                self.assertEqual(code, 0)
                gen.assert_not_called()
                fake_transcriber.assert_not_called()


class CliTests(unittest.TestCase):
    def test_main_no_args_uses_inbox(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            inbox = Path(temp_dir)
            with mock.patch.object(mom_from_audio, "run_pipeline", return_value=0) as run:
                code = mom_from_audio.main(["--inbox", str(inbox)])
                self.assertEqual(code, 0)
                kwargs = run.call_args.kwargs
                self.assertEqual(kwargs["inbox_dir"], inbox)
                self.assertIsNone(kwargs["audio_files"])


if __name__ == "__main__":
    unittest.main()
