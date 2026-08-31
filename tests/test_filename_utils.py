import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import filename_utils as fu

HAS_YTDLP = importlib.util.find_spec("yt_dlp") is not None


class StemEquivalenceTests(unittest.TestCase):
    def test_current_and_legacy_slash_are_equivalent(self):
        current = f"w{fu.YTDLP_SLASH} Max Hodak"
        legacy = f"w{fu.LEGACY_SLASH} Max Hodak"
        self.assertTrue(fu.stems_equivalent(current, legacy))
        self.assertNotEqual(current, legacy)

    def test_distinct_titles_are_not_equivalent(self):
        self.assertFalse(
            fu.stems_equivalent(
                f"Ex-Neuralink Founder w{fu.YTDLP_SLASH} Max Hodak",
                "Meta Buys Moltbook, GPT 5.4, and Fruitfly Brain Upload",
            )
        )


class RecursionLimitTests(unittest.TestCase):
    def test_278_videos_covers_full_pipeline(self):
        # 9 node visits per video * 278 + start/finalize < computed limit
        limit = fu.recursion_limit_for(278)
        self.assertGreaterEqual(limit, 278 * 9 + 50)
        self.assertGreater(limit, 1000)

    def test_small_jobs_keep_previous_floor(self):
        self.assertEqual(fu.recursion_limit_for(1), 1000)
        self.assertEqual(fu.recursion_limit_for(0), 1000)


class ResolveMediaTests(unittest.TestCase):
    def test_finds_file_with_ytdlp_slash_when_info_has_legacy_title(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            wav_name = f"Talk w{fu.YTDLP_SLASH} Guest.wav"
            wav_path = Path(temp_dir) / wav_name
            wav_path.write_bytes(b"RIFF")
            decoy = Path(temp_dir) / "Meta Buys Moltbook.wav"
            decoy.write_bytes(b"RIFF")

            with mock.patch.object(fu, "sanitize_title", return_value=f"Talk w{fu.LEGACY_SLASH} Guest"):
                found = fu.resolve_downloaded_media(
                    temp_dir,
                    {"title": "Talk w/ Guest"},
                    "wav",
                )
            self.assertEqual(found, str(wav_path))

    def test_does_not_return_unrelated_wav(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            decoy = Path(temp_dir) / "Meta Buys Moltbook.wav"
            decoy.write_bytes(b"RIFF")
            with mock.patch.object(fu, "sanitize_title", return_value=f"Talk w{fu.YTDLP_SLASH} Guest"):
                found = fu.resolve_downloaded_media(
                    temp_dir,
                    {"title": "Talk w/ Guest"},
                    "wav",
                )
            self.assertIsNone(found)

    def test_prefers_ytdlp_reported_filepath(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            real = Path(temp_dir) / "real.wav"
            real.write_bytes(b"RIFF")
            decoy = Path(temp_dir) / "decoy.wav"
            decoy.write_bytes(b"RIFF")
            found = fu.resolve_downloaded_media(
                temp_dir,
                {"requested_downloads": [{"filepath": str(real)}]},
                "wav",
            )
            self.assertEqual(found, str(real))


class YoutubeIdTests(unittest.TestCase):
    def test_extracts_from_watch_url_and_playlist_query(self):
        self.assertEqual(
            fu.youtube_video_id("https://www.youtube.com/watch?v=Slle5_AxBzs"),
            "Slle5_AxBzs",
        )
        self.assertEqual(
            fu.youtube_video_id("https://www.youtube.com/watch?v=Slle5_AxBzs&list=PLabc"),
            "Slle5_AxBzs",
        )
        self.assertEqual(fu.youtube_video_id("Slle5_AxBzs"), "Slle5_AxBzs")
        self.assertIsNone(fu.youtube_video_id("https://www.youtube.com/playlist?list=PLabc"))


class FindExistingTranscriptTests(unittest.TestCase):
    def test_finds_by_video_id_when_playlist_title_differs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            fu.invalidate_transcript_index(temp_dir)
            json_path = Path(temp_dir) / "Chinas Endgame Graylin 281.json"
            json_path.write_text(json.dumps({
                "video_info": {
                    "url": "https://www.youtube.com/watch?v=Slle5_AxBzs",
                    "title": "China's Endgame",
                },
                "audio_file": "Chinas Endgame Graylin 281.wav",
            }))
            txt_path = Path(temp_dir) / "Chinas Endgame Graylin 281.txt"
            txt_path.write_text("ok")
            with mock.patch.object(fu, "sanitize_title", side_effect=lambda t: t):
                found_json, found_txt = fu.find_existing_transcript(
                    temp_dir,
                    title="The US-China AI Race: China's Endgame, ASI Timelines | 281",
                    url="https://www.youtube.com/watch?v=Slle5_AxBzs",
                )
            self.assertEqual(found_json, str(json_path))
            self.assertEqual(found_txt, str(txt_path))


class TranscriptMatchTests(unittest.TestCase):
    def test_mismatch_is_detected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            json_path = Path(temp_dir) / f"Ex-Neuralink Founder w{fu.LEGACY_SLASH} Max Hodak.json"
            json_path.write_text(json.dumps({
                "audio_file": "Meta Buys Moltbook, GPT 5.4, and Fruitfly Brain Upload.wav",
            }))
            self.assertFalse(fu.transcript_audio_matches(str(json_path)))

    def test_slash_encoding_only_is_a_match(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            json_path = Path(temp_dir) / f"Talk w{fu.LEGACY_SLASH} Guest.json"
            json_path.write_text(json.dumps({
                "audio_file": f"Talk w{fu.YTDLP_SLASH} Guest.wav",
            }))
            self.assertTrue(fu.transcript_audio_matches(str(json_path)))


class QuarantineTests(unittest.TestCase):
    def test_moves_mismatched_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            json_path = Path(temp_dir) / "bad.json"
            txt_path = Path(temp_dir) / "bad.txt"
            json_path.write_text("{}")
            txt_path.write_text("nope")
            fu.quarantine_output_files(temp_dir, [str(json_path), str(txt_path)], "mismatch")
            self.assertFalse(json_path.exists())
            self.assertTrue((Path(temp_dir) / "_corrupt_transcripts" / "bad.json").exists())
            self.assertTrue((Path(temp_dir) / "_corrupt_transcripts" / "bad.txt").exists())


@unittest.skipUnless(HAS_YTDLP, "yt-dlp not installed")
class SanitizeTitleTests(unittest.TestCase):
    def test_slash_maps_to_big_solidus_not_fullwidth(self):
        title = "Ex-Neuralink Founder: AI Enhanced Bodies Are Nearly Here w/ Max Hodak | EP #171"
        sanitized = fu.sanitize_title(title)
        self.assertIn(fu.YTDLP_SLASH, sanitized)
        self.assertNotIn(fu.LEGACY_SLASH, sanitized)
        self.assertNotIn("/", sanitized)


if __name__ == "__main__":
    unittest.main()
