import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SETUP_SCRIPT = PROJECT_ROOT / "setup_and_run_audio_transcription.sh"
TRANSCRIBE_SCRIPT = PROJECT_ROOT / "src" / "transcribe_m4a.py"


FAKE_PYTHON = """#!/usr/bin/env bash
set -eu

if [[ "${1:-}" == "-c" ]]; then
    if [[ "${2:-}" == *"mlx_whisper"* ]]; then
        if [[ "${FAKE_DEPS_MISSING:-0}" == "1" && ! -f "${FAKE_DEPS_MARKER}" ]]; then
            exit 1
        fi
    fi
    exit 0
fi

if [[ "${1:-}" == "-m" && "${2:-}" == "pip" ]]; then
    printf '%s\\n' "$*" >> "${FAKE_PIP_LOG}"
    if [[ "$*" == *"numpy mlx mlx-whisper"* ]]; then
        : > "${FAKE_DEPS_MARKER}"
    fi
    exit 0
fi

printf '%s\\n' "$@" > "${FAKE_ARG_LOG}"
exit "${FAKE_TRANSCRIBE_EXIT:-0}"
"""


FAKE_PYTHON_CREATOR = """#!/usr/bin/env bash
set -eu

if [[ "${1:-}" == "-c" ]]; then
    exit 0
fi
if [[ "${1:-}" == "-m" && "${2:-}" == "venv" ]]; then
    printf '%s\\n' "$*" > "${FAKE_VENV_LOG}"
    mkdir -p "${3}/bin"
    cp "${FAKE_PYTHON_TEMPLATE}" "${3}/bin/python"
    chmod +x "${3}/bin/python"
    exit 0
fi
exit 1
"""


class SetupAndRunTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.bin_dir = self.temp_path / "bin"
        self.venv_dir = self.temp_path / "venv"
        self.venv_bin_dir = self.venv_dir / "bin"
        self.bin_dir.mkdir()
        self.venv_bin_dir.mkdir(parents=True)

        self._write_executable(self.bin_dir / "ffmpeg", "#!/bin/sh\nexit 0\n")
        self._write_executable(self.venv_bin_dir / "python", FAKE_PYTHON)

        self.audio_path = self.temp_path / "audio recording.m4a"
        self.audio_path.touch()
        self.arg_log = self.temp_path / "args.log"
        self.pip_log = self.temp_path / "pip.log"
        self.deps_marker = self.temp_path / "dependencies-installed"
        self.venv_log = self.temp_path / "venv.log"

        self.env = os.environ.copy()
        self.env.update(
            {
                "PATH": f"{self.bin_dir}{os.pathsep}{self.env['PATH']}",
                "VENV_DIR": str(self.venv_dir),
                "FAKE_ARG_LOG": str(self.arg_log),
                "FAKE_PIP_LOG": str(self.pip_log),
                "FAKE_DEPS_MARKER": str(self.deps_marker),
                "FAKE_VENV_LOG": str(self.venv_log),
            }
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    @staticmethod
    def _write_executable(path, contents):
        path.write_text(contents, encoding="utf-8")
        path.chmod(0o755)

    def run_script(self, *args, **kwargs):
        return subprocess.run(
            [str(SETUP_SCRIPT), *map(str, args)],
            cwd=kwargs.get("cwd", self.temp_path),
            env=kwargs.get("env", self.env),
            capture_output=True,
            text=True,
            check=False,
        )

    def test_script_is_executable_and_valid_bash(self):
        self.assertTrue(os.access(SETUP_SCRIPT, os.X_OK))
        result = subprocess.run(
            ["bash", "-n", str(SETUP_SCRIPT)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_forwards_spaced_paths_and_options_from_another_directory(self):
        output_path = self.temp_path / "output files" / "transcript.txt"
        result = self.run_script(
            self.audio_path,
            "--language",
            "es",
            "--output",
            output_path,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            self.arg_log.read_text(encoding="utf-8").splitlines(),
            [
                str(TRANSCRIBE_SCRIPT),
                str(self.audio_path),
                "--language",
                "es",
                "--output",
                str(output_path),
            ],
        )
        self.assertFalse(self.pip_log.exists())

    def test_installs_only_focused_dependencies_when_missing(self):
        env = self.env.copy()
        env["FAKE_DEPS_MISSING"] = "1"

        result = self.run_script(self.audio_path, env=env)

        self.assertEqual(result.returncode, 0, result.stderr)
        pip_commands = self.pip_log.read_text(encoding="utf-8").splitlines()
        self.assertEqual(pip_commands[0], "-m pip install --upgrade pip")
        self.assertEqual(
            pip_commands[1],
            "-m pip install --upgrade numpy mlx mlx-whisper",
        )
        self.assertTrue(self.deps_marker.exists())

    def test_creates_fresh_virtual_environment_with_python_override(self):
        python_template = self.bin_dir / "fake-venv-python"
        python_creator = self.bin_dir / "python-custom"
        self._write_executable(python_template, FAKE_PYTHON)
        self._write_executable(python_creator, FAKE_PYTHON_CREATOR)
        shutil.rmtree(self.venv_dir)

        env = self.env.copy()
        env["PYTHON_BIN"] = str(python_creator)
        env["FAKE_PYTHON_TEMPLATE"] = str(python_template)

        result = self.run_script(self.audio_path, env=env)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            self.venv_log.read_text(encoding="utf-8").strip(),
            f"-m venv {self.venv_dir}",
        )
        self.assertTrue((self.venv_dir / "bin" / "python").is_file())
        self.assertTrue(self.arg_log.is_file())

    def test_propagates_transcription_exit_code(self):
        env = self.env.copy()
        env["FAKE_TRANSCRIBE_EXIT"] = "7"

        result = self.run_script(self.audio_path, env=env)

        self.assertEqual(result.returncode, 7)

    def test_rejects_missing_input_before_setup(self):
        result = self.run_script(self.temp_path / "missing.m4a")

        self.assertEqual(result.returncode, 1)
        self.assertIn("Audio file not found", result.stderr)
        self.assertFalse(self.arg_log.exists())
        self.assertFalse(self.pip_log.exists())

    def test_help_does_not_run_setup(self):
        result = self.run_script("--help")

        self.assertEqual(result.returncode, 0)
        self.assertIn("Usage:", result.stdout)
        self.assertFalse(self.arg_log.exists())
        self.assertFalse(self.pip_log.exists())

    def test_missing_ffmpeg_fails_before_python_setup(self):
        restricted_bin = self.temp_path / "restricted-bin"
        restricted_bin.mkdir()
        for command in ("bash", "dirname"):
            executable = shutil.which(command)
            self.assertIsNotNone(executable)
            (restricted_bin / command).symlink_to(executable)

        env = self.env.copy()
        env["PATH"] = str(restricted_bin)
        result = self.run_script(self.audio_path, env=env)

        self.assertEqual(result.returncode, 1)
        self.assertIn("FFmpeg", result.stderr)
        self.assertFalse(self.arg_log.exists())


if __name__ == "__main__":
    unittest.main()
