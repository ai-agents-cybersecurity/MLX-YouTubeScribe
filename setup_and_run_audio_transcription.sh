#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
TRANSCRIBE_SCRIPT="${SCRIPT_DIR}/src/transcribe_m4a.py"
VENV_DIR="${VENV_DIR:-${SCRIPT_DIR}/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_PYTHON="${VENV_DIR}/bin/python"
DEPENDENCY_CHECK='import numpy, mlx_whisper'


usage() {
    cat <<EOF
Usage: $(basename "$0") AUDIO_FILE.m4a [TRANSCRIPTION_OPTIONS]

Set up a local Python environment and transcribe an M4A audio file.

Options forwarded to transcribe_m4a.py:
  -o, --output PATH       Write the transcript to a UTF-8 text file
  --language LANGUAGE     Whisper language name or code (default: en)
  -h, --help              Show this help message

Environment overrides:
  PYTHON_BIN              Python used to create the environment (default: python3)
  VENV_DIR                Virtual environment directory (default: .venv)

Examples:
  $(basename "$0") /path/to/audio.m4a
  $(basename "$0") /path/to/audio.m4a --output transcript.txt
  $(basename "$0") /path/to/audio.m4a --language es
EOF
}


log() {
    printf '==> %s\n' "$*" >&2
}


die() {
    printf 'Error: %s\n' "$*" >&2
    exit 1
}


run_as_root() {
    if [[ "${EUID}" -eq 0 ]]; then
        "$@" >&2
    elif command -v sudo >/dev/null 2>&1; then
        sudo "$@" >&2
    else
        die "Installing FFmpeg requires root privileges. Re-run with sudo available or install FFmpeg manually."
    fi
}


ensure_ffmpeg() {
    if command -v ffmpeg >/dev/null 2>&1; then
        return
    fi

    log "FFmpeg was not found; attempting to install it."
    case "${OSTYPE:-}" in
        darwin*)
            if ! command -v brew >/dev/null 2>&1; then
                die "Homebrew is required to install FFmpeg on macOS. Install Homebrew, then run: brew install ffmpeg"
            fi
            brew install ffmpeg >&2
            ;;
        linux*)
            if command -v apt-get >/dev/null 2>&1; then
                run_as_root apt-get update
                run_as_root apt-get install -y ffmpeg
            elif command -v dnf >/dev/null 2>&1; then
                run_as_root dnf install -y ffmpeg
            elif command -v yum >/dev/null 2>&1; then
                run_as_root yum install -y ffmpeg
            elif command -v pacman >/dev/null 2>&1; then
                run_as_root pacman -S --needed --noconfirm ffmpeg
            elif command -v zypper >/dev/null 2>&1; then
                run_as_root zypper --non-interactive install ffmpeg
            else
                die "No supported package manager was found. Install FFmpeg manually and ensure it is on PATH."
            fi
            ;;
        *)
            die "Install FFmpeg manually and ensure it is on PATH."
            ;;
    esac

    hash -r
    command -v ffmpeg >/dev/null 2>&1 || die "FFmpeg installation completed, but ffmpeg is still not on PATH."
}


check_python_version() {
    local python_path="$1"
    if ! "${python_path}" -c 'import sys; raise SystemExit(sys.version_info < (3, 9))' >/dev/null 2>&1; then
        die "Python 3.9 or newer is required: ${python_path}"
    fi
}


ensure_virtual_environment() {
    if [[ -x "${VENV_PYTHON}" ]]; then
        check_python_version "${VENV_PYTHON}"
        log "Using existing virtual environment: ${VENV_DIR}"
        return
    fi

    if [[ -e "${VENV_DIR}" ]]; then
        die "${VENV_DIR} exists but is not a usable Python virtual environment."
    fi
    if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
        die "Python executable not found: ${PYTHON_BIN}"
    fi

    check_python_version "${PYTHON_BIN}"
    log "Creating virtual environment: ${VENV_DIR}"
    "${PYTHON_BIN}" -m venv "${VENV_DIR}" >&2
    [[ -x "${VENV_PYTHON}" ]] || die "Virtual environment creation failed: ${VENV_DIR}"
}


ensure_python_dependencies() {
    if "${VENV_PYTHON}" -c "${DEPENDENCY_CHECK}" >/dev/null 2>&1; then
        log "Python transcription dependencies are already installed."
        return
    fi

    log "Installing Python transcription dependencies..."
    "${VENV_PYTHON}" -m pip install --upgrade pip >&2
    "${VENV_PYTHON}" -m pip install --upgrade numpy mlx mlx-whisper >&2

    if ! "${VENV_PYTHON}" -c "${DEPENDENCY_CHECK}" >/dev/null 2>&1; then
        die "Python dependencies were installed but could not be imported."
    fi
}


main() {
    if [[ $# -eq 0 ]]; then
        usage >&2
        exit 2
    fi
    if [[ "$1" == "-h" || "$1" == "--help" ]]; then
        usage
        exit 0
    fi

    local audio_file="$1"
    shift

    [[ -f "${audio_file}" ]] || die "Audio file not found: ${audio_file}"
    case "${audio_file}" in
        *.[mM]4[aA]) ;;
        *) die "Expected an .m4a audio file: ${audio_file}" ;;
    esac
    [[ -f "${TRANSCRIBE_SCRIPT}" ]] || die "Transcription script not found: ${TRANSCRIBE_SCRIPT}"

    ensure_ffmpeg
    ensure_virtual_environment
    ensure_python_dependencies

    log "Starting transcription: ${audio_file}"
    exec "${VENV_PYTHON}" "${TRANSCRIBE_SCRIPT}" "${audio_file}" "$@"
}


main "$@"
