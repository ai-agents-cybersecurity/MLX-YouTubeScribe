#!/usr/bin/env bash
# Process all new .m4a files in the Desktop drop folder → transcript + MoM.
#
# Usage:
#   ./run_mom_inbox.sh
#   ./run_mom_inbox.sh --force
#   ./run_mom_inbox.sh /path/to/file.m4a
#
# Env overrides:
#   MOM_INBOX_DIR   default: ~/Desktop/AudioToMoM
#   OMLX_BASE_URL   default: http://127.0.0.1:8000/v1
#   OMLX_MODEL      default: DeepSeek-V4-Flash-0731-MXFP4-MLX
#   OMLX_API_KEY    optional; else read from ~/.omlx/settings.json
#   CONDA_ENV       default: youtube

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/src/mom_from_audio.py"
CONDA_ENV="${CONDA_ENV:-youtube}"
INBOX_DIR="${MOM_INBOX_DIR:-${HOME}/Desktop/AudioToMoM}"

die() {
    printf 'Error: %s\n' "$*" >&2
    exit 1
}

[[ -f "${PYTHON_SCRIPT}" ]] || die "Missing ${PYTHON_SCRIPT}"

mkdir -p "${INBOX_DIR}"

resolve_python() {
    if [[ -n "${MOM_PYTHON:-}" && -x "${MOM_PYTHON}" ]]; then
        printf '%s\n' "${MOM_PYTHON}"
        return
    fi

    # Prefer the youtube conda env the user already uses for transcription.
    if command -v conda >/dev/null 2>&1; then
        local conda_base
        conda_base="$(conda info --base 2>/dev/null || true)"
        if [[ -n "${conda_base}" && -x "${conda_base}/envs/${CONDA_ENV}/bin/python" ]]; then
            printf '%s\n' "${conda_base}/envs/${CONDA_ENV}/bin/python"
            return
        fi
    fi

    if [[ -x "${HOME}/miniconda3/envs/${CONDA_ENV}/bin/python" ]]; then
        printf '%s\n' "${HOME}/miniconda3/envs/${CONDA_ENV}/bin/python"
        return
    fi

    if [[ -x "${SCRIPT_DIR}/.venv/bin/python" ]]; then
        printf '%s\n' "${SCRIPT_DIR}/.venv/bin/python"
        return
    fi

    if command -v python3 >/dev/null 2>&1; then
        command -v python3
        return
    fi

    die "No Python found. Set MOM_PYTHON or activate the '${CONDA_ENV}' conda env."
}

PYTHON_BIN="$(resolve_python)"
printf '==> Using Python: %s\n' "${PYTHON_BIN}" >&2
printf '==> Inbox: %s\n' "${INBOX_DIR}" >&2
printf '==> Ensure oMLX is running with your DeepSeek model loaded.\n' >&2

export MOM_INBOX_DIR="${INBOX_DIR}"
export OMLX_BASE_URL="${OMLX_BASE_URL:-http://127.0.0.1:8000/v1}"
export OMLX_MODEL="${OMLX_MODEL:-DeepSeek-V4-Flash-0731-MXFP4-MLX}"

exec "${PYTHON_BIN}" "${PYTHON_SCRIPT}" "$@"
