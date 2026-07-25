#!/usr/bin/env bash
set -Eeuo pipefail

if ! command -v docker >/dev/null 2>&1; then
    echo "Docker is not installed or is not available in PATH." >&2
    exit 1
fi

if ! docker compose version >/dev/null 2>&1; then
    echo "Docker Compose v2 is required (the 'docker compose' command)." >&2
    exit 1
fi

export WEBCAM_DEVICE="${WEBCAM_DEVICE:-/dev/video0}"
export DISPLAY="${DISPLAY:-:0}"
export HOST_UID="$(id -u)"
export HOST_GID="$(id -g)"
export COMPOSE_FILE="${COMPOSE_FILE:-compose.yaml}"
export BUILD_IMAGE="${BUILD_IMAGE:-1}"

if [[ ! -e "${WEBCAM_DEVICE}" ]]; then
    echo "Camera device ${WEBCAM_DEVICE} was not found." >&2
    echo "Available video devices:" >&2
    ls -1 /dev/video* 2>/dev/null || true
    echo "Run with WEBCAM_DEVICE=/dev/videoN ./run-linux.sh if needed." >&2
    exit 1
fi

export CAMERA_GID="$(stat -c '%g' "${WEBCAM_DEVICE}")"
mkdir -p Output_Folder

XHOST_CHANGED=0
XHOST_RULE="SI:localuser:$(id -un)"
if command -v xhost >/dev/null 2>&1; then
    xhost +"${XHOST_RULE}" >/dev/null
    XHOST_CHANGED=1
else
    echo "Warning: xhost is unavailable; the Tkinter window may not open." >&2
fi

cleanup() {
    if [[ "${XHOST_CHANGED}" -eq 1 ]]; then
        xhost -"${XHOST_RULE}" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT INT TERM

if [[ "${BUILD_IMAGE}" == "1" ]]; then
    docker compose -f "${COMPOSE_FILE}" up --build
else
    docker compose -f "${COMPOSE_FILE}" up
fi
