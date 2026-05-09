#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

DATASET_URL="https://huggingface.co/datasets/chris10/EgoForce"
DEFAULT_REF="main"
DEFAULT_RETRIES=5
DEFAULT_RETRY_DELAY=5

usage() {
    cat <<EOF
Usage: $(basename "$0") [--data-root DIR] [--ref REF] [--force]

Download the EgoForce dataset repo from Hugging Face into:
  <data-root>/EgoForce

Options:
  --data-root DIR   Required. Destination root directory.
  --ref REF         Branch or tag to fetch. Default: ${DEFAULT_REF}
  --force           Remove an invalid existing partial checkout and retry.
  -h, --help        Show this help message.

Environment overrides:
  EGOFORCE_DATA_ROOT         Same as --data-root
  EGOFORCE_DATASET_REF       Same as --ref
  EGOFORCE_DOWNLOAD_RETRIES  Retry count for network operations. Default: ${DEFAULT_RETRIES}
  EGOFORCE_RETRY_DELAY       Base retry delay in seconds. Default: ${DEFAULT_RETRY_DELAY}
  EGOFORCE_LFS_CONCURRENCY   git-lfs concurrent transfers. Default: 3

Examples:
  $(basename "$0") --data-root /netscratch/millerdurai/Datasets
  EGOFORCE_DATA_ROOT=/netscratch/millerdurai/Datasets $(basename "$0")
EOF
}

log() {
    printf '[download_datasets] %s\n' "$*"
}

die() {
    printf '[download_datasets] ERROR: %s\n' "$*" >&2
    exit 1
}

require_command() {
    local cmd="$1"
    command -v "${cmd}" >/dev/null 2>&1 || die "Missing required command: ${cmd}"
}

retry() {
    local -i attempts="$1"
    local -i delay="$2"
    shift 2

    local -i try=1
    until "$@"; do
        local exit_code=$?
        if (( try >= attempts )); then
            return "${exit_code}"
        fi
        log "Command failed (attempt ${try}/${attempts}): $*"
        sleep $(( delay * try ))
        try=$(( try + 1 ))
    done
}

FORCE=0
DATA_ROOT="${EGOFORCE_DATA_ROOT:-}"
REF="${EGOFORCE_DATASET_REF:-${DEFAULT_REF}}"
RETRIES="${EGOFORCE_DOWNLOAD_RETRIES:-${DEFAULT_RETRIES}}"
RETRY_DELAY="${EGOFORCE_RETRY_DELAY:-${DEFAULT_RETRY_DELAY}}"
LFS_CONCURRENCY="${EGOFORCE_LFS_CONCURRENCY:-3}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-root)
            [[ $# -ge 2 ]] || die "--data-root requires a value"
            DATA_ROOT="$2"
            shift 2
            ;;
        --ref)
            [[ $# -ge 2 ]] || die "--ref requires a value"
            REF="$2"
            shift 2
            ;;
        --force)
            FORCE=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "Unknown argument: $1"
            ;;
    esac
done

if [[ -z "${DATA_ROOT}" ]]; then
    usage
    die "Missing required --data-root. Example: $(basename "$0") --data-root /netscratch/millerdurai/Datasets"
fi

mkdir -p "${DATA_ROOT}"
DATA_ROOT="$(cd "${DATA_ROOT}" && pwd -P)"
DEST_DIR="${DATA_ROOT}/EgoForce"
PARTIAL_DIR="${DEST_DIR}.partial"

require_command git
git lfs version >/dev/null 2>&1 || die "Missing required command: git lfs"

cleanup_invalid_partial_if_requested() {
    if [[ -e "${PARTIAL_DIR}" && ! -d "${PARTIAL_DIR}/.git" ]]; then
        if [[ "${FORCE}" -eq 1 ]]; then
            log "Removing invalid partial directory: ${PARTIAL_DIR}"
            rm -rf "${PARTIAL_DIR}"
        else
            die "Found invalid partial directory at ${PARTIAL_DIR}. Re-run with --force to remove it."
        fi
    fi
}

prepare_repo_checkout() {
    if [[ -d "${DEST_DIR}/.git" ]]; then
        log "Using existing dataset checkout: ${DEST_DIR}"
        printf '%s\n' "${DEST_DIR}"
        return 0
    fi

    cleanup_invalid_partial_if_requested

    if [[ -d "${PARTIAL_DIR}/.git" ]]; then
        log "Resuming partial dataset checkout: ${PARTIAL_DIR}"
        printf '%s\n' "${PARTIAL_DIR}"
        return 0
    fi

    if [[ -e "${DEST_DIR}" && ! -d "${DEST_DIR}/.git" ]]; then
        die "Destination exists but is not a git checkout: ${DEST_DIR}"
    fi

    log "Cloning dataset repository into partial checkout: ${PARTIAL_DIR}"
    retry "${RETRIES}" "${RETRY_DELAY}" \
        env GIT_LFS_SKIP_SMUDGE=1 git clone \
            --branch "${REF}" \
            --single-branch \
            "${DATASET_URL}" \
            "${PARTIAL_DIR}"
    printf '%s\n' "${PARTIAL_DIR}"
}

WORK_DIR="$(prepare_repo_checkout)"

log "Configuring git-lfs for a more stable transfer profile"
git -C "${WORK_DIR}" lfs install --local >/dev/null
git -C "${WORK_DIR}" config lfs.concurrenttransfers "${LFS_CONCURRENCY}"
git -C "${WORK_DIR}" config lfs.transfer.maxretries "${RETRIES}"
git -C "${WORK_DIR}" config lfs.transfer.maxretrydelay 30

log "Fetching git metadata for ref '${REF}'"
retry "${RETRIES}" "${RETRY_DELAY}" \
    git -C "${WORK_DIR}" fetch --prune origin "${REF}"

log "Checking out ref '${REF}'"
git -C "${WORK_DIR}" checkout -f FETCH_HEAD >/dev/null

log "Pulling git-lfs dataset files"
retry "${RETRIES}" "${RETRY_DELAY}" \
    git -C "${WORK_DIR}" lfs pull origin "${REF}"

if [[ "${WORK_DIR}" == "${PARTIAL_DIR}" ]]; then
    if [[ -e "${DEST_DIR}" ]]; then
        die "Destination already exists, refusing to overwrite: ${DEST_DIR}"
    fi
    mv "${PARTIAL_DIR}" "${DEST_DIR}"
    WORK_DIR="${DEST_DIR}"
fi

log "Dataset is available at: ${WORK_DIR}"
log "If needed, point config.DATASET.DIR to: ${DATA_ROOT}/"
