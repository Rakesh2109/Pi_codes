#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
ENV_FILE="${TM_RPI_ENV_FILE:-$REPO_ROOT/ansible/rpi.env}"
if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

REMOTE_USER="${TM_RPI_USER:-rpi}"
REMOTE_HOST="${TM_RPI_HOST:-}"
if [[ -n "$REMOTE_HOST" && "$REMOTE_HOST" == *@* ]]; then
    REMOTE="$REMOTE_HOST"
else
    REMOTE="${REMOTE_USER}@${REMOTE_HOST}"
fi
REPO_NAME="$(basename "$REPO_ROOT")"
REMOTE_DIR_REL="${REPO_NAME}/src/fuzzy_tm_infer"
REMOTE_DIR_CMD="\$HOME/${REMOTE_DIR_REL}"
LOCAL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SSH_OPTS=(-o StrictHostKeyChecking=accept-new)
SSH_PASS="${TM_RPI_PASSWORD:-${SSHPASS:-}}"

DRY_RUN=0
RUN_BUILD=0
RUN_BENCH=0
WITH_ASSETS=1

usage() {
    cat <<'USAGE'
Usage: ./scripts/rsync_rpi5.sh [options]

Sync src/fuzzy_tm_infer to the Raspberry Pi 5 over SSH/rsync.

Options:
  --host USER@HOST      SSH target, or set TM_RPI_HOST/TM_RPI_USER in ansible/rpi.env
  --no-assets           Do not sync extracted assets/
  --build               Build algorithms/c/fuzzy_tm/v17 on the Pi after sync
  --bench               Run the algorithms/c/fuzzy_tm/v17 benchmark on the Pi after sync; implies --build
  --dry-run             Show what rsync would do
  -h, --help            Show this help

Environment overrides:
  TM_RPI_HOST           Same as --host
  TM_RPI_USER           SSH user when TM_RPI_HOST does not include USER@
  TM_RPI_PASSWORD       Optional password for sshpass; not stored by this script
  TM_RPI_ENV_FILE       Optional env file path. Default: ansible/rpi.env

Examples:
  ./scripts/rsync_rpi5.sh
  ./scripts/rsync_rpi5.sh --build
  ./scripts/rsync_rpi5.sh --bench
USAGE
}

while (($#)); do
    case "$1" in
        --host)
            REMOTE="${2:?missing value for --host}"
            shift 2
            ;;
        --no-assets)
            WITH_ASSETS=0
            shift
            ;;
        --build)
            RUN_BUILD=1
            shift
            ;;
        --bench)
            RUN_BUILD=1
            RUN_BENCH=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ "$REMOTE" == *@ ]]; then
    echo "missing Raspberry Pi host; set TM_RPI_HOST in ansible/rpi.env or pass --host USER@HOST" >&2
    exit 2
fi

RSYNC_ARGS=(
    -az
    --human-readable
    --info=stats2,progress2
    --delete
    --exclude='__pycache__/'
    --exclude='*.pyc'
    --exclude='.mypy_cache/'
    --exclude='tm_infer_c'
    --exclude='libfuzzy_tm_infer.so'
    --exclude='compression_table.png'
)

if ((WITH_ASSETS == 0)); then
    RSYNC_ARGS+=(--exclude='assets/')
fi

if ((DRY_RUN)); then
    RSYNC_ARGS+=(--dry-run --itemize-changes)
fi

if [[ -n "$SSH_PASS" ]]; then
    if ! command -v sshpass >/dev/null 2>&1; then
        echo "TM_RPI_PASSWORD/SSHPASS is set, but sshpass is not installed" >&2
        exit 1
    fi
    export SSHPASS="$SSH_PASS"
    SSH_BASE=(sshpass -e ssh)
    RSYNC_SSH=(sshpass -e ssh -o StrictHostKeyChecking=accept-new)
else
    SSH_BASE=(ssh)
    RSYNC_SSH=(ssh -o StrictHostKeyChecking=accept-new)
fi

echo "==> Ensuring remote directory exists: ${REMOTE}:\$HOME/${REMOTE_DIR_REL}"
"${SSH_BASE[@]}" "${SSH_OPTS[@]}" "$REMOTE" "mkdir -p \"$REMOTE_DIR_CMD\""

echo "==> Syncing ${LOCAL_DIR}/ -> ${REMOTE}:\$HOME/${REMOTE_DIR_REL}/"
rsync -e "${RSYNC_SSH[*]}" \
    "${RSYNC_ARGS[@]}" "${LOCAL_DIR}/" "${REMOTE}:${REMOTE_DIR_CMD}/"

if ((RUN_BUILD)); then
    echo "==> Building on Raspberry Pi"
    "${SSH_BASE[@]}" "${SSH_OPTS[@]}" "$REMOTE" \
        "cd \"$REMOTE_DIR_CMD/algorithms/c/fuzzy_tm/v17\" && make clean all lib ARCHFLAGS='-mcpu=cortex-a76'"
fi

if ((RUN_BENCH)); then
    echo "==> Running benchmark on Raspberry Pi"
    "${SSH_BASE[@]}" "${SSH_OPTS[@]}" "$REMOTE" \
        "cd \"$REMOTE_DIR_CMD/algorithms/c/fuzzy_tm/v17\" && ./tm_infer_c --profile"
fi

echo "==> Done"
