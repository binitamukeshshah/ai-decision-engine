#!/usr/bin/env bash
set -euo pipefail

usage() {
    printf 'Usage: %s [--dry-run] [user@]host\n' "${0##*/}"
    printf '       DEPLOY_HOST=[user@]host %s [--dry-run]\n' "${0##*/}"
}

dry_run=false
target_host=${DEPLOY_HOST:-}

while (($#)); do
    case $1 in
        --dry-run)
            dry_run=true
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --*)
            printf 'Unknown option: %s\n' "$1" >&2
            usage >&2
            exit 2
            ;;
        *)
            if [[ -n $target_host ]]; then
                printf 'Supply the target host only once.\n' >&2
                exit 2
            fi
            target_host=$1
            ;;
    esac
    shift
done

if [[ -z $target_host ]]; then
    printf 'A target host is required as an argument or DEPLOY_HOST.\n' >&2
    usage >&2
    exit 2
fi

if [[ ! $target_host =~ ^([A-Za-z0-9._-]+@)?[A-Za-z0-9.-]+$ ]]; then
    printf 'Target host contains unsupported characters.\n' >&2
    exit 2
fi

remote_path=${DEPLOY_REMOTE_PATH:-'~/services/ai-decision-engine'}
if [[ ! $remote_path =~ ^[A-Za-z0-9_./~-]+$ ]]; then
    printf 'Remote path contains unsupported characters.\n' >&2
    exit 2
fi

repo_root=$(git rev-parse --show-toplevel)
rsync_args=(
    --archive
    --compress
    --human-readable
    --itemize-changes
    --exclude=.git/
    --exclude=.venv/
    --exclude=.env
    --include=.env.example
    --exclude='.env.*'
    --exclude=.data/
    --exclude=__pycache__/
    --exclude='*.py[cod]'
    --exclude=.pytest_cache/
    --exclude=.ruff_cache/
    --exclude=.mypy_cache/
    --exclude=.coverage
    --exclude=htmlcov/
    --exclude=build/
    --exclude=dist/
    --exclude='*.egg-info/'
    --exclude='*.pem'
    --exclude='*.key'
    --exclude='*.p12'
    --exclude='*.pfx'
    --exclude='*credentials*'
    --exclude='*secret*'
)

if [[ $dry_run == true ]]; then
    rsync_args+=(--dry-run)
fi

printf 'Deploying tracked workspace content to %s:%s%s\n' \
    "$target_host" "$remote_path" "$([[ $dry_run == true ]] && printf ' (dry run)')"
rsync "${rsync_args[@]}" "$repo_root/" "$target_host:$remote_path/"
