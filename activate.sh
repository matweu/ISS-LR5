LR5_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "$LR5_ROOT/.venv/bin/activate"

export HF_HOME="$LR5_ROOT/.cache/huggingface"
export XDG_CACHE_HOME="$LR5_ROOT/.cache"
export TOKENIZERS_PARALLELISM=false

unset LR5_ROOT
