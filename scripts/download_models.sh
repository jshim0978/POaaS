#!/bin/bash
# Download models required for POaaS experiments.
#
# Models used in the FEVER 2026 manuscript:
#   - meta-llama/Llama-3.2-3B-Instruct (default, Table 1-4)
#   - meta-llama/Llama-3.1-8B-Instruct (cross-model evaluation, Table 5)
#
# Prerequisites:
#   - A Hugging Face account with access to Llama models
#     (request access at https://huggingface.co/meta-llama)
#   - HUGGING_FACE_HUB_TOKEN set in environment or ~/.huggingface/token
#
# Usage:
#   bash scripts/download_models.sh          # Download both models
#   bash scripts/download_models.sh 3b       # Download 3B model only
#   bash scripts/download_models.sh 8b       # Download 8B model only

set -euo pipefail

MODEL_3B="meta-llama/Llama-3.2-3B-Instruct"
MODEL_8B="meta-llama/Llama-3.1-8B-Instruct"

# Check for huggingface_hub
if ! python -c "import huggingface_hub" 2>/dev/null; then
    echo "Installing huggingface_hub..."
    pip install huggingface_hub
fi

# Check for HF token
if [ -z "${HUGGING_FACE_HUB_TOKEN:-}" ] && [ ! -f "$HOME/.huggingface/token" ] && [ ! -f "$HOME/.cache/huggingface/token" ]; then
    echo "WARNING: No Hugging Face token found."
    echo "Set HUGGING_FACE_HUB_TOKEN or run: huggingface-cli login"
    echo "Llama models require access approval at https://huggingface.co/meta-llama"
    exit 1
fi

download_model() {
    local model_id="$1"
    echo "Downloading ${model_id}..."
    huggingface-cli download "${model_id}"
    echo "Done: ${model_id}"
}

TARGET="${1:-all}"

case "$TARGET" in
    3b)
        download_model "$MODEL_3B"
        ;;
    8b)
        download_model "$MODEL_8B"
        ;;
    all)
        download_model "$MODEL_3B"
        download_model "$MODEL_8B"
        ;;
    *)
        echo "Usage: $0 [3b|8b|all]"
        exit 1
        ;;
esac

echo ""
echo "Models downloaded to Hugging Face cache."
echo "Start vLLM with:"
echo "  python -m vllm.entrypoints.openai.api_server \\"
echo "    --model ${MODEL_3B} --host 0.0.0.0 --port 8000 --seed 13"
