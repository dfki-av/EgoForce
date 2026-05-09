SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
DATA_DIR="${REPO_ROOT}/_DATA"

echo "Downloading model weights to ${DATA_DIR}"

git clone https://huggingface.co/chris10/EgoForce

mv "${REPO_ROOT}/EgoForce/_DATA" "${REPO_ROOT}"
rm -rf "${REPO_ROOT}/EgoForce"