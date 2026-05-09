#!/usr/bin/env bash

ENV_NAME="egoforce"

PYTHON_VERSION="3.10"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"

if ! command -v conda >/dev/null 2>&1; then
    echo "conda is required but was not found on PATH." >&2
    exit 1
fi


if conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
    echo "Using existing conda environment: ${ENV_NAME}"
else
    conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y -c conda-forge --override-channels
fi


conda install -n "${ENV_NAME}" -y -c nvidia -c conda-forge \
    "cuda-nvcc=12.6" \
    "cuda-runtime=12.6" \
    "cuda-cudart-dev=12.6" \
    "cuda-toolkit=12.6" \
    "cudnn=9" \
    "ffmpeg=6.1.1" \
    "fvcore=0.1.5.post20221221" \
    "iopath=0.1.10" \
    git \
    curl \
    git-lfs \
    "yacs=0.1.8" --override-channels



python -m pip install --upgrade pip setuptools==81.0.0 wheel
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install torch_tensorrt==2.8.0+cu126 --find-links https://download.pytorch.org/whl/torch-tensorrt
python -m pip install -r "${SCRIPT_DIR}/requirements.txt"
python -m pip install mmcv==2.1.0 --no-build-isolation


python -m pip  install git+https://github.com/javrtg/AnyCalib.git --no-build-isolation 
python -m pip  install git+https://github.com/mattloper/chumpy.git --no-build-isolation 
python -m pip  install git+https://github.com/facebookresearch/pytorch3d.git --no-build-isolation 


curl --proto '=https' --tlsv1.2 -sSf \
  https://raw.githubusercontent.com/huggingface/xet-core/refs/heads/main/git_xet/install.sh \
| sed "s|INSTALL_DIR=\"/usr/local/bin\"|INSTALL_DIR=\"$CONDA_PREFIX/bin\"|" \
| sh


python -m pip install "$REPO_ROOT/thirdparty/datapipes" 
rm -rf "$REPO_ROOT/thirdparty/datapipes/build" "$REPO_ROOT/thirdparty/datapipes"/*.egg-info

python -m pip install "$REPO_ROOT/thirdparty/mmdetection" --no-build-isolation 
rm -rf "$REPO_ROOT/thirdparty/mmdetection/build" "$REPO_ROOT/thirdparty/mmdetection"/*.egg-info

pip3 install numpy==1.26.4
python -m pip install projectaria_client_sdk==1.1.0 --no-cache-dir
