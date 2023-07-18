#!/bin/sh

# exit when any command fails
set -e

# Set color codes
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# Set default values for named arguments
ENV_NAME_DEFAULT="gevbev"
CONDA_PATH_DEFAULT="~/anaconda3/etc/profile.d/conda.sh" # you may need to modify the conda path.

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  key="$1"

  case $key in
    --env_name)
      ENV_NAME="$2"
      shift
      shift
      ;;
    --conda_path)
      CONDA_PATH="$2"
      shift
      shift
      ;;
    *)
      shift
      ;;
  esac
done

# Set named arguments to default values if not provided
ENV_NAME="${ENV_NAME:-$ENV_NAME_DEFAULT}"
CONDA_PATH="${CONDA_PATH:-$CONDA_PATH_DEFAULT}"

conda init zsh
source $CONDA_PATH

ENVS=$(conda env list | awk '{print $ENV_NAME}' )

if [[ $ENVS = *"$ENV_NAME"* ]]; then
    echo -e "\e[31m[ERR] \"$1\" already exists. Pass the installation.\e[0m"
else
    echo -e "${GREEN}[INFO] Create conda environment: $ENV_NAME ...${NC}"
    conda create -n $ENV_NAME python=3.8 -y
    conda activate $ENV_NAME
    conda install openblas-devel -c anaconda -y
    sudo apt install build-essential python3-dev libopenblas-dev -y
    pip install --upgrade pip

    echo -e "${GREEN}[INFO] Installing pytorch essentials...${NC}"
    pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

    echo - e "${GREEN}[INFO] Installing MinkowskiEngine...${NC}"
    # for old version of pip
    #pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps \
    #    --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" \
    #    --install-option="--blas=openblas"
    pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps \
        --global-option="--blas_include_dirs=${CONDA_PREFIX}/include" \
        --global-option="--blas=openblas"

    echo -e "${GREEN}[INFO] Installing cuda_ops...${NC}"
    cd ops && pip install . && cd ..

    echo "[INFO] Installing requirements...${NC}"
    pip install -r requirements.txt

    echo -e "${GREEN}[INFO] Done.${NC}"

    TORCH="$(python -c "import torch; print(torch.__version__)")"
    export OMP_NUM_THREADS=16
    ME="$(python  -W ignore -c "import MinkowskiEngine as ME; print(ME.__version__)")"

    echo -e "${GREEN}[INFO] Finished the installation!"
    echo "[INFO] ========== Configurations =========="
    echo "[INFO] PyTorch version: $TORCH"
    echo "[INFO] MinkowskiEngine version: $ME"
    echo -e "[INFO] ====================================${NC}"

fi;