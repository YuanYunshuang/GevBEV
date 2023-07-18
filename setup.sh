#!/bin/sh

# exit when any command fails
set -e

#echo "[INFO] Running on Local: You should manually load modules..."
#conda init zsh
#source /home/yuan/anaconda3/etc/profile.d/conda.sh # you may need to modify the conda path.
#CUDA_Version=11.3
#export CUDA_HOME=/usr/local/cuda-$CUDA_Version
#export LD_LIBRARY_PATH=/usr/local/cuda-$CUDA_Version/lib64:/usr/local/cuda-$CUDA_Version/extras/CUPTI/lib64:$LD_LIBRARY_PATH
#export PATH=/usr/local/cuda-$CUDA_Version/bin:$PATH:
#export LIBRARY_PATH=/usr/local/cuda-$CUDA_Version/lib64/stubs

# Set color codes
GREEN='\033[0;32m'
NC='\033[0m' # No Color

ENVS=$(conda env list | awk '{print $1}' )

if [[ $ENVS = *"$1"* ]]; then
    echo "\e[31m[ERR] \"$1\" already exists. Pass the installation.\e[0m"
else
    echo -e "${GREEN}[INFO] Create conda environment...${NC}"
    conda create -n $1 python=3.8 -y
    conda activate $1
    conda install openblas-devel -c anaconda
    sudo apt install build-essential python3-dev libopenblas-dev
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