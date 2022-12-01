#!/bin/sh


echo "[INFO] Running on Local: You should manually load modules..."
conda init zsh
# source /media/hdd/yuan/anaconda3/etc/profile.d/conda.sh # you may need to modify the conda path.
CUDA_Version=11.1
export CUDA_HOME=/usr/local/cuda-$CUDA_Version
export LD_LIBRARY_PATH=/usr/local/cuda-$CUDA_Version/lib64:/usr/local/cuda-$CUDA_Version/extras/CUPTI/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-$CUDA_Version/bin:$PATH:
export LIBRARY_PATH=/usr/local/cuda-$CUDA_Version/lib64/stubs


ENVS=$(conda env list | awk '{print $1}' )

if [[ $ENVS = *"$1"* ]]; then
    echo "[INFO] \"$1\" already exists. Pass the installation."
else
    echo "[INFO] Creating $1..."
    conda create -n $1 python=3.8 -y
    conda activate "$1"
    echo "[INFO] Done."

    echo "[INFO] Installing OpenBLAS and PyTorch..."
    conda install pytorch=1.10.0 torchvision cudatoolkit=11.1 setuptools=58.0.4 -c pytorch -c nvidia -y
    conda install numpy -y
    conda install openblas-devel -c anaconda -y
    echo "[INFO] Done."

    echo "[INFO] Installing other dependencies..."
    conda install -c anaconda pandas scipy h5py scikit-learn matplotlib -y
    conda install -c conda-forge plyfile torchmetrics wandb wrapt gin-config rich einops -y
    conda install -c open3d-admin -c conda-forge open3d -y
    pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+cu111.html
    echo "[INFO] Done."

    echo "[INFO] Installing MinkowskiEngine..."
    cd thirdparty/MinkowskiEngine
    python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas --force_cuda
    cd ../..
    echo "[INFO] Done."

    echo "[INFO] Installing cuda_ops..."
    cd src/ops
    pip3 install .
    cd ../..
    echo "[INFO] Done."

    TORCH="$(python -c "import torch; print(torch.__version__)")"
    ME="$(python -c "import MinkowskiEngine as ME; print(ME.__version__)")"

    echo "[INFO] Finished the installation!"
    echo "[INFO] ========== Configurations =========="
    echo "[INFO] PyTorch version: $TORCH"
    echo "[INFO] MinkowskiEngine version: $ME"
    echo "[INFO] ===================================="
fi;