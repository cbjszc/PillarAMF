# 🔧 Installation Guide

## ✅ Base Environment

Prepare the base conda environment with the following configuration:

- **CUDA**: 11.6  
- **Python**: 3.9  

---

## 📦 System Dependencies

Update your system and install essential packages:

```bash
apt update
apt install -y wget git ffmpeg libsm6 libxext6 vim build-essential libopenblas-dev
```

---

## 🔥 PyTorch + CUDA 11.6

Install PyTorch and torchvision with CUDA 11.6 support:

```bash
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 \
    --extra-index-url https://download.pytorch.org/whl/cu116
```

---

## 🔁 Install `torch_scatter`

Download and install the prebuilt wheel for `torch_scatter`:

```bash
wget https://data.pyg.org/whl/torch-1.13.0%2Bcu116/torch_scatter-2.1.1%2Bpt113cu116-cp39-cp39-linux_x86_64.whl
pip install torch_scatter-2.1.1+pt113cu116-cp39-cp39-linux_x86_64.whl
```

---

## 📚 Python Dependencies

Install additional required Python packages:

```bash
pip install hydra-core numba
pip install spconv-cu116
pip install opencv-python
pip install nuscenes-devkit
pip install protobuf==3.20.*
pip install --no-cache-dir tensorflow==2.6.0
pip install timm
```

---

## 🧱 MinkowskiEngine Installation

### 1. Clone the repository

```bash
git clone https://github.com/shwoo93/MinkowskiEngine.git
cd MinkowskiEngine/
```

### 2. Fix header include (if needed)

Open this file:

```bash
vi /opt/conda/lib/python3.9/site-packages/torch/include/ATen/cuda/CUDAUtils.h
```

Add the following line at the top if it’s missing:

```cpp
#include <ATen/core/Tensor.h>
```

### 3. Install OpenBLAS development headers

```bash
conda install openblas-devel -c anaconda -y
```

### 4. Build and install MinkowskiEngine

```bash
python setup.py install \
    --blas_include_dirs=${CONDA_PREFIX}/include \
    --blas=openblas
```

---

## 🧪 Final Step

After completing all installations, install the project in **editable mode**:

```bash
pip install -e .
```
