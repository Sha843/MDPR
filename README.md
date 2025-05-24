# MDPR
The code of MDPR
## Environment Setup

**Requirements**: Python 3.12.3, CUDA 12.4 (for GPU), Linux (Ubuntu 20.04+ recommended), `conda` & `pip`.

**Steps**:
1. Create Conda env: `conda create -n mdpr_project python=3.12.3 && conda activate mdpr_project`
2. Install deps: `pip install -r requirements.txt` (includes `torch==2.5.1+cu124`, `torchvision==0.20.1+cu124`, `numpy==2.1.3`, `nvidia-*` for CUDA 12.4)
3. Verify: `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"` (expect `2.5.1+cu124 True`)

