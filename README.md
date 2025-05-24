# MDPR

## Environment Setup

**Requirements**: Python 3.12.3, CUDA 12.4 (for GPU), Linux (Ubuntu 20.04+ recommended), `conda` & `pip`.

**Steps**:

1. Create Conda env: `conda create -n mdpr_project python=3.12.3 && conda activate mdpr_project`
   
2. Install deps: `pip install -r requirements.txt`
   
3. Verify: `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"` (expect `2.5.1+cu124 True`)


## Training and Testing

**Run**:

1. Prepare dataset in `./data` (e.g., `data/train`, `data/val`).

2. Run: `python main_maple.py --dataset cifar100  --batch_size 128 --epochs 20 --lr 0.001`

3. Outputs: Model weights in `./checkpoints/cifar100/` (e.g., `best_best.pth.tar`), logs and test results in `./checkpoints/cifar100/` (e.g., `mdpr_clip_maple_log.txt`, `mdpr_clip_maple_best.txt`).
