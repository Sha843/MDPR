import json
import torch
import os
import numpy as np

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def load_semantic_prompts_from_json(json_path, class_names_ordered):
    if not os.path.exists(json_path):
        print(f"[WARN] Semantic prompts file not found: {json_path}. Returning None.")
        return None
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"[INFO] Loaded semantic prompts from {json_path}")
    return data

def load_matrix_pretrain(pt_path):
    if not os.path.exists(pt_path):
        print(f"[WARN] Pretrained matrix file not found: {pt_path}. Returning None.")
        return None
    matrix = torch.load(pt_path, map_location='cpu', weights_only=True)
    print(f"[INFO] Loaded matrix_pretrain from {pt_path} with shape {matrix.shape}")
    return matrix
def write_log(log, text):
    log.write(text + '\n')
    log.flush()
