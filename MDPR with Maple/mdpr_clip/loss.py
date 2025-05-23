import torch
import torch.nn as nn
import torch.nn.functional as F

class CompensatedCrossEntropy(nn.Module):
    def __init__(self, cls_num_list, tau=1.0):
        super().__init__()
        cls_num_list = torch.tensor(cls_num_list, dtype=torch.float32) + 1
        cls_num_ratio = cls_num_list / cls_num_list.sum()
        log_cls_num = torch.log(cls_num_ratio)
        self.log_cls_num = log_cls_num
        self.tau = tau

    def forward(self, logit, target):
        log_cls_num_dev = self.log_cls_num.to(logit.device)
        logit_adjusted = logit + self.tau * log_cls_num_dev.unsqueeze(0)
        return F.cross_entropy(logit_adjusted, target)

class MDPRLoss(nn.Module):
    def __init__(self, cfg, cls_num_list):
        super().__init__()
        self.cfg = cfg
        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_cla = CompensatedCrossEntropy(cls_num_list, tau=1.0)
        self.criterion_ka = nn.KLDivLoss(reduction='batchmean')

    def forward(self, outputs, target, epoch, ka_projection_layer, matrix_prior):
        logits_base, logits_sem, pa_matrix_finetune, x_sem_norm_all_cls, prompt_sem_cls_features_all_cls, _ = outputs
        total_loss = 0.0

        loss_base_cls = self.criterion_ce(logits_base, target)
        total_loss += getattr(self.cfg, 'loss_base_cls_weight', 1.0) * loss_base_cls

        epoch_progress = epoch / getattr(self.cfg, 'epochs', 20)
        dynamic_loss_scale = min(1.0, 0.01 + epoch_progress * 0.99)
        if logits_sem is not None:
            loss_sem_cls = self.criterion_cla(logits_sem, target)
            total_loss += dynamic_loss_scale * getattr(self.cfg, 'loss_sem_cls_weight', 0.1) * loss_sem_cls
        else:
            loss_sem_cls = torch.tensor(0.0).to(logits_base.device)


        if pa_matrix_finetune is not None and getattr(self.cfg, 'loss_pa_weight', 0.0) > 0:
            matrix_prior_dev = matrix_prior.to(pa_matrix_finetune.device)
            pa_matrix_finetune_norm_rows = pa_matrix_finetune / (
                pa_matrix_finetune.norm(dim=-1, keepdim=True) + 1e-5)
            matrix_prior_norm_rows = matrix_prior_dev / (
                matrix_prior_dev.norm(dim=-1, keepdim=True) + 1e-5)
            matrix_prior_expanded_norm_rows = matrix_prior_norm_rows.unsqueeze(0).expand_as(
                pa_matrix_finetune_norm_rows)
            loss_pa = (1.0 - F.cosine_similarity(
                pa_matrix_finetune_norm_rows, matrix_prior_expanded_norm_rows, dim=-1)).mean()
            total_loss += getattr(self.cfg, 'loss_pa_weight', 0.05) * loss_pa
        else:
            loss_pa = torch.tensor(0.0).to(logits_base.device)

        if (x_sem_norm_all_cls is not None and prompt_sem_cls_features_all_cls is not None and
            getattr(self.cfg, 'loss_ka_weight', 0.0) > 0):
            x_sem_gt_for_ka = x_sem_norm_all_cls[torch.arange(target.size(0)), target]
            prompt_sem_cls_gt_for_ka = prompt_sem_cls_features_all_cls[target]
            projected_x_sem_gt = ka_projection_layer(x_sem_gt_for_ka)
            projected_prompt_sem_cls_gt = ka_projection_layer(prompt_sem_cls_gt_for_ka)
            log_p_x_sem_gt = F.log_softmax(projected_x_sem_gt / getattr(self.cfg, 'ka_temperature', 2.0), dim=-1)
            q_prompt_cls_gt = F.softmax(projected_prompt_sem_cls_gt / getattr(self.cfg, 'ka_temperature', 2.0), dim=-1)
            loss_ka = self.criterion_ka(log_p_x_sem_gt, q_prompt_cls_gt.detach())
            total_loss += dynamic_loss_scale * getattr(self.cfg, 'loss_ka_weight', 0.1) * loss_ka
        else:
            loss_ka = torch.tensor(0.0).to(logits_base.device)

        return total_loss, loss_base_cls, loss_sem_cls, loss_pa, loss_ka