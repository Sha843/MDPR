import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import warnings
from .utils import AverageMeter, accuracy, write_log
from torch.amp import GradScaler, autocast as autocast_amp


class MDPRTrainer:
    def __init__(self, model, loss_fn, optimizer, scheduler, cfg, device, scaler_instance):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.cfg = cfg
        self.device = device
        self.scaler = scaler_instance
        self.use_amp_trainer = scaler_instance.is_enabled()
        if self.use_amp_trainer:
            print("[INFO MDPRTrainer] AMP is ENABLED for training.")
        else:
            print("[INFO MDPRTrainer] AMP is DISABLED for training.")

    def train_epoch(self, train_loader, epoch, log_file=None):
        losses = AverageMeter()
        top1_base_train = AverageMeter()
        top1_sem_train = AverageMeter()
        loss_base_cls_meter = AverageMeter()
        loss_sem_cls_meter = AverageMeter()
        loss_pa_meter = AverageMeter()
        loss_ka_meter = AverageMeter()

        self.model.train()
        print_freq_train = getattr(self.cfg, 'print_freq_train', 40)

        for i, (input_data, target) in enumerate(train_loader):
            input_var = input_data.to(self.device, non_blocking=True)
            target_var = target.to(self.device, non_blocking=True)

            with autocast_amp(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp_trainer):
                outputs = self.model(input_var,
                                     target_for_ka=target_var if getattr(self.cfg, 'loss_ka_weight', 0.0) > 0 else None)

                if isinstance(self.model, nn.DataParallel):
                    ka_proj_layer_train = self.model.module.ka_projection
                    matrix_prior_tensor_train = self.model.module.matrix_prior
                else:
                    ka_proj_layer_train = self.model.ka_projection
                    matrix_prior_tensor_train = self.model.matrix_prior

                total_loss, loss_base_cls, loss_sem_cls, loss_pa, loss_ka = self.loss_fn(
                    outputs, target_var, epoch, ka_proj_layer_train, matrix_prior_tensor_train
                )


            self.optimizer.zero_grad(set_to_none=True)

            if self.use_amp_trainer:
                self.scaler.scale(total_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                self.optimizer.step()

            losses.update(total_loss.item(), input_var.size(0))
            if hasattr(loss_base_cls, 'item'): loss_base_cls_meter.update(loss_base_cls.item(), input_var.size(0))
            if hasattr(loss_sem_cls, 'item'): loss_sem_cls_meter.update(loss_sem_cls.item(), input_var.size(0))
            if hasattr(loss_pa, 'item'): loss_pa_meter.update(loss_pa.item(), input_var.size(0))
            if hasattr(loss_ka, 'item'): loss_ka_meter.update(loss_ka.item(), input_var.size(0))

            logits_base_train = outputs[0]
            logits_sem_train = outputs[1]

            prec_base = accuracy(logits_base_train.float(), target_var, topk=(1,))[0]
            top1_base_train.update(prec_base.item(), input_var.size(0))

            if logits_sem_train is not None:
                prec_sem = accuracy(logits_sem_train.float(), target_var, topk=(1,))[0]
                top1_sem_train.update(prec_sem.item(), input_var.size(0))

            if i % print_freq_train == 0:
                lbase_val_log = loss_base_cls.item() if hasattr(loss_base_cls, 'item') else float(loss_base_cls)
                lsem_val_log = loss_sem_cls.item() if hasattr(loss_sem_cls, 'item') else float(loss_sem_cls)
                lpa_val_log = loss_pa.item() if hasattr(loss_pa, 'item') else float(loss_pa)
                lka_val_log = loss_ka.item() if hasattr(loss_ka, 'item') else float(loss_ka)
                msg_train = (f"Train Epoch: [{epoch}][{i}/{len(train_loader)}]\t"
                             f"Loss {losses.val:.4f} ({losses.avg:.4f})\t"
                             f"LBase {lbase_val_log:.4f}\tLSem {lsem_val_log:.4f}\tLPA {lpa_val_log:.4f}\tLKA {lka_val_log:.4f}\t"
                             f"PBase {top1_base_train.val:.3f} ({top1_base_train.avg:.3f})\t"
                             f"PSem {top1_sem_train.val:.3f} ({top1_sem_train.avg:.3f})\t"
                             f"LR {self.optimizer.param_groups[0]['lr']:.2e}")
                print(msg_train)
                if log_file: write_log(log_file, msg_train + "\n")

        summary_train = (f"TRAIN EPOCH {epoch} Summary:\tLoss {losses.avg:.4f}\t"
                         f"Loss_base_cls_avg {loss_base_cls_meter.avg:.4f}\tLoss_sem_cls_avg {loss_sem_cls_meter.avg:.4f}\t"
                         f"Loss_pa_avg {loss_pa_meter.avg:.4f}\tLoss_ka_avg {loss_ka_meter.avg:.4f}\t"
                         f"Prec@1_Base {top1_base_train.avg:.3f}\tPrec@1_Sem {top1_sem_train.avg:.3f}")
        print(summary_train)
        if log_file: write_log(log_file, summary_train + "\n")

        return losses.avg, top1_base_train.avg, top1_sem_train.avg

    @torch.no_grad()
    def evaluate(self, val_loader, epoch=None, log_file=None, mode_name="VALIDATION"):
        self.model.eval()
        top1_base_val = AverageMeter()
        top1_sem_val = AverageMeter()
        top1_fuse_val = AverageMeter()
        all_preds_list = []
        all_targets_list = []
        print_freq_val = 40
        for i, (input_data, target) in enumerate(val_loader):
            input_var = input_data.to(self.device, non_blocking=True)
            target_var = target.to(self.device, non_blocking=True)

            with autocast_amp(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp_trainer):
                outputs = self.model(input_var, target_for_ka=None)
                logits_base_raw = outputs[0]
                logits_sem_raw = outputs[1]

                probs_base = logits_base_raw.softmax(dim=-1).float()
                if logits_sem_raw is not None:
                    probs_sem = logits_sem_raw.softmax(dim=-1).float()
                    probs_fuse = ((logits_base_raw + logits_sem_raw) / 2).softmax(dim=-1).float()
                else:
                    probs_sem = probs_base
                    probs_fuse = probs_base

            current_batch_size = input_var.size(0)
            prec_base = accuracy(probs_base, target_var, topk=(1,))[0]
            top1_base_val.update(prec_base.item(), current_batch_size)

            if logits_sem_raw is not None:
                prec_sem = accuracy(probs_sem, target_var, topk=(1,))[0]
                top1_sem_val.update(prec_sem.item(), current_batch_size)

            prec_fuse = accuracy(probs_fuse, target_var, topk=(1,))[0]
            top1_fuse_val.update(prec_fuse.item(), current_batch_size)

            preds_for_cmatrix = probs_fuse.argmax(dim=1).cpu().numpy()
            all_preds_list.append(preds_for_cmatrix)
            all_targets_list.append(target_var.cpu().numpy())

            if i % print_freq_val == 0:
                msg_val_batch = (f"{mode_name} Epoch {epoch if epoch is not None else '-'}: [{i}/{len(val_loader)}]\t"
                                 f"PBase {top1_base_val.val:.2f} ({top1_base_val.avg:.2f})\t"
                                 f"PSem {top1_sem_val.val:.2f} ({top1_sem_val.avg:.2f})\t"
                                 f"PFuse {top1_fuse_val.val:.2f} ({top1_fuse_val.avg:.2f})")
                print(msg_val_batch)

        summary_parts_eval = [
            f"{mode_name} EPOCH {epoch if epoch is not None else '-'} Summary:",
            f"Prec@1_Base_Avg {top1_base_val.avg:.2f}",
        ]
        if top1_sem_val.count > 0:
            summary_parts_eval.append(f"Prec@1_Sem_Avg {top1_sem_val.avg:.2f}")
        summary_parts_eval.append(f"Prec@1_FuseLogits_Avg {top1_fuse_val.avg:.2f} (Primary Metric)")

        detailed_text_output_eval = ""
        all_preds_np_eval = np.array([])
        all_targets_np_eval = np.array([])

        if hasattr(self.cfg, 'img_num_list') and all_preds_list:
            all_preds_np_eval = np.concatenate(all_preds_list)
            all_targets_np_eval = np.concatenate(all_targets_list)
            num_classes_eval = getattr(self.cfg, 'num_classes',
                                       logits_base_raw.shape[1] if logits_base_raw is not None else 0)

            if num_classes_eval > 0:
                pred_mask_eval = (all_targets_np_eval == all_preds_np_eval).astype(np.float32)
                classwise_correct_eval = torch.zeros(num_classes_eval, device='cpu')
                classwise_num_eval = torch.zeros(num_classes_eval, device='cpu')

                for c_idx in range(num_classes_eval):
                    class_mask_indices_eval = np.where(all_targets_np_eval == c_idx)[0]
                    if len(class_mask_indices_eval) > 0:
                        classwise_correct_eval[c_idx] += torch.from_numpy(pred_mask_eval[class_mask_indices_eval]).sum()
                        classwise_num_eval[c_idx] += len(class_mask_indices_eval)

                classwise_acc_eval = (classwise_correct_eval / (classwise_num_eval + 1e-8))
                per_class_num_eval = torch.tensor(self.cfg.img_num_list)
                section_acc_eval = torch.zeros(3)

                many_pos_eval = torch.where(per_class_num_eval > 100)[0]
                med_pos_eval = torch.where((per_class_num_eval <= 100) & (per_class_num_eval >= 20))[0]
                few_pos_eval = torch.where(per_class_num_eval < 20)[0]

                section_acc_eval[0] = classwise_acc_eval[many_pos_eval].mean() if len(
                    many_pos_eval) > 0 else torch.tensor(0.0)
                section_acc_eval[1] = classwise_acc_eval[med_pos_eval].mean() if len(
                    med_pos_eval) > 0 else torch.tensor(0.0)
                section_acc_eval[2] = classwise_acc_eval[few_pos_eval].mean() if len(
                    few_pos_eval) > 0 else torch.tensor(0.0)

                summary_parts_eval.extend([
                    f"Many: {section_acc_eval[0].item() * 100:.2f}",
                    f"Med: {section_acc_eval[1].item() * 100:.2f}",
                    f"Few: {section_acc_eval[2].item() * 100:.2f}"
                ])

        detailed_text_output_eval = ", ".join(summary_parts_eval)
        print(detailed_text_output_eval)
        if log_file: write_log(log_file, detailed_text_output_eval + "\n")

        return top1_fuse_val.avg, all_preds_np_eval, all_targets_np_eval, detailed_text_output_eval