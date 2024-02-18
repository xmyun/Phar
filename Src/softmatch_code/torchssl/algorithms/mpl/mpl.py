
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from utils import get_optimizer, get_cosine_schedule_with_warmup
from copy import deepcopy
from torchssl.algorithms.argument import SSL_Argument, str2bool
from torchssl.algorithms.algorithmbase import AlgorithmBase
from torchssl.algorithms.utils import ce_loss, consistency_loss, Get_Scalar, smooth_targets


class MPL(AlgorithmBase):
    def __init__(self, args, net_builder, num_classes, ema_m, lambda_u, num_eval_iter=1000, tb_log=None, logger=None):
        super().__init__(args, net_builder, num_classes, ema_m, lambda_u, num_eval_iter, tb_log, logger) 
        self.init(T=args.T, p_cutoff=args.p_cutoff, label_smoothing=args.label_smoothing, num_uda_warmup_iter=args.num_uda_warmup_iter, num_stu_wait_iter=args.num_stu_wait_iter)

        # create teacher model 
        self.teacher_model = net_builder(num_classes=num_classes)
        self.teacher_optimizer = get_optimizer(self.teacher_model, args.optim, args.teacher_lr, args.momentum, args.weight_decay)
        self.teacher_scheduler = get_cosine_schedule_with_warmup(self.teacher_optimizer, args.num_train_iter, num_warmup_steps=args.num_train_iter * 0)
        if not torch.cuda.is_available():
            raise Exception('ONLY GPU TRAINING IS SUPPORTED')
        elif args.distributed:
            if args.gpu is not None:
                self.teacher_model = self.teacher_model.cuda(args.gpu)
                self.teacher_model = nn.SyncBatchNorm.convert_sync_batchnorm(self.teacher_model)
                self.teacher_model = torch.nn.parallel.DistributedDataParallel(self.teacher_model,
                                                                            device_ids=[args.gpu],
                                                                            broadcast_buffers=False,
                                                                            find_unused_parameters=True)
            else:
                # if arg.gpu is None, DDP will divide and allocate batch_size
                # to all available GPUs if device_ids are not set.
                self.teacher_model = self.teacher_model.cuda()
                self.teacher_model = torch.nn.parallel.DistributedDataParallel(self.teacher_model)
        elif args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            self.teacher_model = self.teacher_model.cuda(args.gpu)

        else:
            self.teacher_model = torch.nn.DataParallel(self.teacher_model).cuda()
        self.teacher_scaler = GradScaler()

    def init(self, T, p_cutoff, label_smoothing, num_uda_warmup_iter, num_stu_wait_iter):
        self.t_fn = Get_Scalar(T)  # temperature params function
        self.p_fn = Get_Scalar(p_cutoff)  # confidence cutoff function
        # self.use_hard_label = hard_label
        self.label_smoothing = label_smoothing
        self.num_uda_warmup_iter = num_uda_warmup_iter
        self.num_stu_wait_iter = num_stu_wait_iter
        self.moving_dot_product = torch.zeros(1).cuda(self.args.gpu)


    def train_step(self, idx_lb, x_lb, y_lb, idx_ulb, x_ulb_w, x_ulb_s):
        num_lb = x_lb.shape[0]
        num_ulb = x_ulb_w.shape[0]
        assert num_ulb == x_ulb_s.shape[0]

        inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            logits = self.teacher_model(inputs)
            logits_x_lb = logits[:num_lb]
            logits_x_ulb_w, logits_x_ulb_s = logits[num_lb:].chunk(2)

            # hyper-params for update
            T = self.t_fn(self.it)  # temperature
            p_cutoff = self.args.p_cutoff  # threshold
            tsa = self.TSA(self.args.TSA_schedule, self.it, self.args.num_train_iter, self.args.num_classes)  # Training Signal Annealing
            sup_mask = torch.max(torch.softmax(logits_x_lb, dim=-1), dim=-1)[0].le(tsa).float().detach()
            if self.label_smoothing:
                targets_x_lb = smooth_targets(logits_x_lb, y_lb, self.label_smoothing)
                use_hard_labels = False
            else:
                targets_x_lb = y_lb
                use_hard_labels = True
            sup_loss = (ce_loss(logits_x_lb, targets_x_lb, use_hard_labels, reduction='none') * sup_mask).mean()

            # compute mask
            max_probs = torch.max(torch.softmax(logits_x_ulb_w.detach(), dim=-1), dim=-1)[0]
            mask = max_probs.ge(p_cutoff).to(max_probs.dtype)

            unsup_loss, _ = consistency_loss(logits_x_ulb_s,
                                             logits_x_ulb_w,
                                             'ce',
                                             # TODO: check this 
                                             use_hard_labels=False,
                                             T=T,
                                             mask=mask,
                                             label_smoothing=self.label_smoothing)

            # 1st call to student
            inputs = torch.cat([x_lb, x_ulb_s], dim=0)
            logits = self.model(inputs)
            s_logits_x_lb_old = logits[:num_lb]
            s_logits_x_ulb_s = logits[num_lb:]

            s_max_probs = torch.max(torch.softmax(logits_x_ulb_s.detach(), dim=-1), dim=-1)[0]
            s_mask = s_max_probs.ge(p_cutoff).to(s_max_probs.dtype)

            # update student on unlabeled data
            s_unsup_loss, _ = consistency_loss(s_logits_x_ulb_s,
                                               logits_x_ulb_s,
                                               'ce', 
                                               # TODO: check this
                                               use_hard_labels=False,
                                               T=T, 
                                               mask=s_mask,
                                               label_smoothing=self.label_smoothing)
            

        # update student's parameters
        if self.args.amp:
            self.scaler.scale(s_unsup_loss).backward()
            if (self.args.clip > 0):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            s_unsup_loss.backward()
            if (self.args.clip > 0):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
            self.optimizer.step()


        # 2nd call to student
        with self.amp_cm():
            s_logits_x_lb_new = self.model(x_lb)

            # compute teacher's feedback coefficient
            s_sup_loss_old = F.cross_entropy(s_logits_x_lb_old.detach(), y_lb)
            s_sup_loss_new = F.cross_entropy(s_logits_x_lb_new.detach(), y_lb)
            dot_product = s_sup_loss_old - s_sup_loss_new
            self.moving_dot_product = self.moving_dot_product * 0.99 + dot_product * 0.01
            dot_product = dot_product - self.moving_dot_product
            dot_product = dot_product.detach()

            # compute mpl loss
            _, hard_pseudo_label = torch.max(logits_x_ulb_s.detach(), dim=-1)
            mpl_loss = dot_product * ce_loss(logits_x_ulb_s, hard_pseudo_label).mean()
            
            # compute total loss for update teacher
            weight_u = self.lambda_u * min(1., (self.it+1) / self.num_uda_warmup_iter)
            total_loss = sup_loss + weight_u * unsup_loss + mpl_loss

        # update teacher's parameters
        if self.args.amp:
            self.teacher_scaler.scale(total_loss).backward()
            if (self.args.clip > 0):
                torch.nn.utils.clip_grad_norm_(self.teacher_model.parameters(), self.args.clip)
            self.teacher_scaler.step(self.teacher_optimizer)
            self.teacher_scaler.update()
        else:
            total_loss.backward()
            if (self.args.clip > 0):
                torch.nn.utils.clip_grad_norm_(self.teacher_model.parameters(), self.args.clip)
            self.teacher_optimizer.step()


        self.scheduler.step()
        self.teacher_scheduler.step()
        self.ema.update()
        self.model.zero_grad()
        self.teacher_model.zero_grad()

        tb_dict = {}
        tb_dict['train/sup_loss'] = sup_loss.item()
        tb_dict['train/unsup_loss'] = unsup_loss.item()
        tb_dict['train/s_unsup_loss'] = s_unsup_loss.item()
        tb_dict['train/mpl_loss'] = mpl_loss.item()
        tb_dict['train/total_loss'] = total_loss.item()
        tb_dict['train/mask_ratio'] = 1.0 - mask.float().mean().item()
        return tb_dict

    def TSA(self, schedule, cur_iter, total_iter, num_classes):
        training_progress = cur_iter / total_iter

        if schedule == 'linear':
            threshold = training_progress
        elif schedule == 'exp':
            scale = 5
            threshold = math.exp((training_progress - 1) * scale)
        elif schedule == 'log':
            scale = 5
            threshold = 1 - math.exp((-training_progress) * scale)
        elif schedule == 'none':
            return 1
        tsa = threshold * (1 - 1 / num_classes) + 1 / num_classes
        return tsa

    def save_model(self, save_name, save_path):
        save_filename = os.path.join(save_path, save_name)
        # copy EMA parameters to ema_model for saving with model as temp
        self.model.eval()
        self.ema.apply_shadow()
        ema_model = self.model.state_dict()
        self.ema.restore()
        self.model.train()

        torch.save({'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'it': self.it + 1,
                    'best_it': self.best_it,
                    'best_acc': self.best_eval_acc,
                    'moving_dot_product': self.moving_dot_product.cpu(),
                    'ema_model': ema_model},
                   save_filename)

        self.print_fn(f"model saved: {save_filename}")

    def load_model(self, load_path):
        checkpoint = torch.load(load_path)

        self.model.load_state_dict(checkpoint['model'])
        self.ema_model = deepcopy(self.model)
        self.ema_model.load_state_dict(checkpoint['ema_model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.it = checkpoint['it']
        self.best_it = checkpoint['best_it']
        self.best_eval_acc = checkpoint['best_acc']
        if 'moving_dot_product' in checkpoint:
            self.moving_dot_product = checkpoint['moving_dot_product'].cuda(self.args.gpu)
        self.print_fn('model loaded')


    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--TSA_schedule', str, 'none', 'TSA mode: none, linear, log, exp'),
            SSL_Argument('--T', float, 0.5),
            SSL_Argument('--p_cutoff', float, 0.95),
            SSL_Argument('--teacher_lr', float, 0.03),
            SSL_Argument('--label_smoothing', float, 0.1),
            SSL_Argument('--num_uda_warmup_iter', int, 5000),
            SSL_Argument('--num_stu_wait_iter', int, 3000)
        ]
