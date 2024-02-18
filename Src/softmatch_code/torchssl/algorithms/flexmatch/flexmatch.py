
import os
import torch
from copy import deepcopy
from collections import Counter
from torchssl.algorithms.argument import SSL_Argument, str2bool
from torchssl.algorithms.algorithmbase import AlgorithmBase
from torchssl.algorithms.utils import ce_loss, consistency_loss, Get_Scalar



class FlexMatch(AlgorithmBase):
    def __init__(self, args, net_builder, num_classes, ema_m, lambda_u, num_eval_iter=1000, tb_log=None, logger=None):
        super().__init__(args, net_builder, num_classes, ema_m, lambda_u, num_eval_iter, tb_log, logger) 
        # flexmatch specificed arguments
        self.init(T=args.T, p_cutoff=args.p_cutoff, ulb_dest_len=args.ulb_dest_len, hard_label=args.hard_label, thresh_warmup=args.thresh_warmup)
    
    def init(self, T, p_cutoff, ulb_dest_len, hard_label=True, thresh_warmup=True):
        self.t_fn = Get_Scalar(T)  # temperature params function
        self.p_fn = Get_Scalar(p_cutoff)  # confidence cutoff function
        self.use_hard_label = hard_label
        self.thresh_warmup = thresh_warmup

        # how to init this
        self.ulb_dest_len = ulb_dest_len
        selected_label = torch.ones((self.ulb_dest_len,), dtype=torch.long, ) * -1
        self.selected_label = selected_label.cuda(self.args.gpu)
        self.classwise_acc = torch.zeros((self.num_classes,)).cuda(self.args.gpu)

    def update_classwise_acc(self):
        pseudo_counter = Counter(self.selected_label.tolist())
        if max(pseudo_counter.values()) < self.ulb_dest_len:  # not all(5w) -1
            if self.thresh_warmup:
                for i in range(self.num_classes):
                    self.classwise_acc[i] = pseudo_counter[i] / max(pseudo_counter.values())
            else:
                wo_negative_one = deepcopy(pseudo_counter)
                if -1 in wo_negative_one.keys():
                    wo_negative_one.pop(-1)
                for i in range(self.num_classes):
                    self.classwise_acc[i] = pseudo_counter[i] / max(wo_negative_one.values())
    

    def train_step(self, idx_lb, x_lb, y_lb, idx_ulb, x_ulb_w, x_ulb_s):
        idx_ulb = idx_ulb.to(x_lb.device)
        num_lb = x_lb.shape[0]
        num_ulb = x_ulb_w.shape[0]
        assert num_ulb == x_ulb_s.shape[0]

        self.update_classwise_acc()

        inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            logits = self.model(inputs)
            logits_x_lb = logits[:num_lb]
            logits_x_ulb_w, logits_x_ulb_s = logits[num_lb:].chunk(2)
            sup_loss = ce_loss(logits_x_lb, y_lb, reduction='mean')

            # hyper-params for update
            T = self.t_fn(self.it)
            p_cutoff = self.p_fn(self.it)

            # compute mask
            max_probs, max_idx = torch.max(torch.softmax(logits_x_ulb_w.detach(), dim=-1), dim=-1)
            # mask = max_probs.ge(p_cutoff * (class_acc[max_idx] + 1.) / 2).float()  # linear
            # mask = max_probs.ge(p_cutoff * (1 / (2. - class_acc[max_idx]))).float()  # low_limit
            select = max_probs.ge(p_cutoff * (self.classwise_acc[max_idx] / (2. - self.classwise_acc[max_idx])))  # convex
            # mask = max_probs.ge(p_cutoff * (torch.log(class_acc[max_idx] + 1.) + 0.5)/(math.log(2) + 0.5)).float()  # concave
            mask = select.to(max_probs.dtype)

            unsup_loss, pseudo_lb = consistency_loss(logits_x_ulb_s,
                                                     logits_x_ulb_w,
                                                     'ce',
                                                     use_hard_labels=self.use_hard_label,
                                                     T=T,
                                                     mask=mask)
            
            if idx_ulb[select == 1].nelement() != 0:
                self.selected_label[idx_ulb[select == 1]] = pseudo_lb[select == 1]

            total_loss = sup_loss + self.lambda_u * unsup_loss

        # parameter updates
        if self.args.amp:
            self.scaler.scale(total_loss).backward()
            if (self.args.clip > 0):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            if (self.args.clip > 0):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
            self.optimizer.step()

        self.scheduler.step()
        self.ema.update()
        self.model.zero_grad()

        tb_dict = {}
        tb_dict['train/sup_loss'] = sup_loss.item()
        tb_dict['train/unsup_loss'] = unsup_loss.item()
        tb_dict['train/total_loss'] = total_loss.item()
        tb_dict['train/mask_ratio'] = 1.0 - mask.float().mean().item()
        return tb_dict

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
                    'selected_label': self.selected_label.cpu(),
                    'classwise_acc': self.classwise_acc.cpu(),
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
        if 'selected_label' in checkpoint:
            self.selected_label = checkpoint['selected_label'].cuda(self.args.gpu)
        if 'classwise_acc' in checkpoint:
            self.classwise_acc = checkpoint['classwise_acc'].cuda(self.args.gpu)
        self.print_fn('model loaded')

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--hard_label', str2bool, True),
            SSL_Argument('--T', float, 0.5),
            SSL_Argument('--p_cutoff', float, 0.95),
            SSL_Argument('--thresh_warmup', str2bool, True),
        ]
