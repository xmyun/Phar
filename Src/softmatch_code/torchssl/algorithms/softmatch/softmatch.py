

import os
import math
from copy import deepcopy
import numpy as np
import torch
from torchssl.algorithms.argument import SSL_Argument, str2bool
from torchssl.algorithms.algorithmbase import AlgorithmBase
from torchssl.algorithms.utils import ce_loss, consistency_loss, Get_Scalar, concat_all_gather


class SoftMatch(AlgorithmBase):
    def __init__(self, args, net_builder, num_classes, ema_m, lambda_u, num_eval_iter=1000, tb_log=None, logger=None):
        super().__init__(args, net_builder, num_classes, ema_m, lambda_u, num_eval_iter, tb_log, logger) 
        self.init(T=args.T, hard_label=args.hard_label, dist_align=args.dist_align, ema_p=args.ema_p)
    
    def init(self, T, hard_label=True, dist_align=True, ema_p=0.999):
        self.T = T
        self.use_hard_label = hard_label
        self.dist_align = dist_align
        self.ema_p = ema_p
        

        self.lb_prob_t = torch.ones((self.args.num_classes)).cuda(self.args.gpu) / self.args.num_classes
        self.ulb_prob_t = torch.ones((self.args.num_classes)).cuda(self.args.gpu) / self.args.num_classes
        self.prob_max_mu_t = 1.0 / self.args.num_classes
        self.prob_max_var_t = 1.0

    
    @torch.no_grad()
    def update_prob_t(self, lb_probs, ulb_probs):
        if self.args.distributed and self.args.world_size > 1:
            lb_probs = concat_all_gather(lb_probs)
            ulb_probs = concat_all_gather(ulb_probs)
        
        ulb_prob_t = ulb_probs.mean(0)
        self.ulb_prob_t = self.ema_p * self.ulb_prob_t + (1 - self.ema_p) * ulb_prob_t

        lb_prob_t = lb_probs.mean(0)
        self.lb_prob_t = self.ema_p * self.lb_prob_t + (1 - self.ema_p) * lb_prob_t

        max_probs, max_idx = ulb_probs.max(dim=-1)
        prob_max_mu_t = torch.mean(max_probs)
        prob_max_var_t = torch.var(max_probs, unbiased=True)
        self.prob_max_mu_t = self.ema_p * self.prob_max_mu_t + (1 - self.ema_p) * prob_max_mu_t.item()
        self.prob_max_var_t = self.ema_p * self.prob_max_var_t + (1 - self.ema_p) * prob_max_var_t.item()


    @torch.no_grad()
    def calculate_mask(self, probs):
        max_probs, max_idx = probs.max(dim=-1)

        # compute weight
        mu = self.prob_max_mu_t
        var = self.prob_max_var_t
        mask = torch.exp(-((torch.clamp(max_probs - mu, max=0.0) ** 2) / (2 * var / 4)))

        # if self.it % 250 == 0 and self.args.gpu == 0:
        #     self.print_fn(f"it {self.it}")
        #     self.print_fn(f"mean {self.prob_t}")
        #     self.print_fn(f"mu {self.prob_max_mu_t * (self.prob_t / self.prob_t.max())}")
        #     self.print_fn(f"std {math.sqrt(self.prob_max_var_t)}")
        #     pl_label_cnt = torch.bincount(max_idx, minlength=self.num_classes)
        #     w1_cnt = torch.bincount(max_idx[mask == 1.0], minlength=self.num_classes)
        #     self.print_fn(f"pl {pl_label_cnt}") 
        #     self.print_fn(f"w1 {w1_cnt}") 
        #     self.print_fn("\n")
    

        return max_probs.detach(), mask.detach()


    @torch.no_grad()
    def distribution_alignment(self, probs):
        # da
        probs = probs * self.lb_prob_t / self.ulb_prob_t
        probs = probs / probs.sum(dim=1, keepdim=True)
        return probs.detach()


    def train_step(self, idx_lb, x_lb, y_lb, idx_ulb, x_ulb_w, x_ulb_s):
        num_lb = x_lb.shape[0]
        num_ulb = x_ulb_w.shape[0]
        assert num_ulb == x_ulb_s.shape[0]

        inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            logits = self.model(inputs)
            logits_x_lb = logits[:num_lb]
            logits_x_ulb_w, logits_x_ulb_s = logits[num_lb:].chunk(2)
            sup_loss = ce_loss(logits_x_lb, y_lb, reduction='mean')

            probs_x_lb = torch.softmax(logits_x_lb.detach(), dim=-1)
            probs_x_ulb_w = torch.softmax(logits_x_ulb_w.detach(), dim=-1)

            # update 
            self.update_prob_t(probs_x_lb, probs_x_ulb_w)

            # distribution alignment
            if self.dist_align:
                probs_x_ulb_w = self.distribution_alignment(probs_x_ulb_w)

            # calculate weight
            max_probs, mask = self.calculate_mask(probs_x_ulb_w)


            # calculate loss 
            unsup_loss, _ = consistency_loss(logits_x_ulb_s,
                                             logits_x_ulb_w,
                                             'ce',
                                             use_hard_labels=self.use_hard_label,
                                             T=self.T,
                                             mask=mask)

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
        tb_dict['train/avg_w'] = mask.mean().item()
        tb_dict['train/avg_max_prob'] = max_probs.mean().item()
        tb_dict['train/avg_prob_t'] = self.ulb_prob_t.mean().item()
        tb_dict['train/avg_mu_t'] = self.prob_max_mu_t
        tb_dict['train/avg_var_t'] = self.prob_max_var_t
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
                    'ulb_prob_t': self.ulb_prob_t.cpu(),
                    'lb_prob_t': self.lb_prob_t.cpu(),
                    'prob_max_mu_t': self.prob_max_mu_t,
                    'prob_max_var_t': self.prob_max_var_t,
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
        self.lb_prob_t = checkpoint['lb_prob_t'].cuda(self.args.gpu)
        self.ulb_prob_t = checkpoint['ulb_prob_t'].cuda(self.args.gpu)
        self.prob_max_mu_t = checkpoint['prob_max_mu_t']
        self.prob_max_var_t = checkpoint['prob_max_var_t']
        self.print_fn('model loaded')

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--hard_label', str2bool, True),
            SSL_Argument('--T', float, 0.5),
            SSL_Argument('--dist_align', str2bool, True),
            SSL_Argument('--ema_p', float, 0.999)
        ]
