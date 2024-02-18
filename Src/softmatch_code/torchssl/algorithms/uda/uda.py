
import torch
import math
from torchssl.algorithms.algorithmbase import AlgorithmBase
from torchssl.algorithms.utils import ce_loss, consistency_loss, Get_Scalar
from torchssl.algorithms.argument import SSL_Argument, str2bool


class UDA(AlgorithmBase):
    def __init__(self, args, net_builder, num_classes, ema_m, lambda_u, num_eval_iter=1000, tb_log=None, logger=None):
        super().__init__(args, net_builder, num_classes, ema_m, lambda_u, num_eval_iter, tb_log, logger)
        # uda specificed arguments
        self.init(T=args.T, p_cutoff=args.p_cutoff)

    def init(self, T, p_cutoff):
        self.t_fn = Get_Scalar(T)  # temperature params function
        self.p_fn = Get_Scalar(p_cutoff)  # confidence cutoff function

    def train_step(self, idx_lb, x_lb, y_lb, idx_ulb, x_ulb_w, x_ulb_s):

        num_lb = x_lb.shape[0]

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            inputs = torch.cat([x_lb, x_ulb_w, x_ulb_s], dim=0)
            logits = self.model(inputs)
            logits_x_lb = logits[:num_lb]
            logits_x_ulb_w, logits_x_ulb_s = logits[num_lb:].chunk(2)

            # hyper-params for update
            T = self.t_fn(self.it)  # temperature
            # p_cutoff = self.p_fn(self.it)  # threshold
            p_cutoff = self.args.p_cutoff  # threshold
            tsa = self.TSA(self.args.TSA_schedule, self.it, self.args.num_train_iter,
                           self.args.num_classes)  # Training Signal Annealing
            sup_mask = torch.max(torch.softmax(logits_x_lb, dim=-1), dim=-1)[0].le(tsa).float().detach()

            sup_loss = (ce_loss(logits_x_lb, y_lb, reduction='none') * sup_mask).mean()

            # compute mask
            max_probs = torch.max(torch.softmax(logits_x_ulb_w.detach(), dim=-1), dim=-1)[0]
            mask = max_probs.ge(p_cutoff).to(max_probs.dtype)

            unsup_loss, _ = consistency_loss(logits_x_ulb_s,
                                             logits_x_ulb_w,
                                             'ce',
                                             use_hard_labels=False,
                                             T=T,
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

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--TSA_schedule', str, 'none', 'TSA mode: none, linear, log, exp'),
            SSL_Argument('--T', float, 0.4, 'Temperature sharpening'),
            SSL_Argument('--p_cutoff', float, 0.8, 'confidencial masking threshold'),
            # SSL_Argument('--use_flex', str2bool, False),
        ]