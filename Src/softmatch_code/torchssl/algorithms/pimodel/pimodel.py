

import torch
import numpy as np
from torchssl.algorithms.algorithmbase import AlgorithmBase
from torchssl.algorithms.utils import ce_loss, consistency_loss, Get_Scalar
from torchssl.algorithms.argument import SSL_Argument, str2bool


class PiModel(AlgorithmBase):
    def train_step(self, idx_lb, x_lb, y_lb, idx_ulb, x_ulb_w, x_ulb_s):
        num_lb = x_lb.shape[0]
        num_ulb = x_ulb_w.shape[0]
        assert num_ulb == x_ulb_s.shape[0]

        unsup_warmup = np.clip(self.it / (self.args.unsup_warm_up * self.args.num_train_iter),  a_min=0.0, a_max=1.0)

        # inference and calculate sup/unsup losses
        with self.amp_cm():

            logits_x_lb = self.model(x_lb)
            # calculate BN only for the first batch
            self.bn_controller.freeze_bn(self.model)
            logits_x_ulb_w = self.model(x_ulb_w)
            logits_x_ulb_s = self.model(x_ulb_s)
            self.bn_controller.unfreeze_bn(self.model)


            sup_loss = ce_loss(logits_x_lb, y_lb, reduction='mean')


            unsup_loss, _ = consistency_loss(logits_x_ulb_s,
                                             logits_x_ulb_w,
                                             'mse')

            total_loss = sup_loss + self.lambda_u * unsup_loss * unsup_warmup

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
        return tb_dict

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--unsup_warm_up', float, 0.4, 'warm up ratio for unsupervised loss'),
        ]