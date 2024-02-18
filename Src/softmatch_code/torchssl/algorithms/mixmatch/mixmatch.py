
import torch
import numpy as np
import torch.nn.functional as F

from torchssl.algorithms.argument import SSL_Argument
from torchssl.algorithms.algorithmbase import AlgorithmBase
from torchssl.algorithms.utils import ce_loss, Get_Scalar


class MixMatch(AlgorithmBase):
    def __init__(self, args, net_builder, num_classes, ema_m, lambda_u, num_eval_iter=1000, tb_log=None, logger=None):
        super().__init__(args, net_builder, num_classes, ema_m, lambda_u, num_eval_iter, tb_log, logger)
        # mixmatch specificed arguments
        self.init(T=args.T)

    def init(self, T):
        self.t_fn = Get_Scalar(T)  # temperature params function

    def train_step(self, idx_lb, x_lb, y_lb, idx_ulb, x_ulb_w1, x_ulb_w2):

        num_lb = x_lb.shape[0]

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            with torch.no_grad():
                self.bn_controller.freeze_bn(self.model)
                logits_x_ulb_w1 = self.model(x_ulb_w1)
                logits_x_ulb_w2 = self.model(x_ulb_w2)
                self.bn_controller.unfreeze_bn(self.model)
                # Temperature sharpening
                T = self.t_fn(self.it)
                # avg
                avg_prob_x_ulb = (torch.softmax(logits_x_ulb_w1, dim=1) + torch.softmax(logits_x_ulb_w2, dim=1)) / 2
                avg_prob_x_ulb = (avg_prob_x_ulb / avg_prob_x_ulb.sum(dim=-1, keepdim=True))
                # sharpening
                sharpen_prob_x_ulb = avg_prob_x_ulb ** (1 / T)
                sharpen_prob_x_ulb = (sharpen_prob_x_ulb / sharpen_prob_x_ulb.sum(dim=-1, keepdim=True)).detach()

                # Pseudo Label
                input_labels = torch.cat(
                    [self.one_hot(y_lb, self.args.num_classes, self.args.gpu), sharpen_prob_x_ulb, sharpen_prob_x_ulb], dim=0)

                # Mix up
                inputs = torch.cat([x_lb, x_ulb_w1, x_ulb_w2])
                mixed_x, mixed_y, _ = self.mixup_one_target(inputs, input_labels,
                                                       self.args.gpu,
                                                       self.args.alpha,
                                                       is_bias=True)

                # Interleave labeled and unlabeled samples between batches to get correct batch norm calculation
                mixed_x = list(torch.split(mixed_x, num_lb))
                mixed_x = self.interleave(mixed_x, num_lb)

            logits = [self.model(mixed_x[0])]
            # calculate BN for only the first batch
            self.bn_controller.freeze_bn(self.model)
            for ipt in mixed_x[1:]:
                logits.append(self.model(ipt))

            # put interleaved samples back
            logits = self.interleave(logits, num_lb)
            logits_x = logits[0]
            logits_u = torch.cat(logits[1:], dim=0)
            self.bn_controller.unfreeze_bn(self.model)

            sup_loss = ce_loss(logits_x, mixed_y[:num_lb], use_hard_labels=False)
            sup_loss = sup_loss.mean()
            unsup_loss = self.consistency_loss(logits_u, mixed_y[num_lb:])

            # set ramp_up for lambda_u
            unsup_warmup = float(np.clip(self.it / (self.args.unsup_warm_up * self.args.num_train_iter), 0.0, 1.0))
            lambda_u = self.lambda_u * unsup_warmup

            total_loss = sup_loss + lambda_u * unsup_loss

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

    def one_hot(self, targets, nClass, gpu):
        logits = torch.zeros(targets.size(0), nClass).cuda(gpu)
        return logits.scatter_(1, targets.unsqueeze(1), 1)

    def mixup_one_target(self, x, y, gpu, alpha=1.0, is_bias=False):
        """Returns mixed inputs, mixed targets, and lambda
        """
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        if is_bias:
            lam = max(lam, 1 - lam)

        index = torch.randperm(x.size(0)).cuda(gpu)

        mixed_x = lam * x + (1 - lam) * x[index]
        mixed_y = lam * y + (1 - lam) * y[index]
        return mixed_x, mixed_y, lam

    def interleave_offsets(self, batch, nu):
        groups = [batch // (nu + 1)] * (nu + 1)
        for x in range(batch - sum(groups)):
            groups[-x - 1] += 1
        offsets = [0]
        for g in groups:
            offsets.append(offsets[-1] + g)
        assert offsets[-1] == batch
        return offsets

    def interleave(self, xy, batch):
        nu = len(xy) - 1
        offsets = self.interleave_offsets(batch, nu)
        xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
        for i in range(1, nu + 1):
            xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
        return [torch.cat(v, dim=0) for v in xy]

    def consistency_loss(self, logits_w, y):
        return F.mse_loss(torch.softmax(logits_w, dim=-1), y, reduction='mean')

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--alpha', float, 0.5, 'parameter for Beta distribution of Mix Up'),
            SSL_Argument('--T', float, 0.5, 'parameter for Temperature Sharpening'),
            SSL_Argument('--unsup_warm_up', float, 1 / 64, 'ramp up ratio for unsupervised loss'),
        ]
