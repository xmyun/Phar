
import os
import contextlib
import numpy as np
import torch
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
from torch.autograd import Variable

from torchssl.algorithms.algorithmbase import AlgorithmBase
from torchssl.datasets.sampler import DistributedSampler
from torchssl.algorithms.utils import ce_loss, Bn_Controller, EMA
from torchssl.algorithms.argument import SSL_Argument, str2bool

class VAT(AlgorithmBase):
    def __init__(self, args, net_builder, num_classes, ema_m, lambda_u, num_eval_iter=1000, tb_log=None, logger=None):
        super().__init__(args, net_builder, num_classes, ema_m, lambda_u, num_eval_iter, tb_log, logger)


    def train_step(self, idx_lb, x_lb, y_lb, idx_ulb, x_ulb_w):
        unsup_warmup = np.clip(self.it / (self.args.unsup_warm_up * self.args.num_train_iter),
                               a_min=0.0, a_max=1.0)

        with self.amp_cm():

            logits_x_lb = self.model(x_lb)

            sup_loss = ce_loss(logits_x_lb, y_lb, reduction='mean')
            self.bn_controller.freeze_bn(self.model)
            ul_y = self.model(x_ulb_w)
            unsup_loss = self.vat_loss(self.model, x_ulb_w, ul_y, eps=self.args.vat_eps)
            loss_entmin = self.entropy_loss(ul_y)
            self.bn_controller.unfreeze_bn(self.model)
            total_loss = sup_loss + self.lambda_u * unsup_loss * unsup_warmup + self.args.entmin_weight * loss_entmin

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
        if self.args.optim == 'SGD':
            self.scheduler.step()
        self.ema.update()
        self.model.zero_grad()

        tb_dict = {}
        tb_dict['train/sup_loss'] = sup_loss.item()
        tb_dict['train/unsup_loss'] = unsup_loss.item()
        tb_dict['train/loss_entmin'] = loss_entmin.item()
        tb_dict['train/total_loss'] = total_loss.item()

        return tb_dict

    def train(self):

        ngpus_per_node = torch.cuda.device_count()

        # EMA Init
        self.model.train()
        self.ema = EMA(self.model, self.ema_m)
        self.ema.register()
        if self.args.resume == True:
            self.ema.load(self.ema_model)

        # for gpu profiling
        start_batch = torch.cuda.Event(enable_timing=True)
        end_batch = torch.cuda.Event(enable_timing=True)
        start_run = torch.cuda.Event(enable_timing=True)
        end_run = torch.cuda.Event(enable_timing=True)

        start_batch.record()

        self.scaler = GradScaler()
        self.amp_cm = autocast if self.args.amp else contextlib.nullcontext

        # eval for once to verify if the checkpoint is loaded correctly
        if self.args.resume == True:
            eval_dict = self.evaluate()
            self.print_fn(eval_dict)

        for epoch in range(self.args.epoch):

            if isinstance(self.loader_dict['train_lb'].sampler, DistributedSampler):
                self.loader_dict['train_lb'].sampler.set_epoch(epoch)
            if isinstance(self.loader_dict['train_ulb'].sampler, DistributedSampler):
                self.loader_dict['train_ulb'].sampler.set_epoch(epoch)

            # todo: change the signature of AlgorithmBase.train to incorporate different dataloaders
            for (idx_lb, x_lb, y_lb), (idx_ulb, x_ulb_w) in zip(self.loader_dict['train_lb'],
                                                                self.loader_dict['train_ulb']):

                # prevent the training iterations exceed args.num_train_iter
                if self.it > self.args.num_train_iter:
                    break

                end_batch.record()
                torch.cuda.synchronize()
                start_run.record()

                x_lb, x_ulb_w = x_lb.cuda(self.args.gpu), x_ulb_w.cuda(self.args.gpu)
                y_lb = y_lb.cuda(self.args.gpu)

                tb_dict = self.train_step(idx_lb, x_lb, y_lb, idx_ulb, x_ulb_w)

                end_run.record()
                torch.cuda.synchronize()

                # tensorboard_dict update
                tb_dict['lr'] = self.optimizer.param_groups[0]['lr']
                tb_dict['train/prefecth_time'] = start_batch.elapsed_time(end_batch) / 1000.
                tb_dict['train/run_time'] = start_run.elapsed_time(end_run) / 1000.

                # Save model for each 10K steps and best model for each 1K steps
                if self.it % self.num_eval_iter == 0:
                    save_path = os.path.join(self.args.save_dir, self.args.save_name)
                    if not self.args.multiprocessing_distributed or \
                            (self.args.multiprocessing_distributed and self.args.rank % ngpus_per_node == 0):
                        self.save_model('latest_model.pth', save_path)

                if self.it % self.num_eval_iter == 0:
                    eval_dict = self.evaluate()
                    tb_dict.update(eval_dict)

                    save_path = os.path.join(self.args.save_dir, self.args.save_name)

                    if tb_dict['eval/top-1-acc'] > self.best_eval_acc:
                        self.best_eval_acc = tb_dict['eval/top-1-acc']
                        self.best_it = self.it

                    self.print_fn(
                        f"{self.it} iteration, USE_EMA: {self.ema_m != 0}, {tb_dict}, BEST_EVAL_ACC: {self.best_eval_acc}, at {self.best_it} iters")

                    if not self.args.multiprocessing_distributed or \
                            (self.args.multiprocessing_distributed and self.args.rank % ngpus_per_node == 0):

                        if self.it == self.best_it:
                            self.save_model('model_best.pth', save_path)

                        if not self.tb_log is None:
                            self.tb_log.update(tb_dict, self.it)

                self.it += 1
                del tb_dict
                start_batch.record()
                if self.it > 0.8 * self.args.num_train_iter:
                    self.num_eval_iter = 1000

        eval_dict = self.evaluate()
        eval_dict.update({'eval/best_acc': self.best_eval_acc, 'eval/best_it': self.best_it})
        return eval_dict

    def vat_loss(self, model, ul_x, ul_y, xi=1e-6, eps=6, num_iters=1):

        # find r_adv

        d = torch.Tensor(ul_x.size()).normal_()
        for i in range(num_iters):
            d = xi * self._l2_normalize(d)
            d = Variable(d.cuda(), requires_grad=True)

            y_hat = model(ul_x + d)

            delta_kl = self.kl_div_with_logit(ul_y.detach(), y_hat)
            delta_kl.backward()

            d = d.grad.data.clone().cpu()
            model.zero_grad()

        d = self._l2_normalize(d)
        d = Variable(d.cuda())
        r_adv = eps * d
        # compute lds

        y_hat = model(ul_x + r_adv.detach())

        delta_kl = self.kl_div_with_logit(ul_y.detach(), y_hat)
        return delta_kl

    def _l2_normalize(self, d):

        d = d.numpy()
        d /= (np.sqrt(np.sum(d ** 2, axis=(1, 2, 3))).reshape((-1, 1, 1, 1)) + 1e-16)
        return torch.from_numpy(d)

    def kl_div_with_logit(self, q_logit, p_logit):

        q = F.softmax(q_logit, dim=1)
        logq = F.log_softmax(q_logit, dim=1)
        logp = F.log_softmax(p_logit, dim=1)

        qlogq = (q * logq).sum(dim=1).mean(dim=0)
        qlogp = (q * logp).sum(dim=1).mean(dim=0)

        return qlogq - qlogp

    def entropy_loss(self, ul_y):
        p = F.softmax(ul_y, dim=1)
        return -(p * F.log_softmax(ul_y, dim=1)).sum(dim=1).mean(dim=0)

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--vat_weight', float, 0.3),
            SSL_Argument('--entmin_weight', float, 0.06, 'Entropy minimization weight'),
            SSL_Argument('--vat_eps', float, 6, 'VAT perturbation size.'),
            SSL_Argument('--unsup_warm_up', float, 0.4, 'warm up ratio for unsupervised loss'),
        ]
