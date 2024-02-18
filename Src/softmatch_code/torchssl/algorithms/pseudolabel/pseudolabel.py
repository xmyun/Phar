

import os
import contextlib
import numpy as np
import torch
from torch.cuda.amp import autocast, GradScaler
from torchssl.algorithms.algorithmbase import AlgorithmBase
from torchssl.algorithms.utils import ce_loss, consistency_loss, Get_Scalar, EMA
from torchssl.datasets.sampler import DistributedSampler
from torchssl.algorithms.argument import SSL_Argument, str2bool


class PseudoLabel(AlgorithmBase):
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
            print(eval_dict)

        for epoch in range(self.args.epoch):

            if isinstance(self.loader_dict['train_lb'].sampler, DistributedSampler):
                self.loader_dict['train_lb'].sampler.set_epoch(epoch)
            if isinstance(self.loader_dict['train_ulb'].sampler, DistributedSampler):
                self.loader_dict['train_ulb'].sampler.set_epoch(epoch)

            for (idx_lb, x_lb, y_lb), (idx_ulb, x_ulb) in zip(self.loader_dict['train_lb'],
                                                                self.loader_dict['train_ulb']):

                # prevent the training iterations exceed args.num_train_iter
                if self.it > self.args.num_train_iter:
                    break

                end_batch.record()
                torch.cuda.synchronize()
                start_run.record()

                x_lb, x_ulb = x_lb.cuda(self.args.gpu), x_ulb.cuda(self.args.gpu)
                y_lb = y_lb.cuda(self.args.gpu)

                tb_dict = self.train_step(idx_lb, x_lb, y_lb, idx_ulb, x_ulb)

                end_run.record()
                torch.cuda.synchronize()

                # tensorboard_dict update
                tb_dict['lr'] = self.optimizer.param_groups[0]['lr']
                tb_dict['train/prefecth_time'] = start_batch.elapsed_time(end_batch) / 1000.
                tb_dict['train/run_time'] = start_run.elapsed_time(end_run) / 1000.

                # Save model for each 10K steps and best model for each 1K steps
                if self.it % 10000 == 0:
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

    def train_step(self, idx_lb, x_lb, y_lb, idx_ulb, x_ulb):
        unsup_warmup = np.clip(self.it / (self.args.unsup_warm_up * self.args.num_train_iter),  a_min=0.0, a_max=1.0)

        # inference and calculate sup/unsup losses
        with self.amp_cm():

            logits_x_lb = self.model(x_lb)
            # calculate BN only for the first batch
            self.bn_controller.freeze_bn(self.model)
            logits_x_ulb = self.model(x_ulb)
            self.bn_controller.unfreeze_bn(self.model)


            sup_loss = ce_loss(logits_x_lb, y_lb, reduction='mean')

            # compute mask
            max_probs = torch.max(torch.softmax(logits_x_ulb.detach(), dim=-1), dim=-1)[0]
            mask = max_probs.ge(self.args.p_cutoff).to(max_probs.dtype)

            unsup_loss, _ = consistency_loss(logits_x_ulb,
                                             logits_x_ulb,
                                             'ce',
                                             mask=mask)

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
            SSL_Argument('--p_cutoff', float, 0.95),
            SSL_Argument('--unsup_warm_up', float, 0.4, 'warm up ratio for unsupervised loss'),
            # SSL_Argument('--use_flex', str2bool, False),
        ]