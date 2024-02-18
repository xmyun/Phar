
import os
import contextlib
import json
import numpy as np
import torch
from copy import deepcopy
from torch.cuda.amp import autocast, GradScaler

from torchssl.algorithms.algorithmbase import AlgorithmBase
from torchssl.datasets.sampler import DistributedSampler
from torchssl.algorithms.utils import ce_loss, Get_Scalar, EMA
from torchssl.algorithms.argument import SSL_Argument, str2bool


class ReMixMatch(AlgorithmBase):
    def __init__(self, args, net_builder, num_classes, ema_m, lambda_u, num_eval_iter=1000, tb_log=None, logger=None):
        super().__init__(args, net_builder, num_classes, ema_m, lambda_u, num_eval_iter, tb_log, logger)
        # remixmatch specificed arguments
        self.init(T=args.T, w_match=args.w_match)

    def init(self, T, w_match):
        self.t_fn = Get_Scalar(T)  # temperature params function
        self.w_match = w_match  # weight of distribution matching

        # p(y) based on the labeled examples seen during training
        dist_file_name = r"./data_statistics/" + self.args.dataset + '_' + str(self.args.num_labels) + '.json'
        with open(dist_file_name, 'r') as f:
            p_target = json.loads(f.read())
            p_target = torch.tensor(p_target['distribution'])
            self.p_target = p_target.cuda(self.args.gpu)
        print('p_target:', self.p_target)
        self.p_model = None

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

            for (idx_lb, x_lb, y_lb), (idx_ulb, x_ulb_w, x_ulb_s1, x_ulb_s2, x_ulb_s1_rot, rot_v) in zip(self.loader_dict['train_lb'],
                                                                                                         self.loader_dict['train_ulb']):

                # prevent the training iterations exceed args.num_train_iter
                if self.it > self.args.num_train_iter:
                    break

                end_batch.record()
                torch.cuda.synchronize()
                start_run.record()

                num_lb = x_lb.shape[0]
                num_ulb = x_ulb_w.shape[0]
                num_rot = x_ulb_s1_rot.shape[0]
                assert num_ulb == x_ulb_s1.shape[0]

                x_lb, x_ulb_w = x_lb.cuda(self.args.gpu), x_ulb_w.cuda(self.args.gpu)
                x_ulb_s1, x_ulb_s2 = x_ulb_s1.cuda(self.args.gpu), x_ulb_s2.cuda(self.args.gpu)
                x_ulb_s1_rot = x_ulb_s1_rot.cuda(self.args.gpu)  # rot_image
                rot_v = rot_v.cuda(self.args.gpu)  # rot_label
                y_lb = y_lb.cuda(self.args.gpu)

                tb_dict = self.train_step(idx_lb, x_lb, y_lb, idx_ulb, x_ulb_w, x_ulb_s1, x_ulb_s2, x_ulb_s1_rot, rot_v)

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

    def train_step(self, idx_lb, x_lb, y_lb, idx_ulb, x_ulb_w, x_ulb_s1, x_ulb_s2, x_ulb_s1_rot, rot_v):

        num_lb = x_lb.shape[0]

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            with torch.no_grad():
                self.bn_controller.freeze_bn(self.model)
                # logits_x_lb = self.model(x_lb)[0]
                logits_x_ulb_w = self.model(x_ulb_w)[0]
                # logits_x_ulb_s1 = self.model(x_ulb_s1)[0]
                # logits_x_ulb_s2 = self.model(x_ulb_s2)[0]
                self.bn_controller.unfreeze_bn(self.model)

                # hyper-params for update
                T = self.t_fn(self.it)

                prob_x_ulb = torch.softmax(logits_x_ulb_w, dim=1)

                # p^~_(y): moving average of p(y)
                if self.p_model == None:
                    self.p_model = torch.mean(prob_x_ulb.detach(), dim=0)
                else:
                    self.p_model = self.p_model * 0.999 + torch.mean(prob_x_ulb.detach(), dim=0) * 0.001

                prob_x_ulb = prob_x_ulb * self.p_target / self.p_model
                prob_x_ulb = (prob_x_ulb / prob_x_ulb.sum(dim=-1, keepdim=True))

                sharpen_prob_x_ulb = prob_x_ulb ** (1 / T)
                sharpen_prob_x_ulb = (sharpen_prob_x_ulb / sharpen_prob_x_ulb.sum(dim=-1, keepdim=True)).detach()

                # mix up
                mixed_inputs = torch.cat((x_lb, x_ulb_s1, x_ulb_s2, x_ulb_w))
                input_labels = torch.cat(
                    [self.one_hot(y_lb, self.args.num_classes, self.args.gpu), sharpen_prob_x_ulb, sharpen_prob_x_ulb,
                     sharpen_prob_x_ulb], dim=0)

                mixed_x, mixed_y, _ = self.mixup_one_target(mixed_inputs, input_labels,
                                                       self.args.gpu,
                                                       self.args.alpha,
                                                       is_bias=True)

                # Interleave labeled and unlabeled samples between batches to get correct batch norm calculation
                mixed_x = list(torch.split(mixed_x, num_lb))
                mixed_x = self.interleave(mixed_x, num_lb)

                # inter_inputs = torch.cat([mixed_x, x_ulb_s1], dim=0)
                # inter_inputs = list(torch.split(inter_inputs, num_lb))
                # inter_inputs = self.interleave(inter_inputs, num_lb)

                # calculate BN only for the first batch
            logits = [self.model(mixed_x[0])[0]]

            self.bn_controller.freeze_bn(self.model)
            for ipt in mixed_x[1:]:
                logits.append(self.model(ipt)[0])

            u1_logits = self.model(x_ulb_s1)[0]
            logits_rot = self.model(x_ulb_s1_rot)[1]
            logits = self.interleave(logits, num_lb)
            self.bn_controller.unfreeze_bn(self.model)

            logits_x = logits[0]
            logits_u = torch.cat(logits[1:])

            # calculate rot loss with w_rot
            rot_loss = ce_loss(logits_rot, rot_v, reduction='mean')
            rot_loss = rot_loss.mean()
            # sup loss
            sup_loss = ce_loss(logits_x, mixed_y[:num_lb], use_hard_labels=False)
            sup_loss = sup_loss.mean()
            # unsup_loss
            unsup_loss = ce_loss(logits_u, mixed_y[num_lb:], use_hard_labels=False)
            unsup_loss = unsup_loss.mean()
            # loss U1
            u1_loss = ce_loss(u1_logits, sharpen_prob_x_ulb, use_hard_labels=False)
            u1_loss = u1_loss.mean()
            # ramp for w_match
            w_match = self.args.w_match * float(np.clip(self.it / (self.args.unsup_warm_up * self.args.num_train_iter), 0.0, 1.0))
            w_kl = self.args.w_kl * float(np.clip(self.it / (self.args.unsup_warm_up * self.args.num_train_iter), 0.0, 1.0))

            total_loss = sup_loss + self.args.w_rot * rot_loss + w_kl * u1_loss + w_match * unsup_loss

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
        if is_bias: lam = max(lam, 1 - lam)

        index = torch.randperm(x.size(0)).cuda(gpu)

        mixed_x = lam * x + (1 - lam) * x[index, :]
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
                    'p_model': self.p_model.cpu(),
                    'p_target': self.p_target.cpu(),
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
        if 'p_model' in checkpoint:
            self.p_model = checkpoint['p_model'].cuda(self.args.gpu)
        if 'p_target' in checkpoint:
            self.p_target = checkpoint['p_target'].cuda(self.args.gpu)
        self.print_fn('model loaded')

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--alpha', float, 0.75, 'param for Beta distribution of Mix Up'),
            SSL_Argument('--T', float, 0.5, 'Temperature Sharpening'),
            SSL_Argument('--w_kl', float, 0.5, 'weight for KL loss'),
            SSL_Argument('--w_match', float, 1.5, 'weight for distribution matching loss.'),
            SSL_Argument('--w_rot', float, 0.5, 'weight for rot loss'),
            SSL_Argument('--use_dm', str2bool, True, 'Whether to use distribution matching'),
            SSL_Argument('--use_xe', str2bool, True, 'Whether to use cross-entropy or Brier'),
            SSL_Argument('--unsup_warm_up', float, 1 / 64),
        ]
