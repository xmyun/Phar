

import os
from copy import deepcopy

import torch
from torchssl.algorithms.argument import SSL_Argument, str2bool
from torchssl.algorithms.algorithmbase import AlgorithmBase
from torchssl.algorithms.utils import ce_loss, consistency_loss, Get_Scalar, EMA
from torchssl.datasets.sampler import DistributedSampler


class Dash(AlgorithmBase):
    def __init__(self, args, net_builder, num_classes, ema_m, lambda_u, num_eval_iter=1000, tb_log=None, logger=None):
        super().__init__(args, net_builder, num_classes, ema_m, lambda_u, num_eval_iter, tb_log, logger) 
        # fixmatch specificed arguments
        self.init(T=args.T, gamma=args.gamma, C=args.C, rho_min=args.rho_min, num_wu_iter=args.num_wu_iter, num_wu_eval_iter=args.num_wu_eval_iter)
    
    def init(self, T, gamma=1.27, C=1.0001, rho_min=0.05, num_wu_iter=2048, num_wu_eval_iter=100):
        self.t_fn = Get_Scalar(T)  # temperature params function
        self.rho_init = None  # compute from warup training
        self.gamma = gamma 
        self.C = C
        self.rho_min = rho_min
        self.num_wu_iter = num_wu_iter
        self.num_wu_eval_iter = num_wu_eval_iter

        self.rho_init = None
        self.rho_update_cnt = 0
        self.use_hard_label = False
        self.rho = None
        self.warmup_stage = True

    def warmup(self):

        # determine if still in warmup stage
        if not self.warmup_stage:
            self.print_fn("warmup stage finished")
            return

        import os
        import contextlib
        from torch.cuda.amp import autocast, GradScaler

        ngpus_per_node = torch.cuda.device_count()

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

        scaler = GradScaler()
        amp_cm = autocast if self.args.amp else contextlib.nullcontext
        
        per_epoch_steps = self.args.num_train_iter // self.args.epoch
        warmup_epoch = max(1, self.num_wu_iter // per_epoch_steps)

        for epoch in range(warmup_epoch):

            if isinstance(self.loader_dict['train_lb'].sampler, DistributedSampler):
                self.loader_dict['train_lb'].sampler.set_epoch(epoch)

            for _, x_lb, y_lb in self.loader_dict['train_lb']:

                # prevent the training iterations exceed args.num_train_iter
                if self.it > self.num_wu_iter:
                    break

                end_batch.record()
                torch.cuda.synchronize()
                start_run.record()

                x_lb = x_lb.cuda(self.args.gpu)
                y_lb = y_lb.cuda(self.args.gpu)

                # inference and calculate sup/unsup losses
                with amp_cm():
                    logits_x_lb = self.model(x_lb)
                    sup_loss = ce_loss(logits_x_lb, y_lb, use_hard_labels=True, reduction='mean')

                # parameter updates
                if self.args.amp:
                    scaler.scale(sup_loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    sup_loss.backward()
                    self.optimizer.step()

                self.model.zero_grad()

                end_run.record()
                torch.cuda.synchronize()

                # tensorboard_dict update
                tb_dict = {}
                tb_dict['train/sup_loss'] = sup_loss.item()
                tb_dict['lr'] = self.optimizer.param_groups[0]['lr']
                tb_dict['train/prefecth_time'] = start_batch.elapsed_time(end_batch) / 1000.
                tb_dict['train/run_time'] = start_run.elapsed_time(end_run) / 1000.

                if self.it % self.num_wu_eval_iter == 0:
                    save_path = os.path.join(self.args.save_dir, self.args.save_name)
                    if not self.args.multiprocessing_distributed or \
                        (self.args.multiprocessing_distributed and self.args.rank % ngpus_per_node == 0):
                        self.save_model('latest_model.pth', save_path)
                    self.print_fn(f"warmup {self.it} iteration, {tb_dict}")

                del tb_dict
                start_batch.record()
                self.it += 1

        # compute rho_init
        eval_dict = self.evaluate()
        self.rho_init = eval_dict['eval/loss']
        self.rho_update_cnt = 0
        self.use_hard_label = False
        self.rho = self.rho_init
        # reset self it
        self.warmup_stage = False
        self.it = 0
        return


    def train_step(self, idx_lb, x_lb, y_lb, idx_ulb, x_ulb_w, x_ulb_s):
        num_lb = x_lb.shape[0]
        num_ulb = x_ulb_w.shape[0]
        assert num_ulb == x_ulb_s.shape[0]

        inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))

        # adjust rho every 10 epochs
        if self.it % (10 * 1024) == 0:
            self.rho = self.C * (self.gamma ** -self.rho_update_cnt) * self.rho_init
            self.rho = max(self.rho, self.rho_min)
            self.rho_update_cnt += 1
        
        # use hard labels if rho reduced 0.05
        if self.rho == self.rho_min:
            self.use_hard_label = True
        else:
            self.use_hard_label = False

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            logits = self.model(inputs)
            logits_x_lb = logits[:num_lb]
            logits_x_ulb_w, logits_x_ulb_s = logits[num_lb:].chunk(2)
            sup_loss = ce_loss(logits_x_lb, y_lb, reduction='mean')

            # hyper-params for update
            T = self.t_fn(self.it)

            # compute mask
            if self.use_hard_label:
                pseudo_label = torch.argmax(logits_x_ulb_w, dim=-1).detach()
            else:
                pseudo_label = torch.softmax(logits_x_ulb_w / T, dim=-1).detach()
            loss_w = ce_loss(logits_x_ulb_w, pseudo_label, use_hard_labels=self.use_hard_label, reduction='none').detach()
            mask = loss_w.le(self.rho).to(logits_x_ulb_s.dtype).detach()

            unsup_loss, _ = consistency_loss(logits_x_ulb_s,
                                             logits_x_ulb_w,
                                             'ce',
                                             use_hard_labels=self.use_hard_label,
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
                    'rho_init': self.rho_init,
                    'rho_update_cnt': self.rho_update_cnt,
                    'rho': self.rho,
                    'warmup_stage': self.warmup_stage,
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
        self.rho = checkpoint['rho']
        self.rho_init = checkpoint['rho_init']
        self.warmup_stage = checkpoint['warmup_stage']
        self.rho_update_cnt = checkpoint['rho_update_cnt']
        self.print_fn('model loaded')

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--T', float, 0.5),
            SSL_Argument('--gamma', float, 1.27),
            SSL_Argument('--C', float, 1.0001),
            SSL_Argument('--rho_min', float, 0.05),
            SSL_Argument('--num_wu_iter', int, 2048),
            SSL_Argument('--num_wu_eval_iter', int, 100),
        ]
