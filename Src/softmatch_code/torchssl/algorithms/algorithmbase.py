import os
import contextlib
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from copy import deepcopy
from sklearn.metrics import accuracy_score, top_k_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from torchssl.datasets.sampler import DistributedSampler
from torchssl.algorithms.utils import Bn_Controller, EMA


class AlgorithmBase:
    def __init__(
        self,
        args,
        net_builder,
        num_classes,
        ema_m,
        lambda_u,
        num_eval_iter=1000,
        tb_log=None,
        logger=None,
        **kwargs):
        
        # common arguments
        self.args = args
        self.num_classes = num_classes
        self.ema_m = ema_m
        self.num_eval_iter = num_eval_iter
        self.lambda_u = lambda_u 
        self.tb_log = tb_log
        self.logger = logger 
        self.print_fn = print if logger is None else logger.info

        # common model related parameters
        self.it = 0
        self.best_eval_acc, self.best_it = 0.0, 0
        self.bn_controller = Bn_Controller()
        self.optimizer = None
        self.scheduler = None
        self.loader_dict = {}
        self.model = net_builder(num_classes=self.num_classes)
        self.ema_model = deepcopy(self.model)

        # other arguments specific to this algorithm
        # self.init(**kwargs)

    def init(self, **kwargs):
        raise NotImplementedError

    def set_data_loader(self, loader_dict):
        self.loader_dict = loader_dict
        self.print_fn(f'[!] data loader keys: {self.loader_dict.keys()}')

    def set_optimizer(self, optimizer, scheduler=None):
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s):
        # implement train step for each algorithm
        # compute loss
        # update model 
        # record tb_dict
        # return tb_dict
        raise NotImplementedError

    def train(self):
        # prevent the training iterations exceed args.num_train_iter
        if self.it > self.args.num_train_iter:
            return

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

            # prevent the training iterations exceed args.num_train_iter
            if self.it > self.args.num_train_iter:
                break

            if isinstance(self.loader_dict['train_lb'].sampler, DistributedSampler):
                self.loader_dict['train_lb'].sampler.set_epoch(epoch)
            if isinstance(self.loader_dict['train_ulb'].sampler, DistributedSampler):
                self.loader_dict['train_ulb'].sampler.set_epoch(epoch)

            for (idx_lb, x_lb, y_lb), (idx_ulb, x_ulb_w, x_ulb_s) in zip(self.loader_dict['train_lb'],
                                                                         self.loader_dict['train_ulb']):

                # prevent the training iterations exceed args.num_train_iter
                if self.it > self.args.num_train_iter:
                    break

                end_batch.record()
                torch.cuda.synchronize()
                start_run.record()

                num_lb = x_lb.shape[0]
                num_ulb = x_ulb_w.shape[0]
                assert num_ulb == x_ulb_s.shape[0]

                x_lb, x_ulb_w, x_ulb_s = x_lb.cuda(self.args.gpu), x_ulb_w.cuda(self.args.gpu), x_ulb_s.cuda(self.args.gpu)
                y_lb = y_lb.cuda(self.args.gpu)

                tb_dict = self.train_step(idx_lb, x_lb, y_lb, idx_ulb, x_ulb_w, x_ulb_s)

                end_run.record()
                torch.cuda.synchronize()

                # tensorboard_dict update
                tb_dict['lr'] = self.optimizer.param_groups[0]['lr']
                tb_dict['train/prefecth_time'] = start_batch.elapsed_time(end_batch) / 1000.
                tb_dict['train/run_time'] = start_run.elapsed_time(end_run) / 1000.

                # Save model 
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

    @torch.no_grad()
    def evaluate(self, eval_loader=None):
        self.model.eval()
        self.ema.apply_shadow()
        if eval_loader is None:
            eval_loader = self.loader_dict['eval']
        total_loss = 0.0
        total_num = 0.0
        y_true = []
        y_pred = []
        y_logits = []
        for _, x, y in eval_loader:
            x, y = x.cuda(self.args.gpu), y.cuda(self.args.gpu)
            num_batch = x.shape[0]
            total_num += num_batch
            if self.args.algorithm == 'remixmatch':
                logits, _ = self.model(x)
            else:
                logits = self.model(x)
            loss = F.cross_entropy(logits, y, reduction='mean')
            y_true.extend(y.cpu().tolist())
            y_pred.extend(torch.max(logits, dim=-1)[1].cpu().tolist())
            y_logits.extend(torch.softmax(logits, dim=-1).cpu().tolist())
            total_loss += loss.item() * num_batch
        top1 = accuracy_score(y_true, y_pred)
        top5 = top_k_accuracy_score(y_true, y_logits, k=5)
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        F1 = f1_score(y_true, y_pred, average='macro')
        AUC = roc_auc_score(y_true, y_logits, multi_class='ovo')

        cf_mat = confusion_matrix(y_true, y_pred, normalize='true')
        self.print_fn('confusion matrix:\n' + np.array_str(cf_mat))
        self.ema.restore()
        self.model.train()
        return {'eval/loss': total_loss / total_num, 'eval/top-1-acc': top1, 'eval/top-5-acc': top5,
                'eval/precision': precision, 'eval/recall': recall, 'eval/F1': F1, 'eval/AUC': AUC}

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
        self.print_fn('model loaded')

    @staticmethod
    def get_argument():
        raise NotImplementedError
