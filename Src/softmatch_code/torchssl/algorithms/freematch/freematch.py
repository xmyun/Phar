

import os
from copy import deepcopy
import torch
from torchssl.algorithms.argument import SSL_Argument, str2bool
from torchssl.algorithms.algorithmbase import AlgorithmBase
from torchssl.algorithms.utils import ce_loss, consistency_loss, Get_Scalar


def replace_inf_to_zero(val):
    val[val == float('inf')] = 0.0
    return val

def entropy_loss(mask, logits_s, logits_w, prob_model, label_hist):
    if not mask.sum() > 0:
        return 0.0, 0.0
        
    # select samples
    logits_s = logits_s[mask]

    prob_s = logits_s.softmax(dim=-1)
    _, pred_label_s = torch.max(prob_s, dim=-1)

    hist_s = torch.bincount(pred_label_s, minlength=logits_s.shape[1]).to(logits_w.dtype)
    hist_s = hist_s / hist_s.sum()

    # modulate prob model 
    prob_model = prob_model.reshape(1, -1)
    label_hist = label_hist.reshape(1, -1)
    # prob_model_scaler = torch.nan_to_num(1 / label_hist, nan=0.0, posinf=0.0, neginf=0.0).detach()
    prob_model_scaler = replace_inf_to_zero(1 / label_hist).detach()
    mod_prob_model = prob_model * prob_model_scaler
    mod_prob_model = mod_prob_model / mod_prob_model.sum(dim=-1, keepdim=True)

    # modulate mean prob
    mean_prob_scaler_s = replace_inf_to_zero(1 / hist_s).detach()
    # mean_prob_scaler_s = torch.nan_to_num(1 / hist_s, nan=0.0, posinf=0.0, neginf=0.0).detach()
    mod_mean_prob_s = prob_s.mean(dim=0, keepdim=True) * mean_prob_scaler_s
    mod_mean_prob_s = mod_mean_prob_s / mod_mean_prob_s.sum(dim=-1, keepdim=True)

    loss = mod_prob_model * torch.log(mod_mean_prob_s + 1e-12)
    loss = loss.sum(dim=1)
    return loss.mean(), hist_s.mean()


class FreeMatch(AlgorithmBase):
    def __init__(self, args, net_builder, num_classes, ema_m, lambda_u, num_eval_iter=1000, tb_log=None, logger=None):
        super().__init__(args, net_builder, num_classes, ema_m, lambda_u, num_eval_iter, tb_log, logger) 
        # fixmatch specificed arguments
        self.init(T=args.T, hard_label=args.hard_label)
        self.lambda_e = args.ent_loss_ratio

    def init(self, T, hard_label=True):
        self.t_fn = Get_Scalar(T)  # temperature params function
        self.use_hard_label = hard_label

        # intialize p_model and max_p for freematch
        self.p_model = (torch.ones(self.args.num_classes) / self.args.num_classes).cuda(self.args.gpu)
        self.label_hist = (torch.ones(self.args.num_classes) / self.args.num_classes).cuda(self.args.gpu) 
        self.time_p = self.p_model.mean()
    

    def train_step(self, idx_lb, x_lb, y_lb, idx_ulb, x_ulb_w, x_ulb_s):
        num_lb = x_lb.shape[0]
        num_ulb = x_ulb_w.shape[0]
        assert num_ulb == x_ulb_s.shape[0]

        inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            # logits_x_lb = self.model(x_lb)
            # logits_x_ulb_s = self.model(x_ulb_s)
            # with torch.no_grad():
            #     logits_x_ulb_w = self.model(x_ulb_w)

            logits = self.model(inputs)
            logits_x_lb = logits[:num_lb]
            logits_x_ulb_w, logits_x_ulb_s = logits[num_lb:].chunk(2)

            sup_loss = ce_loss(logits_x_lb, y_lb, reduction='mean')

            # hyper-params for update
            T = self.t_fn(self.it)

            # compute mask
            probs_x_ulb_w = torch.softmax(logits_x_ulb_w.detach(), dim=-1)
            max_probs, max_idx = torch.max(probs_x_ulb_w, dim=-1)

            with torch.no_grad():
                p_cutoff = self.time_p
                p_model_cutoff = self.p_model / torch.max(self.p_model,dim=-1)[0]
                mask = max_probs.ge(p_cutoff * p_model_cutoff[max_idx])

            # compute unsupervised loss
            unsup_loss, _ = consistency_loss(logits_x_ulb_s,
                                             logits_x_ulb_w,
                                             'ce',
                                             use_hard_labels=self.use_hard_label,
                                             T=T,
                                             mask=mask.to(logits_x_ulb_w.dtype))
            
            # compute self-adaptive fairness loss
            # ent_loss, _ = entropy_loss(mask, logits_x_ulb_s, logits_x_ulb_w, self.p_model, self.label_hist)

            total_loss = sup_loss + self.lambda_u * unsup_loss  # + self.lambda_e * ent_loss

            # update 
            self.time_p = self.time_p * 0.999 +  max_probs.mean() * 0.001
            self.p_model = self.p_model * 0.999 + torch.mean(probs_x_ulb_w, dim=0) * 0.001
            # hist = torch.bincount(max_idx, minlength=self.p_model.shape[0]).to(self.p_model.dtype) 
            # self.label_hist = self.label_hist * 0.999 + (hist / hist.sum()) * 0.001

            if self.it % 250 == 0 and self.args.gpu == 0:
                self.print_fn(f"it {self.it}")
                self.print_fn(f"p_t {self.p_model}")
                mu = p_cutoff * p_model_cutoff
                self.print_fn(f"mu {mu}")
                pl_label_cnt = torch.bincount(max_idx, minlength=self.num_classes)
                w1_cnt = torch.bincount(max_idx[mask == 1.0], minlength=self.num_classes)
                self.print_fn(f"pl {pl_label_cnt}") 
                self.print_fn(f"w1 {w1_cnt}") 
                self.print_fn("\n")


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

    @torch.no_grad()
    def cal_time_p_and_p_model(self, logits_x_ulb_w):
        prob_w = torch.softmax(logits_x_ulb_w, dim=1) 
        max_probs, max_idx = torch.max(prob_w, dim=-1)
        if time_p is None:
            time_p = max_probs.mean()
        else:
            time_p = time_p * 0.999 +  max_probs.mean() * 0.001
        if p_model is None:
            p_model = torch.mean(prob_w, dim=0)
        else:
            p_model = p_model * 0.999 + torch.mean(prob_w, dim=0) * 0.001
        # if label_hist is None:
        #     label_hist = torch.bincount(max_idx, minlength=p_model.shape[0]).to(p_model.dtype) 
        #     label_hist = label_hist / label_hist.sum()
        # else:
        #     hist = torch.bincount(max_idx, minlength=p_model.shape[0]).to(p_model.dtype) 
        #     label_hist = label_hist * 0.999 + (hist / hist.sum()) * 0.001
        return time_p,p_model #,label_hist

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
                    'time_p': self.time_p.cpu(),
                    'label_hist': self.label_hist.cpu(),
                    'ema_model': ema_model},
                   save_filename)

        self.print_fn(f"model saved: {save_filename}")

    def load_model(self, load_path):
        checkpoint = torch.load(load_path, map_location='cpu')

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
        if 'label_hist' in checkpoint:
            self.label_hist = checkpoint['label_hist'].cuda(self.args.gpu)
        if 'time_p' in checkpoint:
            self.time_p = checkpoint['time_p'].cuda(self.args.gpu)
        self.print_fn('model loaded')

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--hard_label', str2bool, True),
            SSL_Argument('--T', float, 0.5),
            SSL_Argument('--ent_loss_ratio', float, 0.01),
        ]
