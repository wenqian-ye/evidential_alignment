import torch
from .algorithm import Algorithm
from tqdm import tqdm
from utils import AverageMeter, BestMetric, Timer, time_str, log
from .register import register_algorithm
from data.data_utils import IdxDataset
from .utils import init_optimizer
import os
from torch.utils.data import DataLoader
from models.classifier import Classifier
import torch.nn.functional as F

class AlignmentModel(torch.nn.Module):
    def __init__(self, w0, b0):
        super().__init__()
        self.w0 = w0.detach().clone()
        self.b0 = b0.detach().clone()
        self.linear = torch.nn.Linear(w0.shape[1], w0.shape[0], bias=True)
        
    def forward(self, x):
        y_old = x @ self.w0.t() + self.b0
        y_new = self.linear(x)
        return y_old + y_new

def wxe_fn(logits, y, weights):
    ce = torch.nn.functional.cross_entropy(logits, y, reduction='none')
    l = weights * ce
    return l.sum()

@register_algorithm("evidential_alignment")
class EvidentialAlignment(Algorithm):
    def __init__(self, config):
        super(EvidentialAlignment, self).__init__(config)
        self._init_model()
        self._init_training()
        
    def _init_model(self):
        self.n_classes = self.datasets["train"].n_classes
        self.device = f"cuda:{self.config.gpu}"
        self.model = Classifier(self.config.backbone, self.n_classes, self.config.pretrained)
        
        if self.config.check_point:
            log(f"loading the model checkpoint from {self.config.check_point}")
            if len(self.config.erm_model) > 0:
                log(f"ignoring the ERM trained model from {self.config.erm_model}")
            saved_dict = self.load_check_point(self.config.check_point)
            self.model.load_state_dict(saved_dict["model_sd"])
        
        if len(self.config.check_point) == 0 and len(self.config.erm_model) > 0:
            log(f"loading the ERM trained model from {self.config.erm_model}")
            saved_dict = self.load_check_point(self.config.erm_model)
            # Filter out position_ids from state dict
            filtered_sd = {k: v for k, v in saved_dict["model_sd"].items() 
                          if not k.endswith("position_ids")}
            self.model.load_state_dict(filtered_sd, strict=False)

        self.model.to(self.device)
        
    def _init_training(self):
        self.optimizer_cls = None
        self.scheduler_cls = None
        
        if self.config.split_val < 1.0:
            self.sel_metrics = [("embed_val_subset2_acc", True), ("embed_val_subset2_worst_cls_acc", True), 
                              ("embed_val_subset2_worst_group_acc", True), ("embed_val_subset2_avg_cls_diff", False)]
        else:
            self.sel_metrics = [("embed_val_acc", True), ("embed_val_worst_cls_acc", True), 
                              ("embed_val_worst_group_acc", True), ("embed_val_avg_cls_diff", False)]
        self.best_meters = {m:BestMetric(max_val) for m, max_val in self.sel_metrics}

    def compute_uncertainty_weights(self, loader, balance_classes):
        self.model.eval()
        uncertainties = []
        predictions = []
        class_labels = []
        p_true_list = []

        with torch.no_grad():
            for batch in tqdm(loader, leave=False):
                idx, x, y, g, a = batch
                x, y = x.to(self.device), y.to(self.device)
                
                # Ensure y has proper dimensions
                if y.dim() == 0:
                    y = y.unsqueeze(0)  # Make it [1]
                
                output = self.model.fc(x)
                if output.dim() == 1:
                    output = output.unsqueeze(0)  # Make it [1, num_classes]
                
                evidence = F.sigmoid(output)
                alpha = evidence + 1.0
                
                S = torch.sum(alpha, dim=-1, keepdim=True)
                probs = alpha / S
                
                y_onehot = F.one_hot(y, num_classes=self.n_classes)
                p_true = (probs * y_onehot).sum(-1)
                
                total_evidence = S.squeeze(-1)
                uncertainty_weight = self.n_classes / (total_evidence + 1.0)
                
                # Ensure all appended tensors have proper dimensions
                uncertainties.append(uncertainty_weight.view(-1).cpu())
                predictions.append(torch.argmax(output, dim=-1).view(-1).cpu())
                class_labels.append(y.view(-1).cpu())  # Ensure y is 1D
                p_true_list.append(p_true.view(-1).cpu())

        # Concatenate all batches
        uncertainties = torch.cat(uncertainties)
        predictions = torch.cat(predictions)
        class_labels = torch.cat(class_labels)
        p_true = torch.cat(p_true_list)
        
        # Enhanced misclassification detection
        misclassified = (predictions != class_labels)
        
        uncertainty_threshold = torch.quantile(uncertainties, 0)
        low_uncertainty_mask = uncertainties < uncertainty_threshold
        
        # Focused upweighting: Only high-confidence errors
        high_confidence_misclass = misclassified & (p_true < 0.3)
        upweight_mask = high_confidence_misclass & low_uncertainty_mask
        
        weights = torch.ones_like(uncertainties)
        weights[upweight_mask] = uncertainties[upweight_mask]

        if balance_classes:
            class_counts = torch.bincount(class_labels).float()
            class_weights = class_counts.max() / class_counts
            for y in range(len(class_counts)):
                weights[class_labels == y] *= class_weights[y]

        weights = weights / weights.max()
        return weights

    def train(self, output_dir, split="train"):
        timer = Timer()
        criterion = wxe_fn
        initial_weight = self.model.fc.weight.detach().clone()
        initial_bias = self.model.fc.bias.detach().clone()
        embed_loaders = self.get_embed_loaders(split)
        train_loader = embed_loaders["embed_"+split]
        idxdataset = IdxDataset(train_loader.dataset)
        idx_loader = DataLoader(idxdataset, batch_size=train_loader.batch_size, shuffle=False)
        
        # EDL training phase
        log("Starting EDL training phase...")
        edl_optimizer = init_optimizer(
            self.model.fc,
            self.config.optimizer_cls,
            {"lr":0.01, "weight_decay":0.0001, "momentum":0.9}
        )
        
        for epoch in range(1, self.config.edl_epochs + 1):
            self.model.train()
            epoch_meters = {k: AverageMeter() for k in ["edl_loss", "edl_acc"]}
            
            for batch in tqdm(idx_loader, desc=f"EDL Training {split}", leave=False):
                idx, x, y, g, a = batch
                x, y = x.to(self.device), y.to(self.device)
                
                edl_optimizer.zero_grad()
                output = self.model.fc(x)
                evidence = F.sigmoid(output)
                alpha = evidence + 1.0
            
                S = torch.sum(alpha, dim=-1, keepdim=True)
                probs = alpha / S
                y_oh = F.one_hot(y, self.n_classes)
                edl_loss = (torch.pow(y_oh - probs, 2.0) + alpha * (S - alpha) / (torch.pow(S,2.0)*(S+1))).sum()
                
                # Fix dimension in KL divergence calculation
                alpha_0 = torch.ones_like(alpha)
                alpha = y_oh + (1 - y_oh) * alpha
                kl_div = torch.sum(
                    torch.lgamma(torch.sum(alpha, dim=-1)) - torch.lgamma(torch.sum(alpha_0, dim=-1)) -
                    torch.sum(torch.lgamma(alpha), dim=-1) + torch.sum(torch.lgamma(alpha_0), dim=-1) +
                    torch.sum((alpha - alpha_0) * (torch.digamma(alpha) - 
                    torch.digamma(torch.sum(alpha, dim=-1, keepdim=True))), dim=-1)
                )
                
                total_loss = edl_loss + min(epoch/self.config.annealing_step, 1) * kl_div.sum()
                # total_loss = edl_loss + self.config.kl_reg * kl_div.sum()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.fc.parameters(), 100.)
                edl_optimizer.step()
                
                # Fix dimension in prediction calculation
                pred = torch.argmax(output, dim=-1)
                acc = (pred == y).float().mean()
                
                epoch_meters["edl_loss"].update(total_loss.item(), x.size(0))
                epoch_meters["edl_acc"].update(acc.item(), x.size(0))
            
            train_state = {k: epoch_meters[k].avg for k in epoch_meters}
            msg = ', '.join([f"{k}:{v:.6f}" for k,v in train_state.items()])
            log(f"[EDL Epoch {epoch}] {msg}")

       
        weights = self.compute_uncertainty_weights(idx_loader, self.config.balance_classes)
        
        alignment_model = AlignmentModel(initial_weight, initial_bias)
        alignment_model.to(self.device)
        self.model.fc = alignment_model
        
        
        self.optimizer_cls = init_optimizer(
            self.model.fc.linear,
            self.config.optimizer_cls,
            self.config.optimizer_cls_kwargs 
        )
        
        # Alignment training phase
        for epoch in range(1, self.config.alignment_epochs+1):
            self.model.fc.train()
            epoch_meters = {k:AverageMeter() for k in ["loss", "acc"]}

            for batch in tqdm(idx_loader, desc=f"Alignment {split}", leave=False):
                idx, x, y, g, a = batch
                x, y = x.to(self.device), y.to(self.device)
                
                self.optimizer_cls.zero_grad()
                logits = self.model.fc(x)
                
                # Weighted cross entropy loss
                loss = criterion(logits, y, weights[idx].to(self.device))
                
                # Add regularization
                reg = self.model.fc.linear.weight.pow(2).sum() + self.model.fc.linear.bias.pow(2).sum()
                loss += self.config.reg_weight * reg
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.fc.linear.parameters(), 10.)
                self.optimizer_cls.step()
                
                pred = torch.argmax(logits, dim=1)
                acc = (pred == y).float().mean()
                
                epoch_meters["loss"].update(loss.item(), x.size(0))
                epoch_meters["acc"].update(acc.item(), x.size(0))

            train_state = {k:epoch_meters[k].avg for k in epoch_meters}
            
            if epoch % self.config.eval_freq == 0:
                result_dict = self.evaluate(self._get_split("val_subset2") if self.config.split_val < 1.0 else self._get_split("val"))
                result_dict_test = self.evaluate(self._get_split("test"))
                result_dict.update(result_dict_test)
                for metric, _ in self.sel_metrics:
                    if self.best_meters[metric].add(result_dict[metric]):
                        self.save(epoch, self.best_meters[metric].get(), 
                                os.path.join(output_dir, f"best_{metric}_model.pt"))
                
                msg = ', '.join([f"{k}:{v:.6f}" for k,v in train_state.items()])
                msg += ', ' + ', '.join([f"{k}:{v:.6f}" for k,v in result_dict.items()])
                elapsed_time = timer.t()
                est_all_time = elapsed_time / epoch * self.config.epoch
                log(f"[Epoch {epoch}] {msg}, lr:{self.optimizer_cls.param_groups[0]['lr']:.6f} ({time_str(elapsed_time)}/{time_str(est_all_time)})")

            if self.config.save_freq > 0 and epoch % self.config.save_freq == 0:
                self.save(epoch, self.best_meters[self.sel_metrics[0][0]].get(), 
                         os.path.join(output_dir, f"model_epoch{epoch}.pt"))

        self.save(epoch, self.best_meters[self.sel_metrics[0][0]].get(), 
                 os.path.join(output_dir, "latest_model.pt"))

    def save(self, epoch, sel_metric, file_path):
        save_dict = {}
        save_dict["model_sd"] = self.model.state_dict()
        save_dict["sel_metric"] = sel_metric
        save_dict["config"] = self.config
        save_dict["optimizer"] = self.optimizer_cls.state_dict()
        save_dict["scheduler"] = self.scheduler_cls.state_dict() if self.scheduler_cls else None
        save_dict["epoch"] = epoch
        torch.save(save_dict, file_path)

    def test(self, output_dir, split=["test"], result_path=""):
        model_info = f"evidential_alignment {self.config.dataset} {self.config.backbone} {self.config.train_split} train_ratio:{self.config.split_train:.2f} val_ratio:{self.config.split_val:.2f} class_balanced:{self.config.balance_classes} temperature:{self.config.temperature} kl_reg:{self.config.kl_reg} reg_weight:{self.config.reg_weight} seed:{self.config.seed}"
        if len(result_path) > 0:
            with open(result_path, "a") as fout:
                fout.write(model_info)
                fout.write('\n')
        model_paths = []
        for metric, _ in self.sel_metrics:
            model_path = os.path.join(output_dir, f"best_{metric}_model.pt")
            model_paths.append((model_path,metric))
        model_paths.append((os.path.join(output_dir, "latest_model.pt"),"latest"))
        for model_path, metric in model_paths:
            saved_dict = self.load_check_point(model_path)
            model_dict = saved_dict["model_sd"]
            sel_metric_val = saved_dict["sel_metric"]
            self.model.load_state_dict(model_dict)
            for sp in split:
                results = self.evaluate("embed_"+sp)
                result_str = f"[{sp} ({metric}:{sel_metric_val:.6f})]: " + ', '.join([f"{k}:{results[k]:.6f}" for k in results])
                log(result_str) 
                if len(result_path) > 0:
                    with open(result_path, "a") as fout:
                        fout.write(result_str)
                        fout.write('\n')
    