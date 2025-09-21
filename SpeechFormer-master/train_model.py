#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 18:16:12 CST 2021
@author: lab-chen.weidong
"""

import os
import platform
from tqdm import tqdm
import torch
import torch.optim.lr_scheduler as lr_scheduler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict
import json
import argparse

# add missing internal imports
from config import Cfg, modify_config, create_workshop  # configuration helpers
import utils  # project-wide utility package
class Engine:
    def __init__(self, cfg, local_rank: int, world_size: int):
        self.cfg = cfg
        self.local_rank = local_rank
        self.world_size = world_size
        self.ckpt_save_path = self.cfg.ckpt_save_path
        self.device = self.cfg.train.device
        self.EPOCH = self.cfg.train.EPOCH
        self.current_epoch = 0
        self.iteration = 0

        # lr finder
        if self.cfg.train.find_init_lr:
            self.cfg.train.lr = 1e-6
            self.cfg.train.step_size = 1
            self.cfg.train.gamma = 1.05
            if self.local_rank == 0:
                self.writer = SummaryWriter(self.cfg.workshop)

        # load json configs
        with open('./config/model_config.json', 'r') as f1, open(f'./config/{self.cfg.dataset.database}_feature_config.json', 'r') as f2:
            model_json_all = json.load(f1)
            feas_json = json.load(f2)
        model_json = model_json_all[self.cfg.model.type]
        data_json = feas_json[self.cfg.dataset.feature]
        data_json['meta_csv_file'] = feas_json['meta_csv_file']
        model_json['num_classes'] = feas_json['num_classes']
        model_json['input_dim'] = (data_json['feature_dim'] // model_json['num_heads']) * model_json['num_heads']
        model_json['ffn_embed_dim'] = model_json['input_dim'] // 2
        model_json['hop'] = data_json['hop']

        # model & optim
        self.model = utils.model.load_model(self.cfg.model.type, self.device, **model_json)
        self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.cfg.train.lr, momentum=0.9)
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.calculate_score = utils.toolbox.calculate_score_classification

        # data loaders
        self.data_loader_feactory = utils.dataset.DataloaderFactory(self.cfg)
        self.train_dataloader = self.data_loader_feactory.build(state='train', **data_json)
        self.val_dataloader = None
        self.test_dataloader = None
        if self.cfg.dataset.database in ['meld', 'daic_woz']:
            self.val_dataloader = self.data_loader_feactory.build(state='dev', **data_json)
            self.test_dataloader = self.data_loader_feactory.build(state='test', **data_json)
        else:
            self.test_dataloader = self.data_loader_feactory.build(state='test', **data_json)

        # scheduler
        if self.cfg.train.find_init_lr:
            self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=self.cfg.train.step_size, gamma=self.cfg.train.gamma)
        else:
            self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.EPOCH, eta_min=self.cfg.train.lr / 100)

        # loggers
        self.logger_train = utils.logger.create_logger(self.cfg.workshop, name='train') if self.local_rank == 0 else None
        self.logger_test = utils.logger.create_logger(self.cfg.workshop, name='test') if self.local_rank == 0 else None
        if self.logger_train is not None:
            self.logger_train.info(f'workshop: {self.cfg.workshop}')
            self.logger_train.info(f'seed: {self.cfg.train.seed}')
            self.logger_train.info(f'pid: {os.getpid()}')

        # meters & recorders
        dtype_i64 = torch.int64
        self.loss_meter = utils.avgmeter.AverageMeter(device='cpu')
        self.score_meter = utils.avgmeter.AverageMeter(device='cpu')
        self.predict_recoder = utils.recoder.TensorRecorder(device='cuda', dtype=dtype_i64)
        self.label_recoder = utils.recoder.TensorRecorder(device='cuda', dtype=dtype_i64)
        self.tag_recoder = utils.recoder.StrRecorder(device='cpu', dtype=str)
        # logits simply stored in list to avoid original recorder shape constraints
        self.logits_batches = []

        # metric containers
        self.train_score_1, self.train_score_2, self.train_score_3, self.train_score_4 = [], [], [], []
        self.train_score_6, self.train_score_7, self.train_score_8, self.train_loss = [], [], [], []
        self.test_score_1, self.test_score_2, self.test_score_3, self.test_score_4 = [], [], [], []
        self.test_score_6, self.test_score_7, self.test_score_8, self.test_loss = [], [], [], []
        self.val_score_1, self.val_score_2, self.val_score_3, self.val_score_4 = [], [], [], []
        self.val_score_6, self.val_score_7, self.val_score_8, self.val_loss = [], [], [], []
        self.best_val_metric = -1.0
        self.best_epoch = -1
        self.best_dev_scores = None  # (acc, recall, f1, precision, auc, sens, spec)

    def reset_meters(self):
        self.loss_meter.reset()
        self.score_meter.reset()

    def reset_recorders(self):
        self.predict_recoder.reset()
        self.label_recoder.reset()
        self.tag_recoder.reset()
        self.logits_batches = []

    def gather_distributed_tensor(self, tensor: torch.Tensor):
        if self.world_size == 1:
            return tensor
        outputs = [torch.zeros_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(outputs, tensor, async_op=False)
        return torch.cat(outputs, dim=0)

    def gather_distributed_list(self, lst):
        if self.world_size == 1:
            return lst
        output = [None for _ in range(self.world_size)]
        if hasattr(dist, 'all_gather_object'):
            dist.all_gather_object(output, lst)
        else:
            utils.distributed.all_gather_object(output, lst, self.world_size)
        merged = []
        for part in output:
            merged.extend(part)
        return merged

    @staticmethod
    def _aggregate_logits(tags: list, logits: torch.Tensor, labels: torch.Tensor, modify_tag_func):
        subj_ids = modify_tag_func(tags)  # list same length
        agg_map = {}
        for idx, sid in enumerate(subj_ids):
            if sid not in agg_map:
                agg_map[sid] = {'logits': [], 'label': labels[idx]}
            agg_map[sid]['logits'].append(logits[idx])
        new_logits, new_labels, new_tags = [], [], []
        for sid, v in agg_map.items():
            new_tags.append(sid)
            stack = torch.stack(v['logits'], dim=0)
            new_logits.append(stack.mean(dim=0))
            new_labels.append(v['label'])
        new_logits = torch.stack(new_logits, dim=0)
        new_labels = torch.stack(new_labels, dim=0)
        return new_tags, new_logits, new_labels

    def _finalize_epoch(self, mode: str):
        # collect tensors
        epoch_preds = self.gather_distributed_tensor(self.predict_recoder.data).cpu()
        epoch_labels = self.gather_distributed_tensor(self.label_recoder.data).cpu()
        epoch_tags = self.gather_distributed_list(self.tag_recoder.data)
        local_logits = torch.cat(self.logits_batches, dim=0).to(self.device) if self.logits_batches else torch.empty(0, device=self.device)
        epoch_logits = self.gather_distributed_tensor(local_logits).cpu()

        if self.local_rank != 0:
            return None

        # voting (only for test / dev when vote enabled)
        if hasattr(self.cfg.train, 'vote') and mode in ['test']:
            if self.cfg.dataset.database == 'pitt':
                modify_tag_func = utils.toolbox._majority_target_Pitt
            elif self.cfg.dataset.database == 'daic_woz':
                modify_tag_func = utils.toolbox._majority_target_DAIC_WOZ
            else:
                modify_tag_func = None
            if modify_tag_func is not None and epoch_logits.numel() > 0:
                # aggregate logits & labels then recompute preds
                epoch_tags, epoch_logits, epoch_labels = self._aggregate_logits(epoch_tags, epoch_logits, epoch_labels, modify_tag_func)
                epoch_preds = torch.argmax(epoch_logits, dim=1)

        avg_f1_type = 'weighted' if self.cfg.dataset.database == 'meld' else 'macro'
        probs = torch.softmax(epoch_logits, dim=1) if epoch_logits.numel() > 0 else None
        score_1, score_2, score_3, score_4, score_5, score_6, score_7, score_8 = self.calculate_score(epoch_preds, epoch_labels, avg_f1_type, probs=probs)

        container_map = {
            'train': (self.train_score_1, self.train_score_2, self.train_score_3, self.train_score_4, self.train_score_6, self.train_score_7, self.train_score_8, self.train_loss),
            'test': (self.test_score_1, self.test_score_2, self.test_score_3, self.test_score_4, self.test_score_6, self.test_score_7, self.test_score_8, self.test_loss),
            'val': (self.val_score_1, self.val_score_2, self.val_score_3, self.val_score_4, self.val_score_6, self.val_score_7, self.val_score_8, self.val_loss)
        }
        lists = container_map[mode]
        lists[0].append(score_1)
        lists[1].append(score_2)
        lists[2].append(score_3)
        lists[3].append(score_4)
        lists[4].append(score_6)
        lists[5].append(score_7)
        lists[6].append(score_8)
        lists[7].append(self.loss_meter.avg)

        logger = self.logger_train if mode == 'train' else self.logger_test
        if logger is not None:
            logger.info(f'{mode.capitalize()} epoch: {self.current_epoch}, accuracy: {score_1:.5f}, precision: {score_4:.5f}, recall: {score_2:.5f}, F1: {score_3:.5f}, AUC: {score_6:.5f}, sensitivity: {score_7:.5f}, specificity: {score_8:.5f}, loss: {self.loss_meter.avg:.5f}' + ('' if mode != 'test' else f'\nconfuse_matrix:\n{score_5}'))

        # best model selection (dev only by F1)
        if mode == 'val' and score_3 > self.best_val_metric:
            self.best_val_metric = score_3
            self.best_epoch = self.current_epoch
            self.best_dev_scores = (score_1, score_2, score_3, score_4, score_6, score_7, score_8)
            self.model_save(is_best=True)
        return (score_1, score_2, score_3, score_4, score_6, score_7, score_8)

    def train_epoch(self):
        self.train_dataloader.set_epoch(self.current_epoch)
        if self.local_rank == 0:
            print(f'-------- {self.cfg.workshop} --------')
        pbar = tqdm(self.train_dataloader, disable=self.local_rank != 0)
        pbar.set_description(f'TrainEpoch-{self.current_epoch}/{self.EPOCH}')
        self.reset_meters()
        self.reset_recorders()
        self.model.train()
        for batch in pbar:
            self.iteration += 1
            x = torch.stack(batch[0], dim=0).to(self.device)
            y = torch.tensor(batch[1]).to(self.device)
            vote_tag = batch[2]
            self.optimizer.zero_grad()
            out = self.model(x)
            loss = self.loss_func(out, y)
            loss.backward()
            self.optimizer.step()
            preds = torch.argmax(out, dim=1)
            self.predict_recoder.record(preds)
            self.label_recoder.record(y)
            self.tag_recoder.record(vote_tag)
            self.logits_batches.append(out.detach().cpu())
            score_basic = utils.toolbox.calculate_basic_score(preds.cpu(), y.cpu())
            self.loss_meter.update(loss.item())
            self.score_meter.update(score_basic, y.size(0))
            pbar.set_postfix({'iter': self.iteration, 'lr': self.optimizer.param_groups[0]['lr'], 'acc': f'{self.score_meter.avg:.4f}', 'loss': f'{self.loss_meter.avg:.4f}'})
            if self.cfg.train.find_init_lr:
                if loss.item() > 20:
                    raise ValueError('Loss exploding in lr finder.')
                if self.local_rank == 0:
                    self.writer.add_scalar('Step Loss', loss.item(), self.iteration)
                    self.writer.add_scalar('Total Loss', self.loss_meter.avg, self.iteration)
                    self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], self.iteration)
                self.scheduler.step()
        self._finalize_epoch('train')

    def test(self):
        assert self.test_dataloader is not None
        pbar = tqdm(self.test_dataloader, disable=self.local_rank != 0)
        pbar.set_description(f'TestEpoch-{self.current_epoch}')
        self.reset_meters()
        self.reset_recorders()
        self.model.eval()
        with torch.no_grad():
            for batch in pbar:
                x = torch.stack(batch[0], dim=0).to(self.device)
                y = torch.tensor(batch[1]).to(self.device)
                vote_tag = batch[2]
                out = self.model(x)
                loss = self.loss_func(out, y)
                preds = torch.argmax(out, dim=1)
                self.predict_recoder.record(preds)
                self.label_recoder.record(y)
                self.tag_recoder.record(vote_tag)
                self.logits_batches.append(out.detach().cpu())
                basic = utils.toolbox.calculate_basic_score(preds.cpu(), y.cpu())
                self.loss_meter.update(loss.item())
                self.score_meter.update(basic, y.size(0))
                pbar.set_postfix({'acc': f'{self.score_meter.avg:.4f}', 'loss': f'{self.loss_meter.avg:.4f}'})
        self._finalize_epoch('test')

    def validate(self):
        if self.val_dataloader is None:
            return None
        pbar = tqdm(self.val_dataloader, disable=self.local_rank != 0)
        pbar.set_description(f'DevEpoch-{self.current_epoch}')
        self.reset_meters()
        self.reset_recorders()
        self.model.eval()
        with torch.no_grad():
            for batch in pbar:
                x = torch.stack(batch[0], dim=0).to(self.device)
                y = torch.tensor(batch[1]).to(self.device)
                vote_tag = batch[2]
                out = self.model(x)
                loss = self.loss_func(out, y)
                preds = torch.argmax(out, dim=1)
                self.predict_recoder.record(preds)
                self.label_recoder.record(y)
                self.tag_recoder.record(vote_tag)
                self.logits_batches.append(out.detach().cpu())
                basic = utils.toolbox.calculate_basic_score(preds.cpu(), y.cpu())
                self.loss_meter.update(loss.item())
                self.score_meter.update(basic, y.size(0))
                pbar.set_postfix({'acc': f'{self.score_meter.avg:.4f}', 'loss': f'{self.loss_meter.avg:.4f}'})
        self._finalize_epoch('val')
        return self.best_val_metric

    def model_save(self, is_best=False):
        path = os.path.join(self.ckpt_save_path, 'best.pt') if is_best else os.path.join(self.ckpt_save_path, f'epoch{self.current_epoch}.pt')
        torch.save({'epoch': self.current_epoch, 'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()}, path)

    def run(self):
        if self.cfg.train.find_init_lr:
            while self.current_epoch < self.EPOCH:
                self.train_epoch()
                self.current_epoch += 1
            return
        if self.val_dataloader is None:
            plot_names = ['train_score_1','train_score_2','train_score_3','train_loss','test_score_1','test_score_2','test_score_3','test_loss']
            plot_titles = ['WA-train','UA-train','F1-train','Loss-train','WA-test','UA-test','F1-test','Loss-test']
            while self.current_epoch < self.EPOCH:
                self.train_epoch(); self.scheduler.step(); self.test(); self.current_epoch += 1
                if self.local_rank == 0:
                    utils.write_result.plot_process([getattr(self, n) for n in plot_names], plot_titles, self.cfg.workshop)
        else:
            plot_names = ['train_score_1','train_score_2','train_score_3','train_loss','val_score_1','val_score_2','val_score_3','val_loss']
            plot_titles = ['WA-train','UA-train','F1-train','Loss-train','WA-dev','UA-dev','F1-dev','Loss-dev']
            while self.current_epoch < self.EPOCH:
                self.train_epoch(); self.scheduler.step(); self.validate(); self.current_epoch += 1
                if self.local_rank == 0:
                    utils.write_result.plot_process([getattr(self, n) for n in plot_names], plot_titles, self.cfg.workshop)
            # load best and test
            if self.local_rank == 0 and self.best_epoch >= 0:
                best_path = os.path.join(self.ckpt_save_path, 'best.pt')
                if os.path.isfile(best_path):
                    state = torch.load(best_path, map_location=self.device)
                    self.model.load_state_dict(state['model'])
            self.current_epoch = self.best_epoch
            self.test()
            if self.local_rank == 0 and self.best_dev_scores is not None:
                dev_acc, dev_recall, dev_f1, dev_prec, dev_auc, dev_sens, dev_spec = self.best_dev_scores
                test_acc = self.test_score_1[-1] if self.test_score_1 else 0
                test_recall = self.test_score_2[-1] if self.test_score_2 else 0
                test_f1 = self.test_score_3[-1] if self.test_score_3 else 0
                test_prec = self.test_score_4[-1] if self.test_score_4 else 0
                test_auc = self.test_score_6[-1] if self.test_score_6 else 0
                test_sens = self.test_score_7[-1] if self.test_score_7 else 0
                test_spec = self.test_score_8[-1] if self.test_score_8 else 0
                out_csv = os.path.join(self.cfg.workshop, 'dev_test_result.csv')
                import csv
                with open(out_csv, 'w', newline='') as f:
                    w = csv.writer(f)
                    w.writerow(['dev_accuracy','dev_recall','dev_f1','dev_precision','dev_auc','dev_sensitivity','dev_specificity','test_accuracy','test_recall','test_f1','test_precision','test_auc','test_sensitivity','test_specificity'])
                    w.writerow([f'{dev_acc:.6f}',f'{dev_recall:.6f}',f'{dev_f1:.6f}',f'{dev_prec:.6f}',f'{dev_auc:.6f}',f'{dev_sens:.6f}',f'{dev_spec:.6f}',f'{test_acc:.6f}',f'{test_recall:.6f}',f'{test_f1:.6f}',f'{test_prec:.6f}',f'{test_auc:.6f}',f'{test_sens:.6f}',f'{test_spec:.6f}'])
                if self.logger_test is not None:
                    self.logger_test.info(f'Dev/Test results saved to {out_csv}')
        # close loggers (deduplicated)
        utils.logger.close_logger(self.logger_train)
        utils.logger.close_logger(self.logger_test)

def main_worker(local_rank, cfg, world_size, dist_url):
    utils.environment.set_seed(cfg.train.seed + local_rank)
    torch.cuda.set_device(local_rank)
    backend = 'nccl'
    if platform.system() == 'Windows':
        backend = 'gloo'  # Windows 不支持 nccl
    dist.init_process_group(
        backend=backend,
        init_method=dist_url,
        world_size=world_size,
        rank=local_rank,
    )
    if local_rank == 0:
        print(f'Distributed backend: {backend}')
    
    if cfg.dataset.database == 'iemocap':
        cfg.train.strategy = '5cv'
        folds = [1, 2, 3, 4, 5]
    elif cfg.dataset.database == 'meld':
        folds = [1]
    elif cfg.dataset.database == 'pitt':
        cfg.train.vote = True
        folds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    elif cfg.dataset.database == 'daic_woz':
        cfg.train.vote = True
        folds = [1]
    else:
        raise KeyError(f'Unknown database: {cfg.dataset.database}')

    for f in folds:
        cfg_clone = cfg.clone()
        cfg_clone.train.current_fold = f
        create_workshop(cfg_clone, local_rank)
        engine = Engine(cfg_clone, local_rank, world_size)
        engine.run()
        torch.cuda.empty_cache()

    if local_rank == 0:
        # Include extended metrics (AUC, sensitivity, specificity) in final summary CSV
        criterion = ['accuracy', 'precision', 'recall', 'F1', 'AUC', 'sensitivity', 'specificity']
        # evaluate controls model selection across folds; keeps user setting (likely ['F1'])
        evaluate = cfg.train.evaluate
        outfile = f'result/result_{cfg.model.type}.csv'
        utils.write_result.path_to_csv(os.path.dirname(cfg_clone.workshop), criterion, evaluate, csvfile=outfile)

def main(cfg):
    utils.environment.visible_gpus(cfg.train.device_id)
    utils.environment.set_seed(cfg.train.seed)

    free_port = utils.distributed.find_free_port()
    dist_url = f'tcp://127.0.0.1:{free_port}'   
    world_size = torch.cuda.device_count()    # num_gpus
    print(f'world_size={world_size} Using dist_url={dist_url}')

    mp.spawn(fn=main_worker, args=(cfg, world_size, dist_url), nprocs=world_size)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-mo", "--model.type", help="modify cfg.train.model.type", type=str)
    parser.add_argument("-d", "--dataset.database", help="modify cfg.dataset.database", type=str)
    parser.add_argument("-f", "--dataset.feature", help="modify cfg.dataset.feature", type=str)
    parser.add_argument("-g", "--train.device_id", help="modify cfg.train.device_id", type=str)
    parser.add_argument("-m", "--mark", help="modify cfg.mark", type=str)
    parser.add_argument("-s", "--train.seed", help="modify cfg.train.seed", type=int)
    args = parser.parse_args()

    modify_config(Cfg, args)
    main(Cfg)
    
