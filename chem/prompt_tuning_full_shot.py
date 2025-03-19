#!/usr/bin/env python3
"""
Entry-point for full-shot prompt tuning of pre-trained GNNs.

Clean Code Refactored Version:
 - Loose coupling & High Cohesion: 각 기능별 함수로 분리
 - 단일 책임 원칙: 각 함수는 하나의 역할만 수행
 - 불필요한 주석 제거, 필요한 설명만 추가
 - Law of Demeter 준수
 - 중복 코드 제거
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch_geometric.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from loader import MoleculeDataset
from model import GNN_graphpred
from splitters import scaffold_split, random_scaffold_split, random_split
import graph_prompt as Prompt


# Loss criterion defined once
LOSS_CRITERION = nn.BCEWithLogitsLoss(reduction="none")


def train_epoch(args, model, device, loader, optimizer, prompt):
    model.train()
    for batch in tqdm(loader, desc="Training"):
        batch = batch.to(device)
        preds = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, prompt)
        targets = batch.y.view(preds.shape).to(torch.float64)

        valid_mask = targets**2 > 0
        loss_matrix = LOSS_CRITERION(preds.double(), (targets + 1) / 2)
        loss_matrix = torch.where(valid_mask, loss_matrix, torch.zeros_like(loss_matrix))
        
        optimizer.zero_grad()
        loss = torch.sum(loss_matrix) / torch.sum(valid_mask)
        loss.backward()
        optimizer.step()


def evaluate(args, model, device, loader, prompt):
    model.eval()
    all_targets, all_preds = [], []
    for batch in tqdm(loader, desc="Evaluating"):
        batch = batch.to(device)
        with torch.no_grad():
            preds = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, prompt)
        all_targets.append(batch.y.view(preds.shape))
        all_preds.append(preds)

    targets = torch.cat(all_targets, dim=0).cpu().numpy()
    preds = torch.cat(all_preds, dim=0).cpu().numpy()

    roc_scores = []
    for i in range(targets.shape[1]):
        if np.sum(targets[:, i] == 1) > 0 and np.sum(targets[:, i] == -1) > 0:
            valid = targets[:, i]**2 > 0
            roc_scores.append(roc_auc_score((targets[valid, i] + 1) / 2, preds[valid, i]))

    if len(roc_scores) < targets.shape[1]:
        print("Warning: 일부 타겟의 데이터가 부족합니다.")
    return np.mean(roc_scores) if roc_scores else 0.0


def parse_arguments():
    parser = argparse.ArgumentParser(description='Prompt Tuning for Pre-trained GNNs')
    parser.add_argument('--device', type=int, default=0, help='GPU device id (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (default: 32)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--lr_scale', type=float, default=1, help='LR scale for feature extraction layer (default: 1)')
    parser.add_argument('--decay', type=float, default=0, help='Weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5, help='Number of GNN layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=300, help='Embedding dimension (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.5, help='Dropout ratio (default: 0.5)')
    parser.add_argument('--graph_pooling', type=str, default="mean", help='Graph pooling method (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last", help='Feature combination method (last, sum, max, concat)')
    parser.add_argument('--gnn_type', type=str, default="gin", help='GNN type (gin, gcn, graphsage, gat)')
    parser.add_argument('--tuning_type', type=str, default="gpf", help='Tuning type ("gpf" or "gpf-plus")')
    parser.add_argument('--dataset', type=str, default='tox21', help='Dataset name (tox21, bbbp, etc.)')
    parser.add_argument('--model_file', type=str, default='', help='Path to pre-trained model file')
    parser.add_argument('--seed', type=int, default=42, help='Seed for dataset splitting')
    parser.add_argument('--runseed', type=int, default=0, help='Seed for experiment run')
    parser.add_argument('--split', type=str, default="scaffold", help='Dataset split method (scaffold, random, random_scaffold)')
    parser.add_argument('--eval_train', type=int, default=0, help='Evaluate training set (1: yes, 0: no)')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of MLP layers (default: 1)')
    parser.add_argument('--pnum', type=int, default=5, help='Number of basis for GPF-plus (default: 5)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers (default: 4)')
    return parser.parse_args()


def setup_environment(runseed, device_id):
    torch.manual_seed(runseed)
    np.random.seed(runseed)
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(runseed)
    return device


def load_dataset(args):
    dataset_dir = os.path.join("dataset", args.dataset)
    dataset = MoleculeDataset(dataset_dir, dataset=args.dataset)
    print(dataset)

    if args.split == "scaffold":
        smiles_path = os.path.join(dataset_dir, "processed", "smiles.csv")
        smiles_list = pd.read_csv(smiles_path, header=None)[0].tolist()
        split_fn = scaffold_split
        print("Using scaffold split.")
    elif args.split == "random":
        split_fn = random_split
        print("Using random split.")
        smiles_list = None
    elif args.split == "random_scaffold":
        smiles_path = os.path.join(dataset_dir, "processed", "smiles.csv")
        smiles_list = pd.read_csv(smiles_path, header=None)[0].tolist()
        split_fn = random_scaffold_split
        print("Using random scaffold split.")
    else:
        raise ValueError("Invalid split option.")

    return split_fn(dataset, smiles_list, null_value=0, frac_train=0.8, frac_valid=0.1, frac_test=0.1)


def build_model(args, num_tasks, device):
    model = GNN_graphpred(
        num_layer=args.num_layer,
        emb_dim=args.emb_dim,
        num_tasks=num_tasks,
        JK=args.JK,
        drop_ratio=args.dropout_ratio,
        graph_pooling=args.graph_pooling,
        gnn_type=args.gnn_type,
        head_layer=args.num_layers
    )
    if args.model_file:
        model.from_pretrained(args.model_file)
    print(model)
    return model.to(device)


def create_prompt(args, device):
    if args.tuning_type == 'gpf':
        return Prompt.SimplePrompt(args.emb_dim).to(device)
    elif args.tuning_type == 'gpf-plus':
        return Prompt.GPFplusAtt(args.emb_dim, args.pnum).to(device)
    else:
        raise ValueError("Invalid tuning type.")


def configure_optimizer(args, prompt, model):
    param_groups = [{"params": prompt.parameters()}]
    if args.graph_pooling == "attention":
        param_groups.append({"params": model.pool.parameters(), "lr": args.lr * args.lr_scale})
    param_groups.append({"params": model.graph_pred_linear.parameters(), "lr": args.lr * args.lr_scale})
    optimizer = optim.Adam(param_groups, lr=args.lr, weight_decay=args.decay, amsgrad=False)
    print(optimizer)
    return optimizer


def main():
    args = parse_arguments()
    device = setup_environment(args.runseed, args.device)

    # Set number of tasks based on dataset
    dataset_tasks = {
        "tox21": 12, "hiv": 1, "pcba": 128, "muv": 17,
        "bace": 1, "bbbp": 1, "toxcast": 617, "sider": 27, "clintox": 2
    }
    if args.dataset not in dataset_tasks:
        raise ValueError("Invalid dataset name.")
    num_tasks = dataset_tasks[args.dataset]

    # Load dataset splits
    train_ds, val_ds, test_ds = load_dataset(args)
    print("First training sample:", train_ds[0])

    # Create data loaders
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Build model and prompt module
    model = build_model(args, num_tasks, device)
    prompt = create_prompt(args, device)
    optimizer = configure_optimizer(args, prompt, model)

    train_acc_list, val_acc_list, test_acc_list = [], [], []
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch} (LR: {optimizer.param_groups[-1]['lr']})")
        train_epoch(args, model, device, train_loader, optimizer, prompt)
        print("Evaluating...")
        train_acc = evaluate(args, model, device, train_loader, prompt) if args.eval_train else 0
        val_acc = evaluate(args, model, device, val_loader, prompt)
        test_acc = evaluate(args, model, device, test_loader, prompt)
        print(f"Train: {train_acc:.6f}  Val: {val_acc:.6f}  Test: {test_acc:.6f}\n")
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        test_acc_list.append(test_acc)

    with open("result.log", "a+") as log_file:
        log_file.write(f"{args.dataset} {args.runseed} {test_acc_list[-1]}\n")


if __name__ == "__main__":
    main()
