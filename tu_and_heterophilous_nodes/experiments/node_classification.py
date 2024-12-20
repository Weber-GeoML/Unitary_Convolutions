import torch
import numpy as np
from attrdict import AttrDict
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score
from math import inf

from models.node_model import GCN, UnitaryGCN, OrthogonalGCN

default_args = AttrDict(
    {"learning_rate": 3 * 1e-5,
    "max_epochs": 2000,
    "display": True,
    "device": None,
    "eval_every": 1,
    "stopping_criterion": "validation",
    "stopping_threshold": 1,
    "patience": 2000,
    "train_fraction": 0.5,
    "validation_fraction": 0.25,
    "test_fraction": 0.25,
    "dropout": 0.5,
    "weight_decay": 1e-5,
    "hidden_dim": 512,
    "hidden_layers": None,
    "num_layers": 8,
    "batch_size": 50,
    "layer_type": "Unitary",
    "num_relations": 1,
    "T": 20
    }
    )

class Experiment:
    def __init__(self, args=None, dataset=None, train_mask=None, validation_mask=None, test_mask=None):
        self.args = default_args + args
        self.dataset = dataset
        self.train_mask = train_mask
        self.validation_mask = validation_mask
        self.test_mask = test_mask
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.args.input_dim = self.dataset[0].x.shape[1]
        self.args.output_dim = torch.amax(self.dataset[0].y).item() + 1
        self.num_nodes = self.dataset[0].x.size(axis=0)
        self.metric = 'Accuracy' if self.num_nodes > 20000 and self.num_nodes < 25000 else 'ROC AUC'
        print("Metric: ", self.metric)

        if self.args.device is None:
            self.args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.args.hidden_layers is None:
            self.args.hidden_layers = [self.args.hidden_dim] * self.args.num_layers

        if self.args.layer_type == "Orthogonal":
            self.model = OrthogonalGCN(self.args).to(self.args.device)
        elif self.args.layer_type == "Unitary":
            self.model = UnitaryGCN(self.args).to(self.args.device)
        else:
            self.model = GCN(self.args).to(self.args.device)

        if self.test_mask is None:
            node_indices = list(range(self.num_nodes))
            self.args.test_fraction = 1 - self.args.train_fraction - self.args.validation_fraction
            non_test, self.test_mask = train_test_split(node_indices, test_size=self.args.test_fraction)
            self.train_mask, self.validation_mask = train_test_split(non_test, test_size=self.args.validation_fraction/(self.args.validation_fraction + self.args.train_fraction))
        elif self.validation_mask is None:
            non_test = [i for i in range(self.num_nodes) if not i in self.test_mask]
            self.train_mask, self.validation_mask = train_test_split(non_test, test_size=self.args.validation_fraction/(self.args.validation_fraction + self.args.train_fraction))
        
    def run(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer,  patience=100, factor=0.5)
        step_size = 25  # Specify the number of epochs after which to decrease the learning rate
        gamma = 0.1     # Specify the factor by which to decrease the learning rate


        if self.args.display:
            print("Starting training")
        best_test_acc = 0.0
        best_validation_acc = 0.0
        best_train_acc = 0.0
        train_goal = 0.0
        validation_goal = 0.0
        best_epoch = 0
        epochs_no_improve = 0
        train_size = len(self.train_mask)
        batch = self.dataset.data.to(self.args.device)
        y = batch.y

        for epoch in range(self.args.max_epochs):
            self.model.train()
            total_loss = 0
            sample_size = 0
            optimizer.zero_grad()

            out = self.model(batch)
            loss = self.loss_fn(input=out[self.train_mask], target=y[self.train_mask])
            total_loss += loss.item()
            _, train_pred = out[self.train_mask].max(dim=1)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


            new_best_str = ''

            if epoch % self.args.eval_every == 0:
                # compute Accuracy for Roman Empire and Amazon Ratings
                if self.metric == 'Accuracy':
                    self.model.eval()
                    with torch.no_grad():
                        pred = self.model(batch)
                    train_acc = self.compute_acc(pred, y, mask=self.train_mask)
                    validation_acc = self.compute_acc(pred, y, mask=self.validation_mask)
                    test_acc = self.compute_acc(pred, y, mask=self.test_mask)

                # compute ROC AUC for the rest
                else:
                    self.model.eval()
                    with torch.no_grad():
                        pred = self.model(batch)
                    train_acc = self.compute_roc_auc(pred, y, mask=self.train_mask)
                    validation_acc = self.compute_roc_auc(pred, y, mask=self.validation_mask)
                    test_acc = self.compute_roc_auc(pred, y, mask=self.test_mask)
                scheduler.step(-validation_acc)

                if self.args.stopping_criterion == "train":
                    if train_acc > train_goal:
                        best_train_acc = train_acc
                        best_validation_acc = validation_acc
                        best_test_acc = test_acc
                        epochs_no_improve = 0
                        train_goal = train_acc * self.args.stopping_threshold
                        new_best_str = ' (new best train)'
                    elif train_acc > best_train_acc:
                        best_train_acc = train_acc
                        best_validation_acc = validation_acc
                        best_test_acc = test_acc
                        epochs_no_improve += 1
                    else:
                        epochs_no_improve += 1
                elif self.args.stopping_criterion == 'validation':
                    if validation_acc > validation_goal:
                        best_train_acc = train_acc
                        best_validation_acc = validation_acc
                        best_test_acc = test_acc
                        epochs_no_improve = 0
                        validation_goal = validation_acc * self.args.stopping_threshold
                        new_best_str = ' (new best validation)'
                    elif validation_acc > best_validation_acc:
                        best_train_acc = test_acc
                        best_validation_acc = validation_acc
                        best_test_acc = test_acc
                        epochs_no_improve += 1
                    else:
                        epochs_no_improve += 1
                if self.args.display and self.metric == 'Accuracy':
                    print(f'Epoch {epoch}, Train acc: {train_acc}, Validation acc: {validation_acc}{new_best_str}, Test acc: {test_acc}, Learning Rate: {scheduler.get_last_lr()}', flush=True)
                elif self.args.display and self.metric == 'ROC AUC':
                    print(f'Epoch {epoch}, Train ROC AUC: {train_acc}, Validation ROC AUC: {validation_acc}{new_best_str}, Test ROC AUC: {test_acc}, Learning Rate: {scheduler.get_last_lr()}', flush=True)
                if epochs_no_improve > self.args.patience:
                    if self.args.display:
                        print(f'{self.args.patience} epochs without improvement, stopping training', flush=True)
                        print(f'Best train acc: {best_train_acc}, Best validation loss: {best_validation_acc}, Best test loss: {best_test_acc}', flush=True)
                    return train_acc, validation_acc, test_acc
        return train_acc, validation_acc, test_acc

    def compute_acc(self, pred, y, mask):
        _, pred = pred[mask].max(dim=1)
        sample_size = len(mask)
        total_correct = pred.eq(y[mask]).sum().item()
        acc = total_correct / sample_size
        return acc
        
    def compute_roc_auc(self, pred, y, mask):
        pred = pred[mask]
        probs = F.softmax(pred, dim=1)
        y_pred = probs.cpu().numpy()
        y_true = y[mask].cpu().numpy()
        roc_auc = roc_auc_score(y_true, y_pred[:,1])
        return roc_auc
