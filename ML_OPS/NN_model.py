from typing import Any

import lightning.pytorch as pl
import torch
import torchmetrics
from omegaconf import DictConfig
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam


class my_model(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg

        self.linear_1 = nn.Linear(cfg.model.input_dim, cfg.model.hidden_1)
        self.linear_2 = nn.Linear(cfg.model.hidden_1, cfg.model.hidden_2)
        self.linear_3 = nn.Linear(cfg.model.hidden_2, cfg.model.hidden_3)
        self.linear_4 = nn.Linear(cfg.model.hidden_3, cfg.model.hidden_4)
        self.linear_5 = nn.Linear(cfg.model.hidden_4, cfg.model.output_dim)
        self.act_1 = nn.ReLU()
        self.act_2 = nn.ReLU()
        self.act_3 = nn.ReLU()
        self.act_4 = nn.ReLU()
        self.act_5 = nn.ReLU()
        self.batchnorm_1 = nn.BatchNorm1d(cfg.model.hidden_1)
        self.batchnorm_2 = nn.BatchNorm1d(cfg.model.hidden_2)
        self.batchnorm_3 = nn.BatchNorm1d(cfg.model.hidden_3)
        self.batchnorm_4 = nn.BatchNorm1d(cfg.model.hidden_4)

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.f1_fn = torchmetrics.classification.F1Score(
            task=cfg.model.f1_task, num_classes=cfg.model.output_dim
        )

    def forward(self, x):
        x = self.linear_1(x)
        x = self.batchnorm_1(x)
        x = self.act_1(x)

        x = self.linear_2(x)
        x = self.batchnorm_2(x)
        x = self.act_2(x)

        x = self.linear_3(x)
        x = self.batchnorm_3(x)
        x = self.act_3(x)

        x = self.linear_4(x)
        x = self.batchnorm_4(x)
        x = self.act_4(x)

        x = self.linear_5(x)
        x = self.act_5(x)
        return F.softmax(x, dim=1)

    def configure_optimizers(self) -> Any:
        return Adam(self.parameters(), lr=self.cfg.model.lr)

    def training_step(self, batch: Any, batch_idx: int, dataloader_idx=0):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.loss_fn(outputs, labels)
        predicted = torch.argmax(outputs, dim=1)
        acc = torch.sum(labels == predicted).item() / (len(predicted) * 1.0)
        f1 = self.f1_fn(predicted, labels).item()
        self.log_dict(
            {"train_loss": loss, "train_acc": acc, "train_f1": f1},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        inputs, labels = batch
        outputs: torch.Tensor = self(inputs)
        val_loss = self.loss_fn(outputs, labels)
        pred = torch.argmax(outputs, dim=1)
        val_acc = torch.sum(labels == pred).item() / (len(pred) * 1.0)
        val_f1 = self.f1_fn(pred, labels).item()
        self.log_dict(
            {"val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return val_loss
