import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassRecall,
)
from torchvision.utils import make_grid


class LitImageClassifier(pl.LightningModule):
    def __init__(self, model, num_classes, class_weights, lr=1e-5, weight_decay=1e-2):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        backbone = model(weights="DEFAULT")
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(num_filters, num_classes)
        self.lr = lr
        self.weight_decay = weight_decay
        self.register_buffer("class_weights", torch.Tensor(class_weights))

        self.train_acc = MulticlassAccuracy(num_classes=num_classes, average="micro")
        self.train_bacc = MulticlassRecall(num_classes=num_classes, average="macro")
        self.train_auc = MulticlassAUROC(num_classes=num_classes, average="macro")
        self.val_acc = MulticlassAccuracy(num_classes=num_classes, average="micro")
        self.val_bacc = MulticlassRecall(num_classes=num_classes, average="macro")
        self.val_auc = MulticlassAUROC(num_classes=num_classes, average="macro")
        self.test_acc = MulticlassAccuracy(num_classes=num_classes, average="micro")
        self.test_bacc = MulticlassRecall(num_classes=num_classes, average="macro")
        self.test_auc = MulticlassAUROC(num_classes=num_classes, average="macro")

        self.register_buffer("val_bacc_best", torch.Tensor([0.0]))

    def forward(self, x):
        features = self.feature_extractor(x).flatten(1)
        logits = self.classifier(features)

        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch

        if self.current_epoch == 0 and batch_idx == 0:
            self._log_images(x)

        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y, weight=self.class_weights)
        self.log("train/loss", loss)

        self.train_acc(y_hat, y)
        self.train_bacc(y_hat, y)
        self.train_auc(y_hat, y)
        self.log("train/acc_step", self.train_acc)
        self.log("train/bacc_step", self.train_bacc)
        self.log("train/auc_step", self.train_auc)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y, weight=self.class_weights)
        self.log("val/loss", loss)

        self.val_acc(y_hat, y)
        self.val_bacc(y_hat, y)
        self.val_auc(y_hat, y)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y, weight=self.class_weights)
        self.log("test/loss", loss)

        self.test_acc(y_hat, y)
        self.test_bacc(y_hat, y)
        self.test_auc(y_hat, y)

        return loss

    def on_train_epoch_end(self):
        self.log("train/acc_epoch", self.train_acc)
        self.log("train/bacc_epoch", self.train_bacc)
        self.log("train/auc_epoch", self.train_auc)

    def on_validation_epoch_end(self):
        if self.val_bacc.compute() > self.val_bacc_best:
            self.val_bacc_best = self.val_bacc.compute()

        self.log("val/acc_epoch", self.val_acc)
        self.log("val/bacc_epoch", self.val_bacc)
        self.log("val/bacc_best", self.val_bacc_best)
        self.log("val/auc_epoch", self.val_auc)

    def on_test_epoch_end(self):
        self.log("test/acc_epoch", self.test_acc)
        self.log("test/bacc_epoch", self.test_bacc)
        self.log("test/auc_epoch", self.test_auc)

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer

    def _log_images(self, x):
        tensorboard = self.logger.experiment
        image_grid = make_grid(x, nrow=8, normalize=True)
        tensorboard.add_image("train/x", image_grid, global_step=self.global_step)
