import torch
from torch import nn
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint,LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger

from pytorch_metric_learning import losses
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.utils.inference import CustomKNN


from model import EyeKnowYouTooSimple
from model import EyeKnowYouTooDense

from dataset import create_dataloaders


class EyeKnowYouTooLightning(pl.LightningModule):
    def __init__(self, num_classes, learning_rate= 1e-4):
        super().__init__()

        self.model = EyeKnowYouTooDense(num_classes=num_classes)
        self.ce_loss = nn.CrossEntropyLoss()
        self.ms_loss = losses.MultiSimilarityLoss()
        self.learning_rate = learning_rate

        self.validation_embeddings = []
        self.validation_user_ids = []

        self.knn_func = CustomKNN(CosineSimilarity(), batch_size= 256)
        self.accuracy_calculator = AccuracyCalculator(
            include = ("mean_average_precision_at_r",),
            k = "max_bin_count",
            knn_func = self.knn_func)


    def training_step(self, batch, batch_idx):
        signals, labels, _ = batch

        embeddings, logits = self.model(signals)

        normalized_embeddings = F.normalize(
            embeddings.detach(),
            p=2,
            dim=1)

        cosine_matrix = normalized_embeddings @ normalized_embeddings.T
        coincidence_matrix = labels.unsqueeze(1)  == labels.unsqueeze(0) 
        not_self = ~torch.eye(labels.size(0), dtype = torch.bool, device = labels.device)
        positive_mask = coincidence_matrix & not_self
        negative_mask = ~coincidence_matrix
        positive_cosines = cosine_matrix[positive_mask]
        negative_cosines = cosine_matrix[negative_mask]
        mean_positive_cosine = positive_cosines.mean()
        mean_negative_cosine = negative_cosines.mean()

        
        ms_loss = self.ms_loss(embeddings, labels)
        ce_loss = self.ce_loss(logits, labels)

        loss = ms_loss + 0.1 * ce_loss

        self.log("train_ms_loss", ms_loss, on_step=False, on_epoch=True,
                prog_bar=True, batch_size=signals.size(0))

        self.log("train_ce_loss", ce_loss, on_step=False, on_epoch=True,
                batch_size=signals.size(0))

        self.log("train_loss", loss, on_step=False, on_epoch=True,
                batch_size=signals.size(0))

        self.log(
            "train_pos_cos",
            mean_positive_cosine,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=signals.size(0)
        )

        self.log(
            "train_neg_cos",
            mean_negative_cosine,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=signals.size(0)
        )

        return loss

    def validation_step(self, batch, batch_idx):

        signals, _, user_ids = batch
        embeddings, _ = self.model(signals)

        self.validation_embeddings.append(embeddings.detach().cpu())
        self.validation_user_ids.extend(user_ids)


    def on_validation_epoch_start(self):
        self.validation_embeddings = []
        self.validation_user_ids = []

    def on_validation_epoch_end(self):
        all_embeddings = torch.cat(self.validation_embeddings, dim = 0)

        user_ids = [int(user_id) for user_id in self.validation_user_ids]
        user_ids = torch.tensor(user_ids, dtype=torch.long)

        accuracies = self.accuracy_calculator.get_accuracy(query=all_embeddings, query_labels=user_ids,
                                              reference=all_embeddings, reference_labels=user_ids,
                                              ref_includes_query=True)

        val_map_at_r = accuracies["mean_average_precision_at_r"]

        self.log("val_map_at_r", val_map_at_r, on_step=False, on_epoch=True)


    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate
        )
        total_steps = self.trainer.estimated_stepping_batches
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=1e-2,
            total_steps=total_steps,
            pct_start=0.3,
            anneal_strategy="cos",
            div_factor=100,
            final_div_factor=1000,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train EKYT on one GazeBase fold.")
    parser.add_argument("--fold", type=int, default=0,
                        help="Must match the fold used when running split.py.")
    args = parser.parse_args()

    import json
    with open("splits.json", "r", encoding="utf-8") as file:
        split_fold = json.load(file)["current_fold"]
    if split_fold != args.fold:
        raise SystemExit(
            f"splits.json was made for fold {split_fold}, not {args.fold}. "
            f"Run: python split.py --fold {args.fold} && python normalize.py"
        )

    pl.seed_everything(96, workers=True)
    
    train_loader, val_loader, test_loader, label_map = create_dataloaders(
        batch_size=64, num_workers=2
    )


    model = EyeKnowYouTooLightning(
        num_classes=len(label_map)
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/fold{args.fold}",
        filename="ekyt-{epoch:02d}",
        every_n_epochs=1,
        save_top_k=-1,
        save_last=True,
    )

    lr_monitor = LearningRateMonitor(
        logging_interval="epoch"
    )

    trainer = pl.Trainer(
        max_epochs=100,
        accelerator="auto",  # CUDA, Apple MPS or CPU, whichever is available
        devices=1,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=CSVLogger("lightning_logs", name=f"fold{args.fold}"),
    )

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )


if __name__ == "__main__":
    main()