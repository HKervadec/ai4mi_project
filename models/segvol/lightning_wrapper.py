from lightning.pytorch import LightningModule
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
from models.segvol.base import SegVolConfig
from models.segvol.lora_model import SegVolLoRA
from utils.dataset import VolumetricDataset

# from utils.losses import get_loss


class SegVolLightning(LightningModule):
    def __init__(self, args, batch_size, K, **kwargs):
        super().__init__()
        self.args = args
        self.batch_size = batch_size
        self.K = K

        config = SegVolConfig(test_mode=False)
        self.model = SegVolLoRA(config)
        self.categories = ["background", "esophagus", "heart", "trachea", "aorta"]

        if args.loss == "dicefocal":
            print(">>Changed BCELoss to FocalLoss")
            from monai.losses.focal_loss import FocalLoss
            self.model.model.bce_loss = FocalLoss(include_background=False)
        # if True: # meant to be a flag
        #     self.net = torch.compile(self.net)

        # self.loss_fn = get_loss(
        #     args.loss, self.K, include_background=args.include_background
        # )

        # Dataset part
        self.batch_size: int = args.datasets_params[args.dataset]["B"]  # Batch size
        self.root_dir: Path = Path(args.data_dir) / str(args.dataset)

        args.dest.mkdir(parents=True, exist_ok=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda x: x.requires_grad, self.parameters()),
            lr=self.args.lr,
            weight_decay=0.005,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.args.epochs
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def on_fit_start(self) -> None:
        super().on_fit_start()
        self.log_loss_tra = torch.zeros(self.args.epochs, len(self.train_dataloader()))
        self.log_dice_tra = torch.zeros(self.args.epochs, len(self.train_set), self.K)
        self.log_loss_val = torch.zeros(self.args.epochs, len(self.val_dataloader()))
        self.log_dice_val = torch.zeros(self.args.epochs, len(self.val_set), self.K)
        # TODO: Implement 3D DICE

        # self.log_dice_3d_tra = torch.zeros(
        #     (args.epochs, len(self.gt_shape["train"].keys()), self.K)
        # )
        # self.log_dice_3d_val = torch.zeros(
        #     (args.epochs, len(self.gt_shape["val"].keys()), self.K)
        # )
        self.best_dice = 0

    def train_dataloader(self):
        self.train_set = VolumetricDataset(
            self.root_dir / "train",
            ratio=0.8,
            processor=self.model.processor,
            num_classes=self.K,
            train=True,
            cache_size=0,
        )
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            # pin_memory=True,
        )

    def val_dataloader(self):
        self.val_set = VolumetricDataset(
            self.root_dir / "train",
            ratio=0.8,
            processor=self.model.processor,
            num_classes=self.K,
            train=False,
            cache_size=40,
        )
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            # pin_memory=True,
        )

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        self.eval()

        # self.gt_volumes = {
        #     p: np.zeros((Z, self.K, X, Y), dtype=np.uint8)
        #     for p, (X, Y, Z) in self.gt_shape["val"].items()
        # }

        # self.pred_volumes = {
        #     p: np.zeros((Z, self.K, X, Y), dtype=np.uint8)
        #     for p, (X, Y, Z) in self.gt_shape["val"].items()
        # }

    def _prepare_3d_dice(self, batch_stems, gt, pred_seg):
        for i, seg_class in enumerate(pred_seg):
            stem = batch_stems[i]
            _, patient_n, z = stem.split("_")
            patient_id = f"Patient_{patient_n}"

            X, Y, _ = self.gt_shape["val"][patient_id]

            # self.pred_volumes[patient_id] = resize_and_save_slice(
            #     seg_class, self.K, X, Y, z, self.pred_volumes[patient_id]
            # )
            # self.gt_volumes[patient_id] = resize_and_save_slice(
            #     gt[i], self.K, X, Y, z, self.gt_volumes[patient_id]
            # )

    def forward(self, image, gt):
        # Sanity tests to see we loaded and encoded the data correctly
        raise NotImplementedError
        # assert image.shape[0] == 1, image.shape

        # for k in range(1, self.K):
        #     text_label = self.categories[k]
        #     mask_label = gt[:, k]

        # return self.net(x)
    
    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        self.train()

    def training_step(self, batch, batch_idx):
        img, gt = batch["image"], batch["label"]

        compound_loss = 0.
        for k in range(1, self.K):
            text_label = self.categories[k]
            mask_label = gt[:, k]

            # NOTE: We must patch the source code of SegVol to change the loss function
            loss = self.model.forward_train(
                img, train_organs=text_label, train_labels=mask_label, modality="CT"
            )
            compound_loss += loss

        self.log_loss_tra[self.current_epoch, batch_idx] = compound_loss
        # self.log_dice_tra[
        #     self.current_epoch, batch_idx : batch_idx + img.size(0), :
        # ] = dice_coef(pred_seg, gt)

        self.log("train/loss", loss, prog_bar=True, logger=True)
        return compound_loss

    def validation_step(self, batch, batch_idx):
        img, gt = batch["image"], batch["label"]

        dice = {}
        for k in range(1, self.K):
            # text prompt
            text_prompt = [self.categories[k]]

            # # point prompt
            # point_prompt, point_prompt_map = self.model.processor.point_prompt_b(batch['zoom_out_label'][0, k], device=self.device)   # inputs w/o batch dim, outputs w batch dim

            # # bbox prompt
            # bbox_prompt, bbox_prompt_map = self.model.processor.bbox_prompt_b(batch['zoom_out_label'][0, k], device=self.device)   # inputs w/o batch dim, outputs w batch dim

            logits_mask = self.model.forward_test(
                image=img,
                zoomed_image=batch["zoom_out_image"],
                # point_prompt_group=[point_prompt, point_prompt_map],
                # bbox_prompt_group=[bbox_prompt, bbox_prompt_map],
                text_prompt=text_prompt,
                use_zoom=True,
            )

            dice[f"val/dice/{text_prompt[0]}"] = self.model.processor.dice_score(
                logits_mask[0][0], gt[0, k], self.device
            )

        self.log_dict(dice, logger=True, prog_bar=True, on_epoch=True)
        # self._prepare_3d_dice(batch["stems"], gt, pred_seg)

    def on_validation_epoch_end(self):
        # log_dict = {
        #     "val/loss": self.log_loss_val[self.current_epoch].mean().detach(),
        #     "val/dice/total": self.log_dice_val[self.current_epoch, :, 1:]
        #     .mean()
        #     .detach(),
        # }
        # for k, v in self.get_dice_per_class(
        #     self.log_dice_val, self.K, self.current_epoch
        # ).items():
        #     log_dict[f"val/dice/{k}"] = v
        # if self.args.dataset == "SEGTHOR":
        #     for i, (patient_id, pred_vol) in tqdm_(
        #         enumerate(self.pred_volumes.items()), total=len(self.pred_volumes)
        #     ):
        #         gt_vol = torch.from_numpy(self.gt_volumes[patient_id]).to(self.device)
        #         pred_vol = torch.from_numpy(pred_vol).to(self.device)

        #         dice_3d = dice_batch(gt_vol, pred_vol)
        #         self.log_dice_3d_val[self.current_epoch, i, :] = dice_3d

        #     log_dict["val/dice_3d/total"] = (
        #         self.log_dice_3d_val[self.current_epoch, :, 1:].mean().detach()
        #     )
        #     # log_dict["val/dice_3d_class"] = self.get_dice_per_class(self.log_dice_3d_val, self.K, self.current_epoch)
        #     for k, v in self.get_dice_per_class(
        #         self.log_dice_3d_val, self.K, self.current_epoch
        #     ).items():
        #         log_dict[f"val/dice_3d/{k}"] = v
        # self.log_dict(log_dict)

        # current_dice = self.log_dice_val[self.current_epoch, :, 1:].mean().detach()
        # if current_dice > self.best_dice:
        #     self.best_dice = current_dice
        #     self.save_model()

        super().on_validation_epoch_end()

    # def get_dice_per_class(self, log, K, e):
    #     if self.args.dataset == "SEGTHOR":
    #         class_names = [
    #             (1, "background"),
    #             (2, "esophagus"),
    #             (3, "heart"),
    #             (4, "trachea"),
    #             (5, "aorta"),
    #         ]
    #         dice_per_class = {
    #             f"dice_{k}_{n}": log[e, :, k - 1].mean().item() for k, n in class_names
    #         }
    #     else:
    #         dice_per_class = {
    #             f"dice_{k}": log[e, :, k].mean().item() for k in range(1, K)
    #         }
    #     return dice_per_class

    def save_model(self):
        torch.save(self.model, self.args.dest / "bestmodel.pkl")
        torch.save(self.model.state_dict(), self.args.dest / "bestweights.pt")
        # if self.args.wandb_project_name:
        #     self.logger.save(str(self.args.dest / "bestweights.pt"))
