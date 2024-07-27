from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

from config import Config

def get_ckpt_cb(dir):
    return ModelCheckpoint(
        monitor="val_loss",
        dirpath=dir,
        filename='ckpt-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode="min",
    )

def get_trainer(cfg: Config, logger=None):
    checkpoint_callback = get_ckpt_cb(cfg.checkpoint_dir)
    return Trainer(
        # checkpointing
        callbacks=[checkpoint_callback],
        accelerator='cpu',
        # distribution
        strategy="ddp_spawn",
        devices=cfg.n_workers,
        max_epochs=4,
        # logging
        logger=logger
    )
    
def get_test_trainer(cfg: Config, logger=None):
    return Trainer(
        accelerator='cpu',
        strategy='auto',
        logger=logger
    )
