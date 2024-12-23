import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from split_dataset import VAEDataset
from transformer_pos_enc import Transformer

torch.set_float32_matmul_precision('medium')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 256
max_mech_size = 12
num_classes = 175

dataset = VAEDataset(num_classes=num_classes, max_mech_size=max_mech_size)

dataset_size = len(dataset)
train_size = int(0.8 * dataset_size)
val_size = dataset_size - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, shuffle=True, num_workers=15, batch_size=batch_size)
val_loader = DataLoader(val_dataset, num_workers=15, batch_size=batch_size)

model = Transformer(output_size=2, tgt_seq_len=max_mech_size)

checkpoint_callback = ModelCheckpoint(
    save_weights_only=False,
    dirpath='weights/',
    monitor='train_loss',
    filename='{epoch}',
    save_top_k=1)

wandb_logger = WandbLogger(project='vq_transformer')

# Initialize the trainer without the deprecated argument
if torch.cuda.device_count() == 1:
    trainer = pl.Trainer(logger=wandb_logger, accelerator="gpu", devices=-1, max_epochs=-1, strategy="ddp", callbacks=[checkpoint_callback])
elif torch.cuda.device_count() > 1:
    trainer = pl.Trainer(
        logger=wandb_logger,
        accelerator="gpu",
        devices=torch.cuda.device_count(),
        max_epochs=-1,
        strategy='ddp',  # Enable unused parameter detection
        callbacks=[checkpoint_callback]
    )
else:
    trainer = pl.Trainer(logger=wandb_logger, accelerator="cpu", max_epochs=-1, callbacks=[checkpoint_callback])

# Resume training from the checkpoint
trainer.fit(model, train_loader, val_loader)