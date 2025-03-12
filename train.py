import torch
import os
from torch.utils.data import random_split
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
from torch_geometric.loader import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from tqdm import tqdm
import wandb

from dataset import TransformerDataset  # Your dataset class
from model_torch import Transformer  # Your model class

def mse_loss(predictions, targets, mask_value=-1.0):
    mask = ~(targets == mask_value).all(dim=-1)
    mask = mask.unsqueeze(-1).expand_as(predictions)
    masked_predictions = predictions[mask]
    masked_targets = targets[mask]
    loss = F.mse_loss(masked_predictions, masked_targets, reduction="mean")
    return loss

def setup_ddp():
    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))

def cleanup_ddp():
    dist.destroy_process_group()

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def save_best_checkpoint(model, optimizer, epoch, best_loss, batch_size, lr, save_dir="/gpfs/scratch/anurizada/nobari_weights"):
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, f"bs_{batch_size}_lr_{lr}_new.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'best_loss': best_loss,
        'batch_size': batch_size,
        'learning_rate': lr
    }, checkpoint_path)
    print(f"[Rank {get_rank()}] Best model saved at {checkpoint_path} with training loss {best_loss:.6f}.")

def train():
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    setup_ddp()
    torch.set_float32_matmul_precision('medium')

    batch_size = 128
    max_mech_size = 14
    num_epochs = 300
    lr = 0.0001

    # Initialize wandb for logging
    if rank == 0:
        wandb.init(project="distributed-training", name=f"ddp_bs_{batch_size}_lr_{lr}_masked")

    dataset = TransformerDataset(
        node_features_path='/gpfs/scratch/anurizada/nobari_10_joints/node_features.npy',
        edge_index_path='/gpfs/scratch/anurizada/nobari_10_joints/edge_index.npy',
        curves_path='/gpfs/scratch/anurizada/nobari_10_joints/curves.npy',
        shuffle=True
    )

    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size, num_workers=0)

    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    val_loader = DataLoader(val_dataset, sampler=val_sampler, batch_size=batch_size, num_workers=0)

    model = Transformer(output_size=2, tgt_seq_len=max_mech_size).to(local_rank)
    model = DDP(model, device_ids=[local_rank])

    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.1)

    best_loss = float("inf")  # Initialize the best loss with infinity

    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        model.train()

        epoch_loss = 0.0
        with tqdm(total=len(train_loader), desc=f"Rank {rank} Train Epoch {epoch}", leave=False) as pbar:
            for batch_idx, batch in enumerate(train_loader):
                decoder_input_first = batch["decoder_input_first"].to(local_rank)
                decoder_input_second = batch["decoder_input_second"].to(local_rank)
                mask_first = batch["decoder_mask_first"].to(local_rank)
                mask_second = batch["decoder_mask_second"].to(local_rank)
                curve_data = batch["curve"].to(local_rank)
                graph_data = batch["data"].x.to(local_rank)
                edge_index = batch["data"].edge_index.to(local_rank)
                batch_indices = batch["data"].batch.to(local_rank)
                label_first = batch["label_first"].to(local_rank)
                label_second = batch["label_second"].to(local_rank)

                optimizer.zero_grad()
                pred_first, pred_second = model(
                    decoder_input_first,
                    decoder_input_second,
                    mask_first,
                    mask_second,
                    curve_data,
                    graph_data,
                    edge_index,
                    batch_indices,
                )

                loss_first = mse_loss(pred_first, label_first)
                loss_second = mse_loss(pred_second, label_second)
                loss = loss_first + loss_second

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

                # Log training losses to wandb
                if rank == 0:
                    wandb.log({
                        "train_loss_first": loss_first.item(),
                        "train_loss_second": loss_second.item(),
                        "train_loss_total": loss.item(),
                        "epoch": epoch,
                    })

                pbar.set_postfix({"Loss": loss.item()})
                pbar.update(1)

        scheduler.step()

        # Calculate average loss for the epoch
        avg_train_loss = epoch_loss / len(train_loader)

        # Save the best model based on training loss
        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            save_best_checkpoint(model, optimizer, epoch, best_loss, batch_size, lr)

        # Validation loop
        model.eval()
        val_loss = 0.0
        with tqdm(total=len(val_loader), desc=f"Rank {rank} Validation Epoch {epoch}", leave=False) as pbar:
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    decoder_input_first = batch["decoder_input_first"].to(local_rank)
                    decoder_input_second = batch["decoder_input_second"].to(local_rank)
                    mask_first = batch["decoder_mask_first"].to(local_rank)
                    mask_second = batch["decoder_mask_second"].to(local_rank)
                    curve_data = batch["curve"].to(local_rank)
                    graph_data = batch["data"].x.to(local_rank)
                    edge_index = batch["data"].edge_index.to(local_rank)
                    batch_indices = batch["data"].batch.to(local_rank)
                    label_first = batch["label_first"].to(local_rank)
                    label_second = batch["label_second"].to(local_rank)

                    pred_first, pred_second = model(
                        decoder_input_first,
                        decoder_input_second,
                        mask_first,
                        mask_second,
                        curve_data,
                        graph_data,
                        edge_index,
                        batch_indices,
                    )

                    loss_first = mse_loss(pred_first, label_first)
                    loss_second = mse_loss(pred_second, label_second)
                    loss = loss_first + loss_second

                    val_loss += loss.item()

                    # Log validation losses to wandb
                    if rank == 0:
                        wandb.log({
                            "val_loss_first": loss_first.item(),
                            "val_loss_second": loss_second.item(),
                            "val_loss_total": loss.item(),
                            "epoch": epoch,
                        })

                    pbar.set_postfix({"Val Loss": loss.item()})
                    pbar.update(1)

        if rank == 0:
            print(f"Epoch {epoch} completed. Avg Train Loss: {avg_train_loss:.6f}, Validation Loss: {val_loss / len(val_loader):.6f}")

    cleanup_ddp()


if __name__ == "__main__":
    train()