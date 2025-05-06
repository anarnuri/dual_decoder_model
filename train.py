import torch
import os
from torch.utils.data import random_split
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from tqdm import tqdm
import wandb

from dataset import TransformerDataset
from model import Transformer

def mse_loss(predictions, targets, mask_value=-1.0):
    mask = ~(targets == mask_value).all(dim=-1)
    mask = mask.unsqueeze(-1).expand_as(predictions)
    masked_predictions = predictions[mask]
    masked_targets = targets[mask]
    return F.mse_loss(masked_predictions, masked_targets, reduction="mean")

def contrastive_loss(curve_emb, adj_emb, temperature=0.07):
    curve_norm = F.normalize(curve_emb, p=2, dim=1)
    adj_norm = F.normalize(adj_emb, p=2, dim=1)
    logits = torch.matmul(curve_norm, adj_norm.t()) / temperature
    batch_size = curve_emb.size(0)
    targets = torch.arange(batch_size, device=curve_emb.device)
    loss_c2a = F.cross_entropy(logits, targets)
    loss_a2c = F.cross_entropy(logits.t(), targets)
    return 0.5 * (loss_c2a + loss_a2c)

def setup_ddp():
    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))

def cleanup_ddp():
    dist.destroy_process_group()

def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()

def get_rank():
    return dist.get_rank() if is_dist_avail_and_initialized() else 0

def save_best_checkpoint(model, optimizer, epoch, best_loss, batch_size, lr, save_dir="weights"):
    os.makedirs(save_dir, exist_ok=True)
    
    # Get architecture parameters from the model
    d_model = model.module.d_model if hasattr(model.module, 'd_model') else 1024
    n_heads = model.module.h if hasattr(model.module, 'h') else 32
    n_layers = model.module.N if hasattr(model.module, 'N') else 6
    
    checkpoint_path = os.path.join(save_dir, f"d{d_model}_h{n_heads}_bs{batch_size}_lr{lr}_best.pth")
    torch.save({
        'model_state_dict': model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'best_loss': best_loss,
        'batch_size': batch_size,
        'learning_rate': lr,
        'architecture': {
            'd_model': d_model,
            'n_heads': n_heads,
            'n_layers': n_layers
        }
    }, checkpoint_path)
    print(f"[Rank {get_rank()}] Saved best model at {checkpoint_path} with loss {best_loss:.6f}")

def train():
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    setup_ddp()
    torch.set_float32_matmul_precision('medium')

    # Hyperparameters
    batch_size = 128
    max_mech_size = 20
    num_epochs = 1000
    lr = 1e-4
    clip_loss_weight = 1.0

    # ------------------------------
    # Load Dataset
    # ------------------------------
    dataset = TransformerDataset(data_dir='/home/anurizada/Documents/nobari_10_transformer')
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)

    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    val_loader = DataLoader(val_dataset, sampler=val_sampler, batch_size=batch_size)

    # ------------------------------
    # Model, Optimizer, Scheduler
    # ------------------------------
    model = Transformer(
        output_size=2,
        tgt_seq_len=max_mech_size,
        d_model=1024,
        h=32,
        N=6
    ).to(local_rank)
    
    model = DDP(model, device_ids=[local_rank])

    # Initialize WandB after model creation
    if rank == 0:
        # Get architecture parameters from the model
        d_model = model.module.d_model if hasattr(model.module, 'd_model') else 1024
        n_heads = model.module.h if hasattr(model.module, 'h') else 32
        n_layers = model.module.N if hasattr(model.module, 'N') else 6
        
        wandb.init(
            project="distributed-training",
            name=f"d{d_model}_h{n_heads}_bs{batch_size}_lr{lr}",
            config={
                # Architecture
                "d_model": d_model,
                "n_heads": n_heads,
                "n_layers": n_layers,
                
                # Training
                "batch_size": batch_size,
                "lr": lr,
                "clip_loss_weight": clip_loss_weight
            }
        )

    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=200, gamma=0.1)

    best_loss = float("inf")

    for epoch in range(num_epochs):
        # Training
        train_sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0

        with tqdm(total=len(train_loader), desc=f"Rank {rank} Train Epoch {epoch}", leave=False) as pbar:
            for batch in train_loader:
                d1, d2 = batch["decoder_input_first"].to(local_rank), batch["decoder_input_second"].to(local_rank)
                m1, m2 = batch["decoder_mask_first"].to(local_rank), batch["decoder_mask_second"].to(local_rank)
                curve_data = batch["curve_numerical"].to(local_rank)
                adj_data = batch["adjacency"].to(local_rank)
                lbl1 = batch["label_first"].to(local_rank)
                lbl2 = batch["label_second"].to(local_rank)

                optimizer.zero_grad()
                pred1, pred2, curve_emb, adj_emb = model(d1, d2, m1, m2, curve_data, adj_data)

                loss1 = mse_loss(pred1, lbl1)
                loss2 = mse_loss(pred2, lbl2)
                clip_loss = contrastive_loss(curve_emb, adj_emb)
                total_loss = loss1 + loss2 + clip_loss_weight * clip_loss

                total_loss.backward()
                optimizer.step()

                epoch_loss += total_loss.item()

                if rank == 0:
                    wandb.log({
                        "train/mse_loss_first": loss1.item(),
                        "train/mse_loss_second": loss2.item(),
                        "train/clip_loss": clip_loss.item(),
                        "train/total_loss": total_loss.item(),
                        "epoch": epoch,
                    })

                pbar.set_postfix({"Loss": total_loss.item()})
                pbar.update(1)

        scheduler.step()
        avg_train_loss = epoch_loss / len(train_loader)

        # Save best model
        if avg_train_loss < best_loss and rank == 0:
            best_loss = avg_train_loss
            save_best_checkpoint(model, optimizer, epoch, best_loss, batch_size, lr)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            with tqdm(total=len(val_loader), desc=f"Rank {rank} Val Epoch {epoch}", leave=False) as pbar:
                for batch in val_loader:
                    d1, d2 = batch["decoder_input_first"].to(local_rank), batch["decoder_input_second"].to(local_rank)
                    m1, m2 = batch["decoder_mask_first"].to(local_rank), batch["decoder_mask_second"].to(local_rank)
                    curve_data = batch["curve_numerical"].to(local_rank)
                    adj_data = batch["adjacency"].to(local_rank)
                    lbl1 = batch["label_first"].to(local_rank)
                    lbl2 = batch["label_second"].to(local_rank)

                    pred1, pred2, curve_emb, adj_emb = model(d1, d2, m1, m2, curve_data, adj_data)

                    loss1 = mse_loss(pred1, lbl1)
                    loss2 = mse_loss(pred2, lbl2)
                    clip_loss = contrastive_loss(curve_emb, adj_emb)
                    total_loss = loss1 + loss2 + clip_loss_weight * clip_loss

                    val_loss += total_loss.item()

                    if rank == 0:
                        wandb.log({
                            "val/mse_loss_first": loss1.item(),
                            "val/mse_loss_second": loss2.item(),
                            "val/clip_loss": clip_loss.item(),
                            "val/total_loss": total_loss.item(),
                            "epoch": epoch,
                        })

                    pbar.set_postfix({"Val Loss": total_loss.item()})
                    pbar.update(1)

        if rank == 0:
            print(f"[Epoch {epoch}] Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss/len(val_loader):.6f}")

    cleanup_ddp()

if __name__ == "__main__":
    train()