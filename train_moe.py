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
import numpy as np
from collections import defaultdict

from dataset import TransformerDataset
from moe_model import Transformer, MoEFeedForwardBlock  # Ensure MoEFeedForwardBlock is imported

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

def moe_balancing_loss(model, weight=0.01):
    """Calculate load balancing loss across all MoE layers"""
    total_loss = 0.0
    num_layers = 0
    
    for module in model.modules():
        if isinstance(module, MoEFeedForwardBlock):
            expert_usage = module.get_expert_usage('current')
            if expert_usage is not None:
                importance = expert_usage
                load = (expert_usage > 0).float()
                
                importance_loss = importance.std() / (importance.mean() + 1e-6)
                load_loss = load.std() / (load.mean() + 1e-6)
                
                total_loss += (importance_loss + load_loss)
                num_layers += 1
    
    return (total_loss / max(num_layers, 1)) * weight

def setup_ddp():
    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))

def cleanup_ddp():
    dist.destroy_process_group()

def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()

def get_rank():
    return dist.get_rank() if is_dist_avail_and_initialized() else 0

def save_best_checkpoint(model, optimizer, epoch, best_loss, batch_size, lr, save_dir="weights", checkpoint_name=None):
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, checkpoint_name or f"moe_bs_{batch_size}_lr_{lr}_best.pth")
    torch.save({
        'model_state_dict': model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'best_loss': best_loss,
        'architecture': {
            'd_model': 1024,
            'n_heads': 32,
            'n_layers': 6
        },
        'training_config': {
            'batch_size': batch_size,
            'learning_rate': lr
        },
        'moe_config': {
            'num_experts': 8,
            'experts_used': 2
        }
    }, checkpoint_path)
    print(f"[Rank {get_rank()}] Saved best model at {checkpoint_path} with loss {best_loss:.6f}")

def log_expert_usage(model, epoch, prefix="train"):
    """Log expert usage statistics to wandb"""
    expert_stats = defaultdict(list)
    
    for name, module in model.named_modules():
        if isinstance(module, MoEFeedForwardBlock):
            usage = module.get_expert_usage('current')
            if usage is not None:
                usage = usage.cpu().numpy()
                for expert_idx, percent in enumerate(usage):
                    expert_stats[f"{prefix}/expert_{expert_idx}_usage"].append(percent)
                
                expert_stats[f"{prefix}/expert_usage_mean"].append(usage.mean())
                expert_stats[f"{prefix}/expert_usage_std"].append(usage.std())
                expert_stats[f"{prefix}/expert_usage_min"].append(usage.min())
                expert_stats[f"{prefix}/expert_usage_max"].append(usage.max())
    
    if get_rank() == 0 and expert_stats:
        wandb.log({
            **{k: np.mean(v) for k, v in expert_stats.items()},
            "epoch": epoch
        })

def train():
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    setup_ddp()
    torch.set_float32_matmul_precision('medium')

    # Hyperparameters
    batch_size = 512
    max_mech_size = 10
    num_epochs = 300
    lr = 1e-4
    clip_loss_weight = 1.0
    moe_loss_weight = 0.01
    
    # Model Architecture Parameters
    d_model = 1024  # Dimension of the model
    n_heads = 32    # Number of attention heads
    n_layers = 6    # Number of layers
    num_experts = 8  # Number of MoE experts
    experts_used = 2 # Experts used per forward pass

    def get_checkpoint_name():
        return f"d{d_model}_h{n_heads}_e{num_experts}_bs{batch_size}_lr{lr}_best.pth"

    if rank == 0:
        wandb.init(
            project="distributed-training",
            name=get_checkpoint_name().replace('.pth', ''),
            config={
                # Architecture
                "d_model": d_model,
                "n_heads": n_heads,
                "n_layers": n_layers,
                
                # Training
                "batch_size": batch_size,
                "lr": lr,
                "clip_loss_weight": clip_loss_weight,
                "moe_loss_weight": moe_loss_weight,
                
                # MoE specific
                "num_experts": num_experts,
                "experts_used": experts_used
            }
        )

    # Dataset and DataLoaders
    dataset = TransformerDataset(data_dir='/home/anurizada/Documents/nobari_10_transformer')
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)

    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    val_loader = DataLoader(val_dataset, sampler=val_sampler, batch_size=batch_size)

    # Model setup
    model = Transformer(
        tgt_seq_len=max_mech_size,
        output_size=2,
        d_model=d_model,
        h=n_heads,
        N=n_layers
    ).to(local_rank)
    
    model = DDP(model, device_ids=[local_rank])

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
                moe_loss = moe_balancing_loss(model.module, weight=moe_loss_weight)
                
                total_loss = loss1 + loss2 + clip_loss_weight * clip_loss + moe_loss

                total_loss.backward()
                optimizer.step()

                epoch_loss += total_loss.item()

                if rank == 0:
                    wandb.log({
                        "train/mse_loss_first": loss1.item(),
                        "train/mse_loss_second": loss2.item(),
                        "train/clip_loss": clip_loss.item(),
                        "train/moe_balance_loss": moe_loss.item(),
                        "train/total_loss": total_loss.item(),
                        "epoch": epoch,
                    })

                pbar.set_postfix({
                    "Loss": total_loss.item(),
                    "MoE": moe_loss.item()
                })
                pbar.update(1)

        scheduler.step()
        avg_train_loss = epoch_loss / len(train_loader)

        # Log expert usage
        log_expert_usage(model.module, epoch, prefix="train")

        # Save best model
        if avg_train_loss < best_loss and rank == 0:
            best_loss = avg_train_loss
            save_best_checkpoint(
                model, optimizer, epoch, best_loss,
                batch_size, lr, checkpoint_name=get_checkpoint_name()
            )

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

        # Log validation expert usage
        log_expert_usage(model.module, epoch, prefix="val")

        if rank == 0:
            print(f"[Epoch {epoch}] Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss/len(val_loader):.6f}")

    cleanup_ddp()

if __name__ == "__main__":
    train()