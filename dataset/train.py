import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import MELDNumpyDataset, meld_collate_fn
from model import MultimodalMambaMoE

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # ==========================
    # 1. Setup Configurations
    # ==========================
    # Pick your strongest State-of-the-Art extracted features:
    TEXT_CFG = "TFE-C_cfg1"   # DeBERTa-v3-large: 1024
    AUDIO_CFG = "AFE-B_cfg1"  # WavLM-large: 1024
    VIDEO_CFG = "VFE-A_cfg1"  # ResNet-50: 2048
    
    dims = {'t': 1024, 'a': 1024, 'v': 2048, 'b': 336}

    # ==========================
    # 2. Data Loaders
    # ==========================
    train_dataset = MELDNumpyDataset(data_dir="./train", text_cfg=TEXT_CFG, audio_cfg=AUDIO_CFG, video_cfg=VIDEO_CFG)
    dev_dataset = MELDNumpyDataset(data_dir="./dev", text_cfg=TEXT_CFG, audio_cfg=AUDIO_CFG, video_cfg=VIDEO_CFG)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=meld_collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=16, shuffle=False, collate_fn=meld_collate_fn)

    # ==========================
    # 3. Model & Optimizers
    # ==========================
    model = MultimodalMambaMoE(
        tfe_dim=dims['t'], a_dim=dims['a'], v_dim=dims['v'], b_dim=dims['b'],
        hidden_dim=512, num_classes=7
    ).to(device)

    # Note: ignore_index=-100 prevents padded zeros from updating gradients!
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    
    MOE_LOSS_WEIGHT = 0.01

    # ==========================
    # 4. Training Loop
    # ==========================
    epochs = 10
    for epoch in range(epochs):
        model.train()
        total_loss, total_correct, total_samples = 0, 0, 0
        
        for batch_idx, (t, a, v, b, labels, mask) in enumerate(train_loader):
            t, a, v, b = t.to(device), a.to(device), v.to(device), b.to(device)
            labels, mask = labels.to(device), mask.to(device)
            
            optimizer.zero_grad()
            
            # Forward Pass
            logits, aux_loss = model(t, a, v, b, mask)
            
            # Flatten to compute Cross Entropy (Batch*Seq_Len, Num_Classes)
            logits_flat = logits.view(-1, 7)
            labels_flat = labels.view(-1)
            
            # CE loss handles the mask natively via ignore_index=-100
            ce_loss = criterion(logits_flat, labels_flat)
            loss = ce_loss + (MOE_LOSS_WEIGHT * aux_loss)
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Metrics (only on valid tokens)
            preds = torch.argmax(logits_flat, dim=1)
            valid_idx = (labels_flat != -100)
            total_correct += (preds[valid_idx] == labels_flat[valid_idx]).sum().item()
            total_samples += valid_idx.sum().item()
            total_loss += loss.item()
            
        train_acc = total_correct / total_samples
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.4f}")

if __name__ == "__main__":
    train()
