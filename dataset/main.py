import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import accuracy_score, f1_score
from mamba_ssm import Mamba

# ==========================================
# 1. Robust Multimodal Dataset
# ==========================================
class MELDNumpyDataset(Dataset):
    def __init__(self, data_dir, t_cfg, a_cfg, v_cfg, b_cfg):
        self.data_dir = data_dir
        
        # Load Identifiers and Labels
        self.dialogue_ids = np.load(os.path.join(data_dir, "dialogue_ids.npy"))
        self.utterance_ids = np.load(os.path.join(data_dir, "utterance_ids.npy"))
        self.emotion_labels = np.load(os.path.join(data_dir, "emotion_labels.npy"))
        
        self.num_samples = len(self.dialogue_ids)
        
        # Helper to dynamically load features and detect dimensions
        def load_feature(prefix, cfg, fallback_dim):
            path = os.path.join(data_dir, f"{prefix}_{cfg}.npy")
            if os.path.exists(path):
                arr = np.load(path)
                return arr, arr.shape[1]
            else:
                print(f"[WARN] Missing {path}. Using zero tensors of dim {fallback_dim}.")
                return None, fallback_dim

        self.t_feats, self.t_dim = load_feature("text", t_cfg, 1024)
        self.a_feats, self.a_dim = load_feature("audio", a_cfg, 1024)
        self.v_feats, self.v_dim = load_feature("video", v_cfg, 2048)
        self.b_feats, self.b_dim = load_feature("body", b_cfg, 336)
        
        # Group utterance indices by dialogue_id to preserve Temporal sequence
        self.dialogues = {}
        for idx, d_id in enumerate(self.dialogue_ids):
            if d_id not in self.dialogues:
                self.dialogues[d_id] = []
            self.dialogues[d_id].append((self.utterance_ids[idx], idx))
            
        self.valid_dialogue_ids = list(self.dialogues.keys())
        for d_id in self.valid_dialogue_ids:
            self.dialogues[d_id].sort(key=lambda x: x[0]) 

    def __len__(self):
        return len(self.valid_dialogue_ids)

    def _get_tensor(self, feats, indices, dim):
        if feats is not None:
            return torch.tensor(feats[indices], dtype=torch.float32)
        return torch.zeros((len(indices), dim), dtype=torch.float32)

    def __getitem__(self, idx):
        d_id = self.valid_dialogue_ids[idx]
        indices = [x[1] for x in self.dialogues[d_id]]
        
        t = self._get_tensor(self.t_feats, indices, self.t_dim)
        a = self._get_tensor(self.a_feats, indices, self.a_dim)
        v = self._get_tensor(self.v_feats, indices, self.v_dim)
        b = self._get_tensor(self.b_feats, indices, self.b_dim)
        labels = torch.tensor(self.emotion_labels[indices], dtype=torch.long)
        
        return t, a, v, b, labels

def collate_fn(batch):
    t_feats, a_feats, v_feats, b_feats, labels = zip(*batch)
    t = pad_sequence(t_feats, batch_first=True)
    a = pad_sequence(a_feats, batch_first=True)
    v = pad_sequence(v_feats, batch_first=True)
    b = pad_sequence(b_feats, batch_first=True)
    
    # Pad labels with -100 for CrossEntropy ignore_index
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    mask = (labels != -100)
    return t, a, v, b, labels, mask

# ==========================================
# 2. Joint Artificial Empathy Network
# ==========================================
class LatentProjector(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
    def forward(self, x): return self.proj(x)

class DynamicModalityAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)
    def forward(self, modalities):
        stacked = torch.stack(modalities, dim=2)
        weights = F.softmax(self.attention(stacked), dim=2)
        return torch.sum(stacked * weights, dim=2)

class MassiveMicroMoE(nn.Module):
    def __init__(self, d_model, num_experts, top_k):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(d_model, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU()) for _ in range(num_experts)
        ])
        
    def forward(self, x, mask):
        B, Seq, D = x.shape
        x_flat = x.view(-1, D)
        mask_flat = mask.view(-1)
        
        router_logits = self.router(x_flat)
        routing_probs = F.softmax(router_logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(routing_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        output = torch.zeros_like(x_flat)
        for i in range(self.top_k):
            expert_indices = top_k_indices[:, i]
            expert_probs = top_k_probs[:, i].unsqueeze(-1)
            for e_idx, expert in enumerate(self.experts):
                e_mask = (expert_indices == e_idx) & mask_flat
                if e_mask.sum() > 0:
                    output[e_mask] += expert(x_flat[e_mask]) * expert_probs[e_mask]
                    
        valid_routing_probs = routing_probs[mask_flat]
        aux_loss = self.num_experts * torch.sum(valid_routing_probs.mean(dim=0) ** 2) if valid_routing_probs.size(0) > 0 else torch.tensor(0.0)
        return output.view(B, Seq, D), aux_loss

class EmpathyEngine(nn.Module):
    def __init__(self, dims, d_model=512, num_experts=1024, top_k=4, num_classes=7):
        super().__init__()
        self.proj_t = LatentProjector(dims['t'], d_model)
        self.proj_a = LatentProjector(dims['a'], d_model)
        self.proj_v = LatentProjector(dims['v'], d_model)
        self.proj_b = LatentProjector(dims['b'], d_model)
        self.fusion = DynamicModalityAttention(d_model)
        
        # Empathy Engine Components
        self.moe = MassiveMicroMoE(d_model, num_experts, top_k)
        self.mamba = Mamba(d_model=d_model, d_state=32, d_conv=4, expand=2)
        self.future_predictor = nn.Linear(d_model, d_model)
        
        # Cognitive Interpreter Head
        self.interpreter = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, t, a, v, b, mask):
        h_t, h_a, h_v, h_b = self.proj_t(t), self.proj_a(a), self.proj_v(v), self.proj_b(b)
        fused = self.fusion([h_t, h_a, h_v, h_b])
        
        descriptions, aux_loss = self.moe(fused, mask)
        internal_state = self.mamba(descriptions)
        
        predicted_future = self.future_predictor(internal_state)
        logits = self.interpreter(internal_state)
        
        return logits, descriptions, predicted_future, aux_loss

# ==========================================
# 3. Training & Evaluation Pipeline
# ==========================================
def evaluate(model, dataloader, device, criterion_ce):
    model.eval()
    total_loss, all_preds, all_labels = 0, [],[]
    with torch.no_grad():
        for t, a, v, b, labels, mask in dataloader:
            t, a, v, b, labels, mask =[x.to(device) for x in (t, a, v, b, labels, mask)]
            logits, _, _, _ = model(t, a, v, b, mask)
            
            logits_flat = logits.view(-1, 7)
            labels_flat = labels.view(-1)
            
            loss = criterion_ce(logits_flat, labels_flat)
            total_loss += loss.item()
            
            preds = torch.argmax(logits_flat, dim=1)
            valid_idx = (labels_flat != -100)
            
            all_preds.extend(preds[valid_idx].cpu().numpy())
            all_labels.extend(labels_flat[valid_idx].cpu().numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return total_loss / len(dataloader), acc, f1

def main():
    parser = argparse.ArgumentParser(description="Artificial Empathy Engine for MELD")
    parser.add_argument("--data_dir", type=str, default="./", help="Base dir containing train/dev/test folders")
    parser.add_argument("--text", type=str, default="TFE-C_cfg1")
    parser.add_argument("--audio", type=str, default="AFE-B_cfg1")
    parser.add_argument("--video", type=str, default="VFE-A_cfg1")
    parser.add_argument("--body", type=str, default="BFE-A_cfg1")
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--experts", type=int, default=512, help="Number of MoE Experts")
    parser.add_argument("--top_k", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--empathy_w", type=float, default=0.5, help="Weight for predictive future loss")
    parser.add_argument("--moe_w", type=float, default=0.01, help="Weight for load balancing loss")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Initialization | Device: {device} ---")

    train_ds = MELDNumpyDataset(os.path.join(args.data_dir, "train"), args.text, args.audio, args.video, args.body)
    dev_ds = MELDNumpyDataset(os.path.join(args.data_dir, "dev"), args.text, args.audio, args.video, args.body)
    test_ds = MELDNumpyDataset(os.path.join(args.data_dir, "test"), args.text, args.audio, args.video, args.body)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    dims = {'t': train_ds.t_dim, 'a': train_ds.a_dim, 'v': train_ds.v_dim, 'b': train_ds.b_dim}
    print(f"Detected Dimensions -> Text:{dims['t']}, Audio:{dims['a']}, Video:{dims['v']}, Body:{dims['b']}")

    model = EmpathyEngine(dims, d_model=args.d_model, num_experts=args.experts, top_k=args.top_k).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    criterion_ce = nn.CrossEntropyLoss(ignore_index=-100)
    criterion_pred = nn.CosineEmbeddingLoss()

    best_f1 = 0.0
    best_model_path = "best_empathy_model.pth"

    print("\n--- Beginning Joint Training ---")
    for epoch in range(args.epochs):
        model.train()
        train_loss, train_ce, train_emp = 0, 0, 0
        
        for t, a, v, b, labels, mask in train_loader:
            t, a, v, b, labels, mask =[x.to(device) for x in (t, a, v, b, labels, mask)]
            optimizer.zero_grad()
            
            logits, descriptions, predicted_future, aux_loss = model(t, a, v, b, mask)
            
            # 1. Supervised Loss (Interpreter)
            loss_ce = criterion_ce(logits.view(-1, 7), labels.view(-1))
            
            # 2. Empathy Predictive Loss (Anticipating next utterance)
            # Future prediction at time `t` must match description at `t+1`
            preds = predicted_future[:, :-1, :].reshape(-1, args.d_model)
            targets = descriptions[:, 1:, :].reshape(-1, args.d_model).detach()
            
            # Mask out padding transitions
            valid_transitions = mask[:, 1:].reshape(-1)
            preds = preds[valid_transitions]
            targets = targets[valid_transitions]
            
            loss_emp = torch.tensor(0.0).to(device)
            if preds.size(0) > 0:
                cos_labels = torch.ones(preds.size(0)).to(device)
                loss_emp = criterion_pred(preds, targets, cos_labels)
            
            # Total Joint Loss
            loss = loss_ce + (args.empathy_w * loss_emp) + (args.moe_w * aux_loss)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_ce += loss_ce.item()
            train_emp += loss_emp.item()

        # Validation phase
        val_loss, val_acc, val_f1 = evaluate(model, dev_loader, device, criterion_ce)
        
        print(f"Epoch {epoch+1:02d}/{args.epochs} | "
              f"Train Loss: {train_loss/len(train_loader):.4f} (CE:{train_ce/len(train_loader):.4f}, Emp:{train_emp/len(train_loader):.4f}) | "
              f"Dev Loss: {val_loss:.4f} | Dev Acc: {val_acc:.4f} | Dev F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)
            print(f"[*] New Best Model Saved! (F1: {best_f1:.4f})")

    print("\n--- Final Testing Phase ---")
    model.load_state_dict(torch.load(best_model_path))
    test_loss, test_acc, test_f1 = evaluate(model, test_loader, device, criterion_ce)
    print(f"TEST RESULTS | Loss: {test_loss:.4f} | Accuracy: {test_acc:.4f} | Weighted F1: {test_f1:.4f}")

if __name__ == "__main__":
    main()
