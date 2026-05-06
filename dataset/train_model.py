import os
import sys
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report

from dataset import MELDNumpyDataset, meld_collate_fn
from models import MODEL_REGISTRY

FEATURE_DIR = "extracted features"

TFE_DIMS = {"A": 768, "B": 1024, "C": 1024, "D": 1024, "E": 1024}
AFE_DIMS = {"A": 768, "B": 1024, "C": 1024}
VFE_DIMS = {"A": 2048, "B": 1792, "C": 512}


def parse_feature_dim(cfg_str, dim_map):
    backbone = cfg_str.split("_")[0].split("-")[1]
    return dim_map[backbone]


def build_model(variant, tfe_dim, a_dim, v_dim, b_dim, hidden_dim, num_classes, num_experts):
    kwargs = dict(
        tfe_dim=tfe_dim, a_dim=a_dim, v_dim=v_dim, b_dim=b_dim,
        hidden_dim=hidden_dim, num_classes=num_classes,
        num_experts=num_experts,
    )
    return MODEL_REGISTRY[variant](**kwargs).to(DEVICE)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, loader, loss_fn, return_preds=False):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for t, a, v, b, labels, m in loader:
            t, a, v, b, labels, m = t.to(DEVICE), a.to(DEVICE), v.to(DEVICE), b.to(DEVICE), labels.to(DEVICE), m.to(DEVICE)
            logits, aux_loss = model(t, a, v, b, m)
            ce = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss = ce + 0.01 * aux_loss
            valid_mask = (labels != -100)
            total_loss += loss.item() * valid_mask.sum().item()
            total_tokens += valid_mask.sum().item()
            preds = logits.argmax(dim=-1)
            all_preds.append(preds[valid_mask].cpu().numpy())
            all_labels.append(labels[valid_mask].cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    avg_loss = total_loss / max(total_tokens, 1)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    if return_preds:
        return avg_loss, acc, f1, all_preds, all_labels
    return avg_loss, acc, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["variant1","variant2","variant3","variant4"])
    parser.add_argument("--text_cfg", type=str, default="TFE-C_cfg1")
    parser.add_argument("--audio_cfg", type=str, default="AFE-B_cfg1")
    parser.add_argument("--video_cfg", type=str, default="VFE-A_cfg1")
    parser.add_argument("--data_dir", type=str, default=".")
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_experts", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--log_dir", type=str, default="logs")
    args = parser.parse_args()

    tfe_dim = parse_feature_dim(args.text_cfg, TFE_DIMS)
    afe_dim = parse_feature_dim(args.audio_cfg, AFE_DIMS)
    vfe_dim = parse_feature_dim(args.video_cfg, VFE_DIMS)
    bfe_dim = 336

    train_dir = os.path.join(args.data_dir, FEATURE_DIR, "train")
    dev_dir = os.path.join(args.data_dir, FEATURE_DIR, "dev")
    test_dir = os.path.join(args.data_dir, FEATURE_DIR, "test")

    for d, name in [(train_dir,"train"),(dev_dir,"dev"),(test_dir,"test")]:
        for f in [f"text_{args.text_cfg}.npy", f"audio_{args.audio_cfg}.npy", f"video_{args.video_cfg}.npy"]:
            if not os.path.exists(os.path.join(d, f)):
                print(f"SKIP: missing {f} in {name}"); return

    train_ds = MELDNumpyDataset(train_dir, args.text_cfg, args.audio_cfg, args.video_cfg, bfe_dim)
    dev_ds   = MELDNumpyDataset(dev_dir,   args.text_cfg, args.audio_cfg, args.video_cfg, bfe_dim)
    test_ds  = MELDNumpyDataset(test_dir,  args.text_cfg, args.audio_cfg, args.video_cfg, bfe_dim)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=meld_collate_fn)
    dev_loader   = DataLoader(dev_ds,   batch_size=args.batch_size, shuffle=False, collate_fn=meld_collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, collate_fn=meld_collate_fn)

    model = build_model(args.model, tfe_dim, afe_dim, vfe_dim, bfe_dim, args.d_model, 7, args.num_experts)
    print(f"Model: {args.model} | params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    ce_loss = nn.CrossEntropyLoss(ignore_index=-100)

    run_id = f"{args.model}_{args.text_cfg}_{args.audio_cfg}_{args.video_cfg}_d{args.d_model}_e{args.num_experts}"
    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, f"{run_id}.jsonl")

    best_dev_f1 = 0.0
    best_state = None
    best_epoch = 0
    patience_counter = 0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0
        for t, a, v, b, labels, m in train_loader:
            t, a, v, b, labels, m = t.to(DEVICE), a.to(DEVICE), v.to(DEVICE), b.to(DEVICE), labels.to(DEVICE), m.to(DEVICE)
            logits, aux_loss = model(t, a, v, b, m)
            loss = ce_loss(logits.view(-1, 7), labels.view(-1)) + 0.01 * aux_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        scheduler.step()

        dev_loss, dev_acc, dev_f1 = evaluate(model, dev_loader, ce_loss)

        train_avg = total_loss / max(num_batches, 1)
        line = {"epoch": epoch, "train_loss": round(train_avg, 4),
                "dev_loss": round(dev_loss, 4),
                "dev_acc": round(dev_acc, 4), "dev_f1": round(dev_f1, 4)}
        with open(log_path, "a") as f:
            f.write(json.dumps(line) + "\n")
        print(f"Epoch {epoch:3d} | tr_loss={train_avg:.4f} | de_loss={dev_loss:.4f} | de_acc={dev_acc:.4f} | de_f1={dev_f1:.4f}")

        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch} (patience={args.patience})")
            break

    elapsed = time.time() - start_time

    model.load_state_dict(best_state)
    _, test_acc, test_f1, all_preds, all_labels = evaluate(model, test_loader, ce_loss, return_preds=True)
    cls_report = classification_report(all_labels, all_preds, target_names=["anger","disgust","fear","joy","neutral","sadness","surprise"], zero_division=0)

    summary = {
        "run_id": run_id,
        "model": args.model,
        "text_cfg": args.text_cfg, "audio_cfg": args.audio_cfg, "video_cfg": args.video_cfg,
        "d_model": args.d_model, "num_experts": args.num_experts,
        "tfe_dim": tfe_dim, "afe_dim": afe_dim, "vfe_dim": vfe_dim,
        "best_epoch": best_epoch, "best_dev_f1": round(best_dev_f1, 4),
        "test_acc": round(test_acc, 4), "test_f1": round(test_f1, 4),
        "elapsed_sec": round(elapsed, 1),
    }
    print("=" * 60)
    print(f"BEST EPOCH {best_epoch} | DEV F1 {best_dev_f1:.4f} | TEST ACC {test_acc:.4f} | TEST F1 {test_f1:.4f}")
    print(f"Time: {elapsed:.1f}s")
    print(cls_report)

    summary_path = os.path.join(args.log_dir, f"{run_id}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
