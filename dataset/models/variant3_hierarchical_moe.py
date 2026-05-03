"""
Variant 3 — HierarchicalMoE
Two-level routing: a coarse router assigns tokens to one of K emotion-cluster
"super-experts"; each super-expert is itself a small MoE with its own experts.
This mirrors the hierarchical structure of emotional categories (e.g.
positive vs negative vs neutral at level 1, then fine-grained at level 2).

Motivation: Flat MoE routing forces a single linear router to partition a
high-dimensional affective space in one shot. A hierarchical router can first
separate macro-level valence / arousal regions, then let sub-routers handle
fine distinctions within each region — potentially improving coverage for
minority emotion classes that tend to be confusable with neighbours.

Architecture:
  project → gated-fuse → BiMamba → HierarchicalMoE (2-level) → classify
                                        ↑
                                  [level-1 router]
                               super0   super1  super2  super3
                               [local MoE with 4 experts each]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


# ── Shared utilities ──────────────────────────────────────────────────────────
class ModalityProjector(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.2):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.proj(x)


class GatedMultimodalFusion(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, modalities):
        stacked = torch.stack(modalities, dim=2)
        weights = F.softmax(self.attention(stacked), dim=2)
        return torch.sum(stacked * weights, dim=2)


class BiMambaBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.fwd = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        self.bwd = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        fwd = self.fwd(x)
        bwd = torch.flip(self.bwd(torch.flip(x, [1])), [1])
        return self.norm(x + fwd + bwd)


# ── Level-2: local MoE inside one super-expert ───────────────────────────────
class LocalMoE(nn.Module):
    """Small dense MoE (no masking — always activated within a super-expert)."""

    def __init__(self, d_model: int, num_local_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.num_experts = num_local_experts
        self.top_k = top_k
        self.router = nn.Linear(d_model, num_local_experts)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, d_model * 2),
                    nn.GELU(),
                    nn.Linear(d_model * 2, d_model),
                )
                for _ in range(num_local_experts)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x_flat: torch.Tensor):
        # x_flat: (N, D) — only tokens routed here
        probs = F.softmax(self.router(x_flat), dim=-1)
        top_k_probs, top_k_idx = torch.topk(probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(-1, keepdim=True)

        out = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            eidx = top_k_idx[:, k]
            eprob = top_k_probs[:, k].unsqueeze(-1)
            for e, expert in enumerate(self.experts):
                sel = eidx == e
                if sel.sum() > 0:
                    out[sel] += expert(x_flat[sel]) * eprob[sel]

        density = probs.mean(0)
        aux_loss = self.num_experts * torch.sum(density * density)
        return self.norm(x_flat + out), aux_loss


# ── Level-1: hierarchical MoE ────────────────────────────────────────────────
class HierarchicalMoE(nn.Module):
    """
    Two-level MoE.
    Level 1: coarse router assigns each token to one of `num_super` super-experts.
    Level 2: each super-expert contains its own LocalMoE with fine-grained experts.
    """

    def __init__(
        self,
        d_model: int,
        num_super: int = 4,
        num_local: int = 4,
        top_k_super: int = 1,
        top_k_local: int = 2,
    ):
        super().__init__()
        self.num_super = num_super
        self.top_k_super = top_k_super
        self.coarse_router = nn.Linear(d_model, num_super)
        self.super_experts = nn.ModuleList(
            [LocalMoE(d_model, num_local, top_k_local) for _ in range(num_super)]
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        B, T, D = x.shape
        x_flat = x.view(-1, D)
        mask_flat = mask.view(-1)

        coarse_probs = F.softmax(self.coarse_router(x_flat), dim=-1)
        top_k_probs, top_k_idx = torch.topk(coarse_probs, self.top_k_super, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(-1, keepdim=True)

        out = torch.zeros_like(x_flat)
        total_aux = torch.tensor(0.0, device=x.device)

        for k in range(self.top_k_super):
            super_idx = top_k_idx[:, k]
            super_prob = top_k_probs[:, k].unsqueeze(-1)
            for s, super_expert in enumerate(self.super_experts):
                sel = (super_idx == s) & mask_flat
                if sel.sum() == 0:
                    continue
                local_out, local_aux = super_expert(x_flat[sel])
                out[sel] += local_out * super_prob[sel]
                total_aux = total_aux + local_aux

        # Level-1 load-balancing
        valid_coarse = coarse_probs[mask_flat]
        d1 = valid_coarse.mean(0)
        total_aux = total_aux + self.num_super * torch.sum(d1 * d1)

        return out.view(B, T, D), total_aux


# ── Top-level model ───────────────────────────────────────────────────────────
class HierarchicalMoEModel(nn.Module):
    """
    project → gated-fuse → BiMamba → HierarchicalMoE → classify
    """

    def __init__(
        self,
        tfe_dim: int = 1024,
        a_dim: int = 1024,
        v_dim: int = 2048,
        b_dim: int = 336,
        hidden_dim: int = 512,
        num_classes: int = 7,
    ):
        super().__init__()
        self.proj_t = ModalityProjector(tfe_dim, hidden_dim)
        self.proj_a = ModalityProjector(a_dim, hidden_dim)
        self.proj_v = ModalityProjector(v_dim, hidden_dim)
        self.proj_b = ModalityProjector(b_dim, hidden_dim)

        self.fusion = GatedMultimodalFusion(hidden_dim)
        self.mamba = BiMambaBlock(hidden_dim)

        # 4 super-experts × 4 local experts = 16 leaves total (same capacity
        # as the flat 16-expert baseline)
        self.moe = HierarchicalMoE(
            d_model=hidden_dim,
            num_super=4,
            num_local=4,
            top_k_super=1,
            top_k_local=2,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, t, a, v, b, mask):
        h_t = self.proj_t(t)
        h_a = self.proj_a(a)
        h_v = self.proj_v(v)
        h_b = self.proj_b(b)

        fused = self.fusion([h_t, h_a, h_v, h_b])
        ctx = self.mamba(fused)
        moe_out, aux_loss = self.moe(ctx, mask)
        logits = self.classifier(moe_out)
        return logits, aux_loss
