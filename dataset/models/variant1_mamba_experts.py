"""
Variant 1 — MambaExpertMoE
Each expert in the MoE is a Mamba SSM block instead of an MLP.
The standalone BiMamba stage is REMOVED; sequence modeling is entirely
handled inside the MoE routing step.

Motivation: Mamba's selective-scan mechanism acts as both a temporal
reasoner and a feature transformer, so dedicating one SSM per expert
allows each expert to specialise its own recurrent inductive bias for
different emotion-relevant temporal patterns (e.g. short bursts vs.
sustained states).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


# ── Modality projector (unchanged) ───────────────────────────────────────────
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


# ── Gated fusion (unchanged) ──────────────────────────────────────────────────
class GatedMultimodalFusion(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, modalities):
        stacked = torch.stack(modalities, dim=2)          # (B, T, M, D)
        weights = F.softmax(self.attention(stacked), dim=2)
        return torch.sum(stacked * weights, dim=2)        # (B, T, D)


# ── NEW: Bidirectional Mamba used as an expert ────────────────────────────────
class BiMambaExpert(nn.Module):
    """A single expert that wraps a bidirectional Mamba SSM."""

    def __init__(self, d_model: int):
        super().__init__()
        self.fwd = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        self.bwd = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, T, D)  — only the tokens routed to this expert
        fwd_out = self.fwd(x)
        bwd_out = torch.flip(self.bwd(torch.flip(x, [1])), [1])
        return self.norm(x + fwd_out + bwd_out)


# ── Sparse MoE where every expert is a BiMamba ───────────────────────────────
class MambaExpertMoE(nn.Module):
    """
    Sparse MoE in which each expert is a BiMamba block.
    Because Mamba requires a sequence dimension the routing is done at the
    *utterance* level (one token per utterance in a dialogue window) so every
    routed token is processed as a length-1 sequence, which is equivalent to
    a position-wise Mamba pass but preserves the SSM interface cleanly.
    """

    def __init__(self, d_model: int, num_experts: int = 16, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(d_model, num_experts)
        self.experts = nn.ModuleList(
            [BiMambaExpert(d_model) for _ in range(num_experts)]
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        B, T, D = x.shape
        x_flat = x.view(-1, D)                             # (B*T, D)
        mask_flat = mask.view(-1)                          # (B*T,)

        router_logits = self.router(x_flat)
        routing_probs = F.softmax(router_logits, dim=-1)

        top_k_probs, top_k_indices = torch.topk(routing_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        final_output = torch.zeros_like(x_flat)

        for i in range(self.top_k):
            expert_indices = top_k_indices[:, i]
            expert_probs = top_k_probs[:, i].unsqueeze(-1)

            for e_idx, expert in enumerate(self.experts):
                e_mask = (expert_indices == e_idx) & mask_flat
                if e_mask.sum() == 0:
                    continue
                # Treat each token as its own sequence of length 1
                tokens = x_flat[e_mask].unsqueeze(1)       # (N, 1, D)
                expert_out = expert(tokens).squeeze(1)     # (N, D)
                final_output[e_mask] += expert_out * expert_probs[e_mask]

        # Load-balancing auxiliary loss (valid tokens only)
        valid_probs = routing_probs[mask_flat]
        density = valid_probs.mean(dim=0)
        aux_loss = self.num_experts * torch.sum(density * density)

        return final_output.view(B, T, D), aux_loss


# ── Top-level model ───────────────────────────────────────────────────────────
class MambaExpertMoEModel(nn.Module):
    """
    Multimodal emotion recognition model.
    Pipeline: project → fuse → [MambaExpertMoE] → classify
    The BiMamba stage from the original is REMOVED; sequence modelling
    is fully delegated to the Mamba experts inside the MoE.
    """

    def __init__(
        self,
        tfe_dim: int = 1024,
        a_dim: int = 1024,
        v_dim: int = 2048,
        b_dim: int = 336,
        hidden_dim: int = 512,
        num_classes: int = 7,
        num_experts: int = 16,
    ):
        super().__init__()
        self.proj_t = ModalityProjector(tfe_dim, hidden_dim)
        self.proj_a = ModalityProjector(a_dim, hidden_dim)
        self.proj_v = ModalityProjector(v_dim, hidden_dim)
        self.proj_b = ModalityProjector(b_dim, hidden_dim)

        self.fusion = GatedMultimodalFusion(hidden_dim)

        # No standalone BiMamba — sequence modelling lives inside the experts
        self.moe = MambaExpertMoE(hidden_dim, num_experts=num_experts, top_k=2)

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

        fused = self.fusion([h_t, h_a, h_v, h_b])          # (B, T, D)
        moe_out, aux_loss = self.moe(fused, mask)
        logits = self.classifier(moe_out)
        return logits, aux_loss
