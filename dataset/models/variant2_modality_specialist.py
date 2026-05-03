"""
Variant 2 — ModalitySpecialistMoE
Each expert is assigned to exactly ONE modality (text, audio, video, bio).
Since we have 4 modalities, expert slots are partitioned into 4 groups.
Routing is soft but expert groups are hard-constrained: each group of
experts only sees features from its designated modality stream BEFORE fusion,
then a cross-modal head merges the specialist outputs.

Motivation: different modalities have fundamentally different statistical
properties (discrete vs. continuous, frame-level vs. utterance-level).
Forcing a shared expert set to handle all modalities may dilute
specialisation. Giving each modality its own expert pool and combining
afterwards mirrors the way human cognition integrates separate perceptual
streams.

Architecture:
  text ──► TextExpertGroup  ──┐
  audio ──► AudioExpertGroup ─┤
  video ──► VideoExpertGroup ─┼──► CrossModalFusion ──► BiMamba ──► Classify
  bio ──► BioExpertGroup   ──┘
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


# ── Per-modality sparse expert group ─────────────────────────────────────────
class ModalityExpertGroup(nn.Module):
    """
    A small sparse MoE that only processes one modality's token stream.
    Each expert is a two-layer FFN (same width as d_model).
    """

    def __init__(self, d_model: int, num_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(d_model, num_experts)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, d_model * 2),
                    nn.GELU(),
                    nn.Linear(d_model * 2, d_model),
                )
                for _ in range(num_experts)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        """
        x    : (B, T, D)
        mask : (B, T) bool — True for valid utterance positions
        returns: (B, T, D), scalar aux_loss
        """
        B, T, D = x.shape
        x_flat = x.view(-1, D)
        mask_flat = mask.view(-1)

        probs = F.softmax(self.router(x_flat), dim=-1)
        top_k_probs, top_k_idx = torch.topk(probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(-1, keepdim=True)

        out = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            eidx = top_k_idx[:, k]
            eprob = top_k_probs[:, k].unsqueeze(-1)
            for e, expert in enumerate(self.experts):
                sel = (eidx == e) & mask_flat
                if sel.sum() > 0:
                    out[sel] += expert(x_flat[sel]) * eprob[sel]

        # Load-balancing loss on valid tokens
        valid_probs = probs[mask_flat]
        density = valid_probs.mean(0)
        aux_loss = self.num_experts * torch.sum(density * density)

        return self.norm(x + out.view(B, T, D)), aux_loss


# ── Cross-modal fusion after specialist processing ────────────────────────────
class CrossModalAttentionFusion(nn.Module):
    """
    Multi-head cross-attention that fuses 4 modality streams.
    Queries come from the concatenation of all streams; keys/values
    are each individual stream in turn (like a cross-attention aggregator).
    """

    def __init__(self, d_model: int, num_heads: int = 8):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.proj = nn.Linear(d_model * 4, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, streams):
        # streams: list of 4 tensors, each (B, T, D)
        concat = torch.cat(streams, dim=-1)    # (B, T, 4D)
        query = self.proj(concat)              # (B, T, D)
        # Self-attend over the aggregated stream
        fused, _ = self.mha(query, query, query)
        return self.norm(query + fused)


# ── Top-level model ───────────────────────────────────────────────────────────
class ModalitySpecialistMoEModel(nn.Module):
    """
    Each modality stream has its OWN sparse expert group.
    Experts within a group specialise on that modality's token distribution.
    Outputs are fused via cross-modal attention, then a BiMamba adds
    dialogue-level context before classification.
    """

    def __init__(
        self,
        tfe_dim: int = 1024,
        a_dim: int = 1024,
        v_dim: int = 2048,
        b_dim: int = 336,
        hidden_dim: int = 512,
        num_classes: int = 7,
        experts_per_modality: int = 4,
        top_k: int = 2,
    ):
        super().__init__()
        # Projectors
        self.proj_t = ModalityProjector(tfe_dim, hidden_dim)
        self.proj_a = ModalityProjector(a_dim, hidden_dim)
        self.proj_v = ModalityProjector(v_dim, hidden_dim)
        self.proj_b = ModalityProjector(b_dim, hidden_dim)

        # One specialist MoE per modality
        self.moe_t = ModalityExpertGroup(hidden_dim, experts_per_modality, top_k)
        self.moe_a = ModalityExpertGroup(hidden_dim, experts_per_modality, top_k)
        self.moe_v = ModalityExpertGroup(hidden_dim, experts_per_modality, top_k)
        self.moe_b = ModalityExpertGroup(hidden_dim, experts_per_modality, top_k)

        # Cross-modal fusion + sequence modelling
        self.fusion = CrossModalAttentionFusion(hidden_dim)
        self.mamba = BiMambaBlock(hidden_dim)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, t, a, v, b, mask):
        # Project each modality
        h_t = self.proj_t(t)
        h_a = self.proj_a(a)
        h_v = self.proj_v(v)
        h_b = self.proj_b(b)

        # Specialist MoE per modality
        h_t, loss_t = self.moe_t(h_t, mask)
        h_a, loss_a = self.moe_a(h_a, mask)
        h_v, loss_v = self.moe_v(h_v, mask)
        h_b, loss_b = self.moe_b(h_b, mask)
        aux_loss = loss_t + loss_a + loss_v + loss_b

        # Cross-modal fusion
        fused = self.fusion([h_t, h_a, h_v, h_b])   # (B, T, D)

        # Dialogue-context modelling
        context = self.mamba(fused)

        logits = self.classifier(context)
        return logits, aux_loss
