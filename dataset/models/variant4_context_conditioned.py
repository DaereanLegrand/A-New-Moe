"""
Variant 4 — ContextConditionedMoE
The MoE router is conditioned not only on the current token's representation
but also on a *dialogue-level summary* vector produced by a lightweight
recurrent summariser (a Mamba SSM).  This makes routing decisions
context-aware: the same utterance features may be routed differently
depending on what has been said earlier in the conversation.

Motivation: Emotion is highly context-dependent in dialogue (e.g. "fine" is
neutral after a greeting but sarcastic after a complaint).  Standard MoE
routers only look at the current token, ignoring conversational history.
Conditioning the router on a dialogue-state summary injects temporal
context into the gating decision itself, not just into the expert outputs.

Architecture:
  project → gated-fuse
                ↓
           ContextSummariser (Mamba) → dialogue_state_vector
                ↓                              ↓
           [token representation] ─► [ContextConditionedRouter]
                                              ↓
                                    SparseMoE (FFN experts)
                                              ↓
                                         classifier
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


# ── Dialogue context summariser ───────────────────────────────────────────────
class DialogueSummariser(nn.Module):
    """
    A causal (forward-only) Mamba that reads the token sequence and produces
    a per-position cumulative dialogue state vector.  This vector encodes
    "everything said so far" at each position.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.mamba = Mamba(d_model=d_model, d_state=32, d_conv=4, expand=2)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D) — returns (B, T, D) dialogue-state per position
        return self.norm(self.mamba(x))


# ── Context-conditioned router ────────────────────────────────────────────────
class ContextConditionedRouter(nn.Module):
    """
    Router that concatenates [token_repr ; dialogue_state] before projecting
    to expert logits.  The dialogue state is computed by the summariser above.
    """

    def __init__(self, d_model: int, num_experts: int):
        super().__init__()
        # Input is 2*d_model (token + context), output is num_experts
        self.router = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_experts),
        )

    def forward(self, x_flat: torch.Tensor, ctx_flat: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([x_flat, ctx_flat], dim=-1)
        return self.router(combined)


# ── Context-Conditioned sparse MoE ───────────────────────────────────────────
class ContextConditionedMoE(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_experts: int = 16,
        top_k: int = 2,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.summariser = DialogueSummariser(d_model)
        self.router = ContextConditionedRouter(d_model, num_experts)
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

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        B, T, D = x.shape

        # Compute causal dialogue state
        ctx = self.summariser(x)                           # (B, T, D)

        x_flat = x.view(-1, D)
        ctx_flat = ctx.view(-1, D)
        mask_flat = mask.view(-1)

        router_logits = self.router(x_flat, ctx_flat)
        routing_probs = F.softmax(router_logits, dim=-1)

        top_k_probs, top_k_idx = torch.topk(routing_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(-1, keepdim=True)

        out = torch.zeros_like(x_flat)

        for k in range(self.top_k):
            eidx = top_k_idx[:, k]
            eprob = top_k_probs[:, k].unsqueeze(-1)
            for e, expert in enumerate(self.experts):
                sel = (eidx == e) & mask_flat
                if sel.sum() > 0:
                    out[sel] += expert(x_flat[sel]) * eprob[sel]

        # Load-balancing aux loss on valid tokens
        valid_probs = routing_probs[mask_flat]
        density = valid_probs.mean(0)
        aux_loss = self.num_experts * torch.sum(density * density)

        return out.view(B, T, D), aux_loss


# ── Top-level model ───────────────────────────────────────────────────────────
class ContextConditionedMoEModel(nn.Module):
    """
    project → gated-fuse → ContextConditionedMoE (routing sees history) → classify
    No separate BiMamba; the dialogue summariser inside the MoE provides
    sequential context to the router, while the expert outputs are still
    independently computed per-token.
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

        # Context-conditioned MoE (includes its own Mamba summariser internally)
        self.moe = ContextConditionedMoE(hidden_dim, num_experts=16, top_k=2)

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
        moe_out, aux_loss = self.moe(fused, mask)
        logits = self.classifier(moe_out)
        return logits, aux_loss
