import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba

class ModalityProjector(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.2):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    def forward(self, x): return self.proj(x)

class GatedMultimodalFusion(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
    def forward(self, modalities):
        stacked = torch.stack(modalities, dim=2)
        weights = F.softmax(self.attention(stacked), dim=2)
        return torch.sum(stacked * weights, dim=2)

class BiMambaBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.mamba_fwd = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        self.mamba_bwd = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x):
        fwd = self.mamba_fwd(x)
        x_flipped = torch.flip(x, dims=[1])
        bwd = torch.flip(self.mamba_bwd(x_flipped), dims=[1])
        return self.norm(x + fwd + bwd)

class SparseMoE(nn.Module):
    def __init__(self, d_model, num_experts=16, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(d_model, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, d_model)) 
            for _ in range(num_experts)
        ])
        
    def forward(self, x, mask):
        B, Seq, D = x.shape
        x_flat = x.view(-1, D)
        mask_flat = mask.view(-1)
        
        router_logits = self.router(x_flat)
        routing_probs = F.softmax(router_logits, dim=-1)
        
        top_k_probs, top_k_indices = torch.topk(routing_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        final_output = torch.zeros_like(x_flat)
        
        for i in range(self.top_k):
            expert_indices = top_k_indices[:, i]
            expert_probs = top_k_probs[:, i].unsqueeze(-1)
            
            for e_idx, expert in enumerate(self.experts):
                # Only process valid tokens mapped to this expert
                e_mask = (expert_indices == e_idx) & mask_flat
                if e_mask.sum() > 0:
                    expert_out = expert(x_flat[e_mask])
                    final_output[e_mask] += expert_out * expert_probs[e_mask]
                    
        # Masked Auxiliary Loss (Load balancing only on actual utterance tokens)
        valid_routing_probs = routing_probs[mask_flat]
        density = valid_routing_probs.mean(dim=0)
        aux_loss = self.num_experts * torch.sum(density * density) 
        
        return final_output.view(B, Seq, D), aux_loss

class MultimodalMambaMoE(nn.Module):
    def __init__(self, tfe_dim=1024, a_dim=1024, v_dim=2048, b_dim=336, hidden_dim=512, num_classes=7):
        super().__init__()
        self.proj_t = ModalityProjector(tfe_dim, hidden_dim)
        self.proj_a = ModalityProjector(a_dim, hidden_dim)
        self.proj_v = ModalityProjector(v_dim, hidden_dim)
        self.proj_b = ModalityProjector(b_dim, hidden_dim)
        self.fusion = GatedMultimodalFusion(hidden_dim)
        self.mamba = BiMambaBlock(hidden_dim)
        self.moe = SparseMoE(hidden_dim, num_experts=16, top_k=2)
        self.interpreter = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, t, a, v, b, mask):
        h_t, h_a, h_v, h_b = self.proj_t(t), self.proj_a(a), self.proj_v(v), self.proj_b(b)
        fused = self.fusion([h_t, h_a, h_v, h_b])
        context_state = self.mamba(fused)
        moe_out, aux_loss = self.moe(context_state, mask)
        logits = self.interpreter(moe_out)
        return logits, aux_loss
