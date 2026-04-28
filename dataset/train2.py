import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba

# ==========================================
# Phase 1: Multimodal Subconscious Alignment
# ==========================================
class LatentProjector(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
    def forward(self, x): return self.proj(x)

class DynamicModalityAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)
        
    def forward(self, modalities):
        # Modalities:[Text, Audio, Video, Body] -> (B, Seq, 4, Dim)
        stacked = torch.stack(modalities, dim=2)
        scores = self.attention(stacked) # (B, Seq, 4, 1)
        weights = F.softmax(scores, dim=2)
        return torch.sum(stacked * weights, dim=2)

# ==========================================
# Phase 2: Massive MoE (The Thousands of Experts)
# ==========================================
class MassiveMicroMoE(nn.Module):
    def __init__(self, d_model, num_experts=1024, top_k=4):
        """
        1024 Experts. They do NOT output classes. 
        They output a high-dimensional "description" of reality.
        """
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(d_model, num_experts)
        
        # Lightweight experts to simulate thousands of micro-receptors
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model), 
                nn.SiLU()
            ) for _ in range(num_experts)
        ])
        
    def forward(self, x):
        B, Seq, D = x.shape
        x_flat = x.view(-1, D)
        
        # Route to experts based on micro-temporal features
        router_logits = self.router(x_flat)
        routing_probs = F.softmax(router_logits, dim=-1)
        
        top_k_probs, top_k_indices = torch.topk(routing_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        output = torch.zeros_like(x_flat)
        
        for i in range(self.top_k):
            expert_indices = top_k_indices[:, i]
            expert_probs = top_k_probs[:, i].unsqueeze(-1)
            
            for e_idx, expert in enumerate(self.experts):
                mask = (expert_indices == e_idx)
                if mask.sum() > 0:
                    output[mask] += expert(x_flat[mask]) * expert_probs[mask]
                    
        # Load balancing auxiliary loss to keep all 1024 experts alive
        density = routing_probs.mean(dim=0)
        aux_loss = self.num_experts * torch.sum(density * density) 
        
        return output.view(B, Seq, D), aux_loss

# ==========================================
# Phase 3: The Anticipation Engine (Mamba)
# ==========================================
class ArtificialEmpathyEngine(nn.Module):
    def __init__(self, t_dim=1024, a_dim=1024, v_dim=2048, b_dim=336, d_model=512):
        super().__init__()
        self.proj_t = LatentProjector(t_dim, d_model)
        self.proj_a = LatentProjector(a_dim, d_model)
        self.proj_v = LatentProjector(v_dim, d_model)
        self.proj_b = LatentProjector(b_dim, d_model)
        
        self.fusion = DynamicModalityAttention(d_model)
        
        # The MoE creates the rich descriptor state
        self.moe_reception = MassiveMicroMoE(d_model=d_model, num_experts=1024, top_k=4)
        
        # Unidirectional Mamba: It can only see the past, forcing it to predict the future
        self.temporal_mamba = Mamba(d_model=d_model, d_state=32, d_conv=4, expand=2)
        
        # Predicts the MoE descriptor of the NEXT utterance
        self.future_predictor = nn.Linear(d_model, d_model)

    def forward(self, t, a, v, b):
        # 1. Map reality to the internal continuous state
        h_t, h_a, h_v, h_b = self.proj_t(t), self.proj_a(a), self.proj_v(v), self.proj_b(b)
        fused_reality = self.fusion([h_t, h_a, h_v, h_b])
        
        # 2. Generate the "Human-Like Description" via 1024 Experts
        descriptions, aux_loss = self.moe_reception(fused_reality)
        
        # 3. Track the emotional flow through time
        internal_state = self.temporal_mamba(descriptions)
        
        # 4. Predict the future emotional state
        predicted_future = self.future_predictor(internal_state)
        
        return descriptions, predicted_future, aux_loss

# ==========================================
# The Training Philosophy: Self-Supervised Empathy
# ==========================================
def train_artificial_empathy():
    model = ArtificialEmpathyEngine().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # We DO NOT use MELD CrossEntropy here. We use Cosine Embedding Loss.
    # The model learns to make its "predicted_future" match the actual "description" of the next utterance.
    predictive_loss_fn = nn.CosineEmbeddingLoss()
    
    # Simulating a batch of 1 conversation with 15 utterances
    dummy_t = torch.randn(1, 15, 1024).cuda()
    dummy_a = torch.randn(1, 15, 1024).cuda()
    dummy_v = torch.randn(1, 15, 2048).cuda()
    dummy_b = torch.randn(1, 15, 336).cuda()
    
    optimizer.zero_grad()
    
    actual_descriptions, predicted_futures, aux_loss = model(dummy_t, dummy_a, dummy_v, dummy_b)
    
    # ALIGNMENT: predicted future of time `t` should match actual description of time `t+1`
    predictions = predicted_futures[:, :-1, :].reshape(-1, 512)
    targets = actual_descriptions[:, 1:, :].reshape(-1, 512).detach() # Detach to prevent collapsing targets
    
    labels = torch.ones(predictions.size(0)).cuda() # 1 means they should be highly similar
    
    # The true loss function of emotion: predicting human reactions
    loss = predictive_loss_fn(predictions, targets, labels) + (0.01 * aux_loss)
    
    loss.backward()
    optimizer.step()
    print("The machine has learned to anticipate one step of human emotion.")

train_artificial_empathy()
