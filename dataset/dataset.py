import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class MELDNumpyDataset(Dataset):
    def __init__(self, data_dir, text_cfg="TFE-C_cfg1", audio_cfg="AFE-B_cfg1", video_cfg="VFE-A_cfg1", bfe_dim=336):
        """
        data_dir: path to './train', './dev', or './test'
        text_cfg: Text backbone and config (e.g., 'TFE-C_cfg1' for DeBERTa 1024d)
        audio_cfg: Audio backbone and config (e.g., 'AFE-B_cfg1' for WavLM-Large 1024d)
        video_cfg: Video backbone and config (e.g., 'VFE-A_cfg1' for ResNet50 2048d)
        """
        self.data_dir = data_dir
        self.bfe_dim = bfe_dim
        
        # Load Identifiers and Labels
        self.dialogue_ids = np.load(os.path.join(data_dir, "dialogue_ids.npy"))
        self.utterance_ids = np.load(os.path.join(data_dir, "utterance_ids.npy"))
        self.emotion_labels = np.load(os.path.join(data_dir, "emotion_labels.npy"))
        
        # Load Features
        self.text_feats = np.load(os.path.join(data_dir, f"text_{text_cfg}.npy"))
        self.audio_feats = np.load(os.path.join(data_dir, f"audio_{audio_cfg}.npy"))
        self.video_feats = np.load(os.path.join(data_dir, f"video_{video_cfg}.npy"))
        
        # Group utterance indices by dialogue_id
        self.dialogues = {}
        for idx, d_id in enumerate(self.dialogue_ids):
            if d_id not in self.dialogues:
                self.dialogues[d_id] = []
            self.dialogues[d_id].append((self.utterance_ids[idx], idx))
            
        # Sort utterances within each dialogue temporally
        self.valid_dialogue_ids = list(self.dialogues.keys())
        for d_id in self.valid_dialogue_ids:
            # Sort by utterance_id to preserve temporal flow for Mamba
            self.dialogues[d_id].sort(key=lambda x: x[0]) 

    def __len__(self):
        return len(self.valid_dialogue_ids)

    def __getitem__(self, idx):
        d_id = self.valid_dialogue_ids[idx]
        indices = [x[1] for x in self.dialogues[d_id]]
        
        # Extract sequential features for this dialogue
        t_feat = torch.tensor(self.text_feats[indices], dtype=torch.float32)
        a_feat = torch.tensor(self.audio_feats[indices], dtype=torch.float32)
        v_feat = torch.tensor(self.video_feats[indices], dtype=torch.float32)
        labels = torch.tensor(self.emotion_labels[indices], dtype=torch.long)
        
        # Handle 0% Body Features (Generate dummy zeros)
        seq_len = t_feat.shape[0]
        b_feat = torch.zeros((seq_len, self.bfe_dim), dtype=torch.float32)
        
        return t_feat, a_feat, v_feat, b_feat, labels

def meld_collate_fn(batch):
    """Pads conversations to the longest dialogue in the batch."""
    t_feats, a_feats, v_feats, b_feats, labels = zip(*batch)
    
    # Pad sequences
    t_padded = pad_sequence(t_feats, batch_first=True)
    a_padded = pad_sequence(a_feats, batch_first=True)
    v_padded = pad_sequence(v_feats, batch_first=True)
    b_padded = pad_sequence(b_feats, batch_first=True)
    
    # Pad labels with -100 (standard PyTorch ignore_index for CrossEntropy)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)
    
    # Create mask (True for valid tokens, False for padding)
    mask = (labels_padded != -100)
    
    return t_padded, a_padded, v_padded, b_padded, labels_padded, mask
