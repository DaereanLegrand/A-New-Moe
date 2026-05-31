"""
MELD × Gemma-4-E4B — Multimodal Feature Extraction
====================================================
Extracts text, video (vision), and audio features for every utterance in the
train, dev, and test splits of MELD using Gemma-4-E4B as the backbone.

Key design decisions (all grounded in official HF docs + source):
─────────────────────────────────────────────────────────────────

1. SINGLE FORWARD PASS FOR ALL MODALITIES
   Video and audio are processed TOGETHER with text in a single call to
   model.forward(), because Gemma-4's multimodal architecture fuses them
   inside the LM forward pass (not in separate encoders that can be called
   independently without the full model context).
   - image_hidden_states / audio_hidden_states are returned by the model
     when output_hidden_states=True.
   - text_hidden_states = last_hidden_state (from the LM stack).
   Source: Gemma4ModelOutputWithPast docstring in modeling_gemma4.py
           https://huggingface.co/docs/transformers/main/model_doc/gemma4

2. BUG-1 FIX: ClippableLinear unwrap
   Gemma-4's audio encoder wraps every Linear in ClippableLinear (not a
   subclass of nn.Linear). We unwrap them before any PEFT usage and also
   before forward-only usage, because some internal dispatching checks
   isinstance(module, nn.Linear).
   Source: Google Cloud / Medium Gemma-4 fine-tuning guide.

3. BUG-2 FIX: use_cache=True ALWAYS
   Gemma-4 E4B has num_kv_shared_layers=18; shared layers read from earlier
   cache slots.  use_cache=False corrupts their attention → NaN loss.
   For feature extraction we're not training, but we still pass use_cache=True
   explicitly to model.generate / model.forward to stay safe.
   Source: https://unsloth.ai/docs/models/gemma-4/train#bug-fixes--tips

4. BUG-3 FIX: mm_token_type_ids injection
   Gemma4ForConditionalGeneration.forward() requires mm_token_type_ids to
   route image/video/audio token embeddings. We build it from actual token IDs
   using the published config constants (image_token_id=258880,
   video_token_id=258884, audio_token_id=258881).
   Source: https://huggingface.co/docs/transformers/main/model_doc/gemma4

5. BUG-4 FIX: correct content type for multi-frame video
   Use {"type":"video","image":[frame1,frame2,...]} for ≥2 frames so the
   processor generates a single <|video|> token expanded to N*soft_tokens
   features rather than duplicating per-image tokens.
   Source: https://github.com/huggingface/blog/blob/main/gemma4.md

6. SAVE EVERY 200 UTTERANCES (per split, resumable)
   Each modality's features are saved as a compressed .npz file in a
   checkpoint directory. On restart the script skips already-processed indices.

Output layout
─────────────
{OUTPUT_DIR}/
  {split}/
    features_{start}_{end}.npz   ← one file per checkpoint block
    meta_{start}_{end}.json      ← matching metadata (labels, paths, …)
    done.flag                    ← written when entire split finishes
"""
# ─────────────────────────────────────────────────────────────────────────────
import os, re, csv, json, sys, time, logging, warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("MELD-FE")

# ─────────────────────────────────────────────────────────────────────────────
# Lazy dependency installer
# ─────────────────────────────────────────────────────────────────────────────
def _pip(pkg, cmd):
    try:
        __import__(pkg.split(".")[0])
    except ImportError:
        logger.info(f"Installing {pkg} …")
        os.system(cmd)

_pip("tqdm",            "pip install tqdm --quiet")
_pip("torch",           "pip install torch --quiet")
os.system("pip install git+https://github.com/huggingface/transformers.git --quiet")
_pip("bitsandbytes",    "pip install bitsandbytes --quiet")
_pip("accelerate",      "pip install accelerate --quiet")
_pip("cv2",             "pip install opencv-python-headless --quiet")
_pip("soundfile",       "pip install soundfile --quiet")
_pip("librosa",         "pip install librosa --quiet")

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
INPUT_DIR   = "/kaggle/input/datasets/franksalas2506/meld-raw-complete-and-clean/MELD.Raw"
OUTPUT_DIR  = "/kaggle/working/meld_features"
HF_MODEL_ID = "google/gemma-4-E4B-it"

SPLITS      = ["train", "dev", "test"]
SAVE_EVERY  = 200          # checkpoint cadence (utterances)
N_FRAMES    = 2            # video frames to sample per utterance
MAX_SEQ_LEN = 512

CSV_FILES  = {
    "train": "train_sent_emo.csv",
    "dev":   "dev_sent_emo.csv",
    "test":  "test_sent_emo.csv",
}
VIDEO_DIRS = {
    "train": "train_splits",
    "dev":   "dev_splits",
    "test":  "test_splits",
}

# ── Gemma-4 special token IDs ──────────────────────────────────────────────
# Source: Gemma4Config defaults in HF transformers
# https://huggingface.co/docs/transformers/main/model_doc/gemma4
GEMMA4_IMAGE_TOKEN_ID = 258880
GEMMA4_VIDEO_TOKEN_ID = 258884
GEMMA4_AUDIO_TOKEN_ID = 258881
MMTTID_TEXT   = 0
MMTTID_VISION = 1   # both image and video soft tokens
MMTTID_AUDIO  = 2


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Utterance:
    sr_no:        int
    utterance:    str
    speaker:      str
    emotion:      str
    sentiment:    str
    dialogue_id:  int
    utterance_id: int
    season:       int
    episode:      int
    start_time:   str
    end_time:     str
    split:        str
    video_path:   Optional[str] = None
    audio_path:   Optional[str] = None


def load_split(root_dir: str, split: str) -> List[Utterance]:
    csv_path  = Path(root_dir) / CSV_FILES[split]
    video_dir = Path(root_dir) / VIDEO_DIRS[split]
    utts = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            dia_id = int(row["Dialogue_ID"])
            utt_id = int(row["Utterance_ID"])
            stem   = f"dia{dia_id}_utt{utt_id}"
            vp     = video_dir / f"{stem}.mp4"
            ap     = video_dir / f"{stem}.wav"
            utts.append(Utterance(
                sr_no        = int(row["Sr No."]),
                utterance    = row["Utterance"].strip(),
                speaker      = row["Speaker"].strip(),
                emotion      = row["Emotion"].strip().lower(),
                sentiment    = row["Sentiment"].strip().lower(),
                dialogue_id  = dia_id,
                utterance_id = utt_id,
                season       = int(row.get("Season", 0)),
                episode      = int(row.get("Episode", 0)),
                start_time   = row.get("StartTime", ""),
                end_time     = row.get("EndTime", ""),
                split        = split,
                video_path   = str(vp) if vp.exists() else None,
                audio_path   = str(ap) if ap.exists() else None,
            ))
    n_vid = sum(1 for u in utts if u.video_path)
    n_aud = sum(1 for u in utts if u.audio_path)
    logger.info(f"Loaded {len(utts)} utterances from '{split}' "
                f"(video:{n_vid} audio:{n_aud})")
    return utts


# ─────────────────────────────────────────────────────────────────────────────
# Media helpers
# ─────────────────────────────────────────────────────────────────────────────
def extract_frames(video_path: str, n_frames: int) -> List:
    """Return list of PIL Images sampled evenly from video."""
    import cv2
    from PIL import Image
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    indices = [int(i * total / n_frames) for i in range(n_frames)]
    frames  = []
    for fi in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if ret:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames


def load_audio_safe(
    audio_path: Optional[str],
    video_path: Optional[str],
) -> Optional[Tuple[np.ndarray, int]]:
    """Return (waveform_float32_mono, sample_rate) or None."""
    import soundfile as sf

    if audio_path and Path(audio_path).exists():
        try:
            wav, sr = sf.read(audio_path, dtype="float32", always_2d=False)
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            return wav, sr
        except Exception:
            pass

    if video_path and Path(video_path).exists():
        try:
            import subprocess, tempfile
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.close()
            subprocess.run(
                ["ffmpeg", "-y", "-i", video_path,
                 "-ar", "16000", "-ac", "1", tmp.name],
                capture_output=True, check=True,
            )
            wav, sr = sf.read(tmp.name, dtype="float32", always_2d=False)
            os.unlink(tmp.name)
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            return wav, sr
        except Exception:
            pass

    return None


# ─────────────────────────────────────────────────────────────────────────────
# BUG-1 FIX: post-load ClippableLinear unwrap
# Source: https://medium.com/google-cloud/fine-tuning-gemma-4-…
# ─────────────────────────────────────────────────────────────────────────────
def unwrap_clippable_linears(model: nn.Module) -> int:
    replaced = 0
    for parent_name, parent_module in list(model.named_modules()):
        for child_name, child_module in list(parent_module.named_children()):
            if "ClippableLinear" in type(child_module).__name__ and \
               hasattr(child_module, "linear"):
                setattr(parent_module, child_name, child_module.linear)
                replaced += 1
    logger.info(f"[BUG-1 FIX] Unwrapped {replaced} ClippableLinear → plain nn.Linear")
    return replaced


# ─────────────────────────────────────────────────────────────────────────────
# BUG-3 FIX: build mm_token_type_ids from actual token IDs
# Source: https://huggingface.co/docs/transformers/main/model_doc/gemma4
# ─────────────────────────────────────────────────────────────────────────────
def build_mm_token_type_ids(input_ids: List[int]) -> List[int]:
    result = []
    for tok in input_ids:
        if tok in (GEMMA4_IMAGE_TOKEN_ID, GEMMA4_VIDEO_TOKEN_ID):
            result.append(MMTTID_VISION)
        elif tok == GEMMA4_AUDIO_TOKEN_ID:
            result.append(MMTTID_AUDIO)
        else:
            result.append(MMTTID_TEXT)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Model loader
# ─────────────────────────────────────────────────────────────────────────────
def load_model_and_processor():
    """
    Load Gemma-4-E4B in 4-bit NF4 (read-only / eval mode for feature extraction).

    We use AutoModelForImageTextToText (correct class for Gemma-4, which is a
    text+image conditional generation model).
    Source: https://huggingface.co/docs/transformers/main/model_doc/gemma4
            (AutoModelForImageTextToText usage in docs examples)
    Note:   for audio+video E2B/E4B the docs also show AutoModelForMultimodalLM,
            but AutoModelForImageTextToText resolves to the same
            Gemma4ForConditionalGeneration class and is simpler.
    """
    from transformers import (
        AutoProcessor,
        AutoModelForImageTextToText,
        BitsAndBytesConfig,
    )
    logger.info(f"Loading {HF_MODEL_ID} in 4-bit NF4 (eval/feature extraction) …")

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit              = True,
        bnb_4bit_quant_type       = "nf4",
        bnb_4bit_use_double_quant = True,
        bnb_4bit_compute_dtype    = torch.bfloat16,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        HF_MODEL_ID,
        quantization_config = bnb_cfg,
        torch_dtype         = torch.bfloat16,
        device_map          = "auto",
        attn_implementation = "sdpa",
        max_memory          = {0: "13GiB", 1: "13GiB"},
    )

    # BUG-2 FIX: always keep use_cache=True
    # Shared-KV layers need the cache; setting False corrupts their attention.
    model.config.use_cache = True
    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = True

    # BUG-1 FIX
    unwrap_clippable_linears(model)

    model.eval()

    processor = AutoProcessor.from_pretrained(
        HF_MODEL_ID,
        padding_side  = "left",
        max_soft_tokens = 70,   # 70 vision soft tokens per frame → light on VRAM
    )

    return model, processor


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction for a single utterance
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a helpful assistant analyzing utterances from a TV show. "
    "Provide a brief description of the emotional content."
)

def extract_features_for_utterance(
    utt: Utterance,
    model,
    processor,
    n_frames: int = N_FRAMES,
) -> Dict[str, Optional[np.ndarray]]:
    """
    Run a SINGLE forward pass through Gemma-4 to extract:
      - text_features  : np.ndarray  [seq_len, hidden_dim]
                         Last hidden state of the LM stack, averaged over
                         non-padding tokens → shape [hidden_dim].
      - video_features : np.ndarray  [n_soft_tokens, hidden_dim]  or None
                         image_hidden_states from the vision encoder/projector,
                         mean-pooled across soft tokens → shape [hidden_dim].
      - audio_features : np.ndarray  [n_audio_tokens, hidden_dim] or None
                         audio_hidden_states from the audio encoder/projector,
                         mean-pooled across time → shape [hidden_dim].

    WHY A SINGLE FORWARD PASS?
    ─────────────────────────
    Gemma-4's image/audio encoders are submodules of
    Gemma4ForConditionalGeneration.  They don't have public standalone forward
    methods that return projected features in the LM hidden dimension; the
    projection to LM space happens *inside* the conditional generation forward.
    output_hidden_states=True exposes image_hidden_states and
    audio_hidden_states alongside the LM last_hidden_state, making a single
    combined call both correct and most efficient.
    Source: Gemma4ModelOutputWithPast in modeling_gemma4.py (lines ~60-80):
      image_hidden_states: (batch, num_images, seq_len, hidden_size)
      audio_hidden_states: (batch, num_images, seq_len, hidden_size)
    """
    frames = []
    audio  = None

    # ── Media loading ────────────────────────────────────────────────────────
    if utt.video_path:
        try:
            frames = extract_frames(utt.video_path, n_frames)
        except Exception as e:
            logger.debug(f"Frame extraction failed for {utt.video_path}: {e}")

    audio = load_audio_safe(utt.audio_path, utt.video_path)

    # ── Build content list ───────────────────────────────────────────────────
    # BUG-4 FIX: use {"type":"video"} for ≥2 frames, {"type":"image"} for 1.
    # This maps to a single placeholder token expanded to N*soft_tokens
    # internally, avoiding the 2× token-duplication crash.
    # Source: https://github.com/huggingface/blog/blob/main/gemma4.md
    user_content = []
    has_video = len(frames) > 0
    has_audio = audio is not None

    if has_video:
        if len(frames) >= 2:
            user_content.append({"type": "video", "image": frames})
        else:
            user_content.append({"type": "image", "image": frames[0]})

    if has_audio:
        wav, sr = audio
        user_content.append({"type": "audio", "audio": wav, "sampling_rate": sr})

    user_content.append({"type": "text", "text": f"{utt.speaker}: {utt.utterance}"})

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]

    # ── Tokenise / pre-process ───────────────────────────────────────────────
    # apply_chat_template with tokenize=True + return_dict=True is the
    # canonical way to process multimodal inputs for Gemma-4.
    # Source: https://huggingface.co/docs/transformers/main/model_doc/gemma4
    inputs = processor.apply_chat_template(
        messages,
        tokenize              = True,
        return_dict           = True,
        return_tensors        = "pt",
        add_generation_prompt = False,
    )

    # BUG-3 FIX: inject mm_token_type_ids if the processor didn't emit it.
    if "mm_token_type_ids" not in inputs:
        ids_list = inputs["input_ids"][0].tolist()
        inputs["mm_token_type_ids"] = torch.tensor(
            [build_mm_token_type_ids(ids_list)], dtype=torch.long
        )
    if "token_type_ids" not in inputs:
        inputs["token_type_ids"] = torch.zeros_like(inputs["input_ids"])

    # Move all tensors to model device
    device = next(model.parameters()).device
    inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
              for k, v in inputs.items()}

    # ── Forward pass ─────────────────────────────────────────────────────────
    # output_hidden_states=True exposes:
    #   .last_hidden_state           → text features
    #   .image_hidden_states         → vision encoder output (projected to LM dim)
    #   .audio_hidden_states         → audio encoder output (projected to LM dim)
    # Source: Gemma4ModelOutputWithPast docstring in modeling_gemma4.py
    result: Dict[str, Optional[np.ndarray]] = {
        "text_features":  None,
        "video_features": None,
        "audio_features": None,
        "has_video": has_video,
        "has_audio": has_audio,
    }

    with torch.inference_mode():
        outputs = model(
            **inputs,
            output_hidden_states = True,
            return_dict          = True,
            use_cache            = True,   # BUG-2 FIX
        )

    # ── Text features ────────────────────────────────────────────────────────
    # last_hidden_state: [batch=1, seq_len, hidden_dim]
    # We mean-pool over non-padding tokens to get a fixed-size [hidden_dim] vector.
    if outputs.last_hidden_state is not None:
        lhs = outputs.last_hidden_state[0]   # [seq_len, hidden_dim]
        if "attention_mask" in inputs:
            mask = inputs["attention_mask"][0].bool()   # [seq_len]
            lhs  = lhs[mask]                            # [valid_len, hidden_dim]
        text_feat = lhs.mean(dim=0).float().cpu().numpy()   # [hidden_dim]
        result["text_features"] = text_feat

    # ── Video/image features ─────────────────────────────────────────────────
    # image_hidden_states: (batch, num_images, n_soft_tokens, hidden_dim)
    # We mean-pool across soft tokens → [hidden_dim].
    # If there are multiple frames (video) they map to a single "image" slot.
    # Source: Gemma4ModelOutputWithPast docstring in modeling_gemma4.py
    if has_video and hasattr(outputs, "image_hidden_states") and \
       outputs.image_hidden_states is not None:
        ihs = outputs.image_hidden_states   # (1, n_images, n_soft, hidden)
        # Squeeze batch + image dims; mean-pool soft tokens
        ihs = ihs[0]                        # (n_images, n_soft, hidden)
        vid_feat = ihs.mean(dim=(0, 1)).float().cpu().numpy()   # [hidden_dim]
        result["video_features"] = vid_feat

    # ── Audio features ───────────────────────────────────────────────────────
    # audio_hidden_states: (batch, n_audio_clips, n_audio_tokens, hidden_dim)
    # Mean-pool across time → [hidden_dim].
    # Source: Gemma4ModelOutputWithPast docstring in modeling_gemma4.py
    if has_audio and hasattr(outputs, "audio_hidden_states") and \
       outputs.audio_hidden_states is not None:
        ahs = outputs.audio_hidden_states   # (1, n_clips, n_tokens, hidden)
        ahs = ahs[0]                        # (n_clips, n_tokens, hidden)
        aud_feat = ahs.mean(dim=(0, 1)).float().cpu().numpy()   # [hidden_dim]
        result["audio_features"] = aud_feat

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────
def split_out_dir(split: str) -> Path:
    d = Path(OUTPUT_DIR) / split
    d.mkdir(parents=True, exist_ok=True)
    return d


def done_flag(split: str) -> Path:
    return split_out_dir(split) / "done.flag"


def find_resume_index(split: str) -> int:
    """Return the index of the first utterance NOT yet saved."""
    od = split_out_dir(split)
    saved_indices: set = set()
    for p in od.glob("meta_*.json"):
        try:
            for rec in json.load(open(p)):
                saved_indices.add(rec["global_idx"])
        except Exception:
            pass
    if not saved_indices:
        return 0
    max_saved = max(saved_indices)
    # All contiguous up to max_saved → resume from max_saved+1
    return max_saved + 1


def save_chunk(
    split: str,
    records: List[dict],
    arrays_text:  List[Optional[np.ndarray]],
    arrays_video: List[Optional[np.ndarray]],
    arrays_audio: List[Optional[np.ndarray]],
    chunk_start: int,
    chunk_end:   int,
):
    od = split_out_dir(split)
    tag = f"{chunk_start}_{chunk_end}"

    # ── NPZ: only save non-None arrays; use idx as key ──────────────────────
    npz_dict = {}
    for rec, tf, vf, af in zip(records, arrays_text, arrays_video, arrays_audio):
        idx = rec["global_idx"]
        if tf  is not None: npz_dict[f"text_{idx}"]  = tf
        if vf  is not None: npz_dict[f"video_{idx}"] = vf
        if af  is not None: npz_dict[f"audio_{idx}"] = af

    npz_path  = od / f"features_{tag}.npz"
    meta_path = od / f"meta_{tag}.json"

    if npz_dict:
        np.savez_compressed(str(npz_path), **npz_dict)
    json.dump(records, open(meta_path, "w"), indent=2)

    tqdm.write(f"  ✔ [{split}] chunk {chunk_start}–{chunk_end}: "
               f"{len(records)} utts saved "
               f"(text:{sum(t is not None for t in arrays_text)} "
               f"video:{sum(v is not None for v in arrays_video)} "
               f"audio:{sum(a is not None for a in arrays_audio)})")


# ─────────────────────────────────────────────────────────────────────────────
# Per-split extraction loop
# ─────────────────────────────────────────────────────────────────────────────
def extract_split(split: str, model, processor, n_frames: int = N_FRAMES):
    logger.info(f"\n{'='*64}\nFeature extraction: {split}\n{'='*64}")

    if done_flag(split).exists():
        logger.info(f"[{split}] Already complete (done.flag present). Skipping.")
        return

    utterances  = load_split(INPUT_DIR, split)
    start_idx   = find_resume_index(split)

    if start_idx >= len(utterances):
        logger.info(f"[{split}] All {len(utterances)} utterances already processed.")
        done_flag(split).touch()
        return

    logger.info(f"[{split}] Resuming from idx {start_idx} "
                f"({len(utterances) - start_idx} remaining)")

    chunk_meta:  List[dict]            = []
    chunk_text:  List[Optional[np.ndarray]] = []
    chunk_video: List[Optional[np.ndarray]] = []
    chunk_audio: List[Optional[np.ndarray]] = []
    chunk_start  = start_idx

    pbar = tqdm(
        range(start_idx, len(utterances)),
        total   = len(utterances),
        initial = start_idx,
        desc    = split,
        unit    = "utt",
        dynamic_ncols = True,
        colour  = "cyan",
    )

    for idx in pbar:
        utt = utterances[idx]
        feats: Dict = {
            "text_features":  None,
            "video_features": None,
            "audio_features": None,
            "has_video": False,
            "has_audio": False,
        }

        try:
            feats = extract_features_for_utterance(utt, model, processor, n_frames)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            tqdm.write(f"  ⚠ OOM at idx={idx}; skipping utterance.")
        except Exception as e:
            tqdm.write(f"  ⚠ Error at idx={idx}: {e}")

        # Build metadata record
        meta = {
            "global_idx":   idx,
            "sr_no":        utt.sr_no,
            "split":        split,
            "dialogue_id":  utt.dialogue_id,
            "utterance_id": utt.utterance_id,
            "speaker":      utt.speaker,
            "utterance":    utt.utterance,
            "emotion":      utt.emotion,
            "sentiment":    utt.sentiment,
            "season":       utt.season,
            "episode":      utt.episode,
            "start_time":   utt.start_time,
            "end_time":     utt.end_time,
            "video_path":   utt.video_path,
            "audio_path":   utt.audio_path,
            "has_video":    feats["has_video"],
            "has_audio":    feats["has_audio"],
            "text_ok":      feats["text_features"]  is not None,
            "video_ok":     feats["video_features"] is not None,
            "audio_ok":     feats["audio_features"] is not None,
        }

        chunk_meta.append(meta)
        chunk_text.append(feats["text_features"])
        chunk_video.append(feats["video_features"])
        chunk_audio.append(feats["audio_features"])

        pbar.set_postfix(
            text  = "✓" if feats["text_features"]  is not None else "✗",
            video = "✓" if feats["video_features"] is not None else "✗",
            audio = "✓" if feats["audio_features"] is not None else "✗",
        )

        # Checkpoint every SAVE_EVERY utterances
        if len(chunk_meta) >= SAVE_EVERY:
            save_chunk(split, chunk_meta, chunk_text, chunk_video, chunk_audio,
                       chunk_start, idx)
            chunk_meta, chunk_text, chunk_video, chunk_audio = [], [], [], []
            chunk_start = idx + 1
            torch.cuda.empty_cache()

    pbar.close()

    # Save any remaining utterances
    if chunk_meta:
        save_chunk(split, chunk_meta, chunk_text, chunk_video, chunk_audio,
                   chunk_start, len(utterances) - 1)

    done_flag(split).touch()
    logger.info(f"[{split}] ✔ All done. Output → {split_out_dir(split)}")


# ─────────────────────────────────────────────────────────────────────────────
# Summary helper
# ─────────────────────────────────────────────────────────────────────────────
def print_summary():
    print(f"\n{'═'*64}")
    print(f"  Feature Extraction Summary")
    print(f"{'═'*64}")
    for split in SPLITS:
        od = split_out_dir(split)
        metas = []
        for p in sorted(od.glob("meta_*.json")):
            try:
                metas.extend(json.load(open(p)))
            except Exception:
                pass
        if not metas:
            print(f"  {split:<6}: no records yet")
            continue
        n_text  = sum(1 for m in metas if m.get("text_ok"))
        n_video = sum(1 for m in metas if m.get("video_ok"))
        n_audio = sum(1 for m in metas if m.get("audio_ok"))
        print(f"  {split:<6}: {len(metas):>5} utts | "
              f"text:{n_text:>5} video:{n_video:>5} audio:{n_audio:>5}")
    print(f"{'═'*64}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI / entry point
# ─────────────────────────────────────────────────────────────────────────────
import argparse

def parse_args():
    p = argparse.ArgumentParser(description="MELD × Gemma-4 Feature Extraction")
    p.add_argument("--splits",      nargs="+", default=SPLITS,
                   choices=SPLITS, help="Which splits to process")
    p.add_argument("--n-frames",    type=int,  default=N_FRAMES,
                   help="Video frames per utterance (default 2)")
    p.add_argument("--save-every",  type=int,  default=SAVE_EVERY,
                   help="Save checkpoint every N utterances (default 200)")
    p.add_argument("--input-dir",   default=INPUT_DIR)
    p.add_argument("--output-dir",  default=OUTPUT_DIR)
    p.add_argument("--model-id",    default=HF_MODEL_ID)
    p.add_argument("--summary-only", action="store_true",
                   help="Only print summary of already-extracted features")
    return p.parse_args()


def main():
    args = parse_args()

    global INPUT_DIR, OUTPUT_DIR, HF_MODEL_ID, SAVE_EVERY, N_FRAMES
    INPUT_DIR  = args.input_dir
    OUTPUT_DIR = args.output_dir
    HF_MODEL_ID = args.model_id
    SAVE_EVERY  = args.save_every
    N_FRAMES    = args.n_frames

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    if args.summary_only:
        print_summary()
        return

    # Load model once; reuse across all splits
    model, processor = load_model_and_processor()

    for split in args.splits:
        extract_split(split, model, processor, n_frames=args.n_frames)

    print_summary()
    logger.info(f"All done. Features saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
