# predict_actions_ravar.py
# Inference: (video + person text) -> multi-label action probabilities using Blipv2
# Mirrors your RAVARDataset preprocessing (Normalize 0.45/0.25, key-frame resize, tokenization w/ special tokens).
#
# Usage:
#   python predict_actions_ravar.py \
#       --video_path /path/to/video.mp4 \
#       --person_text "man in red jacket walking to a counter" \
#       --checkpoint ckpts3/pami_method_retry_31_08_try6/pytorch_model.bin.7 \
#       --labels /path/to/labels.txt \
#       --device cuda:0 --max_frames 8 --top_k 10
#
# Optional:
#   --bbox "x1,y1,x2,y2"   (pixel coords on key-frame; defaults to zeros)

from __future__ import annotations
import os
import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from modules.pami_try6_abl_agg import Blipv2
from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer

# --------------------- preprocessing (match your dataloader) ---------------------

MEAN_045 = (0.45, 0.45, 0.45)
STD_025  = (0.25, 0.25, 0.25)

def tf_video_224():
    return Compose([
        ToTensor(),                       # [0,1]
        Resize([224, 224]),               # bilinear by default
        Normalize(MEAN_045, STD_025),
    ])

def tf_key_340x465():
    return Compose([
        Resize([340, 465]),
        ToTensor(),
        Normalize(MEAN_045, STD_025),
    ])

def load_labels(path: str | None, default_n: int = 80) -> list[str]:
    if path and os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            return [ln.strip() for ln in f if ln.strip()]
    return [f"class_{i}" for i in range(default_n)]

def decode_video_rgb(video_path: str) -> list[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise ValueError("No frames decoded from the video.")
    return frames

def uniform_sample_indices(n: int, max_frames: int) -> list[int]:
    if n <= max_frames:
        return list(range(n))
    step = n / float(max_frames)
    return [int(i * step) for i in range(max_frames)]

def frames_to_video_tensor(frames_rgb: list[np.ndarray], max_frames: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      video: [1, T, 3, 224, 224]
      video_mask: [1, T] (ones for valid frames)
    """
    idxs = uniform_sample_indices(len(frames_rgb), max_frames)
    tfrm = tf_video_224()
    tensors = []
    for i in idxs:
        pil = Image.fromarray(frames_rgb[i])  # RGB
        tensors.append(tfrm(pil))             # [3,224,224]
    video = torch.stack(tensors, dim=0).unsqueeze(0)          # [1,T,3,224,224]
    video_mask = torch.ones((1, video.shape[1]), dtype=torch.long)
    return video, video_mask

def make_key_frame(frames_rgb: list[np.ndarray]) -> torch.Tensor:
    """
    Key-frame is the center frame, resized to [340,465] and normalized like your loader.
    Returns [1, 3, 340, 465]
    """
    key_idx = len(frames_rgb) // 2
    pil = Image.fromarray(frames_rgb[key_idx])
    return tf_key_340x465()(pil).unsqueeze(0)

def parse_bbox(bbox_str: str | None) -> np.ndarray:
    if not bbox_str:
        return np.zeros((4,), dtype=np.float32)
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox_str.split(",")]
        return np.array([x1, y1, x2, y2], dtype=np.float32)
    except Exception:
        raise ValueError("Invalid --bbox format. Use: x1,y1,x2,y2")

# --------------------- tokenization (match your _get_text) ---------------------

SPECIAL = {"CLS": "<|startoftext|>", "SEP": "<|endoftext|>"}

def build_text_tensors(text: str, max_words: int = 32):
    """
    Reproduces your _get_text() behavior:
    - tokenization via SimpleTokenizer
    - prepend CLS, append SEP
    - pad to max_words
    Returns (input_ids, input_mask, segment_ids) each [1, max_words] long dtype torch.long
    """
    tok = ClipTokenizer()
    words = tok.tokenize(text)
    words = [SPECIAL["CLS"]] + words
    if len(words) > (max_words - 1):
        words = words[:max_words - 1]
    words = words + [SPECIAL["SEP"]]

    input_ids = tok.convert_tokens_to_ids(words)
    input_mask = [1] * len(input_ids)
    segment_ids = [0] * len(input_ids)

    # pad
    while len(input_ids) < max_words:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
    input_mask = torch.tensor(input_mask, dtype=torch.long).unsqueeze(0)
    segment_ids = torch.tensor(segment_ids, dtype=torch.long).unsqueeze(0)
    return input_ids, input_mask, segment_ids

# --------------------- model loading ---------------------

def load_blipv2(checkpoint_path: str, device: torch.device) -> Blipv2:
    model = Blipv2()
    sd = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(sd, dict) and any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[WARN] Missing keys: {len(missing)} (showing up to 10): {missing[:10]}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {len(unexpected)} (showing up to 10): {unexpected[:10]}")
    model.eval().to(device)
    return model

# --------------------- utils ---------------------

def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return 1 / (1 + torch.exp(-x))

# --------------------- main ---------------------

def parse_args():
    ap = argparse.ArgumentParser("RAVAR-style Action Prediction (Blipv2)")
    ap.add_argument("--video_path", required=True, type=str)
    ap.add_argument("--person_text", required=True, type=str)
    ap.add_argument("--checkpoint", required=True, type=str)
    ap.add_argument("--labels", default=None, type=str)
    ap.add_argument("--device", default="cuda:0", type=str)
    ap.add_argument("--max_frames", default=8, type=int)     # matches your dataset default
    ap.add_argument("--max_words", default=32, type=int)     # matches your dataset default
    ap.add_argument("--bbox", default=None, type=str)        # "x1,y1,x2,y2"
    ap.add_argument("--top_k", default=10, type=int)
    return ap.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu")

    labels = load_labels(args.labels)
    num_classes = len(labels)

    # 1) Decode video and build inputs
    frames = decode_video_rgb(args.video_path)
    key_frame = make_key_frame(frames)                            # [1, 3, 340, 465]
    video, video_mask = frames_to_video_tensor(frames, args.max_frames)  # [1,T,3,224,224], [1,T]

    # 2) Text tensors
    input_ids, input_mask, segment_ids = build_text_tensors(args.person_text, max_words=args.max_words)

    # 3) BBox + dummy labels (ann)
    bbox_np = parse_bbox(args.bbox)                               # [4]
    bbox = torch.from_numpy(bbox_np).unsqueeze(0).float()         # [1,4]
    ann  = torch.zeros((1, num_classes), dtype=torch.float32)     # placeholder (not used in inference)

    # 4) Move to device
    key_frame = key_frame.to(device, non_blocking=True)
    video = video.to(device, non_blocking=True)
    video_mask = video_mask.to(device, non_blocking=True)
    input_ids = input_ids.to(device, non_blocking=True)
    input_mask = input_mask.to(device, non_blocking=True)
    segment_ids = segment_ids.to(device, non_blocking=True)
    bbox = bbox.to(device, non_blocking=True)
    ann = ann.to(device, non_blocking=True)

    # 5) Load model
    print("[INFO] Loading Blipv2...")
    model = load_blipv2(args.checkpoint, device)

    # 6) Forward
    print("[INFO] Running inference...")
    with torch.no_grad():
        # loss, predictions, prediction_bboxes, ann_out, logits
        _, predictions, pred_bboxes, _, logits = model(
            key_frame, input_ids, segment_ids, input_mask, video, video_mask, bbox, ann, training=False
        )

    # Prefer logits -> sigmoid; fall back to predictions if logits not returned
    if logits is not None:
        probs = sigmoid(logits.float()).squeeze(0)   # [C]
    else:
        probs = predictions.float().squeeze(0)

    probs_np = probs.detach().cpu().numpy()
    top_k = min(args.top_k, probs_np.shape[-1])
    idxs = np.argsort(-probs_np)[:top_k]

    print("\n=== Predicted actions (top-{}): ===".format(top_k))
    for r, i in enumerate(idxs, 1):
        label = labels[i] if i < len(labels) else f"class_{i}"
        print(f"{r:2d}. {label:30s}  {probs_np[i]:.4f}")

    # Optional: report mean predicted bbox if available
    if pred_bboxes is not None:
        try:
            bb = pred_bboxes.squeeze(0).detach().cpu().numpy()  # [T,4] or [4]
            if bb.ndim == 2 and bb.shape[-1] == 4:
                bb_mean = bb.mean(axis=0)
                print("\n[INFO] Mean predicted bbox (x1,y1,x2,y2):", np.round(bb_mean, 2).tolist())
            elif bb.ndim == 1 and bb.shape[0] == 4:
                print("\n[INFO] Predicted bbox (x1,y1,x2,y2):", np.round(bb, 2).tolist())
        except Exception:
            pass

if __name__ == "__main__":
    main()
