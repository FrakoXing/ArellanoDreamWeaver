# -*- coding: utf-8 -*-
"""
ONNX Model Inference Script
============================
Use the exported ONNX model for chart (note sequence) generation.

Usage:
  # Generate from an audio file
  python scripts/infer_onnx.py --model logs/model.onnx --audio data/audio/song.mp3

  # Generate from random noise (quick test)
  python scripts/infer_onnx.py --model logs/model.onnx --random

  # Decoder-only mode (provide pre-computed features)
  python scripts/infer_onnx.py --model logs/decoder.onnx --decoder-only --audio data/audio/song.mp3

  # Specify generation parameters
  python scripts/infer_onnx.py --model logs/model.onnx --audio song.mp3 \
      --max-tokens 512 --temperature 0.8 --top-k 50
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# Token Vocabulary (mirrors src/models/note_decoder.py)
# ============================================================

PAD = 0
SOS = 1     # Start of Sequence
EOS = 2     # End of Sequence
MASK = 3

# Note types
TYPE_TAP = 4
TYPE_HOLD = 5
TYPE_SLIDE = 6

# Token ranges
TIME_OFFSET_START = 7
TIME_OFFSET_END = 70
POS_X_START = 71
POS_X_END = 79
HOLD_TIME_START = 80
HOLD_TIME_END = 111
NOTE_ABOVE = 112
NOTE_BELOW = 113

VOCAB_SIZE = 256


def decode_token(token_id: int) -> str:
    """Decode a single token to human-readable string."""
    if token_id == PAD:
        return "PAD"
    elif token_id == SOS:
        return "SOS"
    elif token_id == EOS:
        return "EOS"
    elif token_id == MASK:
        return "MASK"
    elif token_id == TYPE_TAP:
        return "TAP"
    elif token_id == TYPE_HOLD:
        return "HOLD"
    elif token_id == TYPE_SLIDE:
        return "SLIDE"
    elif TIME_OFFSET_START <= token_id <= TIME_OFFSET_END:
        offset_beats = (token_id - TIME_OFFSET_START) * 0.25
        return f"time_offset={offset_beats:.2f}bt"
    elif POS_X_START <= token_id <= POS_X_END:
        x_values = [-1.0, -0.75, -0.50, -0.25, 0, 0.25, 0.50, 0.75, 1.0]
        idx = token_id - POS_X_START
        return f"x={x_values[idx]:.2f}" if idx < len(x_values) else f"x_unknown({token_id})"
    elif HOLD_TIME_START <= token_id <= HOLD_TIME_END:
        hold_beats = (token_id - HOLD_TIME_START) * 0.25
        return f"hold_time={hold_beats:.2f}bt"
    elif token_id == NOTE_ABOVE:
        return "ABOVE"
    elif token_id == NOTE_BELOW:
        return "BELOW"
    else:
        return f"token_{token_id}"


def decode_token_sequence(token_ids: List[int]) -> List[dict]:
    """
    Decode a token sequence into structured note events.

    Returns list of dicts with keys: type, time_offset, x, hold_time, direction
    """
    notes = []
    i = 0
    current_note = {}

    while i < len(token_ids):
        tid = token_ids[i]

        if tid == EOS or tid == PAD:
            break

        if tid in (TYPE_TAP, TYPE_HOLD, TYPE_SLIDE):
            if current_note:
                notes.append(current_note)
            current_note = {
                "type": "tap" if tid == TYPE_TAP else "hold" if tid == TYPE_HOLD else "slide",
                "time_offset": None,
                "x": None,
                "hold_time": None,
                "direction": None,
            }
        elif TIME_OFFSET_START <= tid <= TIME_OFFSET_END:
            if current_note:
                current_note["time_offset"] = (tid - TIME_OFFSET_START) * 0.25
        elif POS_X_START <= tid <= POS_X_END:
            x_values = [-1.0, -0.75, -0.50, -0.25, 0, 0.25, 0.50, 0.75, 1.0]
            idx = tid - POS_X_START
            if current_note and idx < len(x_values):
                current_note["x"] = x_values[idx]
        elif HOLD_TIME_START <= tid <= HOLD_TIME_END:
            if current_note:
                current_note["hold_time"] = (tid - HOLD_TIME_START) * 0.25
        elif tid == NOTE_ABOVE:
            if current_note:
                current_note["direction"] = "above"
        elif tid == NOTE_BELOW:
            if current_note:
                current_note["direction"] = "below"

        i += 1

    if current_note:
        notes.append(current_note)

    return notes


# ============================================================
# Audio preprocessing
# ============================================================

def load_audio_features(
    audio_path: str,
    target_steps: int = 200,
    n_mels: int = 128,
    sample_rate: int = 22050,
) -> np.ndarray:
    """
    Load audio file and extract Mel spectrogram features.

    Args:
        audio_path: Path to audio file (.mp3/.wav/.flac)
        target_steps: Number of time steps needed (must match ONNX model's audio_len)
        n_mels: Number of Mel bands (must match ONNX model's audio_dim)
        sample_rate: Audio sample rate

    Returns:
        features: [1, target_steps, n_mels] float32 numpy array
    """
    try:
        import librosa
    except ImportError:
        print("[WARN] librosa not installed. Using random noise as fallback.")
        print("       Install: pip install librosa")
        return np.random.randn(1, target_steps, n_mels).astype(np.float32)

    print(f"[AUDIO] Loading: {audio_path}")

    # Load audio (resample if needed)
    waveform, sr = librosa.load(audio_path, sr=sample_rate, mono=True)

    # Extract Mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=waveform, sr=sr, n_mels=n_mels,
        n_fft=2048, hop_length=512, power=2.0,
    )

    # Convert to dB scale
    mel_db = librosa.power_to_db(mel_spec, ref=1.0, top_db=80.0)

    # Shape: [n_mels, T] -> [T, n_mels]
    mel_db = mel_db.T.astype(np.float32)

    # Normalize (z-score)
    mean = mel_db.mean()
    std = mel_db.std() + 1e-6
    mel_normalized = (mel_db - mean) / std

    # Resize to target_steps (linear interpolation)
    if mel_normalized.shape[0] != target_steps:
        # Simple interpolation via linear resampling
        indices = np.linspace(0, mel_normalized.shape[0] - 1, target_steps)
        resampled = np.zeros((target_steps, n_mels), dtype=np.float32)
        for j in range(n_mels):
            resampled[:, j] = np.interp(indices, np.arange(mel_normalized.shape[0]), mel_normalized[:, j])
        mel_normalized = resampled

    # Add batch dimension: [1, T, n_mels]
    features = mel_normalized[np.newaxis, :, :]

    print(f"  Features shape: {features.shape}")
    return features


# ============================================================
# ONNX Inference
# ============================================================

def onnx_generate(
    session,
    audio_features: np.ndarray,
    decoder_only: bool = False,
    max_tokens: int = 256,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    verbose: bool = False,
) -> List[int]:
    """
    Autoregressive token generation using ONNX model.

    The model has FIXED input shapes (static export). We pad the growing
    token sequence to the fixed size with PAD tokens. The causal mask ensures
    that each position only attends to previous positions.

    Args:
        session: onnxruntime.InferenceSession
        audio_features: [1, audio_len, audio_dim] audio features
        decoder_only: If True, audio_features is already encoder_output
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (>0: random, <=0: greedy)
        top_k: Top-K filtering (0 = disabled)
        top_p: Nucleus sampling (1.0 = disabled)
        verbose: Print each generated token

    Returns:
        List of generated token IDs including SOS and EOS
    """
    # Determine the fixed sequence length from the model's input shape
    # target_tokens input: [batch, seq_len]
    target_input = session.get_inputs()[1] if not decoder_only else session.get_inputs()[1]
    if decoder_only:
        # encoder_output, target_tokens
        fixed_seq_len = target_input.shape[1]
    else:
        # audio_features, target_tokens
        token_input_idx = 1 if session.get_inputs()[0].name == 'audio_features' else 0
        fixed_seq_len = session.get_inputs()[token_input_idx].shape[1]

    if fixed_seq_len is None or (isinstance(fixed_seq_len, str)):
        # Dynamic shape: use max_tokens
        fixed_seq_len = max_tokens
        print(f"  [INFO] Dynamic seq_len detected, using max_tokens={max_tokens}")
    else:
        print(f"  [INFO] Fixed seq_len={fixed_seq_len} (static model)")

    if max_tokens > fixed_seq_len:
        print(f"  [WARN] max_tokens ({max_tokens}) > model seq_len ({fixed_seq_len}), clamping")
        max_tokens = fixed_seq_len

    # Initialize with SOS token, pad the rest
    target_tokens = np.zeros((1, fixed_seq_len), dtype=np.int64)  # PAD = 0
    target_tokens[0, 0] = SOS
    current_len = 1  # Number of valid tokens (not counting padding)

    for step in range(max_tokens - 1):  # -1 because we already have SOS
        # Prepare inputs (always use fixed shape)
        if decoder_only:
            onnx_inputs = {
                'encoder_output': audio_features.astype(np.float32),
                'target_tokens': target_tokens,
            }
        else:
            onnx_inputs = {
                'audio_features': audio_features.astype(np.float32),
                'target_tokens': target_tokens,
            }

        # Run inference
        outputs = session.run(None, onnx_inputs)
        logits = outputs[0]  # [1, fixed_seq_len, vocab_size]

        # Get logits for the CURRENT last valid position (predicts next token)
        # Position `current_len - 1` predicts token at position `current_len`
        next_logits = logits[0, current_len - 1, :]  # [vocab_size]

        # === Sampling ===
        if temperature <= 0:
            # Greedy
            next_token = int(np.argmax(next_logits))
        else:
            # Temperature scaling
            scaled_logits = next_logits / temperature

            # Top-K filtering
            if top_k > 0:
                top_k_indices = np.argpartition(scaled_logits, -top_k)[-top_k:]
                mask = np.full_like(scaled_logits, -1e10)
                mask[top_k_indices] = scaled_logits[top_k_indices]
                scaled_logits = mask

            # Top-P (nucleus) filtering
            if top_p < 1.0:
                sorted_indices = np.argsort(scaled_logits)[::-1]
                sorted_l = scaled_logits[sorted_indices]
                exp_l = np.exp(sorted_l - np.max(sorted_l))
                cumulative_probs = np.cumsum(exp_l / np.sum(exp_l))
                cutoff = int(np.searchsorted(cumulative_probs, top_p)) + 1
                scaled_logits[sorted_indices[cutoff:]] = -1e10

            # Softmax + sample
            probs = np.exp(scaled_logits - np.max(scaled_logits))
            probs = probs / np.sum(probs)
            next_token = int(np.random.choice(len(probs), p=probs))

        if verbose:
            print(f"  step {step+1}: pos={current_len} -> {decode_token(next_token)}")

        # Update: set next token at the current position
        target_tokens[0, current_len] = next_token
        current_len += 1

        # Stop conditions
        if next_token == EOS:
            break

    # Return only valid tokens (not padding)
    return target_tokens[0, :current_len].tolist()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="ONNX model inference for chart generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--model", type=str, required=True,
                        help="Path to ONNX model (.onnx)")
    parser.add_argument("--audio", type=str, default=None,
                        help="Path to audio file (.mp3/.wav/.flac)")
    parser.add_argument("--random", action="store_true",
                        help="Use random noise instead of real audio (quick test)")
    parser.add_argument("--decoder-only", action="store_true",
                        help="Model is decoder-only (audio_features treated as encoder_output)")
    parser.add_argument("--audio-len", type=int, default=200,
                        help="Audio feature time steps (must match export shape)")
    parser.add_argument("--audio-dim", type=int, default=128,
                        help="Audio feature dimension (must match export shape)")
    parser.add_argument("--max-tokens", type=int, default=256,
                        help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (0=greedy, 1.0=default)")
    parser.add_argument("--top-k", type=int, default=50,
                        help="Top-K sampling filter (0=disabled)")
    parser.add_argument("--top-p", type=float, default=0.95,
                        help="Nucleus sampling threshold (1.0=disabled)")
    parser.add_argument("--num-samples", type=int, default=1,
                        help="Number of samples to generate")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file for generated chart")
    parser.add_argument("--verbose", action="store_true",
                        help="Print each generated token")

    args = parser.parse_args()

    # Check ONNX model
    if not os.path.exists(args.model):
        print(f"[FAIL] ONNX model not found: {args.model}")
        sys.exit(1)

    # Load ONNX Runtime FIRST to detect expected shapes
    try:
        import onnxruntime as ort
    except ImportError:
        print("[FAIL] onnxruntime not installed. Run: pip install onnxruntime")
        sys.exit(1)

    print(f"\n[ONNX] Loading model: {args.model}")
    sess = ort.InferenceSession(args.model, providers=['CPUExecutionProvider'])

    # Auto-detect expected shapes from ONNX model
    model_shapes = {}
    print(f"  Inputs:")
    for inp in sess.get_inputs():
        shape = [d if isinstance(d, int) else 'dynamic' for d in inp.shape]
        model_shapes[inp.name] = shape
        print(f"    - {inp.name}: shape={inp.shape}, type={inp.type}")
    print(f"  Outputs:")
    for out in sess.get_outputs():
        print(f"    - {out.name}: shape={out.shape}, type={out.type}")

    # Detect audio_len, audio_dim, seq_len from model
    auto_audio_len = None
    auto_audio_dim = None
    auto_seq_len = None

    for inp in sess.get_inputs():
        shape = inp.shape
        if inp.name in ('audio_features', 'encoder_output'):
            if len(shape) >= 3:
                auto_audio_len = shape[1] if isinstance(shape[1], int) else None
                auto_audio_dim = shape[2] if isinstance(shape[2], int) else None
        elif inp.name == 'target_tokens':
            if len(shape) >= 2:
                auto_seq_len = shape[1] if isinstance(shape[1], int) else None

    # Use detected shapes (CLI args override if explicitly set for non-defaults)
    audio_len = auto_audio_len if auto_audio_len else args.audio_len
    audio_dim = auto_audio_dim if auto_audio_dim else args.audio_dim
    max_tokens = auto_seq_len if auto_seq_len else args.max_tokens

    if auto_audio_len and auto_audio_len != args.audio_len:
        print(f"  [INFO] Auto-detected audio_len={auto_audio_len} (CLI default was {args.audio_len})")
    if auto_seq_len and auto_seq_len != args.max_tokens:
        print(f"  [INFO] Auto-detected seq_len={auto_seq_len}, setting max_tokens={auto_seq_len}")
        args.max_tokens = auto_seq_len

    # Load audio features with CORRECT shape
    if args.random or args.audio is None:
        print("\n[INFO] Using random noise as input (quick test mode)")
        audio_features = np.random.randn(1, audio_len, audio_dim).astype(np.float32)
    else:
        audio_features = load_audio_features(
            args.audio,
            target_steps=audio_len,
            n_mels=audio_dim,
        )

    # Generate
    for sample_idx in range(args.num_samples):
        print(f"\n{'='*50}")
        print(f"[GENERATE] Sample {sample_idx + 1}/{args.num_samples}")
        print(f"  Temperature: {args.temperature}, Top-K: {args.top_k}, Top-P: {args.top_p}")

        tokens = onnx_generate(
            session=sess,
            audio_features=audio_features,
            decoder_only=args.decoder_only,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            verbose=args.verbose,
        )

        # Decode tokens
        notes = decode_token_sequence(tokens)

        print(f"\n  Generated {len(tokens)} tokens -> {len(notes)} notes")
        print(f"  Token sequence: {' '.join(decode_token(t) for t in tokens[:30])}{'...' if len(tokens) > 30 else ''}")

        if notes:
            print(f"\n  Decoded notes:")
            for i, note in enumerate(notes[:10]):
                print(f"    [{i}] {note}")
            if len(notes) > 10:
                print(f"    ... ({len(notes) - 10} more)")

        # Save output
        if args.output:
            output_path = args.output
            if args.num_samples > 1:
                base, ext = os.path.splitext(output_path)
                output_path = f"{base}_{sample_idx}{ext}"

            result = {
                "tokens": tokens,
                "decoded_tokens": [decode_token(t) for t in tokens],
                "notes": notes,
                "num_tokens": len(tokens),
                "num_notes": len(notes),
            }
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\n  Saved to: {output_path}")

    print(f"\n[DONE] Generation complete.")


if __name__ == "__main__":
    main()
