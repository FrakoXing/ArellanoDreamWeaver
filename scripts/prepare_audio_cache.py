# -*- coding: utf-8 -*-
"""
Audio Feature Cache Preprocessor
=================================
Pre-process all audio files into Mel spectrogram .pt cache files.
This eliminates real-time librosa processing from the training loop,
resulting in 50-100x faster data loading during training.

Usage:
  # Process all audio files matched to tokens
  python scripts/prepare_audio_cache.py

  # Specify directories
  python scripts/prepare_audio_cache.py --tokens-dir data/tokens --audio-dir data/audio --cache-dir data/audio_cache

  # Re-process specific range (for parallel processing)
  python scripts/prepare_audio_cache.py --start 0 --end 1000

  # Use more CPU cores
  python scripts/prepare_audio_cache.py --workers 8
"""

import os
import sys
import json
import argparse
import hashlib
from pathlib import Path
from typing import Optional, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def find_audio_for_token(token_path: Path, audio_dir: Path) -> Optional[Path]:
    """Match token file to its audio file by stem name."""
    base_name = token_path.stem
    for ext in ['.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac']:
        audio_path = audio_dir / f"{base_name}{ext}"
        if audio_path.exists():
            return audio_path
    return None


def extract_mel_features(
    audio_path: Path,
    target_steps: int = 4096,
    n_mels: int = 128,
    sample_rate: int = 22050,
    n_fft: int = 2048,
    hop_length: int = 512,
    duration: Optional[float] = None,
) -> np.ndarray:
    """
    Extract normalized Mel spectrogram from audio file.

    Returns:
        float32 array [target_steps, n_mels]
    """
    import librosa

    # Load audio (mono, target sample rate)
    waveform, sr = librosa.load(str(audio_path), sr=sample_rate, mono=True)

    # Truncate to duration if specified
    if duration is not None:
        max_samples = int(duration * sr)
        waveform = waveform[:max_samples]

    # Mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=waveform, sr=sr,
        n_mels=n_mels, n_fft=n_fft, hop_length=hop_length,
        power=2.0,
    )

    # Convert to dB
    mel_db = librosa.power_to_db(mel_spec, ref=np.max, top_db=80.0)

    # Shape: [n_mels, time] -> [time, n_mels]
    mel_db = mel_db.T.astype(np.float32)

    # Normalize (per-sample z-score)
    mean = mel_db.mean()
    std = mel_db.std() + 1e-6
    mel_norm = (mel_db - mean) / std

    # Resize to target_steps
    if mel_norm.shape[0] != target_steps:
        mel_norm = _interpolate_time(mel_norm, target_steps)

    return mel_norm


def _interpolate_time(features: np.ndarray, target_steps: int) -> np.ndarray:
    """Linear interpolation along time axis."""
    src_len, n_feats = features.shape
    src_indices = np.linspace(0, src_len - 1, src_len)
    tgt_indices = np.linspace(0, src_len - 1, target_steps)

    result = np.zeros((target_steps, n_feats), dtype=np.float32)
    for i in range(n_feats):
        result[:, i] = np.interp(tgt_indices, src_indices, features[:, i])
    return result


def process_one(
    token_path: Path,
    audio_dir: Path,
    cache_dir: Path,
    target_steps: int,
    n_mels: int,
    sample_rate: int,
) -> Tuple[str, bool, str]:
    """
    Process a single token-audio pair.

    Returns:
        (cache_key, success, message)
    """
    cache_key = token_path.stem
    cache_path = cache_dir / f"{cache_key}.pt"

    # Skip if cache already exists and is valid
    if cache_path.exists():
        try:
            cached = torch.load(cache_path, map_location='cpu', weights_only=True)
            if isinstance(cached, torch.Tensor) and cached.shape == (target_steps, n_mels):
                return (cache_key, True, "already cached")
        except Exception:
            cache_path.unlink(missing_ok=True)

    # Find audio file
    audio_path = find_audio_for_token(token_path, audio_dir)
    if audio_path is None:
        return (cache_key, False, "no audio file found")

    # Estimate duration from token count
    duration = None
    try:
        if token_path.suffix == '.json':
            with open(token_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            num_tokens = len(data.get('tokens', []))
            # Rough estimate: average chart ~180s, tokens scale linearly
            # max 61k tokens ~ 180s, so duration = tokens / 340
            duration = max(30, min(600, num_tokens / 340.0))
        else:
            duration = 180.0  # default for .pt files
    except Exception:
        duration = 180.0

    # Extract features
    try:
        features = extract_mel_features(
            audio_path,
            target_steps=target_steps,
            n_mels=n_mels,
            sample_rate=sample_rate,
            duration=duration,
        )
    except Exception as e:
        return (cache_key, False, f"audio processing error: {e}")

    # Save cache
    tensor = torch.from_numpy(features)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, cache_path)

    return (cache_key, True, f"cached ({features.shape[0]}x{features.shape[1]})")


def main():
    parser = argparse.ArgumentParser(description="Pre-process audio files to Mel spectrogram cache")
    parser.add_argument("--tokens-dir", default="data/tokens", help="Token files directory")
    parser.add_argument("--audio-dir", default="data/audio", help="Audio files directory")
    parser.add_argument("--cache-dir", default="data/audio_cache", help="Output cache directory")
    parser.add_argument("--target-steps", type=int, default=4096, help="Target time steps (match max_seq_len)")
    parser.add_argument("--n-mels", type=int, default=128, help="Number of Mel bands")
    parser.add_argument("--sample-rate", type=int, default=22050, help="Audio sample rate")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers for processing")
    parser.add_argument("--start", type=int, default=0, help="Start index (for parallel runs)")
    parser.add_argument("--end", type=int, default=None, help="End index (exclusive)")
    parser.add_argument("--force", action="store_true", help="Re-process even if cache exists")

    args = parser.parse_args()

    tokens_dir = Path(args.tokens_dir)
    audio_dir = Path(args.audio_dir)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Collect token files
    token_files = sorted(
        list(tokens_dir.glob("*.json")) + list(tokens_dir.glob("*.pt"))
    )

    if not token_files:
        print(f"[FAIL] No token files found in {tokens_dir}")
        sys.exit(1)

    # Apply range
    end = args.end if args.end is not None else len(token_files)
    token_files = token_files[args.start:end]

    print(f"[PREP] Processing {len(token_files)} token files")
    print(f"  Audio dir:   {audio_dir}")
    print(f"  Cache dir:   {cache_dir}")
    print(f"  Target shape: [{args.target_steps}, {args.n_mels}]")
    print(f"  Workers:     {args.workers}")
    if args.force:
        print(f"  Mode:        FORCE re-process all")

    # Process (parallel or sequential)
    success_count = 0
    skip_count = 0
    fail_count = 0

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    process_one, tf, audio_dir, cache_dir,
                    args.target_steps, args.n_mels, args.sample_rate
                ): tf
                for tf in token_files
            }

            for future in as_completed(futures):
                key, ok, msg = future.result()
                if ok:
                    if msg == "already cached":
                        skip_count += 1
                    else:
                        success_count += 1
                else:
                    fail_count += 1
                    if fail_count <= 10:
                        print(f"  [FAIL] {key}: {msg}")

                total = success_count + skip_count + fail_count
                if total % 100 == 0:
                    print(f"  Progress: {total}/{len(token_files)} "
                          f"(ok={success_count}, cached={skip_count}, fail={fail_count})")
    else:
        for i, tf in enumerate(token_files):
            key, ok, msg = process_one(
                tf, audio_dir, cache_dir,
                args.target_steps, args.n_mels, args.sample_rate
            )
            if ok:
                if msg == "already cached":
                    skip_count += 1
                else:
                    success_count += 1
            else:
                fail_count += 1
                print(f"  [FAIL] {key}: {msg}")

            if (i + 1) % 100 == 0:
                print(f"  Progress: {i+1}/{len(token_files)} "
                      f"(ok={success_count}, cached={skip_count}, fail={fail_count})")

    print(f"\n[DONE] Processed {len(token_files)} files")
    print(f"  Newly cached:  {success_count}")
    print(f"  Already cached: {skip_count}")
    print(f"  Failed:         {fail_count}")
    print(f"  Cache size:     {_dir_size_mb(cache_dir):.1f} MB")


def _dir_size_mb(directory: Path) -> float:
    """Calculate directory size in MB."""
    total = 0
    for f in directory.rglob("*.pt"):
        total += f.stat().st_size
    return total / 1024 / 1024


if __name__ == "__main__":
    main()
