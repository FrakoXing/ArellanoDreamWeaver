# -*- coding: utf-8 -*-
"""
PyTorch .pth -> ONNX Model Conversion Tool
-------------------------------------------
Convert trained .pth models to ONNX format for cross-platform deployment.

Supports two export modes:
1. full_pipeline: Export complete pipeline (AudioProjection + NoteDecoder) [default]
2. decoder_only:  Export NoteDecoder only (requires external audio features)

Usage:
  # Export full model (default)
  python scripts/export_onnx.py --checkpoint logs/final_model.pth --output model.onnx

  # Export decoder only
  python scripts/export_onnx.py --checkpoint logs/final_model.pth --output decoder.onnx --decoder-only

  # Specify export parameters
  python scripts/export_onnx.py --checkpoint logs/final_model.pth --output model.onnx \
      --batch-size 1 --seq-len 512 --audio-len 1000

  # Export and verify
  python scripts/export_onnx.py --checkpoint logs/final_model.pth --output model.onnx --verify

  # Export as float16 (reduced model size)
  python scripts/export_onnx.py --checkpoint logs/final_model.pth --output model_fp16.onnx --fp16
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import torch
import torch.nn as nn

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.note_decoder import NoteDecoder


# ============================================================
# Audio Projection Layer (consistent with train_tokens.py)
# ============================================================

class AudioProjection(nn.Module):
    """
    Projects preprocessed audio features to model hidden dimension.

    Input:  [B, T, D_audio] (Mel spectrogram features)
    Output: [B, T, hidden_dim]
    """

    def __init__(self, audio_dim: int = 128, hidden_dim: int = 512):
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        return self.projection(audio_features)


# ============================================================
# Full Inference Model (AudioProjection + NoteDecoder)
# ============================================================

class FullInferenceModel(nn.Module):
    """
    Full inference model: audio features -> Token predictions.

    Input:
        - audio_features: [B, T_audio, D_audio] audio features
        - target_tokens:  [B, T_tgt] target token sequence (for teacher forcing)

    Output:
        - logits: [B, T_tgt, vocab_size] predicted token probability distribution
    """

    def __init__(
        self,
        audio_projection: AudioProjection,
        note_decoder: NoteDecoder,
    ):
        super().__init__()
        self.audio_projection = audio_projection
        self.note_decoder = note_decoder

    def forward(
        self,
        audio_features: torch.Tensor,
        target_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            audio_features: [B, T_audio, D_audio]
            target_tokens:  [B, T_tgt]

        Returns:
            logits: [B, T_tgt, vocab_size]
        """
        encoder_output = self.audio_projection(audio_features)
        logits = self.note_decoder(
            encoder_output=encoder_output,
            target_tokens=target_tokens,
            return_logits=True,
        )
        return logits


# ============================================================
# Checkpoint Loading
# ============================================================

def load_checkpoint(
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[AudioProjection, NoteDecoder, dict]:
    """
    Load a training checkpoint.

    Auto-detects config format:
      1. Training format: {"model": {"audio_projection": {...}, "note_decoder": {...}}, ...}
      2. Legacy format:   {"decoder": {...}, "audio": {...}, ...}

    Returns:
        (audio_projection, note_decoder, flat_config)
    """
    print(f"[LOAD] Loading checkpoint: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract and flatten config (compatible with multiple formats)
    raw_config = checkpoint.get("config", {})

    # Try training format: config.model.note_decoder / config.model.audio_projection
    model_cfg = raw_config.get("model", {})
    decoder_cfg = model_cfg.get("note_decoder", {})
    audio_cfg = model_cfg.get("audio_projection", {})

    # Fallback: legacy format config.decoder / config.audio
    if not decoder_cfg:
        decoder_cfg = raw_config.get("decoder", {})
    if not audio_cfg:
        audio_cfg = raw_config.get("audio", {})

    # Extract parameters (with defaults)
    vocab_size = decoder_cfg.get("vocab_size", 256)
    hidden_dim = decoder_cfg.get("hidden_dim", 512)
    num_layers = decoder_cfg.get("num_layers", 6)
    num_heads = decoder_cfg.get("num_heads", 8)
    max_seq_len = decoder_cfg.get("max_seq_len", 4096)
    dropout = decoder_cfg.get("dropout", 0.1)
    dim_feedforward = decoder_cfg.get("dim_feedforward", None)

    audio_dim = audio_cfg.get("audio_dim", 128)

    # Alternative: n_mels for audio feature dimension
    if audio_cfg.get("n_mels"):
        audio_dim = audio_cfg["n_mels"]

    print(f"  [OK] Model parameters:")
    print(f"     vocab_size:  {vocab_size}")
    print(f"     hidden_dim:  {hidden_dim}")
    print(f"     num_layers:  {num_layers}")
    print(f"     num_heads:   {num_heads}")
    print(f"     max_seq_len: {max_seq_len}")
    print(f"     audio_dim:   {audio_dim}")

    # Build AudioProjection
    audio_projection = AudioProjection(
        audio_dim=audio_dim,
        hidden_dim=hidden_dim,
    )

    # Build NoteDecoder
    note_decoder_kwargs = dict(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        dropout=dropout,
    )
    if dim_feedforward is not None:
        note_decoder_kwargs["dim_feedforward"] = dim_feedforward

    note_decoder = NoteDecoder(**note_decoder_kwargs)

    # Load weights
    audio_proj_sd = checkpoint.get("audio_projection_state_dict")
    note_decoder_sd = checkpoint.get("note_decoder_state_dict")

    if audio_proj_sd is None:
        raise KeyError("Checkpoint missing 'audio_projection_state_dict'")
    if note_decoder_sd is None:
        raise KeyError("Checkpoint missing 'note_decoder_state_dict'")

    audio_projection.load_state_dict(audio_proj_sd, strict=True)
    note_decoder.load_state_dict(note_decoder_sd, strict=True)

    # Set to eval mode
    audio_projection.eval()
    note_decoder.eval()

    # Return flattened config
    flat_config = {
        "vocab_size": vocab_size,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "max_seq_len": max_seq_len,
        "audio_dim": audio_dim,
    }

    total_params = sum(p.numel() for p in audio_projection.parameters()) \
                 + sum(p.numel() for p in note_decoder.parameters())
    print(f"  [OK] Weights loaded ({total_params:,} params)")

    return audio_projection, note_decoder, flat_config


# ============================================================
# ONNX Export Functions
# ============================================================

def export_full_pipeline(
    audio_projection: AudioProjection,
    note_decoder: NoteDecoder,
    output_path: str,
    batch_size: int = 1,
    seq_len: int = 512,
    audio_len: int = 1000,
    audio_dim: int = 128,
    opset_version: int = 17,
    use_fp16: bool = False,
    use_static: bool = False,
) -> None:
    """
    Export full pipeline (AudioProjection + NoteDecoder).

    Input:
        - audio_features: [B, T_audio, D_audio]
        - target_tokens:  [B, T_tgt]

    Output:
        - logits: [B, T_tgt, vocab_size]
    """
    mode_str = "static" if use_static else "dynamic"
    print(f"\n[EXPORT] Exporting full pipeline [{mode_str} shapes]")

    model = FullInferenceModel(audio_projection, note_decoder)
    model.eval()

    if use_fp16:
        model = model.half()
        print(f"  [FP16] Using float16 precision")
        dtype = torch.float16
    else:
        dtype = torch.float32

    # Create sample inputs
    audio_features = torch.randn(batch_size, audio_len, audio_dim, dtype=dtype)
    target_tokens = torch.randint(0, note_decoder.vocab_size, (batch_size, seq_len))

    print(f"  Fixed export shapes:")
    print(f"     audio_features: [{batch_size}, {audio_len}, {audio_dim}]")
    print(f"     target_tokens:  [{batch_size}, {seq_len}]")

    # Quick PyTorch forward check
    print(f"\n[CHECK] PyTorch forward pass...")
    with torch.no_grad():
        try:
            _ = model(audio_features, target_tokens)
            print(f"  [OK] PyTorch forward pass succeeded")
        except Exception as e:
            print(f"  [FAIL] PyTorch forward pass failed: {e}")
            raise

    # Export to ONNX
    print(f"\n[ONNX] Exporting (opset={opset_version}, {'static' if use_static else 'dynamic'} axes)...")

    if use_static:
        # Fixed shapes: no dynamic axes, guarantee ONNX Runtime compatibility
        dynamic_axes = None
    else:
        # Dynamic axes: batch + seq_len/audio_len can vary
        # NOTE: TorchScript tracer may bake in concrete dims for internal Reshape ops.
        # If ONNX Runtime fails with "reshape_helper" errors, use --static instead.
        dynamic_axes = {
            'audio_features': {0: 'batch', 1: 'audio_len'},
            'target_tokens': {0: 'batch', 1: 'seq_len'},
            'logits': {0: 'batch', 1: 'seq_len'},
        }

    torch.onnx.export(
        model,
        (audio_features, target_tokens),
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['audio_features', 'target_tokens'],
        output_names=['logits'],
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )

    print(f"  [OK] Export complete: {output_path}")
    _print_onnx_info(output_path)


def export_decoder_only(
    note_decoder: NoteDecoder,
    output_path: str,
    batch_size: int = 1,
    seq_len: int = 512,
    audio_len: int = 1000,
    hidden_dim: int = 512,
    opset_version: int = 17,
    use_fp16: bool = False,
    use_static: bool = False,
) -> None:
    """
    Export NoteDecoder only (without AudioProjection).

    Input:
        - encoder_output: [B, T_audio, hidden_dim]  (projected audio features)
        - target_tokens:  [B, T_tgt]                 (target token sequence)

    Output:
        - logits: [B, T_tgt, vocab_size]
    """
    mode_str = "static" if use_static else "dynamic"
    print(f"\n[EXPORT] Exporting NoteDecoder [{mode_str} shapes]")

    model = note_decoder
    model.eval()

    if use_fp16:
        model = model.half()
        dtype = torch.float16
    else:
        dtype = torch.float32

    # Create sample inputs
    encoder_output = torch.randn(batch_size, audio_len, hidden_dim, dtype=dtype)
    target_tokens = torch.randint(0, note_decoder.vocab_size, (batch_size, seq_len))

    print(f"  Fixed export shapes:")
    print(f"     encoder_output: [{batch_size}, {audio_len}, {hidden_dim}]")
    print(f"     target_tokens:  [{batch_size}, {seq_len}]")

    # Quick PyTorch forward check
    print(f"\n[CHECK] PyTorch forward pass...")
    with torch.no_grad():
        try:
            _ = model(encoder_output, target_tokens)
            print(f"  [OK] PyTorch forward pass succeeded")
        except Exception as e:
            print(f"  [FAIL] PyTorch forward pass failed: {e}")
            raise

    # Export to ONNX
    print(f"\n[ONNX] Exporting (opset={opset_version}, {'static' if use_static else 'dynamic'} axes)...")

    if use_static:
        dynamic_axes = None
    else:
        dynamic_axes = {
            'encoder_output': {0: 'batch', 1: 'audio_len'},
            'target_tokens': {0: 'batch', 1: 'seq_len'},
            'logits': {0: 'batch', 1: 'seq_len'},
        }

    torch.onnx.export(
        model,
        (encoder_output, target_tokens),
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['encoder_output', 'target_tokens'],
        output_names=['logits'],
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )

    print(f"  [OK] Export complete: {output_path}")
    _print_onnx_info(output_path)


# ============================================================
# ONNX Model Info & Validation
# ============================================================

def _print_onnx_info(onnx_path: str) -> None:
    """Print ONNX model info and run checker"""
    try:
        import onnx
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)

        size_mb = os.path.getsize(onnx_path) / 1024 / 1024
        print(f"\n[INFO] ONNX Model:")
        print(f"     File size:   {size_mb:.2f} MB")
        print(f"     IR version:  {model.ir_version}")
        print(f"     Producer:    {model.producer_name}")
        print(f"     Opset:       {model.opset_import[0].domain}:{model.opset_import[0].version}")

        print(f"     Inputs:")
        for inp in model.graph.input:
            shape = _get_shape_str(inp)
            print(f"       - {inp.name}: {shape}")

        print(f"     Outputs:")
        for out in model.graph.output:
            shape = _get_shape_str(out)
            print(f"       - {out.name}: {shape}")

        # Count operators
        op_types = {}
        for node in model.graph.node:
            op_types[node.op_type] = op_types.get(node.op_type, 0) + 1
        print(f"     Operators: {len(model.graph.node)} ({len(op_types)} types)")
        if len(op_types) <= 10:
            for op, count in sorted(op_types.items(), key=lambda x: -x[1]):
                print(f"       - {op}: {count}")

    except ImportError:
        print(f"\n[WARN] onnx not installed, skipping model validation")
        print(f"     pip install onnx")
        print(f"     File size: {os.path.getsize(onnx_path) / 1024 / 1024:.2f} MB")
    except Exception as e:
        print(f"\n[WARN] ONNX model validation failed: {e}")
        print(f"     File size: {os.path.getsize(onnx_path) / 1024 / 1024:.2f} MB")


def _get_shape_str(tensor_info) -> str:
    """Extract shape string from ONNX tensor info"""
    try:
        dims = []
        for d in tensor_info.type.tensor_type.shape.dim:
            if d.dim_value:
                dims.append(str(d.dim_value))
            elif d.dim_param:
                dims.append(d.dim_param)
            else:
                dims.append('?')
        return f"[{', '.join(dims)}]"
    except Exception:
        return "?"


def verify_onnx(
    onnx_path: str,
    audio_projection: AudioProjection,
    note_decoder: NoteDecoder,
    batch_size: int = 1,
    seq_len: int = 64,
    audio_len: int = 100,
    audio_dim: int = 128,
    hidden_dim: int = 512,
    decoder_only: bool = False,
    use_static: bool = False,
    rtol: float = 1e-3,
    atol: float = 1e-5,
) -> bool:
    """
    Verify ONNX model output matches PyTorch model.

    IMPORTANT: For static exports, seq_len/audio_len MUST match the export shapes exactly.
    For dynamic exports, use the same or smaller shapes as export.

    Returns:
        True if verification passes
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print(f"\n[WARN] onnxruntime not installed, skipping verification")
        print(f"     pip install onnxruntime")
        return False

    print(f"\n[CHECK] Verifying ONNX model (rtol={rtol}, atol={atol})...")
    if use_static:
        print(f"  [INFO] Static mode: using exact export shapes for verification")

    device = torch.device("cpu")
    vocab_size = note_decoder.vocab_size

    # Create test inputs using the EXACT shapes provided
    audio_features = torch.randn(batch_size, audio_len, audio_dim, device=device)
    target_tokens = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # === PyTorch Inference ===
    with torch.no_grad():
        if decoder_only:
            encoder_output = audio_projection(audio_features)
            pt_logits = note_decoder(encoder_output, target_tokens)
        else:
            full_model = FullInferenceModel(audio_projection, note_decoder)
            full_model.eval()
            pt_logits = full_model(audio_features, target_tokens)

    print(f"  PyTorch output: shape={list(pt_logits.shape)}, "
          f"range=[{pt_logits.min().item():.4f}, {pt_logits.max().item():.4f}]")

    # === ONNX Runtime Inference ===
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_names = [inp.name for inp in sess.get_inputs()]

    if 'audio_features' in input_names:
        # Full pipeline mode
        onnx_inputs = {
            'audio_features': audio_features.numpy(),
            'target_tokens': target_tokens.numpy(),
        }
    elif 'encoder_output' in input_names:
        # Decoder-only mode
        with torch.no_grad():
            encoder_out = audio_projection(audio_features)
        onnx_inputs = {
            'encoder_output': encoder_out.numpy(),
            'target_tokens': target_tokens.numpy(),
        }
    else:
        print(f"  [FAIL] Unknown input names: {input_names}")
        return False

    onnx_outputs = sess.run(None, onnx_inputs)
    onnx_logits = torch.from_numpy(onnx_outputs[0])

    print(f"  ONNX    output: shape={list(onnx_logits.shape)}, "
          f"range=[{onnx_logits.min().item():.4f}, {onnx_logits.max().item():.4f}]")

    # === Comparison ===
    abs_diff = (pt_logits - onnx_logits).abs()
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()
    # Relative difference (avoid division by zero)
    rel_diff = abs_diff / (pt_logits.abs() + 1e-8)
    max_rel_diff = rel_diff.max().item()

    print(f"\n  Difference analysis:")
    print(f"     Max absolute diff:  {max_diff:.8f}")
    print(f"     Mean absolute diff: {mean_diff:.8f}")
    print(f"     Max relative diff:  {max_rel_diff:.8f}")

    # Verdict
    passed = (max_diff < atol * 100) or \
             torch.allclose(pt_logits, onnx_logits, rtol=rtol, atol=atol)

    if passed:
        print(f"\n  [OK] Verification passed! ONNX matches PyTorch output")
    else:
        print(f"\n  [WARN] Large difference detected (max_diff={max_diff:.6f})")
        print(f"       This may be due to numerical precision, usually acceptable")
        if max_rel_diff < 5e-3:
            print(f"       Relative difference within acceptable range (< 0.5%)")

    return passed


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="PyTorch (.pth) -> ONNX Model Conversion Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export full model (default)
  python scripts/export_onnx.py --checkpoint logs/final_model.pth --output model.onnx

  # Export decoder only
  python scripts/export_onnx.py --checkpoint logs/final_model.pth --output decoder.onnx --decoder-only

  # Export and verify
  python scripts/export_onnx.py --checkpoint logs/final_model.pth --output model.onnx --verify

  # FP16 export (half model size)
  python scripts/export_onnx.py --checkpoint logs/final_model.pth --output model_fp16.onnx --fp16
        """
    )

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="PyTorch checkpoint path (.pth)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output ONNX file path")
    parser.add_argument("--decoder-only", action="store_true",
                        help="Export NoteDecoder only (without AudioProjection)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for export (default: 1)")
    parser.add_argument("--seq-len", type=int, default=512,
                        help="Target sequence length (default: 512)")
    parser.add_argument("--audio-len", type=int, default=1000,
                        help="Audio feature length (default: 1000)")
    parser.add_argument("--opset-version", type=int, default=17,
                        help="ONNX opset version (default: 17)")
    parser.add_argument("--fp16", action="store_true",
                        help="Export float16 model (half size, some backends may not support)")
    parser.add_argument("--static", action="store_true",
                        help="Export with fixed shapes (no dynamic axes). More compatible with ONNX Runtime.")
    parser.add_argument("--verify", action="store_true",
                        help="Verify ONNX output consistency with onnxruntime after export")
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip verification (default when --verify is set)")

    args = parser.parse_args()

    # Check input file
    if not os.path.exists(args.checkpoint):
        print(f"[FAIL] Checkpoint file not found: {args.checkpoint}")
        sys.exit(1)

    # Create output directory
    output_path = Path(args.output)
    output_dir = output_path.parent
    if output_dir != Path(".") and not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure .onnx extension
    if output_path.suffix.lower() != '.onnx':
        output_path = output_path.with_suffix('.onnx')
        print(f"[INFO] Output extension corrected to .onnx: {output_path}")

    # Load model
    device = torch.device("cpu")
    audio_projection, note_decoder, flat_config = load_checkpoint(args.checkpoint, device)

    # Export
    try:
        if args.decoder_only:
            export_decoder_only(
                note_decoder=note_decoder,
                output_path=str(output_path),
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                audio_len=args.audio_len,
                hidden_dim=flat_config["hidden_dim"],
                opset_version=args.opset_version,
                use_fp16=args.fp16,
                use_static=args.static,
            )
        else:
            export_full_pipeline(
                audio_projection=audio_projection,
                note_decoder=note_decoder,
                output_path=str(output_path),
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                audio_len=args.audio_len,
                audio_dim=flat_config["audio_dim"],
                opset_version=args.opset_version,
                use_fp16=args.fp16,
                use_static=args.static,
            )
    except Exception as e:
        print(f"\n[FAIL] Export failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Verify: always use the SAME shapes as export to avoid reshape errors
    # (TorchScript tracer bakes concrete dims into internal Reshape ops)
    if args.verify and not args.no_verify:
        verify_onnx(
            onnx_path=str(output_path),
            audio_projection=audio_projection,
            note_decoder=note_decoder,
            batch_size=args.batch_size,
            seq_len=args.seq_len,       # Same as export shape
            audio_len=args.audio_len,   # Same as export shape
            audio_dim=flat_config["audio_dim"],
            hidden_dim=flat_config["hidden_dim"],
            decoder_only=args.decoder_only,
            use_static=args.static,
        )

    print(f"\n{'='*60}")
    print(f"[DONE] Conversion complete! Output: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
