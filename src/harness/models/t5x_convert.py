"""Convert public Google T5X Flan-T5 checkpoints (GCS, zarr v2 + gzip) to a Hugging Face directory.

Hugging Face Hub is unreachable in the sandbox this project was built in, so the open
Flan-T5 weights are taken from ``gs://t5-data/pretrained_models/t5x/flan_t5_*`` and
converted here with numpy only (no tensorstore / t5x / jax dependency).
"""
from __future__ import annotations

import gzip
import json
import shutil
from pathlib import Path

import numpy as np

# (d_model, d_kv, d_ff, num_layers, num_heads) for the Flan-T5 (T5 v1.1) family.
FLAN_T5_SHAPES: dict[str, tuple[int, int, int, int, int]] = {
    "flan_t5_small": (512, 64, 1024, 8, 6),
    "flan_t5_base": (768, 64, 2048, 12, 12),
    "flan_t5_large": (1024, 64, 2816, 24, 16),
    "flan_t5_xl": (2048, 64, 5120, 24, 32),
}


def read_zarr(path: Path) -> np.ndarray:
    """Read a zarr v2 array with gzip chunks stored on the local filesystem."""
    meta = json.loads((path / ".zarray").read_text())
    shape, chunks = tuple(meta["shape"]), tuple(meta["chunks"])
    dtype = np.dtype(meta["dtype"])
    if meta["compressor"] is not None and meta["compressor"]["id"] != "gzip":
        raise ValueError(f"unsupported compressor {meta['compressor']}")
    out = np.zeros(shape, dtype=dtype)
    if not shape:  # scalar
        raw = (path / "0").read_bytes()
        return np.frombuffer(gzip.decompress(raw), dtype=dtype).reshape(())
    grid = [range(-(-s // c)) for s, c in zip(shape, chunks)]
    import itertools

    for idx in itertools.product(*grid):
        f = path / meta["dimension_separator"].join(str(i) for i in idx)
        if not f.exists():
            continue
        raw = f.read_bytes()
        if meta["compressor"] is not None:
            raw = gzip.decompress(raw)
        block = np.frombuffer(raw, dtype=dtype).reshape(chunks)
        sl = tuple(slice(i * c, min((i + 1) * c, s)) for i, c, s in zip(idx, chunks, shape))
        out[sl] = block[tuple(slice(0, x.stop - x.start) for x in sl)]
    return out


def convert(name: str, ckpt_root: Path, spiece_path: Path, out_dir: Path) -> Path:
    """Convert ``flan_t5_<size>`` to an HF ``T5ForConditionalGeneration`` directory."""
    import torch
    from transformers import T5Config, T5ForConditionalGeneration

    d_model, d_kv, d_ff, n_layers, n_heads = FLAN_T5_SHAPES[name]
    ckpt = next((ckpt_root / name).glob("checkpoint_*"))
    cfg = T5Config(
        vocab_size=32128, d_model=d_model, d_kv=d_kv, d_ff=d_ff, num_layers=n_layers,
        num_decoder_layers=n_layers, num_heads=n_heads, relative_attention_num_buckets=32,
        relative_attention_max_distance=128, dropout_rate=0.0, layer_norm_epsilon=1e-6,
        feed_forward_proj="gated-gelu", tie_word_embeddings=False, is_encoder_decoder=True,
        decoder_start_token_id=0, pad_token_id=0, eos_token_id=1,
    )
    # transformers>=5 forces tying inside __init__ (the kwarg only controls output rescaling);
    # T5 v1.1 / Flan-T5 has an untied, un-rescaled output head.
    cfg.tie_word_embeddings = False
    model = T5ForConditionalGeneration(cfg)
    g = lambda k: torch.from_numpy(read_zarr(ckpt / f"target.{k}").astype(np.float32))  # noqa: E731
    sd: dict[str, torch.Tensor] = {}
    emb = g("token_embedder.embedding")
    sd["shared.weight"] = emb
    sd["encoder.embed_tokens.weight"] = emb
    sd["decoder.embed_tokens.weight"] = emb
    sd["lm_head.weight"] = g("decoder.logits_dense.kernel").T.contiguous()
    sd["encoder.final_layer_norm.weight"] = g("encoder.encoder_norm.scale")
    sd["decoder.final_layer_norm.weight"] = g("decoder.decoder_norm.scale")
    sd["encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"] = g(
        "encoder.relpos_bias.rel_embedding").T.contiguous()
    sd["decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"] = g(
        "decoder.relpos_bias.rel_embedding").T.contiguous()
    attn = {"query": "q", "key": "k", "value": "v", "out": "o"}
    for i in range(n_layers):
        e, dd = f"encoder.layers_{i}", f"decoder.layers_{i}"
        eb, db = f"encoder.block.{i}", f"decoder.block.{i}"
        for a, h in attn.items():
            sd[f"{eb}.layer.0.SelfAttention.{h}.weight"] = g(f"{e}.attention.{a}.kernel").T.contiguous()
            sd[f"{db}.layer.0.SelfAttention.{h}.weight"] = g(f"{dd}.self_attention.{a}.kernel").T.contiguous()
            sd[f"{db}.layer.1.EncDecAttention.{h}.weight"] = g(
                f"{dd}.encoder_decoder_attention.{a}.kernel").T.contiguous()
        sd[f"{eb}.layer.0.layer_norm.weight"] = g(f"{e}.pre_attention_layer_norm.scale")
        sd[f"{eb}.layer.1.layer_norm.weight"] = g(f"{e}.pre_mlp_layer_norm.scale")
        sd[f"{db}.layer.0.layer_norm.weight"] = g(f"{dd}.pre_self_attention_layer_norm.scale")
        sd[f"{db}.layer.1.layer_norm.weight"] = g(f"{dd}.pre_cross_attention_layer_norm.scale")
        sd[f"{db}.layer.2.layer_norm.weight"] = g(f"{dd}.pre_mlp_layer_norm.scale")
        for blk, src, ly in ((eb, e, 1), (db, dd, 2)):
            for w in ("wi_0", "wi_1", "wo"):
                sd[f"{blk}.layer.{ly}.DenseReluDense.{w}.weight"] = g(f"{src}.mlp.{w}.kernel").T.contiguous()
    missing, unexpected = model.load_state_dict(sd, strict=False)
    real_missing = [m for m in missing if "embed_tokens" not in m]
    if real_missing or unexpected:
        raise RuntimeError(f"conversion mismatch missing={real_missing} unexpected={unexpected}")
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    shutil.copy(spiece_path, out_dir / "spiece.model")
    return out_dir
