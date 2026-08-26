# File: compat.py
"""Compatibility shims for models whose trust_remote_code files predate the
installed transformers.

Kept out of run_benchmark.py so the core pipeline stays readable and so each
shim can be deleted independently once its model is dropped or updated. Every
applied shim is recorded in the run manifest, because a shim changes what the
model actually executes and is therefore part of the experiment.

Current contents:

  Aleph-Alpha/Pharia-1-LLM-7B-control-aligned-hf (published August 2024) calls
  DynamicCache.get_max_length(), which transformers renamed to
  get_max_cache_shape() around 4.48 and later removed. Without the shim,
  generation dies with "'DynamicCache' object has no attribute
  'get_max_length'". Pinning transformers old enough for Pharia is not an
  option -- LFM2.5 is a 2026 architecture that needs a recent release.
"""


def apply_compat_shims():
    """Install shims idempotently. Returns the names of the ones applied."""
    applied = []

    try:
        from transformers.cache_utils import DynamicCache
    except Exception:
        return applied

    if not hasattr(DynamicCache, "get_max_length"):
        # The old method returned None for an unbounded dynamic cache, which is
        # what callers branch on, so None is the faithful replacement rather
        # than a placeholder.
        DynamicCache.get_max_length = lambda self: None
        applied.append("DynamicCache.get_max_length")

    return applied
