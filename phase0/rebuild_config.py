#!/usr/bin/env python3
"""rebuild_config.py -- H10: rebuild an official DFloat11 output config.json
from (source config.json + dfloat11_config settings) using ONLY the Python
standard library. No torch, no transformers, no diffusers, no safetensors.

Ground truth for every transformation below was established by diffing:
  - SOURCE  ../bf16-exponent-compression/real_model/config.json
            (Qwen3-0.6B, written by transformers 4.51.0)
  - OUTPUT  phase0/out/official/qwen3-0.6b-layers-only/config.json
            (produced by dfloat11 0.5.0's compress_model(), which calls
             model.config.dfloat11_config = {...}; model.save_pretrained(...),
             i.e. transformers 5.17.0's PretrainedConfig.to_json_file())

and the same diff repeated, with identical results, on the 4-layer truncated
model (phase0/corpus/tier0/qwen3-trunc -> qwen3-trunc-layers-only-dir).

See phase0/H10_H11_RESULTS.md for the full key-by-key table and the verdict.
This script's docstring only records the *mechanism*; the results doc records
the *evidence and judgment calls*.

WHAT model.save_pretrained() ACTUALLY DOES (confirmed from dfloat11.py
lines 606-620): it does not "add a few keys" to the source config. It hands
the fully-parsed transformers Qwen3Config object to that library's own
serializer, which re-normalizes the ENTIRE config to the schema of whatever
transformers version is installed at compress time. Three renames/expansions
were observed between 4.51.0 and 5.17.0 (both arbitrary versions -- nothing
pins them, and a different pair of versions could show different deltas):

  1. "torch_dtype" -> "dtype"                              (verbatim rename)
  2. "rope_theta" + "rope_scaling" -> "rope_parameters"     (dict-of-two -> nested dict)
  3. new key "layer_types": ["full_attention", ...] * num_hidden_layers
     (Qwen3-specific: derived from num_hidden_layers + sliding_window +
     use_sliding_window; only the "no sliding window at all" case is
     confirmed here, since both fixtures have sliding_window=null)
  4. new key "pad_token_id": null
     (PretrainedConfig's base class always tracks pad_token_id/bos_token_id/
     eos_token_id; a source config that omits pad_token_id round-trips to an
     explicit null once transformers rebuilds the object)
  5. "transformers_version" bumped from the value baked into the SOURCE file
     to str(transformers.__version__) of the installed library
     -- THIS FIELD CANNOT BE COMPUTED FROM THE SOURCE CONFIG AT ALL. It is a
     read of the currently-installed package's own version string, which by
     construction a torch/transformers-free binary has no way to produce.
     See RISK FIELDS below.

Everything else (architectures, attention_bias, attention_dropout,
bos_token_id, eos_token_id, head_dim, hidden_act, hidden_size,
initializer_range, intermediate_size, max_position_embeddings,
max_window_layers, model_type, num_attention_heads, num_hidden_layers,
num_key_value_heads, rms_norm_eps, sliding_window, tie_word_embeddings,
use_cache, use_sliding_window, vocab_size) passes through byte-for-byte,
unchanged in value.

Key order in the OUTPUT file is plain alphabetical (sort_keys=True), indent=2,
UTF-8, trailing "\n". Confirmed against phase0/out/official/*/config.json.

RISK FIELDS (library-version-dependent; a Rust binary has no library to ask):
  - transformers_version: cannot be computed. Recommendation: a df11pack.toml
    (or CLI flag) records a *target* transformers-version string the operator
    declares df11pack is emulating for this run, e.g. "5.17.0" or "4.51.0",
    and rebuild_config uses that string verbatim. Ship a small versioned table
    of (transformers_version -> which of the 5 transformations above apply)
    inside df11pack, updated by hand when a new transformers release changes
    the serialization schema again. This turns an unknowable field into a
    declared, auditable one -- exactly like pattern_dict already is a per-model
    choice (FINDINGS.md 0.2c).
  - layer_types / rope_parameters shape: these are *derived*, not read from an
    installed library's version string, so they ARE computable from the
    source config's own fields (num_hidden_layers, sliding_window,
    use_sliding_window, rope_theta, rope_scaling) -- but the derivation rule
    itself belongs to a specific transformers version and a specific model
    architecture (Qwen3Config here). Getting this wrong produces a
    functioning-but-not-byte-identical config, not a crash (see below).
  - dfloat11_config.version: NOT library-version-dependent -- it is the
    dfloat11 PACKAGE's own version string, hardcoded at injection time by
    dfloat11.py itself (`version` is a module-level constant in dfloat11.py,
    not read from an installed transformers/diffusers). df11pack should treat
    it as a df11pack-format-version constant it owns and picks explicitly,
    exactly as FINDINGS.md 0.2 already flagged ("df11pack must emit the right
    version per target and tolerate both on read").

WHY BYTE-IDENTITY MAY NOT MATTER IN PRACTICE (a hypothesis, NOT proven on
this machine -- no transformers install available to test it): transformers'
AutoConfig.from_pretrained is documented and generally observed to accept
legacy/old-style kwargs (rope_theta, rope_scaling, torch_dtype) even on
versions that themselves serialize the new-style schema, because config
classes accept both the current and prior kwarg names for backward
compatibility. If true, a df11pack config.json that keeps the SOURCE's
original field names/shapes and only *adds* dfloat11_config (skipping the
5 renormalization steps entirely) would still load correctly through
AutoConfig on the end user's transformers, even though it is not
byte-identical to what compress_model would have written. This is exactly
the "functional equivalence" fallback mode this script also offers
(--minimal), and it is the recommended default until someone with a
transformers install confirms or refutes the compatibility claim.
"""
import argparse
import copy
import json
import sys


# Keys observed to be removed from the source and replaced/renamed during
# transformers' PretrainedConfig round-trip (see module docstring, item 1-2).
_RENAMED_OR_FOLDED = ("torch_dtype", "rope_theta", "rope_scaling")


def derive_layer_types(config):
    """Qwen3-specific. Confirmed only for sliding_window=None (both fixtures).

    Real Qwen3 hybrid-attention checkpoints (sliding_window set,
    use_sliding_window True) are UNTESTED here -- no such fixture exists in
    phase0/corpus or in the two local outputs. Flagged, not guessed at.
    """
    n = config["num_hidden_layers"]
    if config.get("sliding_window") is None or not config.get("use_sliding_window", False):
        return ["full_attention"] * n
    raise NotImplementedError(
        "layer_types derivation for sliding_window != None is untested on this "
        "machine (no fixture exercises it); refusing to guess rather than emit "
        "a silently wrong value."
    )


def derive_rope_parameters(config):
    rope_theta = config["rope_theta"]
    rope_scaling = config.get("rope_scaling")
    if rope_scaling is None:
        return {"rope_theta": rope_theta, "rope_type": "default"}
    # Untested: no fixture has a non-null rope_scaling. Best-effort merge,
    # flagged so a caller can tell this path was taken.
    out = dict(rope_scaling)
    out.setdefault("rope_theta", rope_theta)
    out.setdefault("rope_type", rope_scaling.get("rope_type", "default"))
    return out


def build_dfloat11_config(version, threads_per_block, bytes_per_thread, pattern_dict):
    return {
        "version": version,
        "threads_per_block": list(threads_per_block),
        "bytes_per_thread": bytes_per_thread,
        "pattern_dict": pattern_dict,
    }


def rebuild_full(source_config, dfloat11_config, transformers_version_target):
    """Reproduce the FULL transformers-normalized schema (5 transformations).

    Byte-identical to the official output IFF transformers_version_target is
    exactly the version the official run used, AND the architecture-specific
    derivations above (layer_types) are correct for this model family.
    """
    out = copy.deepcopy(source_config)

    if "torch_dtype" in out:
        out["dtype"] = out.pop("torch_dtype")

    if "rope_theta" in out:
        out["rope_parameters"] = derive_rope_parameters(out)
        out.pop("rope_theta", None)
        out.pop("rope_scaling", None)

    out["layer_types"] = derive_layer_types(source_config)
    out.setdefault("pad_token_id", None)
    out["transformers_version"] = transformers_version_target
    out["dfloat11_config"] = dfloat11_config
    return out


def rebuild_minimal(source_config, dfloat11_config):
    """Functional-equivalence fallback: keep the source schema verbatim,
    only add dfloat11_config. Not byte-identical to the official output, but
    requires zero knowledge of any library's serialization schema or version.
    Recommended default (see module docstring, "WHY BYTE-IDENTITY MAY NOT
    MATTER IN PRACTICE").
    """
    out = copy.deepcopy(source_config)
    out["dfloat11_config"] = dfloat11_config
    return out


def dump(config_dict):
    return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("source_config", help="path to the SOURCE config.json")
    ap.add_argument("--pattern-dict-json", required=True,
                     help="path to a JSON file holding the dfloat11 pattern_dict")
    ap.add_argument("--dfloat11-version", default="0.5.0")
    ap.add_argument("--threads-per-block", type=int, nargs="+", default=[512])
    ap.add_argument("--bytes-per-thread", type=int, default=8)
    ap.add_argument("--mode", choices=("full", "minimal"), default="minimal")
    ap.add_argument("--transformers-version-target", default=None,
                     help="required for --mode full: the transformers version "
                          "string to emit verbatim (cannot be derived)")
    ap.add_argument("-o", "--out", default=None, help="output path; default stdout")
    args = ap.parse_args(argv)

    with open(args.source_config, "r", encoding="utf-8") as f:
        source_config = json.load(f)
    with open(args.pattern_dict_json, "r", encoding="utf-8") as f:
        pattern_dict = json.load(f)

    dfloat11_config = build_dfloat11_config(
        args.dfloat11_version, args.threads_per_block, args.bytes_per_thread, pattern_dict,
    )

    if args.mode == "full":
        if not args.transformers_version_target:
            print("error: --mode full requires --transformers-version-target", file=sys.stderr)
            return 2
        out = rebuild_full(source_config, dfloat11_config, args.transformers_version_target)
    else:
        out = rebuild_minimal(source_config, dfloat11_config)

    text = dump(out)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
