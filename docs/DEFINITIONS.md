# Model definitions

A definition tells df11pack which tensors of a model to compress, and how to group
them. It is the `pattern_dict` the official compressor takes, plus a few fields.
The 35 built-in ones are listed by `df11pack architectures`.

## Format

```toml
name = "flux-schnell-comfyui"        # what --arch takes
layout = "comfyui-native"            # transformers | diffusers | diffusers-single | comfyui-native
format_version = "0.5.0"             # dfloat11_config.version written to config.json
threads_per_block = [512]            # kernel geometry; always these two values today
bytes_per_thread = 8
source = "where this came from"      # required: a definition without provenance is a guess
file = "model.safetensors"           # optional: single-file name, if not the layout's default

[keys]                               # optional: source name -> output name
strip_prefix = ["net."]              # removed where present
drop = ['^accum_']                   # regexes; matching tensors are left out

[[keys.rename]]                      # applied in order, first match only
pattern = '\.scale$'
replacement = ".weight"

[[unit]]                             # one per pattern_dict entry, in its order
pattern = 'double_blocks\.\d+'       # Python regex, matched against the whole module name
attrs = [                            # concatenation order: it fixes every compressed byte
    "img_mod.lin",
    "img_attn.qkv",
]

[[unit]]
pattern = "lm_head"
attrs = []                           # empty: the module itself is one tensor
```

Rules df11pack enforces, each with an error that says what to fix:

- Patterns match with Python's `re.fullmatch` semantics. A possessive quantifier is
  accepted only as a trailing `\d++`.
- Every attribute of a matched unit must exist, and every tensor to compress must
  be BF16.
- Units may not nest, and a unit that lists attrs may not also have its own
  `.weight` (the official tool would ignore the attrs if it were a Linear).
- Output names are the source names unless `[keys]` says otherwise. Key rules only
  rename; tensor bytes never change. Two source tensors may not map to one name.
- With `tie_word_embeddings` in the source's `config.json`, a standalone `lm_head`
  unit is compressed from the embedding, as the official tool does.
- Layouts: `transformers` and `diffusers` write one shard per unit plus a remainder
  file and `config.json`. `diffusers-single` writes one
  `diffusion_pytorch_model.safetensors` plus `config.json`. `comfyui-native` writes one
  `model.safetensors`, strips the `model.diffusion_model.` checkpoint prefix, and
  writes no config.

## Using your own

```sh
df11pack compress model/ --arch my-model.toml -o out/ --safe
df11pack verify out/ --source model/ --arch my-model.toml --level full
```

Or put TOML files in a folder and set `DF11PACK_ARCH_DIR`: they add to the built-in
set, and replace a built-in one of the same name.

`--safe` proves the output decodes back to the source. It cannot prove the output
matches what the official tool would write for that model. For that, compare
against an official release, or ask for it to be added as a built-in (below).

## Adding a built-in definition

Built-in definitions are generated, so their order can never drift from their
source:

1. Add an entry to `phase0/fixtures/official_pattern_dicts.json`: `repo` (the
   provenance), `layout`, `description`, `version`, `threads_per_block`,
   `bytes_per_thread`, `pattern_dict` (in its exact order), and `file` if needed.
2. Run `python3 phase0/gen_arch_defs.py`, which writes `data/architectures/<name>.toml`.
3. `cargo test`. The drift test checks the TOML against the entry.

A maintainer then adds the byte-identity fixture: a stand-in model run through the
official compressor (`phase0/make_synthetic.py <name>`, needs the Phase 0
environment). If the model has an ungated checkpoint and a published DF11 file, add
them to `phase0/fetch_real_headers.py` too, so the definition is checked against
real tensor names.
