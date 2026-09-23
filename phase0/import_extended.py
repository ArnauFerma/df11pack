"""Transcribe ComfyUI-DFloat11-Extended's pattern_dict.py into JSON, pinned.

The file is parsed with `ast` and read with `literal_eval` -- never executed. Only
the data is kept (the repository ships no licence file, so the source is not
copied). Dict order is preserved: it is the order the official compressor walks
patterns in.

    python phase0/import_extended.py path/to/pattern_dict.py

Output: phase0/fixtures/extended_pattern_dicts.json
"""
import ast, hashlib, json, pathlib, sys, warnings

COMMIT = "414506dda343af616bea5b4987cfbbb41f5177fe"
URL = f"https://raw.githubusercontent.com/mingyi456/ComfyUI-DFloat11-Extended/{COMMIT}/pattern_dict.py"


def main(path):
    raw = pathlib.Path(path).read_bytes()
    with warnings.catch_warnings():
        # Upstream has non-raw strings like "a\.b"; Python keeps the backslash,
        # which is what the loader sees too.
        warnings.simplefilter("ignore", SyntaxWarning)
        tree = ast.parse(raw)
    assign = tree.body[0]
    assert isinstance(assign, ast.Assign) and assign.targets[0].id == "MODEL_TO_PATTERN_DICT"
    data = ast.literal_eval(assign.value)
    out = {
        "source": URL,
        "commit": COMMIT,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "models": {
            model: [[pattern, list(attrs)] for pattern, attrs in pd.items()]
            for model, pd in data.items()
        },
    }
    dest = pathlib.Path(__file__).resolve().parent / "fixtures/extended_pattern_dicts.json"
    dest.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {dest}: {len(data)} models")


if __name__ == "__main__":
    main(sys.argv[1])
