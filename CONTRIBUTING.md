# Contributing

Thanks for helping. By taking part you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).

## Most useful right now

- **Test a model on a GPU.** Compress a model you use, load it in ComfyUI-DFloat11-Extended
  or with `DFloat11Model`, and report what happened, working or not. Image and video
  models have not yet been load-tested this way.
- **Report a model that fails**, or ask for a new one: use the issue templates.
- **Add a model definition**: see [docs/DEFINITIONS.md](docs/DEFINITIONS.md).

Questions go to [Discussions](https://github.com/ArnauFerma/df11pack/discussions);
security problems go through [SECURITY.md](SECURITY.md), not public issues.

## Development

Rust 1.85 or newer.

```sh
cargo build --release -p df11pack
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --release
```

CI runs the same checks on Linux, Windows and macOS.

Tests that compare against the official compressor need the Phase 0 fixtures:
official outputs and small source models, several hundred MB, not in git. Without
them those tests print `SKIP` and pass. To regenerate them, see
[phase0/README.md](phase0/README.md). Anything that touches encoding must be run
with the fixtures before it is merged.

## Rules that keep the output trustworthy

- **Byte-identity is the contract.** Default output must stay byte-identical to the
  official compressor ([docs/COMPATIBILITY.md](docs/COMPATIBILITY.md)). Anything
  that differs is opt-in, off by default, and marks the files it writes.
- **Refuse rather than guess.** If df11pack cannot produce the exact output, it stops
  with an error that names the cause.
- **Tests must be able to fail.** A new check comes with a test that fails when the
  check is removed.
- **Measurements go in [docs/FINDINGS.md](docs/FINDINGS.md)**, with the prediction
  written down before the measurement.

## Pull requests

Keep them focused. Include tests, and a line in the `Unreleased` section of
[CHANGELOG.md](CHANGELOG.md) for anything a user would notice. Contributions are
accepted under the project's [MIT licence](LICENSE).

## Releases (maintainers)

Move `Unreleased` into a version section in `CHANGELOG.md`, bump `version` in
`Cargo.toml`, commit, then `git tag vX.Y.Z && git push origin vX.Y.Z`. The release
workflow builds the binaries and publishes the release from that changelog section.
