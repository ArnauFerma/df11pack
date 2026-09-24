<!-- What does this change, and why? Link the issue if there is one. -->

- [ ] `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace --release` pass
- [ ] Encoding changes were tested with the Phase 0 fixtures (not only with the tests skipping)
- [ ] Default output is still byte-identical to the official compressor, or the change is opt-in and marked ([COMPATIBILITY](https://github.com/ArnauFerma/df11pack/blob/main/docs/COMPATIBILITY.md))
- [ ] `CHANGELOG.md` has a line under `Unreleased` if users would notice
