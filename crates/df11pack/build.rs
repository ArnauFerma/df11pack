//! Embeds every definition in `data/architectures/` into the binary, so a
//! downloaded `df11pack` works on its own. Adding a TOML there is all it takes.

use std::fmt::Write as _;
use std::path::Path;

fn main() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/architectures");
    println!("cargo:rerun-if-changed={}", dir.display());
    let mut entries: Vec<_> = std::fs::read_dir(&dir)
        .expect("data/architectures")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "toml"))
        .collect();
    entries.sort();
    let mut out =
        String::from("/// `(name, toml)` for every shipped definition, sorted by name.\n");
    out.push_str("pub static EMBEDDED: &[(&str, &str)] = &[\n");
    for p in &entries {
        println!("cargo:rerun-if-changed={}", p.display());
        let name = p.file_stem().unwrap().to_string_lossy();
        let abs = p.canonicalize().unwrap();
        writeln!(
            out,
            "    ({name:?}, include_str!({:?})),",
            abs.display().to_string()
        )
        .unwrap();
    }
    out.push_str("];\n");
    let dest = Path::new(&std::env::var("OUT_DIR").unwrap()).join("embedded_defs.rs");
    std::fs::write(dest, out).unwrap();
}
