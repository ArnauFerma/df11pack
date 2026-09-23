//! The binary carries its definitions: a downloaded df11pack works with no
//! `data/` folder anywhere, and DF11PACK_ARCH_DIR adds or overrides by name.

use std::process::Command;

fn run(dir: Option<&std::path::Path>, args: &[&str]) -> (i32, String) {
    let mut c = Command::new(env!("CARGO_BIN_EXE_df11pack"));
    c.args(args).env_remove("DF11PACK_ARCH_DIR");
    // A working directory with no data/ folder in it.
    c.current_dir(std::env::temp_dir());
    if let Some(d) = dir {
        c.env("DF11PACK_ARCH_DIR", d);
    }
    let o = c.output().unwrap();
    (
        o.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&o.stdout).into_owned(),
    )
}

fn listed(out: &str) -> Vec<String> {
    out.lines()
        .skip(1)
        .map(|l| l.split_whitespace().next().unwrap().to_string())
        .collect()
}

#[test]
fn every_shipped_definition_is_built_in() {
    let data = concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/architectures");
    let mut files: Vec<String> = std::fs::read_dir(data)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "toml"))
        .map(|p| p.file_stem().unwrap().to_string_lossy().into_owned())
        .collect();
    files.sort();
    let (code, out) = run(None, &["architectures"]);
    assert_eq!(code, 0);
    assert_eq!(listed(&out), files);
    assert!(!out.contains("INVALID"), "{out}");
}

#[test]
fn a_definition_directory_adds_and_overrides_by_name() {
    let dir = df11_fixtures::scratch("embedded_override");
    let base = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../data/architectures/qwen3-4b.toml"
    ))
    .unwrap();
    std::fs::write(
        dir.join("qwen3-4b.toml"),
        base.replace(
            "source = \"DFloat11/Qwen3-4B-DF11\"",
            "source = \"local override\"",
        ),
    )
    .unwrap();
    std::fs::write(
        dir.join("my-model.toml"),
        base.replace("name = \"qwen3-4b\"", "name = \"my-model\""),
    )
    .unwrap();
    let (code, out) = run(Some(&dir), &["architectures"]);
    assert_eq!(code, 0);
    assert!(listed(&out).contains(&"my-model".to_string()), "added");
    let line = out.lines().find(|l| l.starts_with("qwen3-4b ")).unwrap();
    assert!(line.contains("local override"), "overridden: {line}");
}
