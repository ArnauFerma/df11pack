//! Reading a model that may be one file or many.
//!
//! A diffusers checkpoint is commonly sharded, with a
//! `*.safetensors.index.json` mapping tensor names to files. Sharding is a
//! property of the *input* and changes nothing downstream: units are found by
//! name, so a sharded source and a single-file one produce the same units.

use crate::safetensors::{SafeTensorsFile, StError};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// A model to compress: one safetensors file, or a directory of them.
#[derive(Debug)]
pub struct ModelSource {
    dir: PathBuf,
    files: Vec<SafeTensorsFile>,
    /// tensor name -> index into `files`
    index: BTreeMap<String, usize>,
}

impl ModelSource {
    /// Open a file, or a directory containing one or more `.safetensors`.
    ///
    /// When several files declare the same tensor, that is an error rather than
    /// a silent choice: the two copies could differ.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, StError> {
        let path = path.as_ref();
        let (dir, paths): (PathBuf, Vec<PathBuf>) = if path.is_file() {
            (
                path.parent().unwrap_or(Path::new(".")).to_path_buf(),
                vec![path.to_path_buf()],
            )
        } else if path.is_dir() {
            let mut v: Vec<PathBuf> = std::fs::read_dir(path)
                .map_err(StError::Io)?
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
                .collect();
            // Deterministic order, so a duplicate is always reported against the
            // same pair of files.
            v.sort();
            if v.is_empty() {
                return Err(StError::Malformed(format!(
                    "no .safetensors files in {}",
                    path.display()
                )));
            }
            (path.to_path_buf(), v)
        } else {
            return Err(StError::Malformed(format!(
                "{} is neither a file nor a directory",
                path.display()
            )));
        };

        let mut files = Vec::with_capacity(paths.len());
        let mut index: BTreeMap<String, usize> = BTreeMap::new();
        for (i, p) in paths.iter().enumerate() {
            let f = SafeTensorsFile::open(p)?;
            for n in f.names() {
                if let Some(prev) = index.insert(n.to_string(), i) {
                    // The two copies could differ; picking one silently would be
                    // a coin flip over which model gets compressed.
                    return Err(StError::Malformed(format!(
                        "tensor {n:?} is declared by both {} and {}",
                        paths[prev].display(),
                        p.display()
                    )));
                }
            }
            files.push(f);
        }
        Ok(ModelSource { dir, files, index })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Every tensor name, sorted.
    pub fn names(&self) -> Vec<String> {
        self.index.keys().cloned().collect()
    }

    pub fn len(&self) -> usize {
        self.index.len()
    }

    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// How many files back this source.
    pub fn shards(&self) -> usize {
        self.files.len()
    }

    pub fn info(&self, name: &str) -> Option<&crate::safetensors::TensorInfo> {
        self.index.get(name).and_then(|i| self.files[*i].info(name))
    }

    pub fn read(&self, name: &str) -> Result<Vec<u8>, StError> {
        let i = self
            .index
            .get(name)
            .ok_or_else(|| StError::NotFound(name.to_string()))?;
        self.files[*i].read(name)
    }

    /// Total bytes of tensor data.
    pub fn total_bytes(&self) -> u64 {
        self.index
            .keys()
            .filter_map(|n| self.info(n))
            .map(|i| i.nbytes())
            .sum()
    }
}
