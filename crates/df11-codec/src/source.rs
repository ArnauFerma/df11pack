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
    pub(crate) files: Vec<SafeTensorsFile>,
    /// tensor name -> index into `files`
    pub(crate) index: BTreeMap<String, usize>,
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

    /// Where a tensor's bytes live on disk: `(path, absolute offset, length)`.
    ///
    /// Lets the writer copy a passthrough tensor straight through without ever
    /// holding it, which matters because the largest of them -- an embedding --
    /// would otherwise set the floor on peak memory by itself.
    pub fn locate(&self, name: &str) -> Option<(PathBuf, u64, u64)> {
        let i = *self.index.get(name)?;
        let f = &self.files[i];
        let info = f.info(name)?;
        Some((
            f.path().to_path_buf(),
            f.data_start() + info.offsets.0,
            info.nbytes(),
        ))
    }

    /// The `__metadata__` of the file that holds `name`.
    pub fn metadata_of(&self, name: &str) -> Option<&BTreeMap<String, String>> {
        Some(self.files[*self.index.get(name)?].metadata())
    }

    /// Whether two tensors hold identical bytes, compared in chunks.
    ///
    /// Reading both into memory to compare them costs their combined size, which
    /// for a tied embedding pair is twice the largest tensor in the model -- the
    /// single biggest allocation the writer would otherwise make.
    pub fn tensors_equal(&self, a: &str, b: &str) -> Result<bool, StError> {
        use std::io::Read;
        let (ia, ib) = match (self.info(a), self.info(b)) {
            (Some(x), Some(y)) => (x.clone(), y.clone()),
            _ => return Ok(false),
        };
        if ia.nbytes() != ib.nbytes() {
            return Ok(false);
        }
        let (fa, fb) = (self.index[a], self.index[b]);
        let mut ra = self.files[fa].open_at(&ia)?;
        let mut rb = self.files[fb].open_at(&ib)?;
        let mut left = ia.nbytes();
        let (mut ba, mut bb) = (vec![0u8; 1 << 20], vec![0u8; 1 << 20]);
        while left > 0 {
            let n = (left as usize).min(ba.len());
            ra.read_exact(&mut ba[..n]).map_err(StError::Io)?;
            rb.read_exact(&mut bb[..n]).map_err(StError::Io)?;
            if ba[..n] != bb[..n] {
                return Ok(false);
            }
            left -= n as u64;
        }
        Ok(true)
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

/// The prefix ComfyUI puts on every diffusion-model key in a full checkpoint.
pub const COMFYUI_PREFIX: &str = "model.diffusion_model.";

/// A [`ModelSource`] seen through the names the output will carry.
///
/// Architecture definitions are written against bare names (`double_blocks.0`),
/// while ComfyUI checkpoints usually carry `model.diffusion_model.` on every
/// diffusion key. With a prefix set, only names carrying it are visible, and
/// they are visible without it; everything else in the file -- a VAE or text
/// encoder in a full checkpoint -- is left out, as it is not part of the model
/// being compressed. A definition's key rules, and a tied `lm_head`, map names
/// further ([`crate::keys`]); every read goes to the physical name.
pub struct View<'a> {
    src: &'a ModelSource,
    map: crate::keys::NameMap,
}

impl<'a> View<'a> {
    pub fn new(src: &'a ModelSource, prefix: Option<&str>) -> Self {
        View {
            src,
            map: crate::keys::NameMap::with_prefix(&src.names(), prefix),
        }
    }

    /// A view through a full name map ([`crate::keys::map_names`]).
    pub fn mapped(src: &'a ModelSource, map: crate::keys::NameMap) -> Self {
        View { src, map }
    }

    /// The source view a definition asks for: checkpoint prefix, key rules, and
    /// the tied `lm_head` if the source's `config.json` says the embeddings are
    /// tied.
    pub fn for_def(
        src: &'a ModelSource,
        def: &crate::arch::ArchDef,
    ) -> Result<Self, crate::keys::KeyError> {
        let tied = std::fs::read(src.dir().join("config.json"))
            .ok()
            .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
            .and_then(|v| v.get("tie_word_embeddings").and_then(|t| t.as_bool()))
            .unwrap_or(false);
        Ok(View::mapped(
            src,
            crate::keys::map_names(def, &src.names(), tied)?,
        ))
    }

    /// The prefix being stripped, if any.
    pub fn prefix(&self) -> Option<&str> {
        self.map.prefix()
    }

    fn phys(&self, name: &str) -> String {
        self.map.physical(name).unwrap_or(name).to_string()
    }

    pub fn names(&self) -> Vec<String> {
        self.map.visible()
    }

    pub fn info(&self, name: &str) -> Option<&crate::safetensors::TensorInfo> {
        self.src.info(&self.phys(name))
    }

    pub fn read(&self, name: &str) -> Result<Vec<u8>, StError> {
        self.src.read(&self.phys(name))
    }

    pub fn locate(&self, name: &str) -> Option<(PathBuf, u64, u64)> {
        self.src.locate(&self.phys(name))
    }

    pub fn tensors_equal(&self, a: &str, b: &str) -> Result<bool, StError> {
        self.src.tensors_equal(&self.phys(a), &self.phys(b))
    }

    pub fn dir(&self) -> &Path {
        self.src.dir()
    }

    pub fn total_bytes(&self) -> u64 {
        self.names()
            .iter()
            .filter_map(|n| self.info(n))
            .map(|i| i.nbytes())
            .sum()
    }
}
