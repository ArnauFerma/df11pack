//! A minimal safetensors reader and writer.
//!
//! Written rather than taken from the crate because df11pack needs control over
//! exactly which bytes land where: the header is JSON whose key order and
//! padding are ours to choose, and Phase 3 will write it with the header
//! reserved up front so the file streams out with no holes.
//!
//! What the format guarantees, and what it does not: the loader dispatches
//! purely by tensor name, and physical order and shard grouping were measured
//! not to matter (FINDINGS, H6 and H11). So the writer is free in layout and
//! constrained only in content.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

/// A tensor's dtype, kept as the format's own string so round-trips are lossless.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Dtype(pub String);

impl Dtype {
    pub const BF16: &'static str = "BF16";
    pub const U8: &'static str = "U8";
    pub const I64: &'static str = "I64";

    pub fn new(s: &str) -> Self {
        Dtype(s.to_string())
    }

    /// Bytes per element, or `None` for a dtype we do not know.
    pub fn width(&self) -> Option<usize> {
        Some(match self.0.as_str() {
            "BOOL" | "U8" | "I8" | "F8_E4M3" | "F8_E5M2" => 1,
            "U16" | "I16" | "F16" | "BF16" => 2,
            "U32" | "I32" | "F32" => 4,
            "U64" | "I64" | "F64" => 8,
            _ => return None,
        })
    }
}

/// Where one tensor lives and what it is.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub dtype: Dtype,
    pub shape: Vec<u64>,
    /// Byte range within the data section.
    pub offsets: (u64, u64),
}

impl TensorInfo {
    pub fn nbytes(&self) -> u64 {
        self.offsets.1 - self.offsets.0
    }

    /// Whether the declared shape and dtype account for exactly `nbytes`.
    pub fn is_consistent(&self) -> bool {
        match self.dtype.width() {
            Some(w) => {
                let n: u64 = self.shape.iter().product::<u64>();
                n * w as u64 == self.nbytes()
            }
            None => true,
        }
    }
}

/// Errors from reading a safetensors file.
#[derive(Debug)]
pub enum StError {
    Io(std::io::Error),
    Malformed(String),
    NotFound(String),
}

impl std::fmt::Display for StError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "{e}"),
            Self::Malformed(m) => write!(f, "malformed safetensors: {m}"),
            Self::NotFound(n) => write!(f, "no tensor named {n:?}"),
        }
    }
}

impl std::error::Error for StError {}

impl From<std::io::Error> for StError {
    fn from(e: std::io::Error) -> Self {
        StError::Io(e)
    }
}

/// An opened safetensors file. Tensor data is read on demand, never all at once.
#[derive(Debug)]
pub struct SafeTensorsFile {
    path: PathBuf,
    tensors: BTreeMap<String, TensorInfo>,
    /// Header key order as written, which is not necessarily sorted.
    order: Vec<String>,
    metadata: BTreeMap<String, String>,
    data_start: u64,
}

impl SafeTensorsFile {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, StError> {
        let path = path.as_ref().to_path_buf();
        let mut f = File::open(&path)?;
        let mut len = [0u8; 8];
        f.read_exact(&mut len)?;
        let n = u64::from_le_bytes(len);
        if n > 100 * 1024 * 1024 {
            return Err(StError::Malformed(format!("header claims {n} bytes")));
        }
        let mut buf = vec![0u8; n as usize];
        f.read_exact(&mut buf)?;
        let v: serde_json::Value =
            serde_json::from_slice(&buf).map_err(|e| StError::Malformed(e.to_string()))?;
        let obj = v
            .as_object()
            .ok_or_else(|| StError::Malformed("header is not an object".into()))?;

        let mut tensors = BTreeMap::new();
        let mut order = Vec::new();
        let mut metadata = BTreeMap::new();
        for (k, val) in obj {
            if k == "__metadata__" {
                if let Some(m) = val.as_object() {
                    for (mk, mv) in m {
                        if let Some(s) = mv.as_str() {
                            metadata.insert(mk.clone(), s.to_string());
                        }
                    }
                }
                continue;
            }
            let dtype = val
                .get("dtype")
                .and_then(|d| d.as_str())
                .ok_or_else(|| StError::Malformed(format!("{k}: no dtype")))?;
            let shape: Vec<u64> = val
                .get("shape")
                .and_then(|s| s.as_array())
                .ok_or_else(|| StError::Malformed(format!("{k}: no shape")))?
                .iter()
                .filter_map(|x| x.as_u64())
                .collect();
            let off = val
                .get("data_offsets")
                .and_then(|o| o.as_array())
                .ok_or_else(|| StError::Malformed(format!("{k}: no data_offsets")))?;
            if off.len() != 2 {
                return Err(StError::Malformed(format!(
                    "{k}: data_offsets is not a pair"
                )));
            }
            let (s, e) = (off[0].as_u64().unwrap_or(0), off[1].as_u64().unwrap_or(0));
            if e < s {
                return Err(StError::Malformed(format!("{k}: data_offsets reversed")));
            }
            order.push(k.clone());
            tensors.insert(
                k.clone(),
                TensorInfo {
                    dtype: Dtype::new(dtype),
                    shape,
                    offsets: (s, e),
                },
            );
        }
        Ok(SafeTensorsFile {
            path,
            tensors,
            order,
            metadata,
            data_start: 8 + n,
        })
    }

    /// Tensor names, sorted.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    /// Tensor names in the order the header lists them.
    pub fn physical_order(&self) -> &[String] {
        &self.order
    }

    pub fn info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    /// The directory the file lives in, where a sibling `config.json` would be.
    pub fn dir(&self) -> Option<&Path> {
        self.path.parent()
    }

    pub fn metadata(&self) -> &BTreeMap<String, String> {
        &self.metadata
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// A reader positioned at the start of a tensor's data.
    pub fn open_at(&self, info: &TensorInfo) -> Result<File, StError> {
        let mut f = File::open(&self.path)?;
        f.seek(SeekFrom::Start(self.data_start + info.offsets.0))?;
        Ok(f)
    }

    /// Read one tensor's raw little-endian bytes.
    pub fn read(&self, name: &str) -> Result<Vec<u8>, StError> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| StError::NotFound(name.to_string()))?;
        let mut f = File::open(&self.path)?;
        f.seek(SeekFrom::Start(self.data_start + info.offsets.0))?;
        let mut buf = vec![0u8; info.nbytes() as usize];
        f.read_exact(&mut buf)?;
        Ok(buf)
    }
}

/// One tensor to write.
pub struct OutTensor {
    pub name: String,
    pub dtype: Dtype,
    pub shape: Vec<u64>,
    pub data: Vec<u8>,
}

/// Write a safetensors file.
///
/// Tensors are laid out in the order given, contiguously, with no padding
/// between them. The header lists them in the same order.
pub fn write_file(
    path: impl AsRef<Path>,
    tensors: &[OutTensor],
    metadata: &BTreeMap<String, String>,
) -> Result<(), StError> {
    let mut header = serde_json::Map::new();
    if !metadata.is_empty() {
        let mut m = serde_json::Map::new();
        for (k, v) in metadata {
            m.insert(k.clone(), serde_json::Value::String(v.clone()));
        }
        header.insert("__metadata__".into(), serde_json::Value::Object(m));
    }
    let mut cursor: u64 = 0;
    for t in tensors {
        let end = cursor + t.data.len() as u64;
        let mut e = serde_json::Map::new();
        e.insert("dtype".into(), serde_json::Value::String(t.dtype.0.clone()));
        e.insert(
            "shape".into(),
            serde_json::Value::Array(t.shape.iter().map(|&d| d.into()).collect()),
        );
        e.insert(
            "data_offsets".into(),
            serde_json::Value::Array(vec![cursor.into(), end.into()]),
        );
        header.insert(t.name.clone(), serde_json::Value::Object(e));
        cursor = end;
    }
    let mut json = serde_json::to_vec(&serde_json::Value::Object(header))
        .map_err(|e| StError::Malformed(e.to_string()))?;
    // The data section must start 8-byte aligned; pad the header with spaces,
    // which the format permits.
    while (8 + json.len()) % 8 != 0 {
        json.push(b' ');
    }

    let mut w = BufWriter::new(File::create(path)?);
    w.write_all(&(json.len() as u64).to_le_bytes())?;
    w.write_all(&json)?;
    for t in tensors {
        w.write_all(&t.data)?;
    }
    w.flush()?;
    Ok(())
}
