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

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Byte offset where the data section begins.
    pub fn data_start(&self) -> u64 {
        self.data_start
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

/// Where a tensor's bytes come from.
pub enum Payload {
    /// Already in memory.
    Owned(Vec<u8>),
    /// Copied from another file, in bounded chunks, so a large tensor never
    /// needs to be resident to be written.
    Borrowed {
        path: PathBuf,
        offset: u64,
        len: u64,
    },
}

impl Payload {
    pub fn len(&self) -> u64 {
        match self {
            Payload::Owned(v) => v.len() as u64,
            Payload::Borrowed { len, .. } => *len,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// One tensor to write.
pub struct OutTensor {
    pub name: String,
    pub dtype: Dtype,
    pub shape: Vec<u64>,
    pub data: Payload,
}

impl OutTensor {
    /// An in-memory tensor.
    pub fn owned(name: impl Into<String>, dtype: Dtype, shape: Vec<u64>, data: Vec<u8>) -> Self {
        OutTensor {
            name: name.into(),
            dtype,
            shape,
            data: Payload::Owned(data),
        }
    }
}

/// A tensor's entry in the header, known before its bytes exist.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorDecl {
    pub name: String,
    pub dtype: Dtype,
    pub shape: Vec<u64>,
    pub len: u64,
}

/// A dtype's position in the `safetensors` library's `Dtype` enum (0.8.0), which
/// its `serialize` uses to lay tensors out: **dtype descending, then name**. The
/// official compressor writes every file through that library, so matching the
/// order is part of byte identity. `None` for a dtype the library does not know.
pub fn dtype_rank(dtype: &str) -> Option<u32> {
    const ORDER: [&str; 22] = [
        "BOOL",
        "F4",
        "F6_E2M3",
        "F6_E3M2",
        "U8",
        "I8",
        "F8_E5M2",
        "F8_E4M3",
        "F8_E8M0",
        "F8_E4M3FNUZ",
        "F8_E5M2FNUZ",
        "I16",
        "U16",
        "F16",
        "BF16",
        "I32",
        "U32",
        "F32",
        "C64",
        "F64",
        "I64",
        "U64",
    ];
    ORDER.iter().position(|d| *d == dtype).map(|i| i as u32)
}

/// Sort into the library's layout order. Errors on a dtype it does not know,
/// rather than guessing where that tensor would go.
pub fn canonical_order<T>(
    items: &mut [T],
    key: impl Fn(&T) -> (&str, &str),
) -> Result<(), StError> {
    for it in items.iter() {
        let (name, dtype) = key(it);
        if dtype_rank(dtype).is_none() {
            return Err(StError::Malformed(format!(
                "tensor {name:?} has dtype {dtype:?}, which the safetensors library does not know"
            )));
        }
    }
    items.sort_by(|a, b| {
        let (an, ad) = key(a);
        let (bn, bd) = key(b);
        dtype_rank(bd)
            .cmp(&dtype_rank(ad))
            .then(an.as_bytes().cmp(bn.as_bytes()))
    });
    Ok(())
}

/// The JSON header for `decls`, padded so the data section starts 8-byte aligned.
fn header_bytes(
    decls: &[TensorDecl],
    metadata: &BTreeMap<String, String>,
) -> Result<Vec<u8>, StError> {
    let mut header = serde_json::Map::new();
    if !metadata.is_empty() {
        let mut m = serde_json::Map::new();
        for (k, v) in metadata {
            m.insert(k.clone(), serde_json::Value::String(v.clone()));
        }
        header.insert("__metadata__".into(), serde_json::Value::Object(m));
    }
    let mut cursor: u64 = 0;
    for d in decls {
        let end = cursor + d.len;
        let mut e = serde_json::Map::new();
        e.insert("dtype".into(), serde_json::Value::String(d.dtype.0.clone()));
        e.insert(
            "shape".into(),
            serde_json::Value::Array(d.shape.iter().map(|&x| x.into()).collect()),
        );
        e.insert(
            "data_offsets".into(),
            serde_json::Value::Array(vec![cursor.into(), end.into()]),
        );
        header.insert(d.name.clone(), serde_json::Value::Object(e));
        cursor = end;
    }
    let mut json = serde_json::to_vec(&serde_json::Value::Object(header))
        .map_err(|e| StError::Malformed(e.to_string()))?;
    while (8 + json.len()) % 8 != 0 {
        json.push(b' ');
    }
    Ok(json)
}

fn tmp_path(dest: &Path) -> Result<PathBuf, StError> {
    match dest.file_name() {
        Some(n) => Ok(dest.with_file_name(format!("{}.tmp", n.to_string_lossy()))),
        None => Err(StError::Malformed(format!(
            "{} has no file name",
            dest.display()
        ))),
    }
}

/// fsync a directory so a rename inside it survives a crash. Best effort: not
/// every platform allows opening a directory.
fn sync_dir(dest: &Path) {
    if let Some(dir) = dest.parent() {
        if let Ok(d) = File::open(dir) {
            let _ = d.sync_all();
        }
    }
}

/// Writes a safetensors file whose header is fixed up front and whose tensors
/// then arrive one at a time, **in header order** -- which is the library's
/// canonical order ([`canonical_order`]), not the order declared: `begin` sorts,
/// and [`StreamingWriter::order`] says what to write next.
///
/// This is what lets a single-file output be written without holding every
/// tensor in memory: the caller declares names, dtypes, shapes and lengths first,
/// then streams the bytes. Each write is checked against its declaration, so a
/// tensor that comes out a different size than declared is an error rather than
/// a corrupt file.
///
/// Atomic like [`write_file`]: bytes go to a sibling `.tmp`, fsynced and renamed
/// on [`StreamingWriter::finish`]. Dropping the writer without finishing removes
/// the temporary and leaves the destination untouched.
pub struct StreamingWriter {
    w: Option<BufWriter<File>>,
    tmp: PathBuf,
    dest: PathBuf,
    decls: Vec<TensorDecl>,
    next: usize,
    buf: Vec<u8>,
}

impl StreamingWriter {
    pub fn begin(
        dest: impl AsRef<Path>,
        decls: Vec<TensorDecl>,
        metadata: &BTreeMap<String, String>,
    ) -> Result<Self, StError> {
        let dest = dest.as_ref().to_path_buf();
        let tmp = tmp_path(&dest)?;
        let mut decls = decls;
        canonical_order(&mut decls, |d| (d.name.as_str(), d.dtype.0.as_str()))?;
        let json = header_bytes(&decls, metadata)?;
        let mut me = StreamingWriter {
            w: Some(BufWriter::new(File::create(&tmp)?)),
            tmp,
            dest,
            decls,
            next: 0,
            buf: Vec::new(),
        };
        let w = me.w.as_mut().expect("open");
        w.write_all(&(json.len() as u64).to_le_bytes())?;
        w.write_all(&json)?;
        Ok(me)
    }

    /// The declarations in the order their bytes must be written.
    pub fn order(&self) -> &[TensorDecl] {
        &self.decls
    }

    fn expect(&self, name: &str, len: u64) -> Result<(), StError> {
        let d = self.decls.get(self.next).ok_or_else(|| {
            StError::Malformed(format!(
                "tensor {name:?} written after the last declared one"
            ))
        })?;
        if d.name != name {
            return Err(StError::Malformed(format!(
                "tensor {name:?} written where {:?} was declared",
                d.name
            )));
        }
        if d.len != len {
            return Err(StError::Malformed(format!(
                "tensor {name:?} is {len} bytes but was declared as {}",
                d.len
            )));
        }
        Ok(())
    }

    /// Write the next declared tensor from memory.
    pub fn write(&mut self, name: &str, bytes: &[u8]) -> Result<(), StError> {
        self.expect(name, bytes.len() as u64)?;
        self.w.as_mut().expect("open").write_all(bytes)?;
        self.next += 1;
        Ok(())
    }

    /// Write the next declared tensor by copying a byte range of another file.
    pub fn copy(&mut self, name: &str, path: &Path, offset: u64, len: u64) -> Result<(), StError> {
        self.expect(name, len)?;
        if self.buf.is_empty() {
            self.buf = vec![0u8; 1 << 20];
        }
        let mut f = File::open(path)?;
        f.seek(SeekFrom::Start(offset))?;
        let mut left = len;
        let w = self.w.as_mut().expect("open");
        while left > 0 {
            let n = (left as usize).min(self.buf.len());
            f.read_exact(&mut self.buf[..n])?;
            w.write_all(&self.buf[..n])?;
            left -= n as u64;
        }
        self.next += 1;
        Ok(())
    }

    /// Write the next declared tensor from a payload.
    pub fn put(&mut self, t: &OutTensor) -> Result<(), StError> {
        match &t.data {
            Payload::Owned(v) => self.write(&t.name, v),
            Payload::Borrowed { path, offset, len } => self.copy(&t.name, path, *offset, *len),
        }
    }

    /// Check every declared tensor was written, make it durable, and publish it.
    pub fn finish(mut self) -> Result<(), StError> {
        if self.next != self.decls.len() {
            return Err(StError::Malformed(format!(
                "{} of {} declared tensors were written",
                self.next,
                self.decls.len()
            )));
        }
        let w = self.w.take().expect("open");
        w.into_inner()
            .map_err(|e| StError::Io(e.into_error()))?
            .sync_all()?;
        std::fs::rename(&self.tmp, &self.dest)?;
        sync_dir(&self.dest);
        Ok(())
    }
}

impl Drop for StreamingWriter {
    fn drop(&mut self) {
        // Anything other than a successful finish leaves no trace.
        if self.w.is_some() || self.tmp.exists() {
            drop(self.w.take());
            let _ = std::fs::remove_file(&self.tmp);
        }
    }
}

/// Write a safetensors file **atomically**.
///
/// The bytes go to a sibling `.tmp`, which is flushed and fsynced, then renamed
/// into place; the directory is fsynced after, so the rename itself survives a
/// power loss. On any failure the temporary is removed and the destination is
/// left untouched.
///
/// This matters more than it looks. A truncated safetensors is not obviously
/// broken — the header parses, the tensors it names are simply short — so a
/// half-written shard can be mistaken for a finished one.
///
/// Tensors are laid out contiguously, with no padding between them, in the
/// safetensors library's order (dtype descending, then name), whatever order
/// they are given in. The header lists them in the same order.
pub fn write_file(
    path: impl AsRef<Path>,
    tensors: &[OutTensor],
    metadata: &BTreeMap<String, String>,
) -> Result<(), StError> {
    let mut sorted: Vec<&OutTensor> = tensors.iter().collect();
    canonical_order(&mut sorted, |t| (t.name.as_str(), t.dtype.0.as_str()))?;
    let decls = sorted
        .iter()
        .map(|t| TensorDecl {
            name: t.name.clone(),
            dtype: t.dtype.clone(),
            shape: t.shape.clone(),
            len: t.data.len(),
        })
        .collect();
    let mut w = StreamingWriter::begin(path, decls, metadata)?;
    for t in sorted {
        w.put(t)?;
    }
    w.finish()
}

/// Write any small file atomically: temporary, fsync, rename, fsync the directory.
///
/// For `config.json` and similar sidecars, which previously used a plain write
/// and so could be left truncated by an interrupted run.
pub fn write_bytes_atomic(dest: impl AsRef<Path>, bytes: &[u8]) -> Result<(), StError> {
    let dest = dest.as_ref();
    let tmp = tmp_path(dest)?;
    let r = (|| -> Result<(), StError> {
        let mut f = File::create(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()?;
        std::fs::rename(&tmp, dest)?;
        sync_dir(dest);
        Ok(())
    })();
    if r.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    r
}
