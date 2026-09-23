//! Deciding whether source reads may overlap.
//!
//! On a spinning disk, several readers at once make the head seek between them
//! and throughput collapses; one sequential reader is much faster. On SSD and
//! NVMe the opposite holds. DESIGN §5.4.
//!
//! Detection is Linux-specific and best-effort: resolve the file's backing
//! device and read `/sys/block/<dev>/queue/rotational`. When that cannot be
//! determined, the safe default is **concurrent**, because assuming a spinning
//! disk on an NVMe box would serialise reads for no reason, while the reverse
//! merely loses the optimisation on a disk that is already slow.

use std::path::Path;

/// Whether source reads may overlap.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IoMode {
    /// Detect the device and choose.
    #[default]
    Auto,
    /// One reader at a time. Correct for spinning disks.
    Sequential,
    /// Readers overlap freely. Correct for SSD and NVMe.
    Concurrent,
}

/// What `Auto` resolves to for a given path, and why.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IoPlan {
    pub sequential: bool,
    pub reason: String,
}

/// Resolve `Auto` against the device backing `path`.
pub fn plan(path: &Path, mode: IoMode) -> IoPlan {
    match mode {
        IoMode::Sequential => IoPlan {
            sequential: true,
            reason: "forced sequential".into(),
        },
        IoMode::Concurrent => IoPlan {
            sequential: false,
            reason: "forced concurrent".into(),
        },
        IoMode::Auto => match rotational_for(path) {
            Some(true) => IoPlan {
                sequential: true,
                reason: "device reports rotational=1, so reads are serialised to avoid seeks"
                    .into(),
            },
            Some(false) => IoPlan {
                sequential: false,
                reason: "device reports rotational=0".into(),
            },
            None => IoPlan {
                sequential: false,
                reason: "device type unknown; defaulting to concurrent".into(),
            },
        },
    }
}

/// Read `rotational` for the device backing `path`, if it can be determined.
pub fn rotational_for(path: &Path) -> Option<bool> {
    let dev = device_name_for(path)?;
    read_rotational(&dev)
}

/// `/sys/block/<dev>/queue/rotational`, walking up from a partition to its disk.
fn read_rotational(dev: &str) -> Option<bool> {
    let direct = format!("/sys/block/{dev}/queue/rotational");
    if let Ok(s) = std::fs::read_to_string(&direct) {
        return parse_rotational(&s);
    }
    // A partition like `sda1` lives under its parent disk.
    let trimmed: String = dev
        .trim_end_matches(|c: char| c.is_ascii_digit())
        .to_string();
    if trimmed != dev && !trimmed.is_empty() {
        if let Ok(s) = std::fs::read_to_string(format!("/sys/block/{trimmed}/queue/rotational")) {
            return parse_rotational(&s);
        }
    }
    None
}

/// Parse the contents of a `queue/rotational` file.
pub fn parse_rotational(s: &str) -> Option<bool> {
    match s.trim() {
        "1" => Some(true),
        "0" => Some(false),
        _ => None,
    }
}

/// The kernel device name backing a path, e.g. `sda` or `nvme0n1`.
fn device_name_for(path: &Path) -> Option<String> {
    // st_dev gives major:minor; /sys/dev/block/<major>:<minor> links to the device.
    use std::os::unix::fs::MetadataExt;
    let md = std::fs::metadata(path).ok()?;
    let (major, minor) = (
        (md.dev() >> 8) & 0xfff,
        (md.dev() & 0xff) | ((md.dev() >> 12) & 0xfff00),
    );
    let link = std::fs::read_link(format!("/sys/dev/block/{major}:{minor}")).ok()?;
    link.file_name().map(|n| n.to_string_lossy().into_owned())
}
