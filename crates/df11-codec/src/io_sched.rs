//! Deciding whether source reads may overlap.
//!
//! On a spinning disk, several readers at once make the head seek between them
//! and throughput collapses; one sequential reader is much faster. On SSD and
//! NVMe the opposite holds. DESIGN §5.4.
//!
//! Detection is Linux-specific and best-effort: resolve the file's backing
//! device through `/sys/dev/block/<major>:<minor>` and read its
//! `queue/rotational`, or its disk's for a partition. When that cannot be
//! determined, the safe default is **concurrent**, because assuming a spinning
//! disk on an NVMe box would serialise reads for no reason, while the reverse
//! merely loses the optimisation on a disk that is already slow.

use std::path::{Path, PathBuf};

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
    rotational_at(&sysfs_dir_for(path)?)
}

/// `rotational` for a device's sysfs directory (`/sys/dev/block/<M>:<m>`).
///
/// A whole disk, or a virtual device such as `dm-0`, has its own
/// `queue/rotational`. A partition has none; it is a subdirectory of its
/// disk's directory (`.../block/nvme0n1/nvme0n1p1`) and is marked by a
/// `partition` file, so its disk is `..`. This works for any naming scheme --
/// `sda1` and `nvme0n1p1` alike -- where stripping trailing digits from the
/// name does not (`nvme0n1p1` would become `nvme0n1p`).
pub fn rotational_at(dev_dir: &Path) -> Option<bool> {
    let read = |d: &Path| std::fs::read_to_string(d.join("queue/rotational")).ok();
    if let Some(s) = read(dev_dir) {
        return parse_rotational(&s);
    }
    if dev_dir.join("partition").is_file() {
        // Resolved physically: `..` of the symlink's target, not of the link.
        let disk = std::fs::canonicalize(dev_dir).ok()?.parent()?.to_path_buf();
        return parse_rotational(&read(&disk)?);
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

/// The sysfs directory of the block device backing a path.
#[cfg(not(unix))]
fn sysfs_dir_for(_path: &Path) -> Option<PathBuf> {
    None
}

/// The sysfs directory of the block device backing a path,
/// `/sys/dev/block/<major>:<minor>`.
///
/// Only Linux has it. Elsewhere (macOS included, which takes this `unix` path
/// and finds no /sys), the device type is unknown and reads default to
/// concurrent, which `--io sequential` overrides.
#[cfg(unix)]
fn sysfs_dir_for(path: &Path) -> Option<PathBuf> {
    // st_dev gives major:minor, decoded as glibc's gnu_dev_major/minor do.
    use std::os::unix::fs::MetadataExt;
    let dev = std::fs::metadata(path).ok()?.dev();
    let major = ((dev & 0x0000_0000_000f_ff00) >> 8) | ((dev & 0xffff_f000_0000_0000) >> 32);
    let minor = (dev & 0x0000_0000_0000_00ff) | ((dev & 0x0000_0fff_fff0_0000) >> 12);
    let dir = PathBuf::from(format!("/sys/dev/block/{major}:{minor}"));
    dir.exists().then_some(dir)
}
