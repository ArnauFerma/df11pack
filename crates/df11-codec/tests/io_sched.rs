//! Phase 3 -- the I/O scheduling decision.

use df11_codec::io_sched::{parse_rotational, plan, rotational_at, rotational_for, IoMode};
use std::path::Path;

#[test]
fn explicit_modes_are_obeyed_and_explained() {
    let p = Path::new("/tmp");
    let s = plan(p, IoMode::Sequential);
    assert!(s.sequential);
    assert!(s.reason.contains("forced"), "{}", s.reason);

    let c = plan(p, IoMode::Concurrent);
    assert!(!c.sequential);
    assert!(c.reason.contains("forced"), "{}", c.reason);
}

#[test]
fn auto_always_gives_a_decision_and_a_reason() {
    let p = plan(Path::new("."), IoMode::Auto);
    assert!(!p.reason.is_empty(), "a decision must say why");
}

/// The default when the device cannot be identified must be concurrent:
/// serialising on an NVMe costs real throughput, while failing to serialise on a
/// disk only forgoes an optimisation on hardware that is already slow.
#[test]
fn an_unknown_device_defaults_to_concurrent() {
    let p = plan(
        Path::new("/nonexistent/path/that/cannot/be/resolved"),
        IoMode::Auto,
    );
    assert!(!p.sequential);
    assert!(p.reason.contains("unknown"), "{}", p.reason);
}

/// The flag's meaning, pinned directly. The machine this runs on may have only
/// SSDs, in which case no end-to-end test can tell the mapping from its inverse.
#[test]
fn the_rotational_flag_means_what_the_kernel_says_it_means() {
    assert_eq!(parse_rotational("1"), Some(true), "1 means a spinning disk");
    assert_eq!(parse_rotational("0"), Some(false), "0 means solid state");
    assert_eq!(
        parse_rotational("1\n"),
        Some(true),
        "trailing newline is normal"
    );
    assert_eq!(parse_rotational("0\n"), Some(false));
    assert_eq!(parse_rotational(""), None);
    assert_eq!(parse_rotational("maybe"), None);
}

#[test]
fn rotational_detection_agrees_with_sysfs_on_this_machine() {
    // Whatever the answer, it must match what /sys reports for the same device,
    // or be None. This catches the major:minor decoding being wrong, which would
    // otherwise silently read the wrong device's flag.
    let here = std::env::current_dir().expect("cwd");
    let got = rotational_for(&here);
    #[cfg(target_os = "linux")]
    {
        // When the kernel lists the backing device, detection must find it.
        // (A filesystem with no block device -- overlay, tmpfs -- has none.)
        use std::os::unix::fs::MetadataExt;
        let dev = std::fs::metadata(&here).unwrap().dev();
        let (major, minor) = (
            ((dev >> 8) & 0xfff) | ((dev >> 32) & 0xffff_f000),
            (dev & 0xff) | ((dev >> 12) & 0xffff_ff00),
        );
        let listed = std::path::Path::new(&format!("/sys/dev/block/{major}:{minor}")).exists();
        assert_eq!(
            got.is_some(),
            listed,
            "device {major}:{minor} listed in /sys/dev/block: {listed}, detected: {got:?}"
        );
    }
    if let Some(v) = got {
        // Cross-check: at least one block device must report that value.
        let mut seen = false;
        if let Ok(rd) = std::fs::read_dir("/sys/block") {
            for e in rd.flatten() {
                let p = e.path().join("queue/rotational");
                if let Ok(s) = std::fs::read_to_string(&p) {
                    if s.trim() == if v { "1" } else { "0" } {
                        seen = true;
                    }
                }
            }
        }
        assert!(seen, "reported rotational={v} but no block device says so");
    }
}

/// A partition has no `queue/` of its own; its disk is the parent directory in
/// sysfs. Checked on a fake sysfs tree shaped like an NVMe disk, whose
/// partition names (`nvme0n1p1`) defeat the old rule of stripping trailing
/// digits (`nvme0n1p`, which does not exist).
#[test]
fn a_partition_takes_its_disks_rotational_flag() {
    let root = df11_fixtures::scratch("io_sysfs");
    let disk = root.join("block/nvme0n1");
    let part = disk.join("nvme0n1p1");
    std::fs::create_dir_all(disk.join("queue")).unwrap();
    std::fs::create_dir_all(&part).unwrap();
    std::fs::write(disk.join("queue/rotational"), "1\n").unwrap();
    std::fs::write(part.join("partition"), "1\n").unwrap();
    assert_eq!(rotational_at(&disk), Some(true), "the disk itself");
    assert_eq!(rotational_at(&part), Some(true), "its partition");

    std::fs::write(disk.join("queue/rotational"), "0\n").unwrap();
    assert_eq!(rotational_at(&part), Some(false), "follows the disk");

    // A directory that is neither a device with a queue nor a partition.
    let other = root.join("block/other");
    std::fs::create_dir_all(&other).unwrap();
    assert_eq!(rotational_at(&other), None);
}

/// Every partition the kernel lists resolves to a flag.
#[cfg(target_os = "linux")]
#[test]
fn every_real_partition_resolves() {
    for e in std::fs::read_dir("/sys/dev/block").unwrap().flatten() {
        let p = e.path();
        if p.join("partition").is_file() {
            assert!(
                rotational_at(&p).is_some(),
                "{}: a partition whose disk flag was not found",
                p.display()
            );
        }
    }
}
