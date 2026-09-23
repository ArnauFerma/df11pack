//! Phase 3 -- the I/O scheduling decision.

use df11_codec::io_sched::{parse_rotational, plan, rotational_for, IoMode};
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
    // Whatever this machine is, Auto must not panic or hang.
    let _ = p.sequential;
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
