//! Security regression tests for High-Severity #4:
//! Insecure `--kv-disk` temporary file creation (predictable PID filename,
//! default 0o644 permissions, symlink-following, unchecked `F_NOCACHE` `fcntl`).

use inkling::engine::{create_kv_disk_file_in, set_fd_nocache};
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::fd::AsRawFd;
use std::os::unix::fs::{MetadataExt, PermissionsExt};

#[test]
fn kv_disk_file_has_0600_permissions() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let mut file = create_kv_disk_file_in(dir.path()).expect("create kv disk file");

    // `fstat` works on the open file descriptor even after immediate unlink.
    let mode = file.metadata().expect("fstat kv file").permissions().mode() & 0o777;
    assert_eq!(
        mode, 0o600,
        "kv-disk temp file must be created with 0o600 permissions, got 0o{mode:03o}"
    );
    assert_eq!(
        mode & 0o077,
        0,
        "kv-disk temp file must not be group- or world-accessible"
    );

    // Confirm the unlinked file is still readable and writable via the open FD.
    let payload = b"kv-disk-smoke-payload-0123456789";
    file.write_all(payload).expect("write kv payload");
    file.seek(SeekFrom::Start(0)).expect("seek to start");
    let mut read_back = Vec::new();
    file.read_to_end(&mut read_back).expect("read kv payload");
    assert_eq!(read_back, payload);
}

#[test]
fn kv_disk_file_resists_predictable_pid_symlink_attack() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let canary_path = dir.path().join("canary.txt");
    let sentinel = b"CANARY_SECRET_MUST_NOT_BE_TRUNCATED_OR_MODIFIED";
    std::fs::write(&canary_path, sentinel).expect("write canary");
    let canary_ino = std::fs::metadata(&canary_path).expect("canary stat").ino();

    // Pre-plant the exact legacy predictable filename as a symlink to the canary file.
    let predictable = dir.path().join(format!("inkling-kv-{}.bin", std::process::id()));
    std::os::unix::fs::symlink(&canary_path, &predictable).expect("plant symlink");

    let mut file = create_kv_disk_file_in(dir.path()).expect("create kv disk file");
    file.write_all(b"OVERWRITE_ATTEMPT").expect("write to kv file");
    file.flush().expect("flush kv file");

    // Canary must be completely untouched and untruncated.
    let canary_after = std::fs::read(&canary_path).expect("read canary after");
    assert_eq!(
        canary_after.as_slice(),
        sentinel,
        "symlink attack truncated or modified the canary file!"
    );

    // Returned file must be a distinct inode, not the canary file.
    let kv_ino = file.metadata().expect("fstat kv file").ino();
    assert_ne!(
        kv_ino, canary_ino,
        "kv-disk temp file must not point to the symlink target inode"
    );

    // The pre-planted symlink itself must not have been followed or unlinked.
    let link_meta = std::fs::symlink_metadata(&predictable).expect("symlink still exists");
    assert!(
        link_meta.file_type().is_symlink(),
        "pre-planted symlink should remain untouched"
    );
}

#[test]
fn kv_disk_file_unique_inodes_concurrent_and_unlinked() {
    let dir = tempfile::tempdir().expect("create temp dir");

    // Back-to-back calls in the same process must produce distinct inodes.
    let mut f1 = create_kv_disk_file_in(dir.path()).expect("create f1");
    let mut f2 = create_kv_disk_file_in(dir.path()).expect("create f2");
    let m1 = f1.metadata().expect("stat f1");
    let m2 = f2.metadata().expect("stat f2");
    assert_ne!(
        (m1.dev(), m1.ino()),
        (m2.dev(), m2.ino()),
        "two kv-disk files in the same process collided on the same inode"
    );

    f1.write_all(b"FILE_ONE").expect("write f1");
    f2.write_all(b"FILE_TWO").expect("write f2");
    f1.seek(SeekFrom::Start(0)).expect("seek f1");
    f2.seek(SeekFrom::Start(0)).expect("seek f2");
    let (mut b1, mut b2) = (Vec::new(), Vec::new());
    f1.read_to_end(&mut b1).expect("read f1");
    f2.read_to_end(&mut b2).expect("read f2");
    assert_eq!(b1, b"FILE_ONE");
    assert_eq!(b2, b"FILE_TWO");

    // Concurrent calls across threads must all succeed with unique inodes.
    let dir_path = dir.path().to_path_buf();
    let handles: Vec<_> = (0..8)
        .map(|_| {
            let p = dir_path.clone();
            std::thread::spawn(move || create_kv_disk_file_in(&p).expect("concurrent create"))
        })
        .collect();
    let files: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    let mut inos: Vec<(u64, u64)> = files
        .iter()
        .map(|f| {
            let m = f.metadata().unwrap();
            (m.dev(), m.ino())
        })
        .collect();
    inos.push((m1.dev(), m1.ino()));
    inos.push((m2.dev(), m2.ino()));
    inos.sort();
    inos.dedup();
    assert_eq!(
        inos.len(),
        10,
        "all 10 open kv-disk files must have unique (dev, ino)"
    );

    // Every file must already be unlinked from the directory while still open.
    let remaining: Vec<_> = std::fs::read_dir(dir.path())
        .expect("read_dir")
        .collect::<Result<Vec<_>, _>>()
        .expect("dir entries");
    assert!(
        remaining.is_empty(),
        "expected zero leftover files in temp dir, found: {remaining:?}"
    );
}

#[test]
fn kv_disk_file_f_nocache_fcntl_is_checked() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let file = create_kv_disk_file_in(dir.path()).expect("create kv disk file");

    // Valid open file descriptor must succeed.
    set_fd_nocache(file.as_raw_fd()).expect("F_NOCACHE on valid fd must succeed");

    // Invalid file descriptor must return Err(EBADF), proving fcntl return value is checked.
    let err = set_fd_nocache(-1).expect_err("F_NOCACHE on invalid fd -1 must fail");
    assert_eq!(
        err.raw_os_error(),
        Some(libc::EBADF),
        "expected EBADF for invalid fd, got: {err:?}"
    );
}
