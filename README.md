# Android NNAPI Prove of concept

Native Android NNAPI usage. Prove of concept using Rust. I only implemented add and matmul operations.

## Prerequisites

- NDK >= 27
- API >= 29 ? (I'm not sure)
- Rust with aarch64-linux-android target
- [Cargo apk](https://github.com/rust-mobile/cargo-apk)

## Prepare Cargo Config

under project root, create a `.cargo/config` file with the following content:

```toml
[build]
target = "aarch64-linux-android"

[env]
ANDROID_HOME = "YOUR SDK HOME"
ANDROID_NDK_ROOT = "YOUR NDK HOME"
```

## Run

```sh
cargo apk run
```
