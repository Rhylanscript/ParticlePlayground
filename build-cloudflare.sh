#!/bin/sh
set -eu

# Cloudflare Pages does not provide Rust or Trunk by default.
if ! command -v rustc >/dev/null 2>&1; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    . "$HOME/.cargo/env"
fi

export PATH="$HOME/.cargo/bin:$PATH"
rustup target add wasm32-unknown-unknown

if ! command -v trunk >/dev/null 2>&1; then
    cargo install trunk --locked
fi

cd sim-app
trunk build --release
