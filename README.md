# Particle Playground

Particle Physics simulation

## Cloudflare Pages deployment

This is a Rust/WebAssembly browser application and should be deployed with
Cloudflare Pages, not as a Worker.

Configure the Pages project with:

- **Build command:** `bash build-cloudflare.sh`
- **Build output directory:** `sim-app/dist`
- **Root directory:** `/`

The build script installs Rust, the `wasm32-unknown-unknown` target, and Trunk
because they are not included in the default Cloudflare Pages environment.
