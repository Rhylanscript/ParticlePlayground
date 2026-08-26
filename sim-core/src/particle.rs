use glam::Vec3;

/// A single simulated particle. Phase 2 keeps this generic (just the
/// physical quantities forces act on) real PDG-sourced particle
/// identities (electron, proton, quark, ...) get layered on top in
/// Phase 3 via sim-data
#[derive(Debug, Clone, Copy)]
pub struct Particle {
    pub position: Vec3,
    pub velocity: Vec3,
    pub mass: f32,
    pub charge: f32,
}

impl Particle {
    pub fn new(position: Vec3, mass: f32, charge: f32) -> Self {
        Self {
            position,
            velocity: Vec3::ZERO,
            mass,
            charge,
        }
    }

    pub fn with_velocity(mut self, velocity: Vec3) -> Self {
        self.velocity = velocity;
        self
    }
}
