use glam::Vec3;

use crate::{Particle, COULOMB_CONSTANT, SOFTENING};

/// Owns the set of particles and advances them in time under mutual
/// Coulomb forces. O(n^2) pairwise force calculation, fine for the
/// particle counts we're rendering in Phase 2; revisit (Barnes-Hut,
/// spatial hashing) only if it becomes a bottleneck later.
pub struct Simulation {
    pub particles: Vec<Particle>,
}

impl Simulation {
    pub fn new() -> Self {
        Self {
            particles: Vec::new(),
        }
    }

    /// Advances the simulation by `dt` seconds (sim-time) using
    /// semi-implicit (symplectic) Euler integration, stable and cheap;
    /// swap in Velocity Verlet later if tighter energy conservation is
    /// needed over long runs.
    pub fn step(&mut self, dt: f32) {
        let n = self.particles.len();
        let mut forces = vec![Vec3::ZERO; n];

        for i in 0..n {
            for j in (i + 1)..n {
                let force_on_i = coulomb_force(&self.particles[i], &self.particles[j]);
                forces[i] += force_on_i;
                forces[j] -= force_on_i; // (newtons third law)
            }
        }

        for (particle, force) in self.particles.iter_mut().zip(forces) {
            if particle.mass > 0.0 {
                let acceleration = force / particle.mass;
                particle.velocity += acceleration * dt;
            }
            particle.position += particle.velocity * dt;
        }
    }
}

impl Default for Simulation {
    fn default() -> Self {
        Self::new()
    }
}

/// Coulomb force exerted on `a` by `b`. Positive product of charges
/// (like charges) pushes `a` away from `b`, opposite charges pull `a`
/// toward `b`.
fn coulomb_force(a: &Particle, b: &Particle) -> Vec3 {
    let delta = a.position - b.position;
    let dist_sq = delta.length_squared() + SOFTENING * SOFTENING;
    let force_magnitude = COULOMB_CONSTANT * a.charge * b.charge / dist_sq;
    delta.normalize_or_zero() * force_magnitude
}
