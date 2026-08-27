use std::collections::HashMap;

use glam::Vec3;

use crate::bound_group::BoundGroupInfo;
use crate::forces::{cornell_damping_force, cornell_force, coulomb_force};
use crate::Particle;

/// Owns the set of particles and advances them in time under mutual
/// Coulomb forces. O(n^2) pairwise force calculation, fine for the
/// particle counts we're rendering in Phase 2; revisit (Barnes-Hut,
/// spatial hashing) only if it becomes a bottleneck later.
pub struct Simulation {
    pub particles: Vec<Particle>,
    /// Group-level metadata for bound quark groups, keyed by the same
    /// id stored in each member `Particle::bound_group`. Deliberately
    /// NOT keyed by index into `particles` indices shift on
    /// removal/swap_remove, group ids don't.
    pub bound_groups: HashMap<u32, BoundGroupInfo>,
    next_group_id: u32,
}

/// Bound-group forces (Cornell confinement) are stiffer than plain
/// Coulomb, the attractive-only force means quarks repeatedly pass
/// through each other's softened core, and a single dt=0.01-scale step
/// under-resolves that crossing, numerically injecting energy each
/// pass. Sub-stepping resolves the crossing properly without changing
/// the force law itself. Revisit if this becomes a bottleneck at
/// higher particle counts (Phase 7), options then: substep only
/// bound-group pairs, or move to Velocity Verlet as already flagged
/// above.
const SUBSTEPS: u32 = 8;

impl Simulation {
    pub fn new() -> Self {
        Self {
            particles: Vec::new(),
            bound_groups: HashMap::new(),
            next_group_id: 0,
        }
    }

    /// Allocates a fresh bound-group id and records its metadata.
    /// Callers (e.g. a proton spawn function) use the returned id to
    /// tag each constituent quark's `Particle::bound_group`.
    pub fn create_bound_group(&mut self, baryon_pdg_id: i32) -> u32 {
        let id = self.next_group_id;
        self.next_group_id += 1;
        self.bound_groups
            .insert(id, BoundGroupInfo { baryon_pdg_id });
        id
    }

    pub fn step(&mut self, dt: f32) {
        let sub_dt = dt / SUBSTEPS as f32;
        for _ in 0..SUBSTEPS {
            self.substep(sub_dt);
        }
    }

    fn substep(&mut self, dt: f32) {
        let n = self.particles.len();
        let mut forces = vec![Vec3::ZERO; n];

        for i in 0..n {
            for j in (i + 1)..n {
                let a = &self.particles[i];
                let b = &self.particles[j];

                let mut force_on_i = coulomb_force(a.position, a.charge, b.position, b.charge);

                // Same-bound-group pairs additionally feel the Cornell
                // confinement force on top of their (real, fractional)
                // EM charge interaction. Different groups, or particles
                // with no group at all, only ever feel plain Coulomb.
                if let (Some(group_a), Some(group_b)) = (a.bound_group, b.bound_group) {
                    if group_a == group_b {
                        force_on_i += cornell_force(a.position, b.position);
                        force_on_i += cornell_damping_force(a.velocity, b.velocity);
                    }
                }

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

#[cfg(test)]
mod cornell_confinement_tests {
    use super::*;
    use crate::color::ColorCharge;

    #[test]
    fn isolated_baryon_stays_roughly_bound() {
        let mut sim = Simulation::new();
        let group = sim.create_bound_group(2212); // proton

        let quark = |charge: f32, color: ColorCharge, pos: Vec3| {
            Particle::new(pos, 4.0, charge).with_bound_group(group, color)
        };

        // u, u, d roughly spaced in a small triangle, matching sim-app's
        // spawn_baryon cluster_radius.
        sim.particles = vec![
            quark(2.0 / 3.0, ColorCharge::Red, Vec3::new(0.3, 0.0, 0.0)),
            quark(2.0 / 3.0, ColorCharge::Green, Vec3::new(-0.15, 0.26, 0.0)),
            quark(-1.0 / 3.0, ColorCharge::Blue, Vec3::new(-0.15, -0.26, 0.0)),
        ];

        let mut max_seen = 0.0f32;
        for _ in 0..2000 {
            sim.step(0.01);
            for i in 0..3 {
                for j in (i + 1)..3 {
                    let d = sim.particles[i]
                        .position
                        .distance(sim.particles[j].position);
                    max_seen = max_seen.max(d);
                }
            }
        }

        assert!(
            max_seen < 10.0,
            "isolated baryon should stay roughly confined, max pairwise distance was {max_seen}"
        );
    }
}
