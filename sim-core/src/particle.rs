use glam::Vec3;
use sim_data::ParticleSpecies;

use crate::units;

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
    pub pdg_id: Option<i32>,
}

impl Particle {
    pub fn new(position: Vec3, mass: f32, charge: f32) -> Self {
        Self {
            position,
            velocity: Vec3::ZERO,
            mass,
            charge,
            pdg_id: None,
        }
    }

    /// Builds a particle from a real PDG species, converting its
    /// PDG-sourced mass (MeV/c^2) and charge (thirds of e) into sim
    /// units via `crate::units`.
    pub fn from_species(species: &ParticleSpecies, position: Vec3) -> Self {
        Self {
            position,
            velocity: Vec3::ZERO,
            mass: units::mass_from_mev(species.mass_mev),
            charge: units::charge_from_thirds(species.charge_thirds),
            pdg_id: Some(species.pdg_id),
        }
    }

    pub fn with_velocity(mut self, velocity: Vec3) -> Self {
        self.velocity = velocity;
        self
    }
}
