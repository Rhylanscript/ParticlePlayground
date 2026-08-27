use glam::Vec3;
use sim_data::ParticleSpecies;

use crate::color::ColorCharge;
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

    /// Id shared by all particles confined in the same color-singlet
    /// group (e.g. the 3 quarks making up a proton). `None` for
    /// particles that aren't part of a bound group. Looked up in
    /// `Simulation::bound_groups` for group-level metadata.
    pub bound_group: Option<u32>,

    /// Discrete color tag, only meaningful when `bound_group` is
    /// `Some`. See `ColorCharge` docs — this is a confinement
    /// bookkeeping tag, not a real SU(3) color simulation.
    pub color_charge: Option<ColorCharge>,
}

impl Particle {
    pub fn new(position: Vec3, mass: f32, charge: f32) -> Self {
        Self {
            position,
            velocity: Vec3::ZERO,
            mass,
            charge,
            pdg_id: None,
            bound_group: None,
            color_charge: None,
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
            bound_group: None,
            color_charge: None,
        }
    }

    pub fn with_velocity(mut self, velocity: Vec3) -> Self {
        self.velocity = velocity;
        self
    }

    /// Tags this particle as a member of a color-singlet bound group.
    /// Used when spawning a baryon's constituent quarks, each gets a
    /// distinct `ColorCharge` from `ColorCharge::BARYON_TRIPLET` and
    /// the same `group_id`.
    pub fn with_bound_group(mut self, group_id: u32, color: ColorCharge) -> Self {
        self.bound_group = Some(group_id);
        self.color_charge = Some(color);
        self
    }
}
