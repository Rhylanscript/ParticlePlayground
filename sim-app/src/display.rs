//! Aggregates raw physics particles into presentational entities.
//!
//! `sim_core::Particle` is a pure physics object — a neutron, since
//! Phase 4's quark decomposition, is 3 separate quark particles sharing
//! a `bound_group` id, not 1 particle. That's correct for physics
//! (Cornell confinement needs 3 real bodies to pull together) but wrong
//! for rendering and UI: a user looking at the sim should see "a
//! neutron", not "3 dots that happen to be near each other".
//!
//! `DisplayEntity` is the single-entity-per-visible-thing view used by
//! rendering (`build_instances`) and, later, by hover/picking (Phase 7)
//! — both need exactly the same "list of named, positioned, sized
//! things on screen" data, so they share this one aggregation instead
//! of each re-deriving group membership independently.
use std::collections::HashMap;

use glam::Vec3;

use crate::{base_radius, color_for};

/// One visible thing on screen: either a free particle (1:1 with a
/// `Particle`) or an aggregated bound group (e.g. 3 quarks presented as
/// 1 neutron).
pub struct DisplayEntity {
    pub position: Vec3,
    pub color: [f32; 3],
    pub radius: f32,
    /// Human-readable name, e.g. "Neutron", "Electron". Not used by
    /// rendering yet - reserved for phase 7 hover/picking
    #[allow(dead_code)]
    pub name: &'static str,
    #[allow(dead_code)]
    pub pdg_id: Option<i32>,
}

pub fn build_display_entities(sim: &sim_core::Simulation) -> Vec<DisplayEntity> {
    let mut entities = Vec::new();

    // Aggregate every bound group to its members' centroid first.
    let mut group_sums: HashMap<u32, (Vec3, u32)> = HashMap::new();
    for p in &sim.particles {
        if let Some(gid) = p.bound_group {
            let entry = group_sums.entry(gid).or_insert((Vec3::ZERO, 0));
            entry.0 += p.position;
            entry.1 += 1;
        }
    }
    for (gid, info) in &sim.bound_groups {
        let Some((sum, count)) = group_sums.get(gid) else {
            // Group registered but has no live members (e.g. a future
            // decay removed them) — nothing to draw.
            continue;
        };
        let centroid = *sum / *count as f32;
        let species = sim_data::by_pdg_id(info.baryon_pdg_id);
        entities.push(DisplayEntity {
            position: centroid,
            color: species
                .map(|s| color_for(s.charge_thirds))
                .unwrap_or([0.65, 0.65, 0.7]),
            radius: base_radius(sim_data::ParticleCategory::Baryon),
            name: species.map(|s| s.name.as_str()).unwrap_or("Unknown Baryon"),
            pdg_id: Some(info.baryon_pdg_id),
        });
    }

    // Every particle NOT in a bound group renders individually, same
    // as before Phase 4's baryon decomposition existed.
    for p in &sim.particles {
        if p.bound_group.is_some() {
            continue; // already represented by its group's entity above
        }
        let species = p.pdg_id.and_then(sim_data::by_pdg_id);
        let charge_thirds = species
            .map(|s| s.charge_thirds)
            .unwrap_or(if p.charge > 0.0 {
                3
            } else if p.charge < 0.0 {
                -3
            } else {
                0
            });
        entities.push(DisplayEntity {
            position: p.position,
            color: color_for(charge_thirds),
            radius: species
                .map(|s| base_radius(s.category))
                .unwrap_or(0.12 + 0.04 * p.mass.min(3.0)),
            name: species
                .map(|s| s.name.as_str())
                .unwrap_or("Unknown Particle"),
            pdg_id: p.pdg_id,
        });
    }

    entities
}
