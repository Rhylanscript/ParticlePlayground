//! PDG-sourced particle property database, loaded once from `data/particles.ron`
//! and embedded into the binary at compile time (so it works identically
//! natively and in WASM, with no runtime file I/O).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ParticleCategory {
    Lepton,
    Quark,
    Boson,
    Baryon,
    Meson,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParticleSpecies {
    /// PDG Monte Carlo numbering scheme ID
    pub pdg_id: i32,
    pub name: String,
    pub symbol: String,
    pub category: ParticleCategory,
    /// Rest mass in MeV/c^2 (PDG convention). 0.0 for massless
    pub mass_mev: f64,
    /// Electric charge in units of e/3 (exact integer - e.g. electron = -3,
    /// up quark = +2) Avoids float drift in conservation law checks
    pub charge_thirds: i8,
    /// Spin in units of hbar/2 (e.g. spin-1/2 = 1, spin-1 = 2, spin-0 = 0)
    pub spin_halves: u8,
    /// Mean lifetime in seconds. `None` = stable (or effectively stable
    /// on any timescale this sim will run)
    pub lifetime_s: Option<f64>,
}

impl ParticleSpecies {
    pub fn charge_e(&self) -> f64 {
        self.charge_thirds as f64 / 3.0
    }

    pub fn spin(&self) -> f64 {
        self.spin_halves as f64 / 2.0
    }

    pub fn is_stable(&self) -> bool {
        self.lifetime_s.is_none()
    }
}

const PARTICLES_RON: &str = include_str!("../data/particles.ron");

static DATABASE: OnceLock<HashMap<i32, ParticleSpecies>> = OnceLock::new();

fn database() -> &'static HashMap<i32, ParticleSpecies> {
    DATABASE.get_or_init(|| {
        let species: Vec<ParticleSpecies> =
            ron::from_str(PARTICLES_RON).expect("data/particles.ron should parse");
        species.into_iter().map(|s| (s.pdg_id, s)).collect()
    })
}

pub fn by_pdg_id(pdg_id: i32) -> Option<&'static ParticleSpecies> {
    database().get(&pdg_id)
}

pub fn by_symbol(symbol: &str) -> Option<&'static ParticleSpecies> {
    database().values().find(|s| s.symbol == symbol)
}

pub fn all() -> impl Iterator<Item = &'static ParticleSpecies> {
    database().values()
}

/// Convenience PDG ID constants for the species wired in as of Phase 3.
pub mod pdg {
    pub const DOWN_QUARK: i32 = 1;
    pub const UP_QUARK: i32 = 2;
    pub const STRANGE_QUARK: i32 = 3;
    pub const CHARM_QUARK: i32 = 4;
    pub const BOTTOM_QUARK: i32 = 5;
    pub const TOP_QUARK: i32 = 6;
    pub const ELECTRON: i32 = 11;
    pub const ELECTRON_NEUTRINO: i32 = 12;
    pub const MUON: i32 = 13;
    pub const MUON_NEUTRINO: i32 = 14;
    pub const TAU: i32 = 15;
    pub const TAU_NEUTRINO: i32 = 16;
    pub const GLUON: i32 = 21;
    pub const PHOTON: i32 = 22;
    pub const Z_BOSON: i32 = 23;
    pub const W_BOSON: i32 = 24;
    pub const HIGGS_BOSON: i32 = 25;
    pub const NEUTRON: i32 = 2112;
    pub const PROTON: i32 = 2212;
}

/// Constituent valence-quark PDG ids for baryons that are currently
/// wired in as opaque single-particle entries (proton, neutron). This
/// intentionally does NOT replace those `ParticleSpecies` rows,
/// `by_pdg_id(pdg::PROTON)` still returns the proton's own mass/charge/
/// lifetime metadata, which Phase 5 decay needs. This is a separate
/// lookup used only at spawn time to decide what to actually push into
/// the simulation (3 quarks) instead of 1 baryon particle.
///
/// Order matches `ColorCharge::BARYON_TRIPLET` (Red, Green, Blue) when
/// consumed by the spawn code, which specific quark gets which color
/// is arbitrary (color assignment is a bookkeeping choice, not physical
/// fact at this level of approximation), so any fixed order is fine.
pub fn constituent_quarks(baryon_pdg_id: i32) -> Option<[i32; 3]> {
    match baryon_pdg_id {
        pdg::PROTON => Some([pdg::UP_QUARK, pdg::UP_QUARK, pdg::DOWN_QUARK]),
        pdg::NEUTRON => Some([pdg::UP_QUARK, pdg::DOWN_QUARK, pdg::DOWN_QUARK]),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_all_entries() {
        assert_eq!(database().len(), 19);
    }

    #[test]
    fn electron_looks_right() {
        let e = by_pdg_id(pdg::ELECTRON).unwrap();
        assert_eq!(e.charge_thirds, -3);
        assert!((e.mass_mev - 0.51099895).abs() < 1e-9);
    }

    #[test]
    fn proton_charge_is_exact_positive_one() {
        let p = by_pdg_id(pdg::PROTON).unwrap();
        assert_eq!(p.charge_e(), 1.0);
    }

    #[test]
    fn proton_decomposes_to_uud() {
        let quarks = constituent_quarks(pdg::PROTON).unwrap();
        assert_eq!(quarks, [pdg::UP_QUARK, pdg::UP_QUARK, pdg::DOWN_QUARK]);
    }

    #[test]
    fn neutron_decomposes_to_udd() {
        let quarks = constituent_quarks(pdg::NEUTRON).unwrap();
        assert_eq!(quarks, [pdg::UP_QUARK, pdg::DOWN_QUARK, pdg::DOWN_QUARK]);
    }

    #[test]
    fn electron_has_no_constituent_quarks() {
        assert!(constituent_quarks(pdg::ELECTRON).is_none());
    }

    #[test]
    fn baryon_charge_matches_sum_of_quark_charges() {
        for &baryon_id in &[pdg::PROTON, pdg::NEUTRON] {
            let baryon = by_pdg_id(baryon_id).unwrap();
            let quark_sum: i8 = constituent_quarks(baryon_id)
                .unwrap()
                .iter()
                .map(|&qid| by_pdg_id(qid).unwrap().charge_thirds)
                .sum();
            assert_eq!(
                baryon.charge_thirds, quark_sum,
                "{} charge should equal sum of its quarks charges",
                baryon.name
            );
        }
    }
}
