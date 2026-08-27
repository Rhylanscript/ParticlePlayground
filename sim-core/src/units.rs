//! Unit conventions for the physics engine.
//!
//! sim-core doesn't use SI units. Real EM dynamics happen at femtosecond/
//! femtometer scales -- simulating that literally would demand absurdly
//! tiny timesteps and blow out f32 precision. Instead we use natural
//! units, in the same spirit as atomic units in computational chemistry:
//! mass is a multiple of the electron rest mass, charge a multiple of the
//! elementary charge. Real ratios between particles stay exact; only the
//! absolute magnitudes are rescaled. Length/time/force stay in the
//! sim-calibrated scale already established by `COULOMB_CONSTANT` and
//! `SOFTENING` in Phase 2 -- this module only concerns mass and charge,
//! since that's the axis real PDG data affects.

pub const ELECTRON_MASS_MEV: f64 = 0.51099895;

/// Convert a PDG rest mass (MeV/c^2) into sim mass units (multiples of m_e).
/// Massless species (photon, gluon, neutrinos in this dataset) correctly
/// convert to 0.0 -- callers integrating motion must guard against that
/// rather than dividing by it.
pub fn mass_from_mev(mass_mev: f64) -> f32 {
    (mass_mev / ELECTRON_MASS_MEV) as f32
}

/// Convert a charge stored as thirds-of-e into sim charge units (multiples of e).
pub fn charge_from_thirds(charge_thirds: i8) -> f32 {
    charge_thirds as f32 / 3.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proton_is_about_1836_electron_masses() {
        let ratio = mass_from_mev(938.27208943);
        assert!((ratio - 1836.15).abs() < 0.1);
    }

    #[test]
    fn quark_charges_are_thirds() {
        assert!((charge_from_thirds(2) - 0.6666667).abs() < 1e-6); // up
        assert!((charge_from_thirds(-1) - (-0.3333333)).abs() < 1e-6); // down
    }

    #[test]
    fn massless_species_convert_to_zero() {
        assert_eq!(mass_from_mev(0.0), 0.0);
    }
}
