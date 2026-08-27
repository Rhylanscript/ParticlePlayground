use glam::Vec3;

use crate::{COULOMB_CONSTANT, SOFTENING};

/// Tuned for this sim's visual/gameplay scale, not a literal QCD alpha_s
/// value (which runs with energy scale in reality, we're not modeling
/// that running here; consistent with "Cornell approximation, not
/// lattice QCD").
pub const CORNELL_ALPHA_S: f32 = 0.5;

/// Linear confinement coefficient (the "string tension" analog). Tuned
/// so confinement visibly dominates once quarks separate by a few
/// sim-units, same spirit as `COULOMB_CONSTANT` being tuned for scale
/// rather than SI-accurate.
pub const CORNELL_CONFINEMENT_K: f32 = 2.0;

/// Plain Coulomb force on a particle at `pos_a` (charge `charge_a`) due to
/// a particle at `pos_b` (charge `charge_b`). Factored out of the
/// pairwise loop so it can be composed with `cornell_force` for
/// same-bound-group quark pairs, which feel both forces.
pub fn coulomb_force(pos_a: Vec3, charge_a: f32, pos_b: Vec3, charge_b: f32) -> Vec3 {
    let delta = pos_a - pos_b;
    let dist_sq = delta.length_squared() + SOFTENING * SOFTENING;
    let dist = dist_sq.sqrt();
    let direction = delta / dist;
    let magnitude = COULOMB_CONSTANT * charge_a * charge_b / dist_sq;
    direction * magnitude
}

/// Damping coefficient applied between bound-group members only.
/// Real confined quark systems lose kinetic energy to gluon radiation
/// as they settle toward their ground state; we aren't simulating that
/// radiation directly, so this term approximates its net effect —
/// without it, a purely attractive Cornell force never settles (no
/// equilibrium radius exists for the potential as written) and instead
/// oscillates forever, chaotically, for 3+ bodies. This is an explicit
/// modeling simplification, flagged as such rather than derived from
/// PDG data — tune by feel until triplets visually settle within a
/// couple seconds without looking sluggish/dead.
pub const CORNELL_DAMPING: f32 = 1.5;

/// Drag force on `a` opposing its velocity *relative to* `b` — pulls
/// the pair's relative motion toward zero without touching their
/// shared center-of-mass motion (so a bound group can still drift
/// together as a unit, it just stops internally oscillating).
pub fn cornell_damping_force(vel_a: Vec3, vel_b: Vec3) -> Vec3 {
    -(vel_a - vel_b) * CORNELL_DAMPING
}

/// Cornell potential force on a particle at `pos_a` due to a particle at
/// `pos_b`, for two particles confined in the same color-singlet bound
/// group.
///
/// V(r) = -(4/3) * alpha_s / r  +  k * r
///
/// Both terms pull the pair together: the short-range Coulomb-like term
/// is attractive in QCD regardless of the specific color pairing at this
/// simplified single-tag level, and the linear term is the confinement
/// "string" that grows without bound as quarks separate, this is what
/// makes an isolated quark impossible to produce in this sim, matching
/// real confinement phenomenology.
///
/// Always attractive: there is no repulsive branch here. A true SU(3)
/// treatment would have some pairwise color combinations repel; we're
/// deliberately not modeling color algebra, only confinement of a
/// pre assigned singlet (see `ColorCharge` docs).
pub fn cornell_force(pos_a: Vec3, pos_b: Vec3) -> Vec3 {
    let delta = pos_b - pos_a; // points from A toward B -> attractive
    let dist_sq = delta.length_squared() + SOFTENING * SOFTENING;
    let dist = dist_sq.sqrt();
    let direction = delta / dist;
    let coulomb_like = (4.0 / 3.0) * CORNELL_ALPHA_S / dist_sq;
    let confinement = CORNELL_CONFINEMENT_K;
    direction * (coulomb_like + confinement)
}
