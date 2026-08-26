//! sim-core: solely physics/logic library for the particle simulator
//! zero rendering dependencies so its independently testable

mod particle;
mod simulation;

pub use particle::Particle;
pub use simulation::Simulation;

// TODO: Proper PDG Scaling

/// Coulomb's constant in simulation units (not SI, as real SI units
/// would be either super large or small at the scale rendered)
/// real PDG accurate scaling will be addressed LATER
pub const COULOMB_CONSTANT: f32 = 50.0;

/// Softening length added to r^2 in the Coulomb force so distances near
/// zero don't produce infinite force when two particles pass close to
/// each other.
pub const SOFTENING: f32 = 0.05;

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    #[test]
    fn opposite_charges_attract() {
        let mut sim = Simulation::new();
        sim.particles
            .push(Particle::new(Vec3::new(-1.0, 0.0, 0.0), 1.0, 1.0));
        sim.particles
            .push(Particle::new(Vec3::new(1.0, 0.0, 0.0), 1.0, -1.0));

        let start = sim.particles[0]
            .position
            .distance(sim.particles[1].position);
        sim.step(0.01);
        let end = sim.particles[0]
            .position
            .distance(sim.particles[1].position);

        assert!(end < start, "opposite charges should move closer together");
    }

    #[test]
    fn like_charges_repel() {
        let mut sim = Simulation::new();
        sim.particles
            .push(Particle::new(Vec3::new(-1.0, 0.0, 0.0), 1.0, 1.0));
        sim.particles
            .push(Particle::new(Vec3::new(1.0, 0.0, 0.0), 1.0, 1.0));

        let start = sim.particles[0]
            .position
            .distance(sim.particles[1].position);
        sim.step(0.01);
        let end = sim.particles[0]
            .position
            .distance(sim.particles[1].position);

        assert!(end > start, "like charges should move apart");
    }

    #[test]
    fn momentum_is_conserved_for_isolated_pair() {
        let mut sim = Simulation::new();
        sim.particles
            .push(Particle::new(Vec3::new(-1.0, 0.0, 0.0), 2.0, 1.0));
        sim.particles
            .push(Particle::new(Vec3::new(1.0, 0.0, 0.0), 3.0, -1.0));

        let total_momentum =
            |sim: &Simulation| -> Vec3 { sim.particles.iter().map(|p| p.velocity * p.mass).sum() };

        let start = total_momentum(&sim);
        for _ in 0..100 {
            sim.step(0.01);
        }
        let end = total_momentum(&sim);

        assert!(
            (start - end).length() < 1e-4,
            "total momentum should be conserved (Newton's 3rd law)"
        );
    }
}
