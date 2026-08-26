//! sim-core: solely physics/logic library for the particle simulator
//! zero rendering dependencies so its independently testable

pub fn placeholder() -> &'static str {
    "sim-core is wired up - physics types land in Phase 2"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placeholder_works() {
        assert_eq!(
            placeholder(),
            "sim-core is wired up - physics types land in Phase 2"
        );
    }
}
