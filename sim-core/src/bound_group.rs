/// Metadata about a color-singlet group of confined particles (e.g. the 3
/// quarks making up a proton).
///
/// Indexed by `Particle::bound_group` (a `u32` id), NOT by position in the
/// particle vec — groups survive `Vec<Particle>` reshuffling (removal,
/// swap_remove) because this struct holds no indices, only facts about
/// the group as a whole. Each member `Particle` carries the id itself.
#[derive(Debug, Clone)]
pub struct BoundGroupInfo {
    /// PDG id of the composite hadron this group represents, e.g. 2212
    /// for a proton, 2112 for a neutron. Not derivable from any single
    /// member particle — it's a property of the group, kept here so
    /// Phase 5 (decay) can ask "what lifetime does this group have"
    /// without re-deriving it from quark flavor content every time.
    pub baryon_pdg_id: i32,
}
