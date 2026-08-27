/// Discrete color-charge tag for strong-force bookkeeping.
///
/// This is NOT a real SU(3) color field simulation — no gluon exchange,
/// no color algebra, no running coupling. It's a lightweight tag used to:
///   1. Enforce that a bound group of 3 quarks forms a color-singlet
///      (one Red, one Green, one Blue) rather than 3 same-colored quarks.
///   2. Later (Phase 9) support mesons as color/anticolor pairs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ColorCharge {
    Red,
    Green,
    Blue,
    AntiRed,
    AntiGreen,
    AntiBlue,
}

impl ColorCharge {
    /// The three colors making up a baryon color-singlet, in a fixed order.
    /// Used when spawning a baryon's 3 constituent quarks so each gets a
    /// distinct color deterministically rather than randomly.
    pub const BARYON_TRIPLET: [ColorCharge; 3] =
        [ColorCharge::Red, ColorCharge::Green, ColorCharge::Blue];
}
