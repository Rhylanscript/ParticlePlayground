use glam::{Mat4, Vec2};

/// Simple top-down orthographic camera - pan + zoom, no rotation.
/// Full camera/UI polish (drag-to-pan, smoothing, etc.) lands in Phase 7.
pub struct Camera {
    pub center: Vec2,
    /// world-space height visible on screen, width is derived from aspect
    pub zoom: f32,
}

impl Camera {
    pub fn new() -> Self {
        Self {
            center: Vec2::ZERO,
            zoom: 10.0,
        }
    }

    pub fn view_proj(&self, aspect: f32) -> Mat4 {
        let half_height = self.zoom * 0.5;
        let half_width = half_height * aspect;
        let proj = Mat4::orthographic_rh(
            -half_width,
            half_width,
            -half_height,
            half_height,
            -100.0,
            100.0,
        );
        let view = Mat4::from_translation(-self.center.extend(0.0));
        proj * view
    }

    pub fn pan(&mut self, delta: Vec2) {
        self.center += delta;
    }

    pub fn zoom_by(&mut self, factor: f32) {
        self.zoom = (self.zoom * factor).clamp(1.0, 200.0);
    }
}

impl Default for Camera {
    fn default() -> Self {
        Self::new()
    }
}
