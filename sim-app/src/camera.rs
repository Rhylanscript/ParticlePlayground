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

    pub fn half_extents(&self, aspect: f32) -> Vec2 {
        let half_height = self.zoom * 0.5;
        let half_width = half_height * aspect;
        Vec2::new(half_width, half_height)
    }

    pub fn pan(&mut self, delta: Vec2) {
        self.center += delta;
    }

    // pub fn zoom_by(&mut self, factor: f32) {
    //     self.zoom = (self.zoom * factor).clamp(1.0, 400.0);
    // }

    /// Zooms while keeping `world_point` fixed under the same screen
    /// position -- i.e. zoom-to-cursor instead of zoom-to-center.
    /// Recomputes the *actual* applied factor after clamping, so
    /// hitting the zoom limits doesn't overshoot the re-anchoring.
    pub fn zoom_at(&mut self, factor: f32, world_point: Vec2) {
        let old_zoom = self.zoom;
        self.zoom = (self.zoom * factor).clamp(1.0, 400.0);
        let applied_factor = self.zoom / old_zoom;
        self.center = world_point - (world_point - self.center) * applied_factor;
    }

    /// Converts a screen-space position (physical pixels, origin
    /// top-left, y-down -- winit's convention) into world space.
    pub fn screen_to_world(&self, screen_pos: Vec2, screen_size: Vec2, aspect: f32) -> Vec2 {
        let half_extents = self.half_extents(aspect);
        let ndc = Vec2::new(
            (screen_pos.x / screen_size.x) * 2.0 - 1.0,
            1.0 - (screen_pos.y / screen_size.y) * 2.0, // flip: screen y-down, world y-up
        );
        self.center + ndc * half_extents
    }
}

impl Default for Camera {
    fn default() -> Self {
        Self::new()
    }
}
