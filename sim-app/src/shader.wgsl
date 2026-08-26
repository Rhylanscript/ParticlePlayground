struct CameraUniform {
    view_proj: mat4x4<f32>,
};
@group(0) @binding(0)
var<uniform> camera: CameraUniform;

struct VertexInput {
    @location(0) quad_pos: vec2<f32>, // -0.5..0.5 quad corner
};

struct InstanceInput {
    @location(1) position: vec2<f32>,
    @location(2) color: vec3<f32>,
    @location(3) radius: f32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) uv: vec2<f32>,
};

@vertex
fn vs_main(vertex: VertexInput, instance: InstanceInput) -> VertexOutput {
    var out: VertexOutput;
    let world_pos = instance.position + vertex.quad_pos * instance.radius * 2.0;
    out.clip_position = camera.view_proj * vec4<f32>(world_pos, 0.0, 1.0);
    out.color = instance.color;
    out.uv = vertex.quad_pos;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let dist = length(in.uv);
    if (dist > 0.5) {
        discard;
    }
    let glow = smoothstep(0.5, 0.35, dist);
    return vec4<f32>(in.color * glow, 1.0);
}
