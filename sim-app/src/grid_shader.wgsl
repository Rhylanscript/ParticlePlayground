struct GridUniform {
    // xy = camera center (world space), 
    // zw = half-width/half-height of visible world area
    params: vec4<f32>,
};
@group(0) @binding(0)
var<uniform> grid: GridUniform;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec2<f32>,
};

@vertex
fn vs_main(@location(0) ndc_pos: vec2<f32>) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4<f32>(ndc_pos, 0.0, 1.0);
    let center = grid.params.xy;
    let half_extents = grid.params.zw;
    out.world_pos = center + ndc_pos * half_extents;
    return out;
}

// calcs how visible a grid line should be
//
// coord        = world x or y coordinate
// spacing      = distance between grid lines
// thickness    = thickness/strength of the line
//
// returns:
// 0.0 = not on a line
// 1.0 = directly on a line
fn grid_line(coord: f32, spacing: f32, thickness: f32) -> f32 {
    let dist_to_line = abs(fract(coord / spacing + 0.5) - 0.5) * spacing;
    let aa = fwidth(coord) * thickness;
    return 1.0 - smoothstep(0.0, aa, dist_to_line);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // makes a line every 1 world unit
    let minor_pixel_size = fwidth(in.world_pos.x);
    let minor_fade = 1.0 - smoothstep(0.005, 0.15, minor_pixel_size);
    let minor = max(
        grid_line(in.world_pos.x, 1.0, 1.5), 
        grid_line(in.world_pos.y, 1.0, 1.5)
    ) * minor_fade;

    // makes a line every 5 world units
    let major_pixel_size = fwidth(in.world_pos.x);
    let major_fade = 1.3 - smoothstep(0.01, 0.3, major_pixel_size);
    let major = max(
        grid_line(in.world_pos.x, 5.0, 2.0), 
        grid_line(in.world_pos.y, 5.0, 2.0)
    ) * major_fade;

    var color = vec3<f32>(0.0, 0.0, 0.0);                   // bg color
    color = mix(color, vec3<f32>(0.02, 0.02, 0.04), minor); // minor grid line color
    color = mix(color, vec3<f32>(0.04, 0.04, 0.05), major); // major grid line color

    // axis antialiasing
    let aa_x = fwidth(in.world_pos.x) * 2.0;
    let aa_y = fwidth(in.world_pos.y) * 2.0;

    // world axes
    let y_axis = 1.0 - smoothstep(0.0, aa_x, abs(in.world_pos.x));
    let x_axis = 1.0 - smoothstep(0.0, aa_y, abs(in.world_pos.y));

    // world axes colours
    color = mix(color, vec3<f32>(0.4, 0.15, 0.15), y_axis); // y axis
    color = mix(color, vec3<f32>(0.15, 0.4, 0.15), x_axis); // x axis

    // return the color, the second value is opacity (1.0 - 0.0)
    return vec4<f32>(color, 1.0);
}
