mod camera;

use std::collections::HashSet;
use std::sync::Arc;

use glam::{Mat4, Vec2, Vec3};
use rand::Rng;
use wgpu::util::DeviceExt;
use winit::{
    event::{ElementState, Event, MouseScrollDelta, WindowEvent},
    event_loop::EventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowBuilder},
};

use camera::Camera;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

const DEMO_PARTICLE_COUNT: usize = 40;
const INSTANCE_CAPACITY: usize = 2048;
const MAX_DT: f32 = 0.05; // clamp so a slow frame doesn't blow up the sim

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct QuadVertex {
    position: [f32; 2],
}

impl QuadVertex {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<QuadVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Float32x2,
            }],
        }
    }
}

const QUAD_VERTICES: &[QuadVertex] = &[
    QuadVertex {
        position: [-0.5, -0.5],
    },
    QuadVertex {
        position: [0.5, -0.5],
    },
    QuadVertex {
        position: [0.5, 0.5],
    },
    QuadVertex {
        position: [-0.5, -0.5],
    },
    QuadVertex {
        position: [0.5, 0.5],
    },
    QuadVertex {
        position: [-0.5, 0.5],
    },
];

const FULLSCREEN_QUAD: &[QuadVertex] = &[
    QuadVertex {
        position: [-1.0, -1.0],
    },
    QuadVertex {
        position: [1.0, -1.0],
    },
    QuadVertex {
        position: [1.0, 1.0],
    },
    QuadVertex {
        position: [-1.0, -1.0],
    },
    QuadVertex {
        position: [1.0, 1.0],
    },
    QuadVertex {
        position: [-1.0, 1.0],
    },
];

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GridUniform {
    params: [f32; 4],
}

impl GridUniform {
    fn update(&mut self, camera: &Camera, aspect: f32) {
        let half_extents = camera.half_extents(aspect);
        self.params = [
            camera.center.x,
            camera.center.y,
            half_extents.x,
            half_extents.y,
        ];
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ParticleInstance {
    position: [f32; 2],
    color: [f32; 3],
    radius: f32,
}

impl ParticleInstance {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<ParticleInstance>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: 8,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: 20,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32,
                },
            ],
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
}

impl CameraUniform {
    fn identity() -> Self {
        Self {
            view_proj: Mat4::IDENTITY.to_cols_array_2d(),
        }
    }

    fn update(&mut self, camera: &Camera, aspect: f32) {
        self.view_proj = camera.view_proj(aspect).to_cols_array_2d();
    }
}

fn spawn_demo_particles(count: usize) -> sim_core::Simulation {
    let mut rng = rand::thread_rng();
    let mut sim = sim_core::Simulation::new();
    for _ in 0..count {
        let angle = rng.gen_range(0.0..std::f32::consts::TAU);
        let radius = rng.gen_range(0.5..4.0);
        let position = Vec3::new(angle.cos() * radius, angle.sin() * radius, 0.0);
        let charge = if rng.gen_bool(0.5) { 1.0 } else { -1.0 };
        let mass = rng.gen_range(0.5..2.0);
        sim.particles
            .push(sim_core::Particle::new(position, mass, charge));
    }
    sim
}

fn build_instances(sim: &sim_core::Simulation) -> Vec<ParticleInstance> {
    sim.particles
        .iter()
        .map(|p| {
            let color = if p.charge > 0.0 {
                [1.0, 0.35, 0.25]
            } else {
                [0.3, 0.55, 1.0]
            };
            ParticleInstance {
                position: [p.position.x, p.position.y],
                color,
                radius: 0.12 + 0.04 * p.mass,
            }
        })
        .collect()
}

struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    window: Arc<Window>,

    render_pipeline: wgpu::RenderPipeline,
    quad_vertex_buffer: wgpu::Buffer,
    instance_buffer: wgpu::Buffer,

    background_pipeline: wgpu::RenderPipeline,
    background_vertex_buffer: wgpu::Buffer,
    grid_uniform: GridUniform,
    grid_buffer: wgpu::Buffer,
    grid_bind_group: wgpu::BindGroup,

    camera: Camera,
    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,

    simulation: sim_core::Simulation,
    last_frame: web_time::Instant,
    pressed_keys: HashSet<KeyCode>,
}

impl State {
    async fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();

        let backends = if cfg!(target_arch = "wasm32") {
            wgpu::Backends::GL
        } else {
            wgpu::Backends::all()
        };

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends,
            ..Default::default()
        });

        let surface = instance.create_surface(window.clone()).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("no suitable GPU adapter found");

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    required_features: wgpu::Features::empty(),
                    required_limits: if cfg!(target_arch = "wasm32") {
                        wgpu::Limits::downlevel_webgl2_defaults()
                    } else {
                        wgpu::Limits::default()
                    },
                    label: None,
                },
                None,
            )
            .await
            .expect("failed to create device");

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("particle shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let camera = Camera::new();
        let mut camera_uniform = CameraUniform::identity();
        camera_uniform.update(&camera, config.width as f32 / config.height as f32);

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("camera buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("camera bind group layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera bind group"),
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("render pipeline layout"),
            bind_group_layouts: &[&camera_bind_group_layout],
            push_constant_ranges: &[],
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("render pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[QuadVertex::desc(), ParticleInstance::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        let quad_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("quad vertex buffer"),
            contents: bytemuck::cast_slice(QUAD_VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("instance buffer"),
            size: (INSTANCE_CAPACITY * std::mem::size_of::<ParticleInstance>())
                as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let grid_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("grid shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("grid_shader.wgsl").into()),
        });

        let mut grid_uniform = GridUniform { params: [0.0; 4] };
        grid_uniform.update(&camera, config.width as f32 / config.height as f32);

        let grid_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("grid buffer"),
            contents: bytemuck::cast_slice(&[grid_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let grid_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("grid bind group layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let grid_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("grid bind group"),
            layout: &grid_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: grid_buffer.as_entire_binding(),
            }],
        });

        let background_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("background pipeline layout"),
                bind_group_layouts: &[&grid_bind_group_layout],
                push_constant_ranges: &[],
            });

        let background_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("background pipeline"),
            layout: Some(&background_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &grid_shader,
                entry_point: "vs_main",
                buffers: &[QuadVertex::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &grid_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        let background_vertex_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("background vertex buffer"),
                contents: bytemuck::cast_slice(FULLSCREEN_QUAD),
                usage: wgpu::BufferUsages::VERTEX,
            });

        let simulation = spawn_demo_particles(DEMO_PARTICLE_COUNT);

        Self {
            surface,
            device,
            queue,
            config,
            size,
            window,
            render_pipeline,
            quad_vertex_buffer,
            instance_buffer,
            background_pipeline,
            background_vertex_buffer,
            grid_uniform,
            grid_buffer,
            grid_bind_group,
            camera,
            camera_uniform,
            camera_buffer,
            camera_bind_group,
            simulation,
            last_frame: web_time::Instant::now(),
            pressed_keys: HashSet::new(),
        }
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    fn handle_key(&mut self, code: KeyCode, state: ElementState) {
        match state {
            ElementState::Pressed => {
                self.pressed_keys.insert(code);
            }
            ElementState::Released => {
                self.pressed_keys.remove(&code);
            }
        }
    }

    fn handle_scroll(&mut self, delta: MouseScrollDelta) {
        let scroll_y = match delta {
            MouseScrollDelta::LineDelta(_, y) => y,
            MouseScrollDelta::PixelDelta(pos) => (pos.y / 100.0) as f32,
        };
        let zoom_factor = 1.0 - scroll_y * 0.1;
        self.camera.zoom_by(zoom_factor);
    }

    fn update(&mut self) {
        let now = web_time::Instant::now();
        let dt = (now - self.last_frame).as_secs_f32().min(MAX_DT);
        self.last_frame = now;

        let pan_speed = self.camera.zoom * 0.6 * dt;
        let mut pan = Vec2::ZERO;
        if self.pressed_keys.contains(&KeyCode::KeyW)
            || self.pressed_keys.contains(&KeyCode::ArrowUp)
        {
            pan.y += pan_speed;
        }
        if self.pressed_keys.contains(&KeyCode::KeyS)
            || self.pressed_keys.contains(&KeyCode::ArrowDown)
        {
            pan.y -= pan_speed;
        }
        if self.pressed_keys.contains(&KeyCode::KeyA)
            || self.pressed_keys.contains(&KeyCode::ArrowLeft)
        {
            pan.x -= pan_speed;
        }
        if self.pressed_keys.contains(&KeyCode::KeyD)
            || self.pressed_keys.contains(&KeyCode::ArrowRight)
        {
            pan.x += pan_speed;
        }
        self.camera.pan(pan);

        self.simulation.step(dt);

        let aspect = self.config.width as f32 / self.config.height as f32;
        self.camera_uniform.update(&self.camera, aspect);
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );

        self.grid_uniform.update(&self.camera, aspect);
        self.queue.write_buffer(
            &self.grid_buffer,
            0,
            bytemuck::cast_slice(&[self.grid_uniform]),
        );

        let instances = build_instances(&self.simulation);
        self.queue
            .write_buffer(&self.instance_buffer, 0, bytemuck::cast_slice(&instances));
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("render encoder"),
            });

        let instance_count = self.simulation.particles.len().min(INSTANCE_CAPACITY) as u32;

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("render pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            render_pass.set_pipeline(&self.background_pipeline);
            render_pass.set_bind_group(0, &self.grid_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.background_vertex_buffer.slice(..));
            render_pass.draw(0..FULLSCREEN_QUAD.len() as u32, 0..1);

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.quad_vertex_buffer.slice(..));
            render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
            render_pass.draw(0..QUAD_VERTICES.len() as u32, 0..instance_count);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

pub async fn run() {
    #[cfg(target_arch = "wasm32")]
    {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        console_log::init_with_level(log::Level::Info).expect("failed to init logger");
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::init();
    }

    let event_loop = EventLoop::new().unwrap();
    let window = Arc::new(
        WindowBuilder::new()
            .with_title("Particle Physics Simulator")
            .build(&event_loop)
            .unwrap(),
    );

    #[cfg(target_arch = "wasm32")]
    {
        use winit::platform::web::WindowExtWebSys;
        web_sys::window()
            .and_then(|win| win.document())
            .and_then(|doc| doc.get_element_by_id("wasm-canvas"))
            .and_then(|dst| {
                let canvas = window.canvas()?;
                canvas.set_id("sim-canvas");
                dst.append_child(&canvas).ok()
            })
            .expect("couldn't append canvas to document");
    }

    let mut state = State::new(window.clone()).await;

    event_loop
        .run(move |event, elwt| {
            if let Event::WindowEvent { window_id, event } = event {
                if window_id != state.window.id() {
                    return;
                }
                match event {
                    WindowEvent::CloseRequested => elwt.exit(),
                    WindowEvent::Resized(physical_size) => state.resize(physical_size),
                    WindowEvent::MouseWheel { delta, .. } => state.handle_scroll(delta),
                    WindowEvent::KeyboardInput {
                        event:
                            winit::event::KeyEvent {
                                physical_key: PhysicalKey::Code(code),
                                state: key_state,
                                ..
                            },
                        ..
                    } => {
                        if code == KeyCode::Escape && key_state == ElementState::Pressed {
                            elwt.exit();
                        } else {
                            state.handle_key(code, key_state);
                        }
                    }
                    WindowEvent::RedrawRequested => {
                        state.update();
                        match state.render() {
                            Ok(_) => {}
                            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                                state.resize(state.size)
                            }
                            Err(wgpu::SurfaceError::OutOfMemory) => elwt.exit(),
                            Err(e) => log::warn!("surface error: {e:?}"),
                        }
                        state.window.request_redraw();
                    }
                    _ => {}
                }
            }
        })
        .unwrap();
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
pub fn wasm_main() {
    wasm_bindgen_futures::spawn_local(run());
}
