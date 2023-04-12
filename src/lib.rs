use std::{collections::HashMap, time::{Duration, SystemTimeError}};

use block::{RENDER_DIST, World, CHUNK_SIZE};
use cgmath::{Point3, Vector3};
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

use wgpu_text::{section::{Section, Text, Layout, HorizontalAlign}, TextBrush, font::FontRef};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
use wgpu::util::DeviceExt;
use winit::window::Window;
use wasm_timer::SystemTime;

mod camera;
mod texture;
mod model;
mod block;
mod collisions;
use camera::*;

pub const MAX_RUNNING_CHUNKS: i32 = 5;


#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
//right now this only stores the position, but it used to store other stuff
//so that's why I don't just use a array
pub struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
    texture_coords: [f32; 2],
}

impl Vertex {
    pub const fn new(x: f32, y: f32, z:f32, nx:f32, ny:f32, nz:f32, tx:f32, ty:f32) -> Self{
        Self {
            position: [x, y, z],
            normal: [nx, ny, nz],
            texture_coords: [tx, ty],
        }
    }

    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 6]>() as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x2,
                },
            ],
        }
    }
}

#[derive(Clone)]
pub struct Instance {
    position: cgmath::Vector3<f32>,
    color: [f32; 3],
    texture_offsets: [[f32; 2]; 6],
}

impl Instance {
    fn to_raw(&self) -> InstanceRaw {
        InstanceRaw {
            pos: self.position.into(),
            color: self.color,
            texture_offsets: self.texture_offsets,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct InstanceRaw {
    pos: [f32; 3],
    color: [f32; 3],
    texture_offsets: [[f32; 2]; 6],
}

impl InstanceRaw {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<InstanceRaw>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 5,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 6,
                    format: wgpu::VertexFormat::Float32x3,
                },
                //all texture offsets
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 6]>() as wgpu::BufferAddress,
                    shader_location: 7,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 8]>() as wgpu::BufferAddress,
                    shader_location: 8,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 10]>() as wgpu::BufferAddress,
                    shader_location: 9,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 12]>() as wgpu::BufferAddress,
                    shader_location: 10,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 14]>() as wgpu::BufferAddress,
                    shader_location: 11,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 16]>() as wgpu::BufferAddress,
                    shader_location: 12,
                    format: wgpu::VertexFormat::Float32x2,
                },
            ],
        }
    }
}

#[allow(dead_code)]
struct State {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    render_pipeline: wgpu::RenderPipeline,
    text_brush: TextBrush<FontRef<'static>>,
    texture_bind_group: wgpu::BindGroup,
    texture: texture::Texture,
    blocks: model::Model,
    generator: block::Generator,
    //world: block::World,
    worlds: HashMap<[i32; 3], block::World>,
    camera: Camera,
    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    camera_controller: CameraController,
    instances: Vec<Instance>,
    instance_buffer: wgpu::Buffer,
    depth_texture: texture::Texture,
    time: SystemTime,
    first_frame: bool,
    last_broken: i32,
}
//I'm gonna keep some of the comments from the tutorial in here because I like them
//tutorial comments will start with a *
impl State {
    // *Creating some of the wgpu types requires async code
    async fn new(window: &Window) -> Self {
        let size = window.inner_size();

        // *The instance is a handle to our GPU
        // *Backends::all => Vulkan + Metal + DX12 + Browser WebGPU
        let instance = wgpu::Instance::new(wgpu::Backends::all());
        let surface = unsafe { instance.create_surface(window) };
        let adapter = instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            },
        ).await.unwrap();
            let (device, queue) = adapter.request_device(
                &wgpu::DeviceDescriptor {
                    features: wgpu::Features::empty(),
                    // *WebGL doesn't support all of wgpu's features, so if
                    // *we're building for the web we'll have to disable some.
                    limits: if cfg!(target_arch = "wasm32") {
                        wgpu::Limits::downlevel_webgl2_defaults()
                    } else {
                        wgpu::Limits::default()
                    },
                    label: None,
                },
                None,
            ).await.unwrap();
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface.get_preferred_format(&adapter).unwrap(),
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Immediate,
        };
        surface.configure(&device, &config);
        let diffuse_bytes = include_bytes!("textures.png");
        let diffuse_texture =
            texture::Texture::from_bytes(&device, &queue, diffuse_bytes, "textures.png").unwrap();
        let texture_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    // This should match the filterable field of the
                    // corresponding Texture entry above.
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
            label: Some("texture_bind_group_layout"),
        });
        let diffuse_bind_group = device.create_bind_group(
            &wgpu::BindGroupDescriptor {
                layout: &texture_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&diffuse_texture.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&diffuse_texture.sampler),
                    }
                ],
                label: Some("diffuse_bind_group"),
            }
        );
        let mut camera = Camera {
            eye: (0.0, -1.0, 2.0).into(),
            target: (1.0, 0.0, 0.0).into(),
            up: cgmath::Vector3::unit_y(),
            aspect: config.width as f32 / config.height as f32,
            fovy: 45.0,
            znear: 0.1,
            zfar: 10000.0,
        };
        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update_view_proj(&camera);

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
                label: Some("camera_bind_group_layout"),
            });
        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
            label: Some("camera_bind_group"),
        });
        let depth_texture =
            texture::Texture::create_depth_texture(&device, &config, "depth_texture");

        let shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&texture_bind_group_layout, &camera_bind_group_layout],
                push_constant_ranges: &[],
            });
        let instances = [Instance { position: cgmath::Vector3::new(0.0, 0.0, 0.0), /*rotation: cgmath::Quaternion::new(1.0, 0.0, 0.0, 0.0), */color: [1.0,1.0,1.0], texture_offsets: [[0.0, 0.0]; 6]}].to_vec();
        let instance_data = instances.iter().map(Instance::to_raw).collect::<Vec<_>>();
        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Instance Buffer"),
            contents: bytemuck::cast_slice(&instance_data),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[Vertex::desc(), InstanceRaw::desc()],
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: texture::Texture::DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less, // 1.
                stencil: wgpu::StencilState::default(), // 2.
                bias: wgpu::DepthBiasState::default(),
            }),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent::REPLACE,
                        alpha: wgpu::BlendComponent::REPLACE,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                }],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                // Setting this to anything other than Fill requires Features::POLYGON_MODE_LINE
                // or Features::POLYGON_MODE_POINT
                polygon_mode: wgpu::PolygonMode::Fill,
                // Requires Features::DEPTH_CLIP_CONTROL
                unclipped_depth: false,
                // Requires Features::CONSERVATIVE_RASTERIZATION
                conservative: false,
            },
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            // If the pipeline will be used with a multiview render pass, this
            // indicates how many array layers the attachments will have.
            multiview: None,
        });
        let camera_controller = CameraController::new(0.006);
        let blocks = block::create_all_meshes(&device);
        let generator = block::Generator::new();
        let world = block::World::world(&generator, 0,0,0, &device, true);
        //let world2 = block::World::world(&generator, 0,world_size as i32 * -1,0,world_size, &device, false);
        let mut worlds: HashMap<[i32; 3], block::World> = HashMap::new();
        worlds.insert([0,0,0], world);
        let middle = (CHUNK_SIZE/ 2) as f32 + 0.5;
        camera.eye = Point3::new(middle, 75.0, middle);
        let time = SystemTime::now();
        let brush = wgpu_text::BrushBuilder::using_font_bytes(include_bytes!("Inconsolata-Regular.ttf")).unwrap()
        /* .initial_cache_size((1024, 1024))) */ // use this to avoid resizing cache texture
        /* .with_depth_testing(true) */ // enable/disable depth testing
            .build(&device, &config);
        Self {
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline,
            text_brush: brush,
            texture: diffuse_texture,
            texture_bind_group: diffuse_bind_group,
            blocks,
            generator,
            //world,
            worlds,
            camera,
            camera_uniform,
            camera_buffer,
            camera_bind_group,
            camera_controller,
            instances,
            instance_buffer,
            depth_texture,
            time,
            first_frame: true,
            last_broken: 0,
        }
    }

    fn get_running_chunks(&self, threshold: i32) -> i32 {
        let mut counter = 0;
        for world in self.worlds.values().into_iter() {
            if !world.finished {
                counter += 1;
            }
            if counter > threshold {
                return counter;
            }
        }
        return counter;
    }

    fn get_center_chunk(&self) -> [i32; 3] {
        let center_x = (self.camera.eye.x /CHUNK_SIZE as f32).floor() as i32;
        let center_y = (self.camera.eye.y /CHUNK_SIZE as f32).floor() as i32;
        let center_z = (self.camera.eye.z /CHUNK_SIZE as f32).floor() as i32;
        return [center_x, center_y, center_z];
        //return [0,0,0];
    }
    
    #[allow(unused_mut)]
    fn resize(&mut self, mut new_size: winit::dpi::PhysicalSize<u32>) {
        cfg_if::cfg_if! {
            if #[cfg(target_arch = "wasm32")] {
                new_size.width = new_size.width.min(2048);
                new_size.height = new_size.height.min(2048);
            }
        }
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.depth_texture = texture::Texture::create_depth_texture(&self.device, &self.config, "depth_texture");
            self.camera.aspect = self.config.width as f32 / self.config.height as f32;
            self.surface.configure(&self.device, &self.config);
            self.text_brush.resize_view(new_size.width as f32, new_size.height as f32, &self.queue);
        }
    }

    fn input(&mut self, window: &Window, event: &WindowEvent) -> bool {
        #[cfg(target_arch = "wasm32")] {
            match event {
                WindowEvent::MouseInput {
                    state,
                    button,
                    ..
                } => {
                    let is_pressed = *state == ElementState::Pressed;
                    match button {
                        MouseButton::Left => {
                            log::info!("It works!");
                        
                            use winit::platform::web::WindowExtWebSys;
                            web_sys::window()
                                .and_then(|win| win.document())
                                .and_then(|doc| {
                                    let canvas = doc.get_element_by_id("game_canvas")?;
                                    canvas.request_pointer_lock();
                                    Some(())
                                })
                                .expect("Couldn't append canvas to document body.");
                        },
                        _ => {},
                    }
                },
                _ => {}
            }
        }
        if self.camera_controller.process_events(event) {
            true
        } else {
            match event {
                WindowEvent::KeyboardInput {
                    input: KeyboardInput {
                        state,
                        virtual_keycode: Some(keycode),
                        ..
                    },
                    ..
                } => {
                    let is_pressed = *state == ElementState::Pressed;
                    match keycode {
                        VirtualKeyCode::F11 => {
                            if is_pressed {
                                window.set_fullscreen(None);
                            }
                            true
                        }
                        _ => false,
                    }
                }
                _ => false,
            }
        }
    }

    fn update(&mut self, delta_time: u128) {
        self.last_broken += 1;
        self.camera_controller.update_camera(&mut self.camera, &self.worlds, delta_time as f32);
        if self.camera_controller.left_mouse_pressed && self.last_broken >= 15 {
            let raycast_result = /*self.worlds.get(&[0,0,0]).unwrap().*/block::raycast(Vector3::new(self.camera.eye.x, self.camera.eye.y, self.camera.eye.z), self.camera_controller.get_forward_vec(), 1000.0, &self.worlds);
            if raycast_result.0 {
                let hit_world_pos = block::get_chunk_pos(raycast_result.1[0], raycast_result.1[1], raycast_result.1[2], block::CHUNK_SIZE);
                self.worlds.get_mut(&hit_world_pos.0).unwrap().change_block(hit_world_pos.1[0], hit_world_pos.1[1], hit_world_pos.1[2], false, 0, &self.device);
                self.last_broken = 0;
            }
        }
        /*if self.camera_controller.right_mouse_pressed {
            let raycast_result = self.worlds.get(&[0,0,0]).unwrap().raycast(Vector3::new(self.camera.eye.x, self.camera.eye.y, self.camera.eye.z), self.camera_controller.get_forward_vec(), 1000.0);
            if raycast_result.0 {
                self.worlds.get_mut(&[0,0,0]).unwrap().change_block((raycast_result.1[0] as f32 + raycast_result.3.x) as i32, (raycast_result.1[1] as f32 + raycast_result.3.y) as i32, (raycast_result.1[2] as f32 + raycast_result.3.z) as i32, true, 0, &self.device);
            }
        }*/
        self.camera_uniform.update_view_proj(&self.camera);
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );
        for world in &mut self.worlds {
            world.1.try_finish(&self.device);
        }
        /*'first: for x in -RENDER_DIST..RENDER_DIST+1 {
            for y in -RENDER_DIST..RENDER_DIST+1 {
                for z in -RENDER_DIST..RENDER_DIST+1 {
                    if !self.worlds.contains_key(&[x, y, z]) {
                        if self.get_running_chunks(5) < 5 {
                            self.worlds.insert([x, y, z], World::world(&self.generator, x, y, z, &self.device, false));
                        } else {
                            break 'first;
                        }
                    }
                }
            }
        }*/
        let mut fill: Vec<i32> = Vec::new();
        let fill_size = RENDER_DIST*2+1;
        fill.resize(fill_size.pow(3) as usize, 0);
        //should set the center to 1
        fill[(RENDER_DIST*fill_size*fill_size + RENDER_DIST*fill_size + RENDER_DIST) as usize] = 1;
        let mut changed = true;
        let center = self.get_center_chunk();
        if self.get_running_chunks(MAX_RUNNING_CHUNKS) < MAX_RUNNING_CHUNKS  && false {
            'outer: while changed {
                changed = false;
                let mut new_fill = fill.clone();
                for x in 0..fill_size {
                    for y in 0..fill_size {
                        for z in 0..fill_size {
                            if fill[(x*fill_size*fill_size + y*fill_size + z) as usize] == 1 {
                                for offset in [[1,0,0], [-1,0,0], [0,-1,0], [0,1,0], [0,0,-1], [0,0,1]] {
                                    if x+offset[0] > -1 && x+offset[0] < fill_size && y+offset[1] > -1 && y+offset[1] < fill_size && z+offset[2] > -1 && z+offset[2] < fill_size {
                                        if fill[((x+offset[0])*fill_size*fill_size + (y+offset[1])*fill_size + z+offset[2]) as usize] == 0 {
                                            new_fill[((x+offset[0])*fill_size*fill_size + (y+offset[1])*fill_size + z+offset[2]) as usize] = 1;
                                            let real_chunk_pos = [x-RENDER_DIST+center[0], y-RENDER_DIST+center[1], z-RENDER_DIST+center[2]];
                                            if !self.worlds.contains_key(&real_chunk_pos) {
                                                //println!("{:?}", real_chunk_pos);
                                                self.worlds.insert(real_chunk_pos, World::world(&self.generator, real_chunk_pos[0]*CHUNK_SIZE as i32, real_chunk_pos[1]*CHUNK_SIZE as i32, real_chunk_pos[2]*CHUNK_SIZE as i32, &self.device, false));
                                                break 'outer;
                                            }
                                        }
                                    }
                                }
                                new_fill[(x*fill_size*fill_size + y*fill_size + z) as usize] = 2;
                                changed = true;
                            }
                        }
                    }
                }
                fill = new_fill;
            }
        }
        let mut to_remove = Vec::new();
        for pos in self.worlds.keys().into_iter() {
            if (((pos[0]-center[0]).pow(2) + (pos[1]-center[1]).pow(2) + (pos[2]-center[2]).pow(2)) as f32).sqrt() as i32 >= RENDER_DIST as i32 +2 {
                to_remove.push(pos.clone());
            }
        }
        for pos in to_remove {
            self.worlds.remove(&pos).unwrap().destroy();
        }
        //self.world.load_around([self.camera.eye.x, self.camera.eye.y, self.camera.eye.z], self.first_frame, &self.device);
        self.first_frame = false;
    }

    fn render(&mut self, delta_time: u128, update_time: Result<Duration, SystemTimeError>) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
                        store: true,
                    },
                }],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: true,
                    }),
                    stencil_ops: None,
                }),
            });

            render_pass.set_pipeline(&self.render_pipeline);
            //render_pass.set_bind_group(0, &self.diffuse_bind_group, &[]); // NEW!
            render_pass.set_bind_group(0, &self.texture_bind_group, &[]);
            render_pass.set_bind_group(1, &self.camera_bind_group, &[]);
            //let blocks = self.world.blocks.as_ref();
            //let blocks2 = self.world2.blocks.as_ref();
            render_pass.set_vertex_buffer(0, self.blocks.vertex_buffer.slice(..));
            render_pass.set_index_buffer(self.blocks.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            let mut indices_drawn = 0;
            for i in 0..64 {
                let indices_to_draw = block::number_to_bits(i).iter().filter(|&n| *n == true).count() as u32 * 6;
                for world in self.worlds.values().into_iter() {
                    if world.blocks.is_some() {
                        render_pass.set_vertex_buffer(1, world.blocks.as_ref().unwrap()[i as usize].instance_buffer.slice(..));
                        render_pass.draw_indexed(indices_drawn..indices_drawn+indices_to_draw, 0, 0..world.blocks.as_ref().unwrap()[i as usize].num_instances as u32);
                    }
                }
                indices_drawn += indices_to_draw;
            }
        }
        let running_chunks_text = format!("{} Running Chunks\nChunk: {:?}\nReal Position {:?}\nVelocity {:?}\nDelta Time {:?}\nUpdate Time {:?}", self.get_running_chunks(MAX_RUNNING_CHUNKS), self.get_center_chunk(), self.camera.eye, self.camera_controller.velocity, delta_time, update_time);
        let mut color = [0.0, 0.0, 0.0, 1.0];
        if delta_time <= 18 {
            color = [0.0, 1.0, 0.0, 1.0];
        } else if delta_time >= 25 {
            color = [1.0, 0.0, 0.0, 1.0];
        }
        let section = Section::default()
            .add_text(Text::new(running_chunks_text.as_str()).with_scale(50.0).with_color(color))
            .with_layout(Layout::default().h_align(HorizontalAlign::Left));
        self.text_brush.queue(&section);

        let text_buffer = self.text_brush.draw(&self.device, &view, &self.queue);

        self.queue.submit([encoder.finish(), text_buffer]);

        // submit will accept anything that implements IntoIter
        output.present();
        Ok(())
    }
}

#[cfg_attr(target_arch="wasm32", wasm_bindgen(start))]
pub async fn run() {
    cfg_if::cfg_if! {
        if #[cfg(target_arch = "wasm32")] {
            std::panic::set_hook(Box::new(console_error_panic_hook::hook));
            console_log::init_with_level(log::Level::Warn).expect("Couldn't initialize logger");
        } else {
            env_logger::init();
        }
    }

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().build(&event_loop).unwrap();
    #[cfg(target_arch = "wasm32")]
    {
        // Winit prevents sizing with CSS, so we have to set
        // the size manually when on web.
        use winit::dpi::PhysicalSize;
        
        use winit::platform::web::WindowExtWebSys;
        web_sys::window()
            .and_then(|win| win.document())
            .and_then(|doc| {
                let dst = doc.get_element_by_id("wasm-example")?;
                let canvas = web_sys::Element::from(window.canvas());
                canvas.set_id("game_canvas");
                dst.append_child(&canvas).ok()?;
                Some(())
            })
            .expect("Couldn't append canvas to document body.");
        let width = match web_sys::window().unwrap().inner_width() {
            Ok(w) => w.as_f64().unwrap() as i32,
            Err(_e) => 450,
        };
        let height = match web_sys::window().unwrap().inner_height() {
            Ok(h) => h.as_f64().unwrap() as i32,
            Err(_e) => 400,
        };
        window.set_inner_size(PhysicalSize::new(width.min(2048), height.min(2048)));
    }   
    // let moniter = window.primary_monitor().unwrap();
    // window.set_outer_position(moniter.position());
    // window.set_inner_size(moniter.size());
    cfg_if::cfg_if! {
        if #[cfg(target_arch = "wasm32")] {

        } else {
            window.set_fullscreen(Some(winit::window::Fullscreen::Borderless(None)));
        }
    }

    let mut state = State::new(&window).await;
    let _cp = window.set_cursor_position(winit::dpi::PhysicalPosition::new(state.config.width/2, state.config.height/2));
    //window.set_cursor_visible(false);

    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::DeviceEvent {
                event: DeviceEvent::MouseMotion{ delta, },
                .. // We're not using device_id currently
            } => if true {
                state.camera_controller.process_mouse(delta.0 as f32, delta.1 as f32);
                let _cp = window.set_cursor_position(winit::dpi::PhysicalPosition::new(state.config.width/2, state.config.height/2));
            }
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == window.id() => {
                if !state.input(&window, event) {
                    // UPDATED!
                    match event {
                        WindowEvent::CloseRequested
                        | WindowEvent::KeyboardInput {
                            input:
                                KeyboardInput {
                                    state: ElementState::Pressed,
                                    virtual_keycode: Some(VirtualKeyCode::Escape),
                                    ..
                                },
                            ..
                        } => {
                            cfg_if::cfg_if! {
                                if #[cfg(target_arch = "wasm32")] {
                                    #[cfg(target_arch = "wasm32")]
                                    {
                                        log::info!("It works!");
                                        
                                        use winit::platform::web::WindowExtWebSys;
                                        web_sys::window()
                                            .and_then(|win| win.document())
                                            .and_then(|doc| {
                                                let canvas = doc.get_element_by_id("game_canvas")?;
                                                canvas.request_pointer_lock();
                                                Some(())
                                            })
                                            .expect("Couldn't append canvas to document body.");
                                    }
                                } else {
                                    *control_flow = ControlFlow::Exit;
                                }
                            }
                        },
                        WindowEvent::Resized(physical_size) => {
                            state.resize(*physical_size);
                        }
                        WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                            // new_inner_size is &&mut so w have to dereference it twice
                            state.resize(**new_inner_size);
                        }
                        _ => {}
                    }
                }
            }
            Event::RedrawRequested(window_id) if window_id == window.id() => {
                //let time = SystemTime::now();
                let delta_time = match state.time.elapsed() {
                    Ok(dur) => dur.as_millis(),
                    Err(_e) => 15,
                };
                state.time = SystemTime::now();
                state.update(delta_time);
                match state.render(delta_time, SystemTime::now().duration_since(state.time)) {
                    Ok(_) => {}
                    // Reconfigure the surface if it's lost or outdated
                    Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                        state.resize(state.size)
                    }
                    // The system is out of memory, we should probably quit
                    Err(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
                    Err(wgpu::SurfaceError::Timeout) => log::warn!("Surface timeout"),
                }
                //println!("took {:?} to render", SystemTime::now().duration_since(time));
            }
            Event::RedrawEventsCleared => {
                // RedrawRequested will only trigger once, unless we manually
                // request it.
                window.request_redraw();
            }
            _ => {}
        }
    });
}


#[test]
fn test_side_indices() {
    assert_eq!(block::sides_to_index([false; 6]), 0);
    assert_eq!(block::sides_to_index([true; 6]), 63);
    assert_eq!(block::number_to_bits(0), [false; 6]);

    assert_eq!(block::get_chunk_pos(0, 0, 0, 100), ([0,0,0], [0,0,0]));
    assert_eq!(block::get_chunk_pos(50, 0, 0, 100), ([0,0,0], [50,0,0]));
    assert_eq!(block::get_chunk_pos(-50, 0, 0, 100), ([-1,0,0], [49,0,0]));
    assert_eq!(block::get_chunk_pos(-150, 75, 0, 100), ([-2,0,0], [49,75,0]));
}