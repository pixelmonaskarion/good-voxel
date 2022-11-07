use cgmath::{Point3, Vector3};
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

// use cgmath::prelude::*;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
use wgpu::util::DeviceExt;
use winit::window::Window;
use std::time::SystemTime;

mod camera;
mod texture;
mod model;
mod block;
//mod chunk;
//mod world;
use camera::*;

// lib.rs
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    position: [f32; 3],
    //color: [f32; 3],
}

impl Vertex {
    pub const fn new(x: f32, y: f32, z:f32, r: f32, g: f32, b: f32) -> Self{
        Self {
            position: [x, y, z],
            //color: [r, g, b],
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
                /*wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3, // NEW!
                },*/
            ],
        }
    }
}

/*const VERTICES: &[Vertex] = &[
    Vertex { position: [-0.0868241, 0.49240386, 0.0], tex_coords: [0.4131759, 0.00759614], }, // A
    Vertex { position: [-0.49513406, 0.06958647, 0.0], tex_coords: [0.0048659444, 0.43041354], }, // B
    Vertex { position: [-0.21918549, -0.44939706, 0.0], tex_coords: [0.28081453, 0.949397], }, // C
    Vertex { position: [0.35966998, -0.3473291, 0.0], tex_coords: [0.85967, 0.84732914], }, // D
    Vertex { position: [0.44147372, 0.2347359, 0.0], tex_coords: [0.9414737, 0.2652641], }, // E
];

const INDICES: &[u16] = &[0, 1, 4, 1, 2, 4, 2, 3, 4];*/

#[derive(Clone)]
pub struct Instance {
    position: cgmath::Vector3<f32>,
    //rotation: cgmath::Quaternion<f32>,
    color: [f32; 3],
}

impl Instance {
    fn to_raw(&self) -> InstanceRaw {
        InstanceRaw {
            /*model: (cgmath::Matrix4::from_translation(self.position)
                * cgmath::Matrix4::from(self.rotation))
            .into(),*/
            pos: self.position.into(),
            color: self.color,
        }
    }
}

// const NUM_INSTANCES_PER_ROW: u32 = 10;
// const INSTANCE_DISPLACEMENT: cgmath::Vector3<f32> = cgmath::Vector3::new(
//     NUM_INSTANCES_PER_ROW as f32 * 0.5,
//     0.0,
//     NUM_INSTANCES_PER_ROW as f32 * 0.5,
// );

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct InstanceRaw {
    //model: [[f32; 4]; 4],
    pos: [f32; 3],
    color: [f32; 3],
}

impl InstanceRaw {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<InstanceRaw>() as wgpu::BufferAddress,
            // We need to switch from using a step mode of Vertex to Instance
            // This means that our shaders will only change to use the next
            // instance when the shader starts processing a new instance
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    // While our vertex shader only uses locations 0, and 1 now, in later tutorials we'll
                    // be using 2, 3, and 4, for Vertex. We'll start at slot 5 not conflict with them later
                    shader_location: 5,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // A mat4 takes up 4 vertex slots as it is technically 4 vec4s. We need to define a slot
                // for each vec4. We'll have to reassemble the mat4 in
                // the shader.
                /*wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                    shader_location: 6,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 8]>() as wgpu::BufferAddress,
                    shader_location: 7,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 12]>() as wgpu::BufferAddress,
                    shader_location: 8,
                    format: wgpu::VertexFormat::Float32x4,
                },*/
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 6,
                    format: wgpu::VertexFormat::Float32x3,
                }
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
    //world: world::World,
    blocks: Vec<model::Model>,
    block_ind: block::World,
    diffuse_bind_group: wgpu::BindGroup,
    diffuse_texture: texture::Texture,
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
}

impl State {
    // Creating some of the wgpu types requires async code
    async fn new(window: &Window) -> Self {
        let size = window.inner_size();

        // The instance is a handle to our GPU
        // Backends::all => Vulkan + Metal + DX12 + Browser WebGPU
        let instance = wgpu::Instance::new(wgpu::Backends::all());
        let surface = unsafe { instance.create_surface(window) };
        let adapter = instance
            .enumerate_adapters(wgpu::Backends::all())
            .filter(|adapter| {
                // Check if this adapter supports our surface
                surface.get_preferred_format(&adapter).is_some()
            })
            .next()
            .unwrap();
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    features: wgpu::Features::POLYGON_MODE_LINE,
                    // WebGL doesn't support all of wgpu's features, so if
                    // we're building for the web we'll have to disable some.
                    limits: if cfg!(target_arch = "wasm32") {
                        wgpu::Limits::downlevel_webgl2_defaults()
                    } else {
                        wgpu::Limits::default()
                    },
                    label: None,
                },
                None, // Trace path
            )
            .await
            .unwrap();
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface.get_preferred_format(&adapter).unwrap(),
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
        };
        surface.configure(&device, &config);
        let diffuse_bytes = include_bytes!("textures.png"); // CHANGED!
        let diffuse_texture =
            texture::Texture::from_bytes(&device, &queue, diffuse_bytes, "textures.png").unwrap(); // CHANGED!

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
        let diffuse_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&diffuse_texture.view), // CHANGED!
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&diffuse_texture.sampler), // CHANGED!
                },
            ],
            label: Some("diffuse_bind_group"),
        });
        let mut camera = Camera {
            // position the camera one unit up and 2 units back
            // +z is out of the screen
            eye: (0.0, -1.0, 2.0).into(),
            // have it look at the origin
            target: (1.0, 0.0, 0.0).into(),
            // which way is "up"
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
        let instances = [Instance { position: cgmath::Vector3::new(0.0, 0.0, 0.0), /*rotation: cgmath::Quaternion::new(1.0, 0.0, 0.0, 0.0), */color: [1.0,1.0,1.0]}].to_vec();
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
        /*let mut blocks: Vec<block::Block> = Vec::new();
        let perlin = Perlin::new();
        for x in 0..WORLD_SIZE {
            for y in 0..WORLD_SIZE {
                for z in 0..WORLD_SIZE {
                    let val = perlin.get([x as f64 /10.0, y as f64 /10.0, z as f64 /10.0]);
                    blocks.push(block::Block::new([x, y, z], val > 0.0, &device));
                }
            }
        }
        for i in 0..WORLD_SIZE*WORLD_SIZE*WORLD_SIZE {
            let mut block = block::Block::new(blocks[i as usize].pos, blocks[i as usize].solid, &device);
            block.set_mesh(&blocks, &device);
            blocks[i as usize] = block;
        }*/
        //let mut world = world::World::new(2, 2, 2, &device);
        //world.lock().unwrap().test();
        //world.create_mesh(&device);
        let camera_controller = CameraController::new(0.006);
        let blocks = block::create_all_meshes(&device);
        let world_size = 100;
        let block_ind = block::world(world_size, &device);
        let middle = (world_size/ 2) as f32 + 0.5;
        camera.eye = Point3::new(middle, 75.0, middle);
        let time = SystemTime::now();
        Self {
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline,
            //world,
            blocks,
            block_ind,
            diffuse_bind_group,
            diffuse_texture,
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
        }
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.depth_texture = texture::Texture::create_depth_texture(&self.device, &self.config, "depth_texture");
            self.camera.aspect = self.config.width as f32 / self.config.height as f32;
            self.surface.configure(&self.device, &self.config);
        }
    }

    fn input(&mut self, window: &Window, event: &WindowEvent) -> bool {
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

    fn update(&mut self) {
        let mut delta_time = match self.time.elapsed() {
            Ok(dur) => dur.as_millis(),
            Err(_e) => panic!(),
        };
        self.time = SystemTime::now();
        delta_time = match self.time.elapsed() {
            Ok(dur) => delta_time-dur.as_millis(),
            Err(_e) => panic!(),
        };
        self.camera_controller.update_camera(&mut self.camera, &self.block_ind, delta_time as f32);
        if self.camera_controller.left_mouse_pressed {
            let raycast_result = block::raycast(Vector3::new(self.camera.eye.x, self.camera.eye.y, self.camera.eye.z), self.camera_controller.get_forward_vec(), 1000.0, &self.block_ind);
            if raycast_result.0 {
                block::change_block(&mut self.block_ind, raycast_result.1[0], raycast_result.1[1], raycast_result.1[2], false, 0, &self.device);
            }
        }
        if self.camera_controller.right_mouse_pressed {
            let raycast_result = block::raycast(Vector3::new(self.camera.eye.x, self.camera.eye.y, self.camera.eye.z), self.camera_controller.get_forward_vec(), 1000.0, &self.block_ind);
            if raycast_result.0 {
                block::change_block(&mut self.block_ind, (raycast_result.1[0] as f32 + raycast_result.3.x) as i32, (raycast_result.1[1] as f32 + raycast_result.3.y) as i32, (raycast_result.1[2] as f32 + raycast_result.3.z) as i32, true, 0, &self.device);
            }
        }
        self.camera_uniform.update_view_proj(&self.camera);
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );

        //self.world.load_around([self.camera.eye.x, self.camera.eye.y, self.camera.eye.z], self.first_frame, &self.device);
        self.first_frame = false;
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
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
            render_pass.set_bind_group(0, &self.diffuse_bind_group, &[]); // NEW!
            render_pass.set_bind_group(1, &self.camera_bind_group, &[]);

            for i in 0..64 {
                //println!("buffer {} has length {}", i, self.block_ind[i].num_instances);
                render_pass.set_vertex_buffer(0, self.blocks[i].vertex_buffer.slice(..));
                render_pass.set_vertex_buffer(1, self.block_ind.blocks[i].instance_buffer.slice(..));
                render_pass.set_index_buffer(self.blocks[i].index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                render_pass.draw_indexed(0..self.blocks[i].num_indices, 0, 0..self.block_ind.blocks[i].num_instances as u32);
            }
            /*for i in 0..self.world.loaded_chunks.len() {
                //println!("drawing chunk {}", i);
                if self.world.loaded_chunks[i].model.is_some() {
                    let model = self.world.loaded_chunks[i].model.as_ref().unwrap();
                    render_pass.set_vertex_buffer(0, model.vertex_buffer.slice(..));
                    render_pass.set_vertex_buffer(1, self.world.loaded_chunks[i].instance_buffer.as_ref().unwrap().slice(..));
                    render_pass.set_index_buffer(model.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
    
                    render_pass.draw_indexed(0..model.num_indices, 0, 0..1);
                }
            }*/
            //self.block.render(&mut render_pass);
            /*
            FROM DRAWING INDIVIDUAL BLOCKS
            for x in 0..WORLD_SIZE {
                for y in 0..WORLD_SIZE {
                    for z in 0..WORLD_SIZE {
                        let i = ((x * WORLD_SIZE * WORLD_SIZE) + (y*WORLD_SIZE) + z) as usize;
                        render_pass.set_vertex_buffer(0, self.blocks[i].model.vertex_buffer.slice(..));
                        render_pass.set_vertex_buffer(1, self.blocks[i].instance_buffer.slice(..));
                        //render_pass.set_index_buffer(self.blocks[i].model.index_buffer.slice(..), wgpu::IndexFormat::Uint16);

                        //render_pass.draw_indexed(0..self.blocks[i].model.num_indices, 0, 0..1);
                        render_pass.draw(0..self.blocks[i].model.num_vertices, 0..1);
                    }
                }
            }*/
            /*render_pass.set_vertex_buffer(0, self.model.vertex_buffer.slice(..));
            render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
            render_pass.set_index_buffer(self.model.index_buffer.slice(..), wgpu::IndexFormat::Uint16);

            // UPDATED!
            render_pass.draw_indexed(0..self.model.num_indices, 0, 0..self.instances.len() as _);*/
        }
        // submit will accept anything that implements IntoIter
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        Ok(())
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen(start))]
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
        window.set_inner_size(PhysicalSize::new(450, 400));
        use winit::platform::web::WindowExtWebSys;
        web_sys::window()
            .and_then(|win| win.document())
            .and_then(|doc| {
                let dst = doc.get_element_by_id("wasm-example")?;
                let canvas = web_sys::Element::from(window.canvas());
                dst.append_child(&canvas).ok()?;
                Some(())
            })
            .expect("Couldn't append canvas to document body.");
    }
    // let moniter = window.primary_monitor().unwrap();
    // window.set_outer_position(moniter.position());
    // window.set_inner_size(moniter.size());
    window.set_fullscreen(Some(winit::window::Fullscreen::Borderless(None)));

    let mut state = State::new(&window).await;
    let _cp = window.set_cursor_position(winit::dpi::PhysicalPosition::new(state.config.width/2, state.config.height/2));
    window.set_cursor_visible(false);

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
                        } => *control_flow = ControlFlow::Exit,
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
                state.update();
                match state.render() {
                    Ok(_) => {}
                    // Reconfigure the surface if it's lost or outdated
                    Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                        state.resize(state.size)
                    }
                    // The system is out of memory, we should probably quit
                    Err(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
                    Err(wgpu::SurfaceError::Timeout) => log::warn!("Surface timeout"),
                }
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
    use std::mem;
    assert_eq!(block::sides_to_index([false; 6]), 0);
    assert_eq!(block::sides_to_index([true; 6]), 63);
    assert_eq!(block::number_to_bits(0), [false; 6]);
    //println!("SIZE OF VERTEX: {}", mem::size_of::<Vertex>());
    assert_eq!(mem::size_of::<Instance>(), 1);
}