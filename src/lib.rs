use cgmath::{Point3, Vector3};
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
use wgpu::util::DeviceExt;
use winit::window::Window;
use wasm_timer::SystemTime;

mod camera;
mod texture;
mod model;
mod block;
use camera::*;


#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
//right now this only stores the position, but it used to store other stuff
//so that's why I don't just use a array
pub struct Vertex {
    position: [f32; 3],
}

impl Vertex {
    pub const fn new(x: f32, y: f32, z:f32) -> Self{
        Self {
            position: [x, y, z],
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
                //old code for adding something else to the vertex
                /*wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3, // NEW!
                },*/
            ],
        }
    }
}

#[derive(Clone)]
pub struct Instance {
    position: cgmath::Vector3<f32>,
    color: [f32; 3],
}

impl Instance {
    fn to_raw(&self) -> InstanceRaw {
        InstanceRaw {
            pos: self.position.into(),
            color: self.color,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct InstanceRaw {
    pos: [f32; 3],
    color: [f32; 3],
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
    blocks: Vec<model::Model>,
    block_ind: block::World,
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
            present_mode: wgpu::PresentMode::Fifo,
        };
        surface.configure(&device, &config);
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
                bind_group_layouts: &[/*&texture_bind_group_layout, */&camera_bind_group_layout],
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
        let block_ind = block::World::world(world_size, &device);
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
            //diffuse_bind_group,
            //diffuse_texture,
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
    
    //I don't know how to remove this warning
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

    fn update(&mut self) {
        let mut delta_time = match self.time.elapsed() {
            Ok(dur) => dur.as_millis(),
            Err(_e) => 15,
        };
        self.time = SystemTime::now();
        delta_time = match self.time.elapsed() {
            Ok(dur) => delta_time-dur.as_millis(),
            Err(_e) => 15,
        };
        self.camera_controller.update_camera(&mut self.camera, &self.block_ind, delta_time as f32);
        if self.camera_controller.left_mouse_pressed {
            let raycast_result = self.block_ind.raycast(Vector3::new(self.camera.eye.x, self.camera.eye.y, self.camera.eye.z), self.camera_controller.get_forward_vec(), 1000.0);
            if raycast_result.0 {
                self.block_ind.change_block(raycast_result.1[0], raycast_result.1[1], raycast_result.1[2], false, 0, &self.device);
            }
        }
        if self.camera_controller.right_mouse_pressed {
            let raycast_result = self.block_ind.raycast(Vector3::new(self.camera.eye.x, self.camera.eye.y, self.camera.eye.z), self.camera_controller.get_forward_vec(), 1000.0);
            if raycast_result.0 {
                self.block_ind.change_block((raycast_result.1[0] as f32 + raycast_result.3.x) as i32, (raycast_result.1[1] as f32 + raycast_result.3.y) as i32, (raycast_result.1[2] as f32 + raycast_result.3.z) as i32, true, 0, &self.device);
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
            //render_pass.set_bind_group(0, &self.diffuse_bind_group, &[]); // NEW!
            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);

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