use std::collections::HashMap;

use cgmath::{Vector3, Point3};
use winit::{
    event::*,
};

use crate::block::{self, CHUNK_SIZE};
use crate::collisions::{colliding_box, Box};

pub const GM: i32 = 1;

#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
);

pub struct Camera {
        pub eye: cgmath::Point3<f32>,
        pub target: cgmath::Point3<f32>,
        pub up: cgmath::Vector3<f32>,
        pub aspect: f32,
        pub fovy: f32,
        pub znear: f32,
        pub zfar: f32,
}

impl Camera {
    fn build_view_projection_matrix(&self) -> cgmath::Matrix4<f32> {
        // 1.
        let view = cgmath::Matrix4::look_at_rh(self.eye, self.target, self.up);
        // 2.
        let proj = cgmath::perspective(cgmath::Deg(self.fovy), self.aspect, self.znear, self.zfar);

        // 3.
        return OPENGL_TO_WGPU_MATRIX * proj * view;
    }
}

// We need this for Rust to store our data correctly for the shaders
#[repr(C)]
// This is so we can store this in a buffer
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    // We can't use cgmath with bytemuck directly so we'll have
    // to convert the Matrix4 into a 4x4 f32 array
    view_proj: [[f32; 4]; 4],
}

impl CameraUniform {
    pub fn new() -> Self {
        use cgmath::SquareMatrix;
        Self {
            view_proj: cgmath::Matrix4::identity().into(),
        }
    }

    pub fn update_view_proj(&mut self, camera: &Camera) {
        self.view_proj = camera.build_view_projection_matrix().into();
    }
}

pub struct CameraController {
    speed: f32,
    pub velocity: Vector3<f32>,
    in_air: u32,
    ground: f32,
    sky: f32,
    w_pressed: bool,
    s_pressed: bool,
    a_pressed: bool,
    d_pressed: bool,
    up_pressed: bool,
    down_pressed: bool,
    right_pressed: bool,
    left_pressed: bool,
    space_pressed: bool,
    pub shift_pressed: bool,
    pub left_mouse_pressed: bool,
    pub right_mouse_pressed: bool,
}

impl CameraController {
    pub fn new(mut speed: f32) -> Self {
        if GM == 1 {
            speed = speed*5.0;
        }
        Self {
            speed,
            velocity: Vector3::new(0.0,0.0,0.0),
            in_air: 0,
            ground: 0.0,
            sky: 0.0,
            w_pressed: false,
            s_pressed: false,
            a_pressed: false,
            d_pressed: false,
            up_pressed: false,
            down_pressed: false,
            right_pressed: false,
            left_pressed: false,
            space_pressed: false,
            shift_pressed: false,
            left_mouse_pressed: false,
            right_mouse_pressed: false,
        }
    }

    pub fn process_events(&mut self, event: &WindowEvent) -> bool {
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
                    VirtualKeyCode::W => {
                        self.w_pressed = is_pressed;
                        true
                    }
                    VirtualKeyCode::A => {
                        self.a_pressed = is_pressed;
                        true
                    }
                    VirtualKeyCode::S => {
                        self.s_pressed = is_pressed;
                        true
                    }
                    VirtualKeyCode::D => {
                        self.d_pressed = is_pressed;
                        true
                    }
                    VirtualKeyCode::Up => {
                        self.up_pressed = is_pressed;
                        true
                    }
                    VirtualKeyCode::Left => {
                        self.left_pressed = is_pressed;
                        true
                    }
                    VirtualKeyCode::Down => {
                        self.down_pressed = is_pressed;
                        true
                    }
                    VirtualKeyCode::Right => {
                        self.right_pressed = is_pressed;
                        true
                    }
                    VirtualKeyCode::Space => {
                        self.space_pressed = is_pressed;
                        true
                    }
                    VirtualKeyCode::LShift => {
                        self.shift_pressed = is_pressed;
                        true
                    }
                    _ => false,
                }
            },
            WindowEvent::MouseInput {
                state,
                button,
                ..
            } => {
                let is_pressed = *state == ElementState::Pressed;
                match button {
                    MouseButton::Left => {
                        self.left_mouse_pressed = is_pressed;
                        true
                    },
                    MouseButton::Right => {
                        self.right_mouse_pressed = is_pressed;
                        true
                    }
                    _ => false,
                }
            }
            _ => false,
        }
    }

    pub fn process_mouse(&mut self, x: f32, y: f32) {
        self.ground += x/500.0;
        self.sky -= y/500.0;
    }

    pub fn update_camera(&mut self, camera: &mut Camera, worlds: &HashMap<[i32; 3], block::World>, delta_time: f32) {
        if self.right_pressed {
            self.ground += 10.0*self.speed*delta_time * std::f32::consts::PI/180.0;
        }
        if self.left_pressed {
            self.ground -= 10.0*self.speed*delta_time * std::f32::consts::PI/180.0;
        }
        if self.up_pressed {
            println!("{}", self.sky);
            self.sky += 10.0*self.speed*delta_time * std::f32::consts::PI/180.0;
        }
        if self.down_pressed {
            println!("{}", self.sky);
            self.sky -= 10.0*self.speed*delta_time * std::f32::consts::PI/180.0;
        }
        self.sky = self.sky.clamp(std::f32::consts::PI*-0.499, std::f32::consts::PI*0.499);

        let forward = self.get_forward_vec();
        //let forward_norm = forward.normalize();
        //let forward_mag = forward.magnitude();
        if self.w_pressed {
            //move_camera(cgmath::Vector3::new(self.ground.cos(), 0.0, self.ground.sin()) * self.speed*delta_time, camera, worlds);
            self.velocity.x += self.ground.cos();
            self.velocity.z += self.ground.sin();
        }
        if self.s_pressed {
            //move_camera(cgmath::Vector3::new(self.ground.cos(), 0.0, self.ground.sin()) * self.speed*delta_time*-1.0, camera, worlds);
            self.velocity.x -= self.ground.cos();
            self.velocity.z -= self.ground.sin();
        }
        let right = cgmath::Vector3::new(self.ground.cos(), 0.0, self.ground.sin()).cross(camera.up);
        if self.a_pressed {
            //move_camera(right * self.speed*delta_time*-1.0, camera, worlds);
            let move_vec = right *-1.0;
            self.velocity.x += move_vec.x;
            self.velocity.z += move_vec.z;
        }
        if self.d_pressed {
            //move_camera(right * self.speed*delta_time, camera, worlds);
            self.velocity.x += right.x;
            self.velocity.z += right.z;
        }
        let dist = (self.velocity.x.powf(2.0)+self.velocity.z.powf(2.0)).sqrt();
        if dist != 0.0 {
            move_camera(Vector3 { x: self.velocity.x/dist*self.speed*delta_time, y: 0.0, z: self.velocity.z/dist*self.speed*delta_time}, camera, worlds);
        }
        self.velocity.x = 0.0;
        self.velocity.z = 0.0;
        if self.space_pressed {
            if GM == 0 {
                if self.in_air < 6 {
                    self.velocity.y = 0.2;
                }
            } else {
                move_camera(cgmath::Vector3::unit_y() * self.speed*delta_time, camera, worlds);
            }
        }
        if self.shift_pressed {
            if GM == 1 {
                move_camera(cgmath::Vector3::unit_y() * self.speed*delta_time * -1.0, camera, worlds);
            }
        }

        if GM == 0 {
            self.velocity.y -= 0.001 * delta_time;
        }
        self.in_air += 1;
        //println!("{}", delta_time);
        if move_camera(Vector3 { x: 0.0, y: self.velocity.y, z: 0.0 }, camera, worlds) {
            self.velocity.y = 0.0;
            self.in_air = 0;
        }
        if GM == 1 {
            self.velocity.y = 0.0;
        }

        camera.target = camera.eye + forward;

        // Redo radius calc in case the fowrard/backward is pressed.
        /*let forward = camera.target - camera.eye;
        let forward_mag = forward.magnitude();

        if self.right_pressed {
            // Rescale the distance between the target and eye so 
            // that it doesn't change. The eye therefore still 
            // lies on the circle made by the target and eye.
            camera.eye = camera.target - (forward + right * self.speed).normalize() * forward_mag;
        }
        if self.left_pressed {
            camera.eye = camera.target - (forward - right * self.speed).normalize() * forward_mag;
        }*/
    }
    
    pub fn get_forward_vec(&self) -> Vector3<f32> {
        cgmath::Vector3::new(self.ground.cos()*self.sky.cos(), self.sky.sin(), self.ground.sin()*self.sky.cos())
    }
    
}
fn move_camera(vec: Vector3<f32>, camera: &mut Camera, worlds: &HashMap<[i32; 3], block::World>) -> bool {
    //return move_camera_x(vec.x, camera, worlds) || move_camera_y(vec.y, camera, worlds) || move_camera_z(vec.z, camera, worlds);
    camera.eye += vec;

    //COLLISIONS

    // let self_box = Box {pos: (camera.eye+Vector3::new(0.0, -0.4, 0.0)).into(), size: [0.8, 1.9, 0.8]};
    // for (pos, solid) in get_in_block(camera.eye, worlds) {
    //     //println!("{:?} {:?}", pos, solid);
    //     if solid {
    //         let block_box = Box {pos: [pos[0] as f32 + 0.5, pos[1] as f32 + 0.5, pos[2] as f32 + 0.5], size: [1.0, 1.0, 1.0]};
    //         let result = colliding_box(&self_box, &block_box);
    //         println!("{:?}", result);
    //         if result.0 {
    //             camera.eye -= Vector3::from(result.1);
    //             return true;
    //         }
    //     }
    // }
    
    return false;
}

/*fn move_camera_x(x: f32, camera: &mut Camera, worlds: &HashMap<[i32; 3], block::World>) -> bool {
    let in_at_start = get_in_block(camera.eye, worlds);
    camera.eye.x += x;
    let mut collided = false;
    while get_in_block(camera.eye, worlds) && !in_at_start {
        camera.eye.x -= x*0.01;
        collided = true;
    }
    return collided;
}

fn move_camera_y(y: f32, camera: &mut Camera, worlds: &HashMap<[i32; 3], block::World>) -> bool {
    let in_at_start = get_in_block(camera.eye, worlds);
    camera.eye.y += y;
    let mut collided = false;
    while get_in_block(camera.eye, worlds) && !in_at_start {
        camera.eye.y -= y*0.01;
        collided = true;
    }
    return collided;
}

fn move_camera_z(z: f32, camera: &mut Camera, worlds: &HashMap<[i32; 3], block::World>) -> bool {
    let in_at_start = get_in_block(camera.eye, worlds);
    camera.eye.z += z;
    let mut collided = false;
    while get_in_block(camera.eye, worlds) && !in_at_start {
        camera.eye.z -= z*0.01;
        collided = true;
    }
    return collided;
}*/

fn get_in_block(pos: Point3<f32>, worlds: &HashMap<[i32; 3], block::World>) -> HashMap<[i32; 3], bool> {
    if GM == 1 {
        //return HashMap::new();
    }
    let mut positions: HashMap<[i32; 3], bool> = HashMap::new();
    for offset in [Vector3::new(0.4 as f32, 0.4, 0.4), Vector3::new(0.4 as f32, 0.4, -0.4), Vector3::new(-0.4 as f32, 0.4, 0.4), Vector3::new(-0.4 as f32, 0.4, -0.4), Vector3::new(0.4 as f32, -1.5, 0.4), Vector3::new(0.4 as f32, -1.5, -0.4), Vector3::new(-0.4 as f32, -1.5, 0.4), Vector3::new(-0.4 as f32, -1.5, -0.4)] {
        let world_pos = block::get_chunk_pos((pos+offset).x as i32, (pos+offset).y as i32, (pos+offset).z as i32, CHUNK_SIZE);
        let world = worlds.get(&world_pos.0);
        if world.is_none() {
            positions.insert([(pos+offset).x as i32, (pos+offset).y as i32, (pos+offset).z as i32], true);
        } else {
            let world = world.unwrap();
            if world.solid_blocks.is_some() {
                let solid_blocks = world.solid_blocks.as_ref().unwrap();
                let block = solid_blocks.get(block::index(world_pos.1[0] as usize, world_pos.1[1] as usize, world_pos.1[2] as usize));
                if block.is_some() {
                    positions.insert([(pos+offset).x as i32, (pos+offset).y as i32, (pos+offset).z as i32], *block.unwrap());
                }
            }
        }
    }
    return positions;
}