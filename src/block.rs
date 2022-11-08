use crate::Instance;
use crate::Vertex;
use crate::model;
use cgmath::InnerSpace;
use wgpu::util::DeviceExt;
use rand::prelude::*;
use bitvec::prelude::*;
use noise::{NoiseFn, OpenSimplex, Perlin};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::panic;
use cgmath::{Vector3, MetricSpace};
use wasm_timer::SystemTime;


#[allow(unused_imports)]
use mockall::predicate::*;
#[allow(unused_imports)]
use mockall::*;

pub const VERTICES: &[Vertex] = &[
    Vertex {
        position: [0.0, 0.0, 0.0],
        //color: [0.1, 0.1, 0.1],
    },
    Vertex {
        position: [1.0, 0.0, 0.0],
        //color: [0.1, 0.1, 0.1],
    },
    Vertex {
        position: [1.0, 1.0, 0.0],
        //color: [0.1, 0.1, 0.1],
    },
    Vertex {
        position: [0.0, 1.0, 0.0],
        //color: [0.1, 0.1, 0.1],
    },
    Vertex {
        position: [0.0, 0.0, 1.0],
        //color: [0.1, 0.1, 0.1],
    },
    Vertex {
        position: [0.0, 1.0, 1.0],
        //color: [0.1, 0.1, 0.1],
    },
    Vertex {
        position: [1.0, 1.0, 1.0],
        //color: [0.1, 0.1, 0.1],
    },
    Vertex {
        position: [1.0, 0.0, 1.0],
        //color: [0.1, 0.1, 0.1],
    },
];
const INDICES: &[u16] = &[
    //bottom
    0, 2, 1, 0, 3, 2, //top
    7, 6, 4, 6, 5, 4, //north
    0, 4, 5, 0, 5, 3, //south
    2, 6, 1, 6, 7, 1, //west
    7, 4, 0, 1, 7, 0, //east
    3, 6, 2, 5, 6, 3,
];

pub struct Block {
    pub pos: [f32; 3],
    pub color: [f32; 3],
    pub model: Option<model::Model>,
}

/*pub const TEXTURE_SIZE: f32 = 256.0;
pub const BLOCK_SIZE: f32 = 16.0;*/

pub fn create_instance(pos: [f32; 3], device: &dyn DeviceExt) -> wgpu::Buffer {
    let instances = [Instance {
        position: cgmath::Vector3::new(pos[0], pos[1], pos[2]),
        //rotation: cgmath::Quaternion::new(1.0, 0.0, 0.0, 0.0),
        color: [0.5, 0.5, 0.5],
    }]
    .to_vec();
    let instance_data = instances.iter().map(Instance::to_raw).collect::<Vec<_>>();
    let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Instance Buffer"),
        contents: bytemuck::cast_slice(&instance_data),
        usage: wgpu::BufferUsages::VERTEX,
    });
    instance_buffer
}

pub struct Blocks {
    pub instance_buffer: wgpu::Buffer,
    pub num_instances: usize,
}

pub struct World {
    pub blocks: Vec<Blocks>,
    pub solid_blocks: BitVec,
    pub block_types: Vec<u8>,
    pub size: usize,
    pub instances: Vec<HashMap<[i32; 3], Instance>>,
}

pub fn world(size: usize, device: &dyn DeviceExt) -> World {
    let mut instances: Vec<HashMap<[i32; 3], Instance>> = std::iter::repeat(HashMap::new()).take(64).collect::<Vec<_>>();
    let mut solid_blocks: BitVec = bitvec![mut 1; 1];
    solid_blocks.resize(size.pow(3), true);
    let mut block_types: Vec<u8> = Vec::new(); 
    block_types.resize(size.pow(3), 0);
    //println!("MADE ARRAY");
    let time = SystemTime::now();
    let mut rng = rand::thread_rng();
    let seed = rng.gen::<u32>();
    let noise = OpenSimplex::new(seed);
    let cave_noise = Perlin::new(seed);
    for x in 0..size {
        for z in 0..size {
            let height = noise.get([x as f64 / 100.0, z as f64 / 100.0]);
            //println!("{}", height);
            for y in 0..size {
                //let solid = x == 0 && y == 0 && z == 0;
                let mut solid = (y as f32) < (height+1.0) as f32 *50.0;
                if (y as f32 + 10.0) < (height+1.0) as f32 *50.0 {
                    solid = cave_noise.get([x as f64 / 30.0, y as f64 / 30.0, z as f64 / 30.0]) < 0.5;
                    block_types[index(x, y, z, size)] = 2;
                } else {
                    block_types[index(x, y, z, size)]  = 1;
                }
                *solid_blocks.get_mut(index(x, y, z, size)).unwrap() = solid;
            }
        }
        println!("Generating {}%", ((x as f32 +1.0)/size as f32)*100.0);
    }
    for x in 0..size {
        for y in 0..size {
            for z in 0..size {
                let mut sides = [false; 6];
                let index = index(x, y, z, size);
                let block_type = block_types[index];
                let x = x as i32;
                let y = y as i32;
                let z = z as i32;
                if get_solid(&mut solid_blocks, x, y, z, size) {
                    sides = get_sides(&solid_blocks, x, y, z, size);
                }
                if sides != [false; 6] {
                    instances[sides_to_index(sides)].insert([x, y, z],Instance {
                        position: cgmath::Vector3::new(x as f32, y as f32, z as f32),
                        //rotation: cgmath::Quaternion::new(1.0, 0.0, 0.0, 0.0),
                        color: get_color_random(index.try_into().unwrap(), block_type, 0.5),
                    });
                }
            }
        }
        println!("Instancing {}%", ((x as f32 +1.0)/size as f32)*100.0);
    }
    println!("took {:?}", SystemTime::now().duration_since(time));
    let mut blockss = Vec::new();
    for instance in &instances {
        let instance_data = instance.values().map(Instance::to_raw).collect::<Vec<_>>();
        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Instance Buffer"),
            contents: bytemuck::cast_slice(&instance_data),
            usage: wgpu::BufferUsages::VERTEX,
        });
        blockss.push(Blocks {
            instance_buffer,
            num_instances: instance.len(),
        });
    }
    println!("fr tho it actually took {:?}", SystemTime::now().duration_since(time));
    World {
        blocks: blockss,
        solid_blocks,
        block_types,
        size,
        instances,
    }
}

pub fn get_sides(solid_blocks: &BitVec, x: i32, y: i32, z: i32, size: usize) -> [bool; 6] {
    [!get_solid(solid_blocks, x, y+1, z, size), !get_solid(solid_blocks, x, y-1, z, size), !get_solid(solid_blocks, x+1, y, z, size), !get_solid(solid_blocks, x-1, y, z, size), !get_solid(solid_blocks, x, y, z+1, size), !get_solid(solid_blocks, x, y, z-1, size)]
}

pub fn change_block(world: &mut World, x: i32, y: i32, z: i32, solid: bool, block_type: u8, device: &dyn DeviceExt) {
    if x < 0 || x >= world.size as i32 || y < 0 || y >= world.size as i32 || z < 0 || z >= world.size as i32 {
        return;
    }
    world.block_types[index(x as usize, y as usize, z as usize, world.size)] = block_type;
    let block_index = sides_to_index(get_sides(&world.solid_blocks, x, y, z, world.size));
    let mut update_sides = false;
    let original_solid = *world.solid_blocks.get(index(x as usize, y as usize, z as usize, world.size)).unwrap();
    if  original_solid == true && solid == false {
        world.instances[block_index].remove(&[x,y,z]);
        update_sides = true;
    }
    if original_solid == true && solid == true {
        world.instances[block_index].insert([x, y, z], Instance {
            position: cgmath::Vector3::new(x as f32, y as f32, z as f32),
            //rotation: cgmath::Quaternion::new(1.0, 0.0, 0.0, 0.0),
            color: get_color_random(index(x as usize, y as usize, z as usize, world.size).try_into().unwrap(), block_type, 0.5),
        });
    }
    if original_solid == false && solid == true {

        world.instances[block_index].insert([x, y, z], Instance {
            position: cgmath::Vector3::new(x as f32, y as f32, z as f32),
            //rotation: cgmath::Quaternion::new(1.0, 0.0, 0.0, 0.0),
            color: get_color_random(index(x as usize, y as usize, z as usize, world.size).try_into().unwrap(), block_type, 0.5),
        });
        update_sides = true;
    }
    if update_sides {
        for offset in [[1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]] {
            if *world.solid_blocks.get(index((x+offset[0]) as usize, (y+offset[1]) as usize, (z+offset[2]) as usize, world.size)).unwrap() == false {
                continue;
            }
            let side_index = sides_to_index(get_sides(&world.solid_blocks, x+offset[0], y+offset[1], z+offset[2], world.size));
            *world.solid_blocks.get_mut(index(x as usize, y as usize, z as usize, world.size)).unwrap() = solid;
            let new_sides = get_sides(&world.solid_blocks, x+offset[0], y+offset[1], z+offset[2], world.size);
            *world.solid_blocks.get_mut(index(x as usize, y as usize, z as usize, world.size)).unwrap() = original_solid;
            world.instances[side_index].remove(&[x+offset[0], y+offset[1], z+offset[2]]);
            world.instances[sides_to_index(new_sides)].insert([x+offset[0], y+offset[1], z+offset[2]], Instance {
                position: cgmath::Vector3::new((x+offset[0]) as f32, (y+offset[1]) as f32, (z+offset[2]) as f32),
                //rotation: cgmath::Quaternion::new(1.0, 0.0, 0.0, 0.0),
                color: get_color_random(index((x+offset[0]) as usize, (y+offset[1]) as usize, (z+offset[2]) as usize, world.size).try_into().unwrap(), world.block_types[index((x+offset[0]) as usize, (y+offset[1]) as usize, (z+offset[2]) as usize, world.size)], 0.5),
            });
        }
    }
    *world.solid_blocks.get_mut(index(x as usize, y as usize, z as usize, world.size)).unwrap() = solid;
    let mut blockss = Vec::new();
    for instance in &world.instances {
        let instance_data = instance.values().map(Instance::to_raw).collect::<Vec<_>>();
        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Instance Buffer"),
            contents: bytemuck::cast_slice(&instance_data),
            usage: wgpu::BufferUsages::VERTEX,
        });
        blockss.push(Blocks {
            instance_buffer,
            num_instances: instance.len(),
        });
    }
    world.blocks = blockss;
}

pub fn get_color_random(index: u64, x: u8, randomness: f32) -> [f32; 3] {
    let mut rng = rand::rngs::StdRng::seed_from_u64(index);
    let color_offset = (rng.next_u64() as f32 / u64::MAX as f32)*randomness;
    let mut color = get_color(x);
    color[0] *= color_offset;
    color[1] *= color_offset;
    color[2] *= color_offset;
    return color;
}

pub fn get_color(x: u8) -> [f32; 3] {
    if x == 1 {
        return [0.0,1.0,0.0];
    }
    if x == 2 {
        return [0.3,0.3,0.3];
    }
    return [0.0,0.0,0.0];
}

pub fn get_solid(bits: &BitVec, x: i32, y: i32, z: i32, size: usize) -> bool {
    if x < 0 || y < 0 || z < 0 || x >= size as i32 || y >= size as i32 || z >= size as i32 {
        return false;
    }
    let bit = bits.get(index(x as usize, y as usize, z as usize, size));
    if bit.is_some() {
        return *bit.unwrap();
    } else {
        return false;
    }
}

pub fn raycast(start_pos: Vector3<f32>, dir: Vector3<f32>, max_dist: f32, world: &World) -> (bool, [i32; 3], u8, Vector3<f32>) {
    let mut pos = start_pos;
    while pos.distance(start_pos) <= max_dist && pos.x >= 0.0 && pos.x < world.size as f32 && pos.y >= 0.0 && pos.y < world.size as f32 && pos.z >= 0.0 && pos.z < world.size as f32 {
        let solid_here = get_solid(&world.solid_blocks, pos.x as i32, pos.y as i32, pos.z as i32, world.size);
        if solid_here {
            let block_pos = [pos.x as i32, pos.y as i32, pos.z as i32];
            let block_here = world.block_types[index(pos.x as usize, pos.y as usize, pos.z as usize, world.size)];
            while [pos.x as i32, pos.y as i32, pos.z as i32] == block_pos {
                pos -= dir*0.0001;
            }
            return (true, block_pos, block_here, Vector3::new(pos.x - block_pos[0] as f32, pos.y - block_pos[1] as f32, pos.z - block_pos[2] as f32).normalize());
        }
        pos += dir;
    }
    return (false, [pos.x as i32, pos.y as i32, pos.z as i32], 0, Vector3::new(0.0,0.0,0.0));
}

pub fn index(x: usize, y: usize, z: usize, size: usize) -> usize {
    if x < size && y < size && z < size {
        return x*size*size + y * size + z;
    }
    return 0;
}

pub fn sides_to_index(sides: [bool; 6]) -> usize {
    let mut index = 0;
    for i in (0..6).rev() {
        index += sides[i] as usize * (2 as usize).pow(5-i as u32);

    }
    index
}

impl Block {
    pub fn new(
        pos: [f32; 3],
        color: [f32; 3],
    ) -> Self {
        Self {
            pos,
            color,
            model: None,
        }
    }

    pub fn get_mesh(&mut self, sides: [bool; 6]) -> Vec<u16> {
        let mut indices: Vec<u16> = Vec::new();
        for i in 0..6 {
            if sides[i] {
                indices.extend_from_slice(&INDICES[(i-1)*6..i*6]);
            }
        }
        //println!("created block mesh");
        indices
    }

    /*fn render<'b>(&'a self, render_pass: &'b mut wgpu::RenderPass<'a>) where 'b: 'a, {
        render_pass.set_vertex_buffer(0, self.model.vertex_buffer.slice(..));
        render_pass.set_index_buffer(self.model.index_buffer.slice(..), wgpu::IndexFormat::Uint16);

        // UPDATED!
        render_pass.draw_indexed(0..self.model.num_indices, 0, 0..1);
    }*/
}

pub fn create_mesh(sides: [bool; 6], device: &dyn DeviceExt) -> model::Model {
    let mut indices: Vec<u16> = Vec::new();
    for i in 0..6 {
        if sides[i] {
            indices.extend_from_slice(&INDICES[i*6..(i+1)*6]);
        }
    }
    //println!("created block mesh");
    model::Model::new(VERTICES, &indices, device)
}

pub fn number_to_bits(x: i32) -> [bool; 6] {
    let mut bits = [false; 6];
    let mut quotient = x;
    let mut i = 0;
    while quotient > 0 {
        let remainder = quotient%2;
        quotient = quotient/2;
        if remainder == 1 {
            bits[i] = true;
        }
        i += 1;
    }
    return bits;
}

pub fn create_all_meshes(device: &dyn DeviceExt) -> Vec<model::Model> {
    let mut meshes = Vec::new();
    for i in 0..64 {
        meshes.push(create_mesh(number_to_bits(i), device));
    }
    meshes
}

/*pub fn add_vertice(vertices: &mut Vec<Vertex>, vertex: Vertex) {
    let tex_mult = BLOCK_SIZE/TEXTURE_SIZE;
    let new_vertex = Vertex::new(vertex.position[0], vertex.position[1], vertex.position[2], vertex.tex_coords[0]*tex_mult, vertex.tex_coords[1]*tex_mult);
    vertices.push(new_vertex);
}*/