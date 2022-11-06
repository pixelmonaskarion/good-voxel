use crate::Instance;
use crate::Vertex;
use crate::model;
use wgpu::util::DeviceExt;
use rand::prelude::*;
use bitvec::prelude::*;
use noise::{NoiseFn, OpenSimplex, Perlin};
use std::time::SystemTime;

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
}

pub fn world(size: usize, device: &dyn DeviceExt) -> World {
    let mut instances: Vec<Vec<Instance>> = std::iter::repeat(vec![]).take(64).collect::<Vec<_>>();
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
                    sides = [!get_solid(&mut solid_blocks, x, y+1, z, size), !get_solid(&mut solid_blocks, x, y-1, z, size), !get_solid(&mut solid_blocks, x+1, y, z, size), !get_solid(&mut solid_blocks, x-1, y, z, size), !get_solid(&mut solid_blocks, x, y, z+1, size), !get_solid(&mut solid_blocks, x, y, z-1, size)];
                }
                if sides != [false; 6] {
                    instances[sides_to_index(sides)].push(Instance {
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
    for instance in instances {
        let instance_data = instance.iter().map(Instance::to_raw).collect::<Vec<_>>();
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
        blocks: blockss
    }
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

pub fn get_solid(bits: &mut BitVec, x: i32, y: i32, z: i32, size: usize) -> bool {
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

pub fn index(x: usize, y: usize, z: usize, size: usize) -> usize {
    x*size*size + y * size + z
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