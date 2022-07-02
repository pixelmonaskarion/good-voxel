use crate::block;
use crate::model;
use crate::block::WORLD_SIZE;
use crate::Vertex;
use crate::world::World;
use crate::Instance;

use noise::*;
use wgpu::util::DeviceExt;

/*static mut VERTICES: [Vertex; (WORLD_SIZE*WORLD_SIZE*WORLD_SIZE) as usize] = [Vertex::new(0.0, 0.0, 0.0, 0.5, 0.5, 0.5); (WORLD_SIZE*WORLD_SIZE*WORLD_SIZE) as usize];

pub fn init() {
    for x in 0..WORLD_SIZE {
        for y in 0..WORLD_SIZE {
            for z in 0..WORLD_SIZE {
                unsafe {
                    VERTICES[(x*WORLD_SIZE*WORLD_SIZE + y*WORLD_SIZE + z) as usize] = Vertex::new(x as f32, y as f32, z as f32, 0.5, 0.5, 0.5);
                }
            }
        }
    }
}*/

pub struct Chunk {
    pub blocks: Vec<block::Block>,
    pub model: Option<model::Model>,
    pub instance_buffer: Option<wgpu::Buffer>,
    pub pos: [i32; 3],
    pub has_any_blocks: bool,
}

impl Clone for Chunk {
    fn clone(&self) -> Self {
        Self {
            blocks: self.blocks.clone(),
            model: None,
            instance_buffer: None,
            pos: self.pos,
            has_any_blocks: self.has_any_blocks,
        }
    }
}

pub fn create_instance(pos: [i32; 3], device: &dyn DeviceExt) -> wgpu::Buffer {
    let instances = [Instance {
        position: cgmath::Vector3::new((pos[0]*WORLD_SIZE) as f32, (pos[1]*WORLD_SIZE) as f32, (pos[2]*WORLD_SIZE) as f32),
        rotation: cgmath::Quaternion::new(1.0, 0.0, 0.0, 0.0),
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

fn normalize_coords(pos: [f32; 3]) -> [f32; 3] {
    let mut newpos = pos.clone();
    newpos[0] = (pos[0]/pos[0].abs()+1.0)/2.0;
    newpos[1] = (pos[1]/pos[1].abs()+1.0)/2.0;
    newpos[2] = (pos[2]/pos[2].abs()+1.0)/2.0;
    newpos
}

impl Chunk {
    pub fn new(cx: i32, cy: i32, cz: i32, device: &dyn DeviceExt) -> Self {
        //println!("creating chunk {} {} {}", cx, cy, cz);
        let perlin = Perlin::new();
        let mut blocks: Vec<block::Block> = Vec::with_capacity((WORLD_SIZE*WORLD_SIZE*WORLD_SIZE) as usize);
        unsafe {
        blocks.set_len((WORLD_SIZE*WORLD_SIZE*WORLD_SIZE) as usize);
        }
        let mut has_any_blocks = false;
        for x in 0..WORLD_SIZE {
            for z in 0..WORLD_SIZE {
                let val = 0;//(perlin.get([(x+cx*WORLD_SIZE) as f64 /20.0, (z+cz*WORLD_SIZE) as f64 /20.0])*10.0) as i32;
                for y in 0..WORLD_SIZE {
                    //println!("creating block {} {} {}", x, y, z);
                    //[x as f32 / WORLD_SIZE as f32, y as f32 / WORLD_SIZE as f32, z as f32 / WORLD_SIZE as f32]
                    let solid = val >= y+cy*WORLD_SIZE;
                    if solid {
                        has_any_blocks = true;
                    }
                    let i = x*WORLD_SIZE*WORLD_SIZE + y*WORLD_SIZE + z;
                    blocks[i as usize] = block::Block::new(false, [x, y, z], [cx, cy, cz], normalize_coords([(x+cx*WORLD_SIZE) as f32, (y+cy*WORLD_SIZE) as f32, (z+cz*WORLD_SIZE) as f32]), 1.0, solid, device);
                }
            }
        }
        let model = Some(model::Model::new(&[], &[], device));
        let instance_buffer = Some(create_instance([cx, cy, cz], device));
        Self {
            blocks,
            model,
            instance_buffer,
            pos: [cx, cy, cz],
            has_any_blocks,
        }
    }

    pub fn create_mesh(&mut self, world: &World, device: &dyn DeviceExt) -> bool {
        if !self.has_any_blocks {
            return false;
        }
        //let time = SystemTime::now();
        let mut vertices = [Vertex::new(0.0,0.0,0.0,0.0,0.0,0.0); (8*WORLD_SIZE*WORLD_SIZE*WORLD_SIZE) as usize];
        //let mut vertices: Vec<Vertex> = Vec::new();
        let mut indices: Vec<u16> = Vec::new();
        for i in 0..WORLD_SIZE*WORLD_SIZE*WORLD_SIZE {
            //println!("meshing block {}", i);
            if self.blocks[i as usize].solid {
                let block_mesh = self.blocks[i as usize].get_mesh(world);
                
                for v in 0..8 {
                    let block = self.blocks[i as usize].clone();
                    let vertex = block::VERTICES[v];
                    let new_vertex = Vertex::new(vertex.position[0]+block.pos[0] as f32, vertex.position[1]+block.pos[1] as f32, vertex.position[2]+block.pos[2] as f32, block.color[0], block.color[1], block.color[2]);
                    //vertices.push(new_vertex);
                    vertices[(i*8) as usize + v] = new_vertex;
                }
                for b in 0..block_mesh.len() {
                    indices.push(block_mesh[b] + (i*8) as u16);
                }
            }
        }
        self.model = Some(model::Model::new(&vertices, &indices, device));
        return true;
        /*let millis = match time.elapsed() {
            Ok(dur) => dur.as_millis(),
            Err(_e) => panic!(),
        };
        println!("took {} seconds to mesh block", millis as f64/1000.0);*/
    }
}