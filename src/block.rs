use crate::model;
use crate::world::World;
use crate::Instance;
use crate::Vertex;
use rand::prelude::*;
use wgpu::util::DeviceExt;

#[allow(unused_imports)]
use mockall::predicate::*;
#[allow(unused_imports)]
use mockall::*;

pub const VERTICES: &[Vertex] = &[
    Vertex {
        position: [0.0, 0.0, 0.0],
        color: [0.1, 0.1, 0.1],
    },
    Vertex {
        position: [1.0, 0.0, 0.0],
        color: [0.1, 0.1, 0.1],
    },
    Vertex {
        position: [1.0, 1.0, 0.0],
        color: [0.1, 0.1, 0.1],
    },
    Vertex {
        position: [0.0, 1.0, 0.0],
        color: [0.1, 0.1, 0.1],
    },
    Vertex {
        position: [0.0, 0.0, 1.0],
        color: [0.1, 0.1, 0.1],
    },
    Vertex {
        position: [0.0, 1.0, 1.0],
        color: [0.1, 0.1, 0.1],
    },
    Vertex {
        position: [1.0, 1.0, 1.0],
        color: [0.1, 0.1, 0.1],
    },
    Vertex {
        position: [1.0, 0.0, 1.0],
        color: [0.1, 0.1, 0.1],
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

pub const WORLD_SIZE: i32 = 16;

pub struct Block {
    pub solid: bool,
    pub pos: [i32; 3],
    pub color_mul: f32,
    pub chunk_pos: [i32; 3],
    pub color: [f32; 3],
    pub model: Option<model::Model>,
    pub instance_buffer: Option<wgpu::Buffer>,
}

impl Clone for Block {
    fn clone(&self) -> Self {
        Self {
            solid: self.solid,
            pos: self.pos,
            color_mul: self.color_mul,
            chunk_pos: self.chunk_pos,
            color: self.color,
            model: None,
            instance_buffer: None,
        }
    }
}

/*pub const TEXTURE_SIZE: f32 = 256.0;
pub const BLOCK_SIZE: f32 = 16.0;*/

fn create_instance(pos: [i32; 3], device: &dyn DeviceExt) -> wgpu::Buffer {
    let instances = [Instance {
        position: cgmath::Vector3::new(pos[0] as f32, pos[1] as f32, pos[2] as f32),
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

impl Block {
    pub fn new(
        buffers: bool,
        pos: [i32; 3],
        chunk_pos: [i32; 3],
        color_in: [f32; 3],
        mul_mul: f32,
        solid: bool,
        device: &dyn DeviceExt,
    ) -> Self {
        let mut rng = rand::thread_rng();
        let color_mul: f32 = rng.gen::<f32>().sqrt() * mul_mul;
        let color = [
            color_in[0] - color_mul,
            color_in[1] - color_mul,
            color_in[2] - color_mul,
        ];
        if buffers {
            let model = Some(model::Model::new(VERTICES, INDICES, device));
            let instance_buffer = Some(create_instance(pos, device));
            Block {
                solid,
                pos,
                color_mul,
                chunk_pos,
                color,
                model,
                instance_buffer,
            }
        } else {
            Block {
                solid,
                pos,
                color_mul,
                chunk_pos,
                color,
                model: None,
                instance_buffer: None,
            }
        }
    }

    pub fn get_mesh(&mut self, world: &World) -> Vec<u16> {
        let mut indices: Vec<u16> = Vec::new();
        //println!("creating block mesh");
        let pos = [
            self.pos[0] + self.chunk_pos[0] * WORLD_SIZE,
            self.pos[1] + self.chunk_pos[1] * WORLD_SIZE,
            self.pos[2] + self.chunk_pos[2] * WORLD_SIZE,
        ];
        if self.solid {
            if !world.get_block(pos[0], pos[1], pos[2] - 1).is_none() {
                if !world.get_block(pos[0], pos[1], pos[2] - 1).unwrap().solid {
                    indices.extend_from_slice(&INDICES[0..6]);
                }
            }
            if !world.get_block(pos[0], pos[1], pos[2] + 1).is_none() {
                if !world.get_block(pos[0], pos[1], pos[2] + 1).unwrap().solid {
                    indices.extend_from_slice(&INDICES[6..12]);
                }
            }
            if !world.get_block(pos[0] - 1, pos[1], pos[2]).is_none() {
                if !world.get_block(pos[0] - 1, pos[1], pos[2]).unwrap().solid {
                    indices.extend_from_slice(&INDICES[12..18]);
                }
            }
            if !world.get_block(pos[0] + 1, pos[1], pos[2]).is_none() {
                if !world.get_block(pos[0] + 1, pos[1], pos[2]).unwrap().solid {
                    indices.extend_from_slice(&INDICES[18..24]);
                }
            }
            if !world.get_block(pos[0], pos[1] - 1, pos[2]).is_none() {
                if !world.get_block(pos[0], pos[1] - 1, pos[2]).unwrap().solid {
                    indices.extend_from_slice(&INDICES[24..30]);
                }
            }
            if !world.get_block(pos[0], pos[1] + 1, pos[2]).is_none() {
                if !world.get_block(pos[0], pos[1] + 1, pos[2]).unwrap().solid {
                    indices.extend_from_slice(&INDICES[30..36]);
                }
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

/*pub fn add_vertice(vertices: &mut Vec<Vertex>, vertex: Vertex) {
    let tex_mult = BLOCK_SIZE/TEXTURE_SIZE;
    let new_vertex = Vertex::new(vertex.position[0], vertex.position[1], vertex.position[2], vertex.tex_coords[0]*tex_mult, vertex.tex_coords[1]*tex_mult);
    vertices.push(new_vertex);
}*/

#[cfg(test)]
mod tests {
    use super::*;
    use wgpu::util::*;

    mock! {
        MyDeviceExt <T: Clone + 'static> {
        }
        impl<T: Clone + 'static> DeviceExt for MyDeviceExt<T> {
            fn create_buffer_init<'a>(&self, desc: &BufferInitDescriptor<'a>) -> wgpu::Buffer;

            fn create_texture_with_data<'a>(
                &self,
                queue: &wgpu::Queue,
                desc: &wgpu::TextureDescriptor<'a>,
                data: &[u8],
            ) -> wgpu::Texture;
        }
    }

    mock! {
         World {
            fn get_block(&self, x: i32, y: i32, z: i32);
        }
    }

    #[test]
    fn test_first_chunk() {
        //let mut device_context = MockMyDeviceExt::new();
        //let world = World::new(1, 1, 1, &device_context);
        //let block = world.get_block(1, 1, 1);
    }
}
