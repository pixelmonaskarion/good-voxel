use crate::model::Model;
use crate::Instance;
use crate::Vertex;
use crate::model;
use cgmath::InnerSpace;
use noise::NoiseFn;
use noise::Perlin;
use wasm_timer::SystemTime;
use wgpu::util::DeviceExt;
use rand::prelude::*;
use bitvec::prelude::*;
use wgpu::Buffer;
use std::collections::HashMap;
use cgmath::{Vector3, MetricSpace};

#[allow(unused_imports)]
use mockall::predicate::*;
#[allow(unused_imports)]
use mockall::*;

pub const BLOCK_SIZE: f32 = 1.0;
pub const CHUNK_SIZE: usize = 32;
pub const RENDER_DIST: i32 = 10;
pub const DIRECTIONS: [[i32; 3]; 6] = [[1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]];

//holds the basic model of a cube, all sides shown
pub const VERTICES: &[Vertex] = &[
    //+y
    Vertex {position: [0.0, 1.0, 0.0], normal: [0.0,1.0,0.0], texture_coords: [1.0, 1.0]},
    Vertex {position: [1.0, 1.0, 1.0], normal: [0.0,1.0,0.0], texture_coords: [0.0, 0.0]},
    Vertex {position: [1.0, 1.0, 0.0], normal: [0.0,1.0,0.0], texture_coords: [0.0, 1.0]},

    Vertex {position: [0.0, 1.0, 0.0], normal: [0.0,1.0,0.0], texture_coords: [1.0, 1.0]},
    Vertex {position: [0.0, 1.0, 1.0], normal: [0.0,1.0,0.0], texture_coords: [1.0, 0.0]},
    Vertex {position: [1.0, 1.0, 1.0], normal: [0.0,1.0,0.0], texture_coords: [0.0, 0.0]},

    //-y
    Vertex {position: [1.0, 0.0, 1.0], normal: [0.0,-1.0,0.0], texture_coords: [0.0, 1.0]},
    Vertex {position: [0.0, 0.0, 0.0], normal: [0.0,-1.0,0.0], texture_coords: [1.0, 0.0]},
    Vertex {position: [1.0, 0.0, 0.0], normal: [0.0,-1.0,0.0], texture_coords: [1.0, 1.0]},

    Vertex {position: [0.0, 0.0, 1.0], normal: [0.0,-1.0,0.0], texture_coords: [0.0, 0.0]},
    Vertex {position: [0.0, 0.0, 0.0], normal: [0.0,-1.0,0.0], texture_coords: [1.0, 0.0]},
    Vertex {position: [1.0, 0.0, 1.0], normal: [0.0,-1.0,0.0], texture_coords: [0.0, 1.0]},

    //+x
    Vertex {position: [1.0, 0.0, 0.0], normal: [1.0,0.0,0.0], texture_coords: [1.0, 1.0]},
    Vertex {position: [1.0, 1.0, 1.0], normal: [1.0,0.0,0.0], texture_coords: [0.0, 0.0]},
    Vertex {position: [1.0, 0.0, 1.0], normal: [1.0,0.0,0.0], texture_coords: [0.0, 1.0]},

    Vertex {position: [1.0, 0.0, 0.0], normal: [1.0,0.0,0.0], texture_coords: [1.0, 1.0]},
    Vertex {position: [1.0, 1.0, 0.0], normal: [1.0,0.0,0.0], texture_coords: [1.0, 0.0]},
    Vertex {position: [1.0, 1.0, 1.0], normal: [1.0,0.0,0.0], texture_coords: [0.0, 0.0]},

    //-x
    Vertex {position: [0.0, 0.0, 1.0], normal: [-1.0,0.0,0.0], texture_coords: [1.0, 1.0]},
    Vertex {position: [0.0, 1.0, 0.0], normal: [-1.0,0.0,0.0], texture_coords: [0.0, 0.0]},
    Vertex {position: [0.0, 0.0, 0.0], normal: [-1.0,0.0,0.0], texture_coords: [0.0, 1.0]},

    Vertex {position: [0.0, 0.0, 1.0], normal: [-1.0,0.0,0.0], texture_coords: [1.0, 1.0]},
    Vertex {position: [0.0, 1.0, 1.0], normal: [-1.0,0.0,0.0], texture_coords: [1.0, 0.0]},
    Vertex {position: [0.0, 1.0, 0.0], normal: [-1.0,0.0,0.0], texture_coords: [0.0, 0.0]},

    //+z
    Vertex {position: [1.0, 0.0, 1.0], normal: [0.0,0.0,1.0], texture_coords: [1.0, 1.0]},
    Vertex {position: [1.0, 1.0, 1.0], normal: [0.0,0.0,1.0], texture_coords: [1.0, 0.0]},
    Vertex {position: [0.0, 0.0, 1.0], normal: [0.0,0.0,1.0], texture_coords: [0.0, 1.0]},

    Vertex {position: [1.0, 1.0, 1.0], normal: [0.0,0.0,1.0], texture_coords: [1.0, 0.0]},
    Vertex {position: [0.0, 1.0, 1.0], normal: [0.0,0.0,1.0], texture_coords: [0.0, 0.0]},
    Vertex {position: [0.0, 0.0, 1.0], normal: [0.0,0.0,1.0], texture_coords: [0.0, 1.0]},

    //-z
    Vertex {position: [0.0, 0.0, 0.0], normal: [0.0,0.0,-1.0], texture_coords: [1.0, 1.0]},
    Vertex {position: [1.0, 1.0, 0.0], normal: [0.0,0.0,-1.0], texture_coords: [0.0, 0.0]},
    Vertex {position: [1.0, 0.0, 0.0], normal: [0.0,0.0,-1.0], texture_coords: [0.0, 1.0]},

    Vertex {position: [0.0, 0.0, 0.0], normal: [0.0,0.0,-1.0], texture_coords: [1.0, 1.0]},
    Vertex {position: [0.0, 1.0, 0.0], normal: [0.0,0.0,-1.0], texture_coords: [1.0, 0.0]},
    Vertex {position: [1.0, 1.0, 0.0], normal: [0.0,0.0,-1.0], texture_coords: [0.0, 0.0]},
];
const INDICES: &[u16] = &[
    0, 1, 2, 3, 4, 5,//bottom
    6, 7, 8, 9, 10, 11,//top
    12, 13, 14, 15, 16, 17,//north
    18, 19, 20, 21, 22, 23,//south
    24, 25, 26, 27, 28, 29,//west
    30, 31, 32, 33, 34, 35,//east
];

pub fn get_vertices(size: f32) -> [Vertex; VERTICES.len()] {
    let mut new_vertices = [Vertex {position:[0.0,0.0,0.0], normal:[0.0,0.0,0.0], texture_coords: [0.0,0.0]}; VERTICES.len()];
    for (i, v) in VERTICES.iter().enumerate() {
        new_vertices[i] = Vertex{position:[v.position[0]*size, v.position[1]*size, v.position[2]*size], normal: v.normal, texture_coords: v.texture_coords};
    }
    return new_vertices;
}

/*holds world information,
    blocks: all 64 Blocks objects
    solid_blocks: a bunch of bits storing whether or not the block at that index is solid
    block_types: a list of bytes storing a number corresponding to the color of the block in the get_color function
    size: width/height/depth of the world
    instances: for holding each individual position for blocks, explained in change_block funcion
*/
#[derive(Debug)]
pub struct Chunk {
    // pub blocks: Option<Vec<Blocks>>,
    pub model: Option<Model>,
    pub instance_buffer: Option<Buffer>,
    pub solid_blocks: Option<BitVec>,
    pub block_types: Option<Vec<u8>>,
    // pub receiver: Receiver<(Vec<HashMap<[i32; 3], Instance>>, BitVec<usize, Lsb0>, Vec<u8>)>,
    pub pos: [i32; 3],
    pub chunk_pos: [i32; 3],
    pub finished: bool,
    pub neighbor_remesh: [bool; 6],
    pub any_air: bool,
}

#[derive(Clone)]
pub struct Generator {
    pub surface_noise: Perlin,
    pub second_surface_noise: Perlin,
    pub cave_noise: Perlin,
    pub seed: u32,
    pub second_seed: u32,
}

impl Generator {
    pub fn new() -> Self {
        let mut rng = rand::thread_rng();
        let seed = rng.gen::<u32>();
        let second_seed = rng.gen::<u32>();
        let surface_noise = Perlin::new(seed);
        let second_surface_noise = Perlin::new(second_seed);
        let cave_noise = Perlin::new(seed);
        Self {
            surface_noise,
            second_surface_noise,
            cave_noise,
            seed,
            second_seed,
        }
    }

    pub fn height_at(&self, x: f64, z: f64) -> f64 {
        let height = (self.surface_noise.get([x, z])*20.0+self.second_surface_noise.get([x/30.0, z/30.0])*40.0)*(self.surface_noise.get([x/100.0, z/100.0]).abs() * 50.0) + 30.0;
        return height.floor();
    }
}

impl Chunk {
    //creates a new world with the selected size
    // generates the blocks
    // creates instances for them
    // sends them to the gpu and gets buffers
    pub fn new(generator: Generator, chunks: &mut HashMap<[i32; 3], Chunk>, wx: i32, wy: i32, wz: i32, chunk_pos: [i32; 3], device: &dyn DeviceExt, _synchronous: bool) -> Self {
        let mut time = SystemTime::now();
        //initializes variables
        let mut solid_blocks: BitVec = bitvec![mut 1; 1];
        solid_blocks.resize(CHUNK_SIZE.pow(3), true);
        let mut block_types: Vec<u8> = Vec::new(); 
        block_types.resize(CHUNK_SIZE.pow(3), 0);
        let mut rng = rand::thread_rng();
        // println!("generation started");
        //loops through all blocks and generates them
        for x in 0..CHUNK_SIZE {
            for z in 0..CHUNK_SIZE {
                let exact_x = x as i32 + wx;
                let exact_z = z as i32 + wz;
                //the height is only determined by x and z so it is created here and the loop is is in this weird order
                let height = generator.height_at(exact_x as f64 / 100.0, exact_z as f64 / 100.0);
                for y in 0..CHUNK_SIZE {
                    let exact_y = y as i32 + wy;
                    //ground or sky?
                    #[allow(unused_mut)]
                    let mut solid = (exact_y as f32) < (height+1.0) as f32 && generator.cave_noise.get([exact_x as f64/ 30.0, exact_y as f64/ 30.0, exact_z as f64/ 30.0]) < 0.4;
                    //is it ten blocks underneath the surface?
                    if (exact_y as f32 + 5.0) < (height+1.0) as f32 {
                        //sets the block type to be stone block type
                        block_types[index(x, y, z)] = 2;
                        let ore_random = generator.second_surface_noise.get([exact_x as f64/ 30.0, exact_y as f64/ 30.0, exact_z as f64/ 30.0]).abs();
                        if ore_random > 0.4 && ore_random < 0.5 {
                            if (rng.next_u32() as f32 / u32::MAX as f32) < 0.1 {
                                block_types[index(x, y, z)] = 4;
                            }
                        }
                    } else if (exact_y as f32 + 1.0) < (height+1.0) as f32 {
                        //solid if in a cave
                        //solid = 
                        //sets the block type to be stone block type
                        block_types[index(x, y, z)] = 3;
                    } else {
                        //sets the block type to be grass block type
                        block_types[index(x, y, z)]  = 1;
                    }
                    //sets the bit at the index of the position to be solid's value
                    *solid_blocks.get_mut(index(x, y, z)).unwrap() = solid;
                }
            }
            // println!("Generating {}%", ((x as f32 +1.0)/CHUNK_SIZE as f32)*100.0);
        }
        // for _ in 0..50 {
        //     let mut pos = [rng.next_u32() as usize %CHUNK_SIZE, rng.next_u32() as usize %CHUNK_SIZE, rng.next_u32() as usize %CHUNK_SIZE];
        //     let mut vien_size = 0;
        //     while (rng.next_u32() as f32 / u32::MAX as f32) < 0.2 || vien_size < 5 && vien_size <= 10 {
        //         if block_types[index(pos[0], pos[1], pos[2])] == 2 && solid_blocks.get(index(pos[0], pos[1], pos[2])).unwrap() == true {
        //             block_types[index(pos[0], pos[1], pos[2])] = 5;
        //             vien_size += 1;
        //         }
        //         let dir = DIRECTIONS[rng.next_u32() as usize %6];
        //         for d in 0..3 {
        //             if dir[d] > 0 {
        //                 if pos[d] != CHUNK_SIZE-1 {
        //                     pos[d] += dir[d] as usize;
        //                 }
        //             }
        //             if dir[d] < 0 {
        //                 if pos[d] != 0 {
        //                     pos[d] -= dir[d].abs() as usize;
        //                 }
        //             }
        //         }
        //     }
        // }
        //let (tx, rx) = channel();
        //loops over every block and instances it
        // for x in 0..CHUNK_SIZE {
        //     for y in 0..CHUNK_SIZE {
        //         for z in 0..CHUNK_SIZE {
        //             //init sides
        //             let mut sides = [false; 6];
        //             let index = index(x, y, z);
        //             let block_type = block_types[index];
        //             //casts position to i32 now for easier readability, index takes usize and get_solid/get_sides take i32
        //             let x = x as i32;
        //             let y = y as i32;
        //             let z = z as i32;
        //             let exact_x = x + wx;
        //             let exact_z = z + wz;
        //             let exact_y = y + wy;
        //             if get_solid(&mut solid_blocks, x, y, z) {
        //                 sides = get_sides(&solid_blocks, x, y, z);
        //             }
        //             if sides != [false; 6] {
        //                 instances[sides_to_index(sides)].insert([x, y, z],Instance {
        //                     position: get_position(exact_x as f32, exact_y as f32, exact_z as f32, BLOCK_SIZE),
        //                     color: [1.0,1.0,1.0],//get_color_random(exact_x, exact_y, exact_z, block_type),
        //                     texture_offsets: get_texture_offsets(block_type),
        //                 });
        //             }
        //         }
        //     }
        //     //println!("Instancing {}%", ((x as f32 +1.0)/size as f32)*100.0);
        // }
        // println!("Generating took {:?}", SystemTime::now().duration_since(time));
        time = SystemTime::now();
        // let mut blockss = Vec::new();
        // //sends each vec of instances to the gpu and puts the buffer into a vec
        // for instance in instances.iter() {
        //     let instance_data = instance.values().map(Instance::to_raw).collect::<Vec<_>>();
        //     let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        //         label: Some("Instance Buffer"),
        //         contents: bytemuck::cast_slice(&instance_data),
        //         usage: wgpu::BufferUsages::VERTEX,
        //     });
        //     blockss.push(Blocks {
        //         instance_buffer,
        //         num_instances: instance.len(),
        //     });
        // }
        let mut vertices = Vec::new();
        // let mut vertices_map = HashMap::new();
        let mut indices = Vec::new();
        let mut neighbor_remesh = [false; 6];
        let mut any_air = false;
        for x in 0..CHUNK_SIZE {
            for y in 0..CHUNK_SIZE {
                for z in 0..CHUNK_SIZE {
                    let solid = get_solid(chunk_pos, &mut solid_blocks, chunks, x as i32, y as i32, z as i32);
                    if solid.0 {
                        let sides_neighbors = get_sides(chunk_pos, &solid_blocks, chunks, x as i32, y as i32, z as i32);
                        let sides: [bool; 6] = sides_neighbors.iter().enumerate().map(|(i, s)| {
                            if s.1 {
                                neighbor_remesh[i] = true;
                            }
                            s.0
                        }).collect::<Vec<bool>>()
                        .try_into().unwrap();
                        if sides != [false; 6] {
                            let mut block_mesh = get_vertices(BLOCK_SIZE);
                            let texture_offsets = get_texture_offsets(block_types[index(x, y, z)]);
                            for (i, block_vertex) in block_mesh.iter_mut().enumerate() {
                                if !sides_neighbors[i/6].0 {
                                    continue;
                                }
                                block_vertex.texture_coords = [block_vertex.texture_coords[0] + texture_offsets[i/6][0], block_vertex.texture_coords[1] + texture_offsets[i/6][1]];
                                block_vertex.position = [block_vertex.position[0] + x as f32, block_vertex.position[1] + y as f32, block_vertex.position[2] + z as f32];
                                // if vertices_map.contains_key(&block_vertex.position_as_string()) {
                                //     indices.push(*vertices_map.get(&block_vertex.position_as_string().clone()).unwrap());
                                // } else {
                                    // vertices_map.insert(block_vertex.position_as_string(), vertices.len() as u16);
                                    indices.push(vertices.len() as u16);
                                    vertices.push(block_vertex.clone());
                                // }
                            }
                        }
                    } else {
                        any_air = true;
                    }
                }
            }
        }
        let model = Model::new(&vertices, &indices, device);
        let instance = Instance {
            position: Vector3::new(wx as f32 *BLOCK_SIZE, wy as f32 *BLOCK_SIZE, wz as f32 *BLOCK_SIZE),
            color: [1.0, 1.0, 1.0],
            texture_offsets: [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        };
        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Instance Buffer"),
            contents: bytemuck::cast_slice(&[instance.to_raw()]),
            usage: wgpu::BufferUsages::VERTEX,
        });
        // println!("Instancing took {:?}", SystemTime::now().duration_since(time));
        return Self {
            // blocks: Some(blockss),
            instance_buffer: Some(instance_buffer),
            model: Some(model),
            solid_blocks: Some(solid_blocks),
            block_types: Some(block_types),
            // receiver: rx,
            pos: [wx, wy, wz],
            chunk_pos,
            finished: true,
            neighbor_remesh,
            any_air,
        };
    }

        //changes the block at a position and regenerates meshes and buffers as needed
    pub fn change_block(&mut self, chunks: &HashMap<[i32; 3], Chunk>, x: i32, y: i32, z: i32, solid: bool, block_type: u8, device: &dyn DeviceExt) {
        if self.block_types.is_none() {return}
        let block_types = self.block_types.as_mut().unwrap();
        let solid_blocks = self.solid_blocks.as_mut().unwrap();
        //return if its not in the world, happens when the raycast goes out of bounds
        if x < 0 || x >= CHUNK_SIZE as i32 || y < 0 || y >= CHUNK_SIZE as i32 || z < 0 || z >= CHUNK_SIZE as i32 {
            return;
        }
        block_types[index(x as usize, y as usize, z as usize)] = block_type;
        *solid_blocks.get_mut(index(x as usize, y as usize, z as usize)).unwrap() = solid;
        self.create_mesh(chunks, true, device);
    }

    pub fn destroy(&mut self) {
        if !self.finished {
            return;
        }
        if let Some(model) = &self.model {
            model.vertex_buffer.destroy();
            model.index_buffer.destroy();
        }
        self.finished = false;
    }

    pub fn create_mesh(&mut self, chunks: &HashMap<[i32; 3], Chunk>, _synchronous: bool, device: &dyn DeviceExt) {
        let block_types = self.block_types.as_mut().unwrap();
        let solid_blocks = self.solid_blocks.as_mut().unwrap();
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        let mut any_air = false;
        for x in 0..CHUNK_SIZE {
            for y in 0..CHUNK_SIZE {
                for z in 0..CHUNK_SIZE {
                    if get_solid(self.chunk_pos, solid_blocks, chunks, x as i32, y as i32, z as i32).0 {
                        let sides_neighbors = get_sides(self.chunk_pos, &solid_blocks, chunks, x as i32, y as i32, z as i32);
                        let sides: [bool; 6] = sides_neighbors.iter().enumerate().map(|(i, s)| {
                            if s.1 {
                                self.neighbor_remesh[i] = true;
                            }
                            s.0
                        }).collect::<Vec<bool>>()
                        .try_into().unwrap();
                        if sides != [false; 6] {
                            let mut block_mesh = get_vertices(BLOCK_SIZE);
                            let texture_offsets = get_texture_offsets(block_types[index(x, y, z)]);
                            for (i, block_vertex) in block_mesh.iter_mut().enumerate() {
                                if !sides[i/6] {
                                    continue;
                                }
                                block_vertex.texture_coords = [block_vertex.texture_coords[0] + texture_offsets[i/6][0], block_vertex.texture_coords[1] + texture_offsets[i/6][1]];
                                block_vertex.position = [block_vertex.position[0] + x as f32, block_vertex.position[1] + y as f32, block_vertex.position[2] + z as f32];
                                indices.push(vertices.len() as u16);
                                vertices.push(block_vertex.clone());
                            }
                        }
                    } else {
                        any_air = true;
                    }
                }
            }
        }
        let model = Model::new(&vertices, &indices, device);
        self.model = Some(model);
        self.any_air = any_air;
    }
}

pub fn raycast(start_pos: Vector3<f32>, dir: Vector3<f32>, max_dist: f32, chunks: &HashMap<[i32; 3], Chunk>) -> (bool, [i32; 3], u8, Vector3<f32>) {
    let mut pos = start_pos;
    while pos.distance(start_pos) <= max_dist {
        let world_pos = get_chunk_pos(pos.x as i32, pos.y as i32, pos.z as i32, CHUNK_SIZE);
        if chunks.contains_key(&world_pos.0) {
            if chunks.get(&world_pos.0).unwrap().finished {
                let solid_here = get_solid(world_pos.0, &chunks.get(&world_pos.0).unwrap().solid_blocks.as_ref().unwrap(), &HashMap::new(), world_pos.1[0], world_pos.1[1], world_pos.1[2]).0;
                if solid_here {
                    let block_pos = [pos.x as i32, pos.y as i32, pos.z as i32];
                    let block_here = chunks.get(&world_pos.0).unwrap().block_types.as_ref().unwrap()[index(pos.x as usize, pos.y as usize, pos.z as usize)];
                    while [pos.x as i32, pos.y as i32, pos.z as i32] == block_pos {
                        pos -= dir*0.0001;
                    }
                    return (true, block_pos, block_here, Vector3::new(pos.x - block_pos[0] as f32, pos.y - block_pos[1] as f32, pos.z - block_pos[2] as f32).normalize());
                }
            }
        }
        pos += dir;
    }
    return (false, [pos.x as i32, pos.y as i32, pos.z as i32], 0, Vector3::new(0.0,0.0,0.0));
}

pub fn get_texture_offsets(block_type: u8) -> [[f32; 2]; 6] {
    //no its +y -y +x -x +z -z
    //special cases
    if block_type == 0 {
        return [[0.0,0.0]; 6];
    }
    if block_type == 1 {
        return [[0.0, 0.0],[3.0, 0.0],[1.0, 0.0],[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]];
    }
    return [[(block_type) as f32,0.0]; 6];
}

//returns which sides of the block at a given position are visible
pub fn get_sides(chunk_pos: [i32; 3], solid_blocks: &BitVec, chunks: &HashMap<[i32; 3], Chunk>, x: i32, y: i32, z: i32) -> [(bool, bool); 6] {
    [get_solid(chunk_pos, solid_blocks, chunks, x, y+1, z), get_solid(chunk_pos, solid_blocks, chunks, x, y-1, z), get_solid(chunk_pos, solid_blocks, chunks, x+1, y, z), get_solid(chunk_pos, solid_blocks, chunks, x-1, y, z), get_solid(chunk_pos, solid_blocks, chunks, x, y, z+1), get_solid(chunk_pos, solid_blocks, chunks, x, y, z-1)].map(|s| (!s.0, s.1))
}

pub fn get_chunk_pos(x: i32, y: i32, z: i32, size: usize) -> ([i32; 3], [i32; 3]){
    let mut block_x = x%size as i32;
    let mut block_y = y%size as i32;
    let mut block_z = z%size as i32;

    let mut chunk_x = x/size as i32;
    let mut chunk_y = y/size as i32;
    let mut chunk_z = z/size as i32;
    if block_x < 0 {
        chunk_x -= 1;
        block_x += size as i32 - 1;
    }
    if block_y < 0 {
        chunk_y -= 1;
        block_y += size as i32 - 1;
    }
    if block_z < 0 {
        chunk_z -= 1;
        block_z += size as i32 - 1;
    }

    return ([chunk_x, chunk_y, chunk_z], [block_x, block_y, block_z]);
}

//gets the block color, and adds some random darkness to it
pub fn get_color_random(x: i32, y: i32, z: i32, block_type: u8) -> [f32; 3] {
    let mut rng = rand::rngs::StdRng::seed_from_u64((x*(CHUNK_SIZE*CHUNK_SIZE) as i32 +y*CHUNK_SIZE as i32 +z) as u64);
    let color_offset = (rng.next_u32() / (u32::MAX / 4)) as f32 + 1.0;
    //println!("{}", color_offset);
    let mut color = get_color(block_type);
    color[0] /= color_offset;
    color[1] /= color_offset;
    color[2] /= color_offset;
    return color;
}

//gets the color of the block based on the id
pub fn get_color(x: u8) -> [f32; 3] {
    if x == 1 {
        return [0.0,0.8,0.0];
    }
    if x == 2 {
        return [0.3,0.3,0.3];
    }
    return [0.0,0.0,0.0];
}

//gets if the block is solid
pub fn get_solid(chunk_pos: [i32; 3], bits: &BitVec, chunks: &HashMap<[i32; 3], Chunk>, x: i32, y: i32, z: i32) -> (bool, bool) {
    if x < 0 || y < 0 || z < 0 || x >= CHUNK_SIZE as i32|| y >= CHUNK_SIZE as i32 || z >= CHUNK_SIZE as i32 {
        let ([offset_chunk_x, offset_chunk_y, offset_chunk_z], [block_x, block_y, block_z]) = get_chunk_pos(x, y, z, CHUNK_SIZE);
        let chunk_option = chunks.get(&[offset_chunk_x+chunk_pos[0], offset_chunk_y+chunk_pos[1], offset_chunk_z+chunk_pos[2]]);
        if chunk_option.is_none() {
            // println!("chunk {:?} not loaded 😔", [offset_chunk_x+chunk_pos[0], offset_chunk_y+chunk_pos[1], offset_chunk_z+chunk_pos[2]]);
            return (true, true);
        }
        let chunk_bits = chunk_option.as_ref().unwrap().solid_blocks.as_ref().unwrap();
        let bit = chunk_bits.get(index(block_x as usize, block_y as usize, block_z as usize));
        if bit.is_some() {
            // println!("good! ✅");
            return (*bit.unwrap(), false);
        } else {
            println!("bit was out of bounds ❌");
            return (false, false);
        }
    }
    let bit = bits.get(index(x as usize, y as usize, z as usize));
    if bit.is_some() {
        return (*bit.unwrap(), false);
    } else {
        println!("bit was out of bounds here 🚨");
        return (false, false);
    }
}

pub fn get_position<T: Into<f32>>(x: T, y: T, z: T, block_size: f32) -> cgmath::Vector3<f32> {
    let x = x.into() * block_size;
    let y = y.into() * block_size;
    let z = z.into() * block_size;
    //println!("{} {} {}", x, y, z);
    return cgmath::Vector3::new(x, y, z);
}

//gets the index in from x y z
pub fn index(x: usize, y: usize, z: usize) -> usize {
    if x < CHUNK_SIZE && y < CHUNK_SIZE && z < CHUNK_SIZE {
        return x*CHUNK_SIZE*CHUNK_SIZE + y * CHUNK_SIZE + z;
    }
    return 0;
}

//interprets false as 0 and true as 1 to create a 6 bit number
pub fn sides_to_index(sides: [bool; 6]) -> usize {
    let mut index = 0;
    for i in (0..6).rev() {
        index += sides[i] as usize * (2 as usize).pow(5-i as u32);

    }
    index
}

//creates a mesh of visible sides
pub fn create_mesh(sides: [bool; 6]) -> Vec<u16> {
    let mut indices: Vec<u16> = Vec::new();
    for i in 0..6 {
        if sides[i] {
            indices.extend_from_slice(&INDICES[i*6..(i+1)*6]);
        }
    }
    indices
}

//returns the bits of a 6 bit number as a boolean array
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

//creates all 64 possible cube meshes 
pub fn create_all_meshes(device: &dyn DeviceExt) -> model::Model {
    let mut indices = Vec::new();
    for i in 0..64 {
        indices.extend_from_slice(&create_mesh(number_to_bits(i)));
    }
    model::Model::new(&get_vertices(BLOCK_SIZE), &indices, device)
}
//banana
// -Meredith