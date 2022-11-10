use crate::Instance;
use crate::Vertex;
use crate::model;
use cgmath::InnerSpace;
use noise::NoiseFn;
use noise::Perlin;
use wgpu::util::DeviceExt;
use rand::prelude::*;
use bitvec::prelude::*;
use std::collections::HashMap;
use std::sync::mpsc::Receiver;
use cgmath::{Vector3, MetricSpace};
use wasm_timer::SystemTime;
use std::thread;
use std::sync::mpsc::channel;

#[allow(unused_imports)]
use mockall::predicate::*;
#[allow(unused_imports)]
use mockall::*;

pub const BLOCK_SIZE: f32 = 1.0;
pub const CHUNK_SIZE: usize = 100;

//holds the basic model of a cube, all sides shown
pub const VERTICES: &[Vertex] = &[
    Vertex {position: [0.0, 0.0, 0.0]},
    Vertex {position: [1.0, 0.0, 0.0]},
    Vertex {position: [1.0, 1.0, 0.0]},
    Vertex {position: [0.0, 1.0, 0.0]},
    Vertex {position: [0.0, 0.0, 1.0]},
    Vertex {position: [0.0, 1.0, 1.0]},
    Vertex {position: [1.0, 1.0, 1.0]},
    Vertex {position: [1.0, 0.0, 1.0]},
];
const INDICES: &[u16] = &[
    0, 2, 1, 0, 3, 2,//bottom 
    7, 6, 4, 6, 5, 4,//top
    0, 4, 5, 0, 5, 3,//north
    2, 6, 1, 6, 7, 1,//south
    7, 4, 0, 1, 7, 0,//west
    3, 6, 2, 5, 6, 3,//east
];

pub fn get_vertices(size: f32) -> [Vertex; VERTICES.len()] {
    let mut new_vertices = [Vertex {position:[0.0,0.0,0.0]}; VERTICES.len()];
    for (i, v) in VERTICES.iter().enumerate() {
        new_vertices[i] = Vertex{position:[v.position[0]*size, v.position[1]*size, v.position[2]*size]};
    }
    return new_vertices;
}

//blocks: struct for holding buffers of all positions of one mesh of block
pub struct Blocks {
    pub instance_buffer: wgpu::Buffer,
    pub num_instances: usize,
}

/*holds world information,
    blocks: all 64 Blocks objects
    solid_blocks: a bunch of bits storing whether or not the block at that index is solid
    block_types: a list of bytes storing a number corresponding to the color of the block in the get_color function
    size: width/height/depth of the world
    instances: for holding each individual position for blocks, explained in change_block funcion
*/
pub struct World {
    pub blocks: Option<Vec<Blocks>>,
    pub solid_blocks: Option<BitVec>,
    pub block_types: Option<Vec<u8>>,
    pub size: usize,
    pub instances: Option<Vec<HashMap<[i32; 3], Instance>>>,
    pub receiver: Receiver<(Vec<HashMap<[i32; 3], Instance>>, BitVec<usize, Lsb0>, Vec<u8>)>,
}

#[derive(Clone)]
pub struct Generator {
    pub surface_noise: Perlin,
    pub cave_noise: Perlin,
    pub seed: u32,
}

impl Generator {
    pub fn new() -> Self {
        let mut rng = rand::thread_rng();
        let seed = rng.gen::<u32>();
        let surface_noise = Perlin::new(seed);
        let cave_noise = Perlin::new(seed);
        Self {
            surface_noise,
            cave_noise,
            seed,
        }
    }
}

impl World {
    //creates a new world with the selected size
    // generates the blocks
    // creates instances for them
    // sends them to the gpu and gets buffers
    pub fn world(generator: &Generator, wx: i32, wy: i32, wz: i32, size: usize, device: &dyn DeviceExt, synchronous: bool) -> World {
        let (tx, rx) = channel();
        let generator_moved = generator.clone();
        thread::spawn(move|| {
        //initializes variables
        let mut instances: Vec<HashMap<[i32; 3], Instance>> = std::iter::repeat(HashMap::new()).take(64).collect::<Vec<_>>();
        let mut solid_blocks: BitVec = bitvec![mut 1; 1];
        solid_blocks.resize(size.pow(3), true);
        let mut block_types: Vec<u8> = Vec::new(); 
        block_types.resize(size.pow(3), 0);
        let time = SystemTime::now();
        //loops through all blocks and generates them
        for x in 0..size {
            for z in 0..size {
                let exact_x = x as i32 + wx;
                let exact_z = z as i32 + wz;
                //the height is only determined by x and z so it is created here and the loop is is in this weird order
                let height = generator_moved.surface_noise.get([exact_x as f64 / 100.0, exact_z as f64 / 100.0]);
                for y in 0..size {
                    let exact_y = y as i32 + wy;
                    //ground or sky?
                    #[allow(unused_mut)]
                    let mut solid = (exact_y as f32) < (height+1.0) as f32 *50.0 && generator_moved.cave_noise.get([exact_x as f64/ 30.0, exact_y as f64/ 30.0, exact_z as f64/ 30.0]) < 0.4;
                    //is it ten blocks underneath the surface?
                    if (exact_y as f32 + 10.0) < (height+1.0) as f32 *50.0 {
                        //solid if in a cave
                        //solid = 
                        //sets the block type to be stone block type
                        block_types[index(x, y, z, size)] = 2;
                    } else {
                        //sets the block type to be grass block type
                        block_types[index(x, y, z, size)]  = 1;
                    }
                    //sets the bit at the index of the position to be solid's value
                    *solid_blocks.get_mut(index(x, y, z, size)).unwrap() = solid;
                }
            }
            //println!("Generating {}%", ((x as f32 +1.0)/size as f32)*100.0);
        }
        //let (tx, rx) = channel();
        //loops over every block and instances it
        for x in 0..size {
            //thread::spawn(move|| {
                for y in 0..size {
                    for z in 0..size {
                        //init sides
                        let mut sides = [false; 6];
                        let index = index(x, y, z, size);
                        let block_type = block_types[index];
                        //castes position to i32 now for easier readability, index takes usize and get_solid/get_sides take i32
                        let x = x as i32;
                        let y = y as i32;
                        let z = z as i32;
                        let exact_x = x + wx;
                        let exact_z = z + wz;
                        let exact_y = y + wy;
                        if get_solid(&mut solid_blocks, x, y, z, size) {
                            sides = get_sides(&solid_blocks, x, y, z, size);
                        }
                        if sides != [false; 6] {
                            instances[sides_to_index(sides)].insert([x, y, z],Instance {
                                position: get_position(exact_x as f32, exact_y as f32, exact_z as f32, BLOCK_SIZE),
                                color: get_color_random(index.try_into().unwrap(), block_type, 0.5),
                            });
                        }
                    }
                }
            //});
            //println!("Instancing {}%", ((x as f32 +1.0)/size as f32)*100.0);
        }
        println!("took {:?}", SystemTime::now().duration_since(time));
        let _result = tx.send((instances, solid_blocks, block_types));
        });
        if synchronous {
            let data = rx.recv().unwrap();
            let mut blockss = Vec::new();
            //sends each vec of instances to the gpu and puts the buffer into a vec
            for instance in &data.0 {
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
            return World {
                blocks: Some(blockss),
                solid_blocks: Some(data.1),
                block_types: Some(data.2),
                size,
                instances: Some(data.0),
                receiver: rx,
            };
        }
        //println!("fr tho it actually took {:?}", SystemTime::now().duration_since(time));
        return World {
            blocks: None,
            solid_blocks: None,
            block_types: None,
            size,
            instances: None,
            receiver: rx,
        };
    }

    pub fn try_finish(&mut self, device: &dyn DeviceExt) {
        match self.receiver.try_recv() {
            Ok(data) => {
                let mut blockss = Vec::new();
                //sends each vec of instances to the gpu and puts the buffer into a vec
                for instance in &data.0 {
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
                self.instances = Some(data.0);
                self.solid_blocks = Some(data.1);
                self.block_types = Some(data.2);
                self.blocks = Some(blockss);
            },
            Err(_e) => {},
        };
    }

        //changes the block at a position and regenerates meshes and buffers as needed
    pub fn change_block(&mut self, x: i32, y: i32, z: i32, solid: bool, block_type: u8, device: &dyn DeviceExt) {
        if self.block_types.is_none() {return}
        let block_types = self.block_types.as_mut().unwrap();
        let solid_blocks = self.solid_blocks.as_mut().unwrap();
        let instances = self.instances.as_mut().unwrap();
        //return if its not in the world, happens when the raycast goes out of bounds
        if x < 0 || x >= CHUNK_SIZE as i32 || y < 0 || y >= CHUNK_SIZE as i32 || z < 0 || z >= CHUNK_SIZE as i32 {
            return;
        }
        block_types[index(x as usize, y as usize, z as usize, CHUNK_SIZE)] = block_type;
        let block_index = sides_to_index(get_sides(&solid_blocks, x, y, z, CHUNK_SIZE));
        let mut update_sides = false;
        let original_solid = *solid_blocks.get(index(x as usize, y as usize, z as usize, CHUNK_SIZE)).unwrap();
        if  original_solid == true && solid == false {
            instances[block_index].remove(&[x,y,z]);
            update_sides = true;
        }
        if original_solid == true && solid == true {
            instances[block_index].insert([x, y, z], Instance {
                position: get_position(x as f32, y as f32, z as f32, BLOCK_SIZE),
                color: get_color_random(index(x as usize, y as usize, z as usize, CHUNK_SIZE).try_into().unwrap(), block_type, 0.5),
            });
        }
        if original_solid == false && solid == true {

            instances[block_index].insert([x, y, z], Instance {
                position: get_position(x as f32, y as f32, z as f32, BLOCK_SIZE),
                color: get_color_random(index(x as usize, y as usize, z as usize, CHUNK_SIZE).try_into().unwrap(), block_type, 0.5),
            });
            update_sides = true;
        }
        if update_sides {
            for offset in [[1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]] {
                if *solid_blocks.get(index((x+offset[0]) as usize, (y+offset[1]) as usize, (z+offset[2]) as usize, CHUNK_SIZE)).unwrap() == false {
                    continue;
                }
                //TODO: this code is very weird, explain it, Christopher!
                let side_index = sides_to_index(get_sides(&solid_blocks, x+offset[0], y+offset[1], z+offset[2], CHUNK_SIZE));
                //
                *solid_blocks.get_mut(index(x as usize, y as usize, z as usize, CHUNK_SIZE)).unwrap() = solid;
                //
                let new_sides = get_sides(&solid_blocks, x+offset[0], y+offset[1], z+offset[2], CHUNK_SIZE);
                *solid_blocks.get_mut(index(x as usize, y as usize, z as usize, CHUNK_SIZE)).unwrap() = original_solid;
                instances[side_index].remove(&[x+offset[0], y+offset[1], z+offset[2]]);
                instances[sides_to_index(new_sides)].insert([x+offset[0], y+offset[1], z+offset[2]], Instance {
                    position: get_position((x+offset[0]) as f32, (y+offset[1]) as f32, (z+offset[2]) as f32, BLOCK_SIZE),
                    color: get_color_random(index((x+offset[0]) as usize, (y+offset[1]) as usize, (z+offset[2]) as usize, CHUNK_SIZE).try_into().unwrap(), block_types[index((x+offset[0]) as usize, (y+offset[1]) as usize, (z+offset[2]) as usize, CHUNK_SIZE)], 0.5),
                });
            }
        }
        *solid_blocks.get_mut(index(x as usize, y as usize, z as usize, CHUNK_SIZE)).unwrap() = solid;
        let mut blockss = Vec::new();
        for instance in instances {
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
        self.blocks = Some(blockss);
    }

    //pretty much a basic raycasting function, but it doesn't work very well and I need to redo it soon
    pub fn raycast(&self, start_pos: Vector3<f32>, dir: Vector3<f32>, max_dist: f32) -> (bool, [i32; 3], u8, Vector3<f32>) {
        let mut pos = start_pos;
        while pos.distance(start_pos) <= max_dist && pos.x >= 0.0 && pos.x < CHUNK_SIZE as f32 && pos.y >= 0.0 && pos.y < CHUNK_SIZE as f32 && pos.z >= 0.0 && pos.z < CHUNK_SIZE as f32 {
            let solid_here = get_solid(&self.solid_blocks.as_ref().unwrap(), pos.x as i32, pos.y as i32, pos.z as i32, CHUNK_SIZE);
            if solid_here {
                let block_pos = [pos.x as i32, pos.y as i32, pos.z as i32];
                let block_here = self.block_types.as_ref().unwrap()[index(pos.x as usize, pos.y as usize, pos.z as usize, CHUNK_SIZE)];
                while [pos.x as i32, pos.y as i32, pos.z as i32] == block_pos {
                    pos -= dir*0.0001;
                }
                return (true, block_pos, block_here, Vector3::new(pos.x - block_pos[0] as f32, pos.y - block_pos[1] as f32, pos.z - block_pos[2] as f32).normalize());
            }
            pos += dir;
        }
        return (false, [pos.x as i32, pos.y as i32, pos.z as i32], 0, Vector3::new(0.0,0.0,0.0));
    }
}

//returns which sides of the block at a given position are visible
pub fn get_sides(solid_blocks: &BitVec, x: i32, y: i32, z: i32, size: usize) -> [bool; 6] {
    [!get_solid(solid_blocks, x, y+1, z, size), !get_solid(solid_blocks, x, y-1, z, size), !get_solid(solid_blocks, x+1, y, z, size), !get_solid(solid_blocks, x-1, y, z, size), !get_solid(solid_blocks, x, y, z+1, size), !get_solid(solid_blocks, x, y, z-1, size)]
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
        block_x += size as i32;
    }
    if block_y < 0 {
        chunk_y -= 1;
        block_y += size as i32;
    }
    if block_z < 0 {
        chunk_z -= 1;
        block_z += size as i32;
    }

    return ([chunk_x, chunk_y, chunk_z], [block_x, block_y, block_z]);
}

//gets the block color, and adds some random darkness to it
pub fn get_color_random(index: u64, x: u8, randomness: f32) -> [f32; 3] {
    let mut rng = rand::rngs::StdRng::seed_from_u64(index);
    let color_offset = (rng.next_u64() as f32 / u64::MAX as f32)*randomness;
    let mut color = get_color(x);
    color[0] *= color_offset;
    color[1] *= color_offset;
    color[2] *= color_offset;
    return color;
}

//gets the color of the block based on the id
pub fn get_color(x: u8) -> [f32; 3] {
    if x == 1 {
        return [0.0,1.0,0.0];
    }
    if x == 2 {
        return [0.3,0.3,0.3];
    }
    return [0.0,0.0,0.0];
}

//gets if the block is solid
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

pub fn get_position<T: Into<f32>>(x: T, y: T, z: T, block_size: f32) -> cgmath::Vector3<f32> {
    let x = x.into() * block_size;
    let y = y.into() * block_size;
    let z = z.into() * block_size;
    //println!("{} {} {}", x, y, z);
    return cgmath::Vector3::new(x, y, z);
}

//gets the index in from x y z
pub fn index(x: usize, y: usize, z: usize, size: usize) -> usize {
    if x < size && y < size && z < size {
        return x*size*size + y * size + z;
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
pub fn create_mesh(sides: [bool; 6], device: &dyn DeviceExt) -> model::Model {
    let mut indices: Vec<u16> = Vec::new();
    for i in 0..6 {
        if sides[i] {
            indices.extend_from_slice(&INDICES[i*6..(i+1)*6]);
        }
    }
    model::Model::new(&get_vertices(BLOCK_SIZE), &indices, device)
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
pub fn create_all_meshes(device: &dyn DeviceExt) -> Vec<model::Model> {
    let mut meshes = Vec::new();
    for i in 0..64 {
        meshes.push(create_mesh(number_to_bits(i), device));
    }
    meshes
}
//banana
// -Meredith