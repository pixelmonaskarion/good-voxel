use crate::chunk::*;
use crate::block::*;
use std::thread;
use wgpu::util::DeviceExt;
use colored::Colorize;
use std::collections::HashMap;

#[allow(dead_code)]
pub struct World {
    pub loaded_chunks: Vec<Chunk>,
    pub unloaded_chunks: HashMap<[i32; 3], Chunk>,
    pub last_pos: [i32; 3],
    width: i32,
    height: i32,
    depth: i32,
}

impl World {
    pub fn new(width: i32, height: i32, depth: i32, device: &dyn DeviceExt) -> Self {
        let mut chunks: Vec<Chunk> = Vec::new();
        for x in -width..width+1 {
            for y in -height..height+1 {
                for z in -depth..depth+1 {
                    let chunk = Chunk::new(x, y, z, device);
                    chunks.push(chunk);
                }
            }
        }
        Self {
            loaded_chunks: chunks,
            unloaded_chunks: HashMap::new(),
            last_pos: [0, 10000, 0],
            width: width*3,
            height: height*3,
            depth: depth*3,
        }
    }

    pub fn load_around(&mut self, camera_pos: [f32; 3], all_in_one_frame: bool, device: &dyn DeviceExt) {
        //let cx = (camera_pos[0] as i32)/WORLD_SIZE;
        //let cy = (camera_pos[1] as i32)/WORLD_SIZE;
        //let cz = (camera_pos[2] as i32)/WORLD_SIZE;
        /*unsafe {
        let handle = thread::spawn(|| {*/
        let cx = (camera_pos[0] as i32 - weird_mod(camera_pos[0] as i32, WORLD_SIZE))/WORLD_SIZE;
        let cy = (camera_pos[1] as i32 - weird_mod(camera_pos[1] as i32, WORLD_SIZE))/WORLD_SIZE;
        let cz = (camera_pos[2] as i32 - weird_mod(camera_pos[2] as i32, WORLD_SIZE))/WORLD_SIZE;
        if self.last_pos == [cx, cy, cz] {
            //println!("last {} {} {}, current {} {} {}", self.last_pos[0], self.last_pos[1], self.last_pos[2], cx, cy, cz);
            return;
        }
        let mut i = self.loaded_chunks.len() as i32 - 1;
        while i >= 0 {
            let chunk = self.loaded_chunks[i as usize].clone();
            //println!("chunk {} {} {} pos {} {} {}", chunk.pos[0], chunk.pos[1], chunk.pos[2], cx, cy, cz);
            if chunk.pos[0]-cx > self.width || chunk.pos[0]-cx < -self.width || chunk.pos[1]-cy > self.height || chunk.pos[1]-cy < -self.height || chunk.pos[2]-cz > self.depth || chunk.pos[2]-cz < -self.depth {
                self.loaded_chunks.remove(i as usize);
                self.unloaded_chunks.insert(chunk.pos, chunk);
            }
            i -= 1;
        }
        //let mut total = 0.0;
        for x in cx-self.width..self.width+cx+1 {
            for y in cy-self.height..self.height+cy+1 {
                'z: for z in cz-self.depth..self.depth+cz+1 {
                    for i in 0..self.loaded_chunks.len() {
                        if self.loaded_chunks[i].pos == [x, y, z] {
                            continue 'z;
                        }
                    }
                    if self.unloaded_chunks.contains_key(&[x, y, z]) {
                        let chunk = self.unloaded_chunks.remove_entry(&[x, y, z]).unwrap().1.clone();
                        self.loaded_chunks.push(chunk);
                        if !all_in_one_frame && self.create_mesh_around(x, y, z, device) {
                            return;
                        }
                    } else {
                        let chunk = Chunk::new(x, y, z, device);
                        self.loaded_chunks.push(chunk);
                        if !all_in_one_frame && self.create_mesh_around(x, y, z, device) {
                            return;
                        }
                    }
                    
                    //self.create_mesh(device);
                    //let time = SystemTime::now();
                    
                    /*let millis = match time.elapsed() {
                        Ok(dur) => dur.as_millis(),
                        Err(_e) => panic!(),
                    };
                    total += millis as f32 / 1000.0;
                    println!("took {} seconds to mesh chunk", millis as f32/1000.0);*/
                }
            }
        }
        self.last_pos = [cx, cy, cz];
    //});
//}
        //println!("total of {} seconds", total);
        /*if loaded_any {
            println!("had to load chunk this frame");
            self.create_mesh(device);
        }*/
    }

    pub fn create_mesh(&mut self, device: &dyn DeviceExt) {
        let mut chunks = self.loaded_chunks.clone();
        for i in 0..chunks.len() {
            chunks[i].instance_buffer = Some(create_instance(chunks[i].pos, device));
            chunks[i].create_mesh(self, device);
        }
        self.loaded_chunks = chunks;
    }

    pub fn create_mesh_around(&mut self, cx:i32, cy: i32, cz: i32, device: &dyn DeviceExt) -> bool {
        //println!("creating mesh around {} {} {}", cx, cy, cz);
        let mut any_chunks_with_blocks = false;
        for i in 0..self.loaded_chunks.len() {
            if self.loaded_chunks[i].pos == [cx, cy, cz] || 
            self.loaded_chunks[i].pos == [cx+1, cy, cz] || self.loaded_chunks[i].pos == [cx-1, cy, cz] || 
            self.loaded_chunks[i].pos == [cx, cy+1, cz] || self.loaded_chunks[i].pos == [cx, cy-1, cz] || 
            self.loaded_chunks[i].pos == [cx, cy, cz+1] || self.loaded_chunks[i].pos == [cx, cy, cz-1] {
                let mut chunk = self.loaded_chunks[i].clone();
                chunk.instance_buffer = Some(create_instance(chunk.pos, device));
                if chunk.create_mesh(self, device) {
                    any_chunks_with_blocks = true;
                }
                self.loaded_chunks[i] = chunk;
            }
        }
        return any_chunks_with_blocks;
    }

    pub fn get_block(&self, x: i32, y: i32, z: i32) -> Option<Block> {
        //finds the chunk that the block is in
        let cx = (x - weird_mod(x, WORLD_SIZE))/WORLD_SIZE;
        let cy = (y - weird_mod(y, WORLD_SIZE))/WORLD_SIZE;
        let cz = (z - weird_mod(z, WORLD_SIZE))/WORLD_SIZE;
        //println!("ci {} cx {} cy {} cz {}", chunk, cx, cy, cz);
        for chunk in 0..self.loaded_chunks.len() {
            if self.loaded_chunks[chunk].pos == [cx, cy, cz] {
                //finds the coords of the block in the chunk
                let bx = weird_mod(x, WORLD_SIZE);
                let by = weird_mod(y, WORLD_SIZE);
                let bz = weird_mod(z, WORLD_SIZE);
                //finds the blocks index in the chunk
                let i = bx*WORLD_SIZE*WORLD_SIZE + by*WORLD_SIZE + bz;
                //println!("i {} x {} y {} z {}", i, bx, by, bz);
                return Some(self.loaded_chunks[chunk].blocks[i as usize].clone());
            }
        }
        //println!("x {} y {} z {}", x, y, z);
        None
    }

    #[allow(dead_code)]
    pub fn test(&self) {
        for x in -1..10 {
            for y in -1..10 {
                for z in -1..10 {
                    if x > -1 && x < 9 && y > -1 && y < 9 && z > -1 && z < 9 {
                        let block = self.get_block(x, y, z).unwrap();
                        if block.pos[0]+block.chunk_pos[0]*WORLD_SIZE == x && block.pos[1]+block.chunk_pos[1]*WORLD_SIZE == y && block.pos[2]+block.chunk_pos[2]*WORLD_SIZE == z {
                            println!("{}", format!("got the right block").green());
                        } else {
                            println!("{} {} {}", x, y, z);
                            println!("{}", format!("got the wrong block, wrong position").red());
                            std::process::exit(1);
                        }
                    } else {
                        let block = self.get_block(x, y, z);
                        if block.is_none() {
                            println!("{}", format!("got the right block").green());
                        } else {
                            println!("{} {} {}", x, y, z);
                            println!("{}", format!("got the wrong block, should be null").red());
                            std::process::exit(1);
                        }
                    }
                    
                }
            }
        }
        std::process::exit(1);
    }
}

pub fn weird_mod(x: i32, modulus: i32) -> i32{
    (x % modulus + modulus) % modulus
}