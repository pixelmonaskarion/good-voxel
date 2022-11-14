pub fn colliding_box(this: &Box, other: &Box) -> (bool, [f32; 3]) {
    let colliding = other.pos[0]+other.size[0]/2.0 > this.pos[0]-this.size[0]/2.0 && other.pos[0]-other.size[0]/2.0 < this.pos[0]+this.size[0]/2.0 &&
    other.pos[1]+other.size[1]/2.0 > this.pos[1]-this.size[1]/2.0 && other.pos[1]-other.size[1]/2.0 < this.pos[1]+this.size[1]/2.0 && 
    other.pos[2]+other.size[2]/2.0 > this.pos[2]-this.size[2]/2.0 && other.pos[2]-other.size[2]/2.0 < this.pos[2]+this.size[2]/2.0;
    if colliding {
        let corner_x1 = sign(this.pos[0]-other.pos[0])*other.size[0]/2.0 + other.pos[0];
        let corner_y1 = sign(this.pos[1]-other.pos[1])*other.size[1]/2.0 + other.pos[1];
        let corner_z1 = sign(this.pos[2]-other.pos[2])*other.size[2]/2.0 + other.pos[2];

        let corner_x2 = sign(other.pos[0]-this.pos[0])*this.size[0]/2.0 + this.pos[0];
        let corner_y2 = sign(other.pos[1]-this.pos[1])*this.size[1]/2.0 + this.pos[1];
        let corner_z2 = sign(other.pos[2]-this.pos[2])*this.size[2]/2.0 + this.pos[2];

        let corners = [corner_x1-corner_x2, corner_y1-corner_y2, corner_z1-corner_z2];
        let mut lowest = 0;
        for i in 0..3 {
            if corners[i] < corners[lowest] {
                lowest = i;
            }
        }
        let mut return_array = [0.0,0.0,0.0];
        return_array[lowest] = -corners[lowest];
        return (colliding, return_array);
    } else {
        return (colliding, [0.0,0.0,0.0]);
    }
}

fn sign(x: f32) -> f32 {
    if x == 0.0 {
        println!("x == 0 :0");
    }
    return x/x.abs();
}

pub struct Box {
    pub pos: [f32; 3],
    pub size: [f32; 3],
}