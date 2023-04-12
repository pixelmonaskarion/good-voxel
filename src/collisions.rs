pub fn colliding_box_old(this: &Box, other: &Box) -> (bool, [f32; 3]) {
    let colliding = is_colliding(this, other);
    if colliding {
<<<<<<< HEAD
        return (true, find_collision_fix_vector(this, other));
=======
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
        if lowest == 1 {
            println!("WEEWOO");
        }
        let mut return_array = [0.0,0.0,0.0];
        return_array[lowest] = -corners[lowest];
        return (colliding, return_array);
>>>>>>> 82510d53306726d5110362cd865cf4168fa28d62
    } else {
        return (colliding, [0.0,0.0,0.0]);
    }
}

fn is_point_in_box(point: [f32; 3], cube: &Box) -> bool {
    let mut colliding = true;
    for i in 0..3 {
        let positive = cube.pos[i]+cube.size[i]/2.0 > point[i];
        let negative = cube.pos[i]-cube.size[i]/2.0 < point[i];
        colliding = colliding && positive && negative;
        println!("i: {}, pos: {}, negative: {}, colliding: {}", i, positive, negative, colliding);
    }
    return colliding;
}

fn find_collision_fix_vector(this: &Box, other: &Box) -> [f32; 3] {
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
    return_array[lowest] = corners[lowest];
    return return_array;
}

fn is_colliding(this: &Box, other: &Box) -> bool {
    let mut colliding = true;
    for i in 0..3 {
        let other_pos = other.pos[i]+other.size[i]/2.0;
        let this_negative = this.pos[i]-this.size[i]/2.0;
        let other_negative = other.pos[i]-other.size[i]/2.0;
        let this_pos = this.pos[i]+this.size[i]/2.0;
        colliding = colliding && other_pos > this_negative && other_negative < this_pos;
    }
    return colliding;
}

fn get_collision_points(this: &Box, other: &Box) -> [Vec<[f32; 3]>; 2] {
    let mut first_corners = Vec::new();
    for corner in this.get_all_points() {
        println!("{:?}", corner);
        if is_point_in_box(corner, other) {
            first_corners.push(corner);
        }
    }
    let mut second_corners = Vec::new();
    for corner in other.get_all_points() {
        if is_point_in_box(corner, this) {
            second_corners.push(corner);
        }
    }
    println!("{:?} {:?}", first_corners, second_corners);
    return [first_corners, second_corners];
}

fn sign(x: f32) -> f32 {
    if x >= 0.0 {
        return 1.0;
    }
    return -1.0;
}

fn get_intersection_rect(box1: &Box, box2: &Box) -> Box {
    let x5 = box1.get_all_points()[0][0].max(box2.get_all_points()[0][0]);
    let y5 = box1.get_all_points()[0][1].max(box2.get_all_points()[0][1]);
    let z5 = box1.get_all_points()[0][2].max(box2.get_all_points()[0][2]);

    let x6 = box1.get_all_points()[1][0].min(box2.get_all_points()[1][0]);
    let y6 = box1.get_all_points()[1][1].min(box2.get_all_points()[1][1]);
    let z6 = box1.get_all_points()[1][2].min(box2.get_all_points()[1][2]);
    let intersection = Box::from_corners([[x5, y5, z5], [x6, y6, z6]]);
    return intersection;
}

pub fn colliding_box_inter(box1: &Box, box2: &Box) -> (bool, [f32; 3]) {
    let intersection = get_intersection_rect(box1, box2);
    return (is_colliding(box1, box2), intersection.size);
}

pub fn colliding_box(box1: &Box, box2: &Box) -> (bool, [f32; 3]) {
    let pos_y = box2.pos[1]-box1.pos[1]-box1.size[1]+box2.size[1];
    return (is_colliding(box1, box2), [0.0, pos_y, 0.0]);
}

#[derive(Debug)]
pub struct Box {
    pub pos: [f32; 3],
    pub size: [f32; 3],
}

impl Box {
    pub fn get_all_points(&self) -> Vec<[f32; 3]> {
        let mut points = Vec::new();
        
        for offset in [[-1, 1, 1], [-1, -1, 1], [-1, -1, -1], [1, -1, 1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [1, 1, 1]] {
            let x = self.pos[0]+self.size[0]/2.0*offset[0] as f32;
            let y = self.pos[1]+self.size[1]/2.0*offset[1] as f32;
            let z = self.pos[2]+self.size[2]/2.0*offset[2] as f32;
            points.push([x, y, z]);
        }

        return points;
    }

    pub fn get_min_and_max_corners(&self) -> [[f32; 3]; 2] {
        return [[self.pos[0]-self.size[0]/2.0, self.pos[1]-self.size[1]/2.0, self.pos[2]-self.size[2]/2.0],
                [self.pos[0]+self.size[0]/2.0, self.pos[1]+self.size[1]/2.0, self.pos[2]+self.size[2]/2.0]]
    }

    pub fn from_corners(corners: [[f32; 3]; 2]) -> Box {
        let center = [(corners[0][0]+corners[1][0])/2.0, (corners[0][1]+corners[1][1])/2.0, (corners[0][2]+corners[1][2])/2.0];
        let size = [(corners[1][0]-center[0])*2.0, (corners[1][1]-center[1])*2.0, (corners[1][2]-center[2])*2.0];
        Box {
            pos: center,
            size,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    pub fn test() {
        let box1 = Box{pos:[50.0, 50.0, 50.0], size:[1.0, 1.0, 1.0]};
        let box2 = Box{pos:[50.5, 51.0, 50.0], size:[0.8, 1.9, 0.8]};
        let result = colliding_box(&box1, &box2);
        assert!(result.0);
    }

    #[test]
    pub fn neg_test() {
        let box1 = Box{pos:[50.0, 50.0, 50.0], size:[1.0, 1.0, 1.0]};
        let box2 = Box{pos:[50.5, 52.800, 50.0], size:[0.8, 1.9, 0.8]};
        let result = colliding_box(&box1, &box2);
        assert!(!result.0);
    }

    #[test]
    pub fn point_test() {
        let cube = Box{pos:[0.0,0.0,0.0], size: [1.0, 2.0, 3.0]};
        assert!(is_point_in_box([0.0, 0.0, 0.0], &cube));
        assert!(!is_point_in_box([10.0, 0.0, 0.0], &cube));
        assert!(is_point_in_box([0.25, 0.0, 0.0], &cube));
        assert!(is_point_in_box([0.499, 0.0, 0.0], &cube));
        assert_eq!(get_collision_points(&Box{pos:[0.0,0.0,0.0], size: [1.0, 1.0, 1.0]}, &Box{pos:[10.0,0.0,0.0], size: [1.0, 1.0, 1.0]})[0].len(), 0);
        assert_eq!(get_collision_points(&Box{pos:[0.0,0.0,0.0], size: [1.0, 1.0, 1.0]}, &Box{pos:[0.1,0.0,0.0], size: [1.0, 1.0, 1.0]})[0].len(), 1);
    }

    #[test]
    pub fn intersection_test() {

    }
}