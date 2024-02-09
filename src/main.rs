use tiny_voxel::run;

fn main() {
    pollster::block_on(run());
}

#[cfg(test)]
mod test {
    use std::collections::HashMap;

    #[test]
    fn mod_test() {
        let a: i32 = -3;
        let b: i32 = 16;
    
        assert_eq!(a.rem_euclid(b), 13);
        let mut map = HashMap::new();
        map.insert([0,0,0], "Balls");
        map.insert([1,0,0], "Balls 1");
        map.insert([2,0,0], "Balls 2");
        assert_eq!(*map.get(&[0,0,0]).unwrap(), "Balls");
    }
}