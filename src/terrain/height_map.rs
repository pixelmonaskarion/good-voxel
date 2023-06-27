pub struct HeightMap {
    pub height_functions: Vec<HeightFunction>
}

pub struct HeightFunction {
    pub function: fn (f32, f32) -> f32,
    pub magnitude: f32,
    pub function_type: HeightFunctionType,
}

impl HeightMap {
    pub fn height_at(&self, x: f32, z: f32) -> f32 {
        let mut composite = 0.0;
        for height_function in &self.height_functions {
            let height = (height_function.function)(x, z)*height_function.magnitude;
            match height_function.function_type {
                HeightFunctionType::Multiply => composite *= height,
                HeightFunctionType::Add => composite += height,
            }
        }
        
        return composite;
    }

    pub fn new() -> Self {
        Self {
            height_functions: Vec::new(),
        }
    }
}

pub enum HeightFunctionType {
    Multiply,
    Add,
}