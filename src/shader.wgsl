// Vertex shader
struct CameraUniform {
    view_proj: mat4x4<f32>;
};
[[group(1), binding(0)]]
var<uniform> camera: CameraUniform;

struct VertexInput {
    [[location(0)]] position: vec3<f32>;
    [[location(1)]] normal: vec3<f32>;
    [[location(2)]] texture_coords: vec2<f32>;
};

struct VertexOutput {
    [[builtin(position)]] clip_position: vec4<f32>;
    [[location(0)]] color: vec3<f32>;
    [[location(1)]] texture_coords: vec2<f32>;
};

//struct InstanceInput {
//    [[location(5)]] model_matrix_0: vec4<f32>;
//    [[location(6)]] model_matrix_1: vec4<f32>;
//    [[location(7)]] model_matrix_2: vec4<f32>;
//    [[location(8)]] model_matrix_3: vec4<f32>;
//    [[location(9)]] color: vec3<f32>;
//};
struct InstanceInput {
    [[location(5)]] pos: vec3<f32>;
    [[location(6)]] color: vec3<f32>;
    [[location(7)]] texture_coords_0: vec2<f32>;
    [[location(8)]] texture_coords_1: vec2<f32>;
    [[location(9)]] texture_coords_2: vec2<f32>;
    [[location(10)]] texture_coords_3: vec2<f32>;
    [[location(11)]] texture_coords_4: vec2<f32>;
    [[location(12)]] texture_coords_5: vec2<f32>;
};

[[stage(vertex)]]
fn vs_main(
    model: VertexInput,
    instance: InstanceInput,
) -> VertexOutput {
    // let model_matrix = mat4x4<f32>(
    //     instance.model_matrix_0,
    //     instance.model_matrix_1,
    //     instance.model_matrix_2,
    //     instance.model_matrix_3,
    // );
    var out: VertexOutput;
    var multiplier = 1.0;
    var texture_coords = instance.texture_coords_0;
    if (model.normal.x == 0.0 && model.normal.y == 1.0 && model.normal.z == 0.0) {
        multiplier = 1.0;
        texture_coords = instance.texture_coords_5;
    } else if (model.normal.x == 0.0 && model.normal.y == 0.0 && model.normal.z == 1.0) {
        multiplier = 0.8;
        texture_coords = instance.texture_coords_1;
    } else if (model.normal.x == 0.0 && model.normal.y == 0.0 && model.normal.z == -1.0) { 
        multiplier = 0.8;
        texture_coords = instance.texture_coords_0;
    } else if (model.normal.x == 1.0 && model.normal.y == 0.0 && model.normal.z == 0.0) {
        multiplier = 0.6;
        texture_coords = instance.texture_coords_3;
    } else if ((model.normal.x == -1.0 && model.normal.y == 0.0 && model.normal.z == 0.0)) {
        multiplier = 0.6;
        texture_coords = instance.texture_coords_2;
    } else {
        multiplier = 0.5;
        texture_coords = instance.texture_coords_4;
    }
    out.color = instance.color*multiplier;
    out.clip_position = camera.view_proj * vec4<f32>(model.position+instance.pos, 1.0);
    out.texture_coords = (model.texture_coords+texture_coords) / 16.0;
    return out;
}
 
// Fragment shader

[[group(0), binding(0)]]
var t_diffuse: texture_2d<f32>;
[[group(0), binding(1)]]
var s_diffuse: sampler;

[[stage(fragment)]]
fn fs_main(in: VertexOutput) -> [[location(0)]] vec4<f32> {
    return vec4<f32>(in.color, 1.0)*textureSample(t_diffuse, s_diffuse, in.texture_coords);
    //return vec4<f32>(in.texture_coords, 1.0, 1.0);
}