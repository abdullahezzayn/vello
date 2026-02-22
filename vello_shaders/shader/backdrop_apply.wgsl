// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

struct BackdropApplyParams {
    frame_size: vec2<u32>,
    tint_rgba: u32,
    _pad0: u32,
    // x, y, width, height
    roi: vec4<u32>,
    // [m00, m01, m02, _]
    inv_row0: vec4<f32>,
    // [m10, m11, m12, _]
    inv_row1: vec4<f32>,
    // [center_x, center_y, half_w, half_h]
    rect: vec4<f32>,
    // [corner_radius, edge_softness, _, _]
    radius_soft: vec4<f32>,
}

@group(0) @binding(0)
var<uniform> params: BackdropApplyParams;

@group(0) @binding(1)
var scene_image: texture_2d<f32>;

@group(0) @binding(2)
var blurred_prefix: texture_2d<f32>;

@group(0) @binding(3)
var dst: texture_storage_2d<rgba8unorm, write>;

fn rounded_rect_sdf(local: vec2<f32>, center: vec2<f32>, half_size: vec2<f32>, radius: f32) -> f32 {
    let q = abs(local - center) - (half_size - vec2(radius));
    let outside = length(max(q, vec2(0.0)));
    let inside = min(max(q.x, q.y), 0.0);
    return outside + inside - radius;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.roi.z || gid.y >= params.roi.w {
        return;
    }

    let p_i = vec2<i32>(i32(params.roi.x + gid.x), i32(params.roi.y + gid.y));
    if p_i.x >= i32(params.frame_size.x) || p_i.y >= i32(params.frame_size.y) {
        return;
    }
    let scene = textureLoad(scene_image, p_i, 0);
    let blurred = textureLoad(blurred_prefix, p_i, 0);
    let tint = unpack4x8unorm(params.tint_rgba);
    let material = tint + blurred * (1.0 - tint.a);

    let p = vec2<f32>(f32(p_i.x) + 0.5, f32(p_i.y) + 0.5);
    let local = vec2<f32>(
        params.inv_row0.x * p.x + params.inv_row0.y * p.y + params.inv_row0.z,
        params.inv_row1.x * p.x + params.inv_row1.y * p.y + params.inv_row1.z,
    );
    let center = params.rect.xy;
    let half_size = params.rect.zw;
    let radius = params.radius_soft.x;
    let softness = max(params.radius_soft.y, 1.0e-3);
    let dist = rounded_rect_sdf(local, center, half_size, radius);
    let mask = clamp(0.5 - dist / softness, 0.0, 1.0);
    textureStore(dst, p_i, mix(scene, material, mask));
}
