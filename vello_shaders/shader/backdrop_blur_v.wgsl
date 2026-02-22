// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

struct BackdropBlurParams {
    frame_size: vec2<u32>,
    radius: u32,
    _pad0: u32,
    // x, y, width, height
    roi: vec4<u32>,
    sigma: f32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0)
var<uniform> params: BackdropBlurParams;

@group(0) @binding(1)
var src: texture_2d<f32>;

@group(0) @binding(2)
var dst: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.roi.z || gid.y >= params.roi.w {
        return;
    }
    let p = vec2<i32>(i32(params.roi.x + gid.x), i32(params.roi.y + gid.y));
    if p.x >= i32(params.frame_size.x) || p.y >= i32(params.frame_size.y) {
        return;
    }
    if params.radius == 0u || params.sigma <= 1.0e-6 {
        textureStore(dst, p, textureLoad(src, p, 0));
        return;
    }

    let sigma2 = params.sigma * params.sigma;
    let inv_two_sigma2 = 0.5 / sigma2;
    let radius = i32(params.radius);
    let min_y = i32(params.roi.y);
    let max_y = i32(params.roi.y + params.roi.w) - 1;
    var accum = vec4<f32>(0.0);
    var wsum = 0.0;

    for (var i = -radius; i <= radius; i += 1) {
        let sy = clamp(p.y + i, min_y, max_y);
        let sample = textureLoad(src, vec2<i32>(p.x, sy), 0);
        let x = f32(i);
        let w = exp(-(x * x) * inv_two_sigma2);
        accum += sample * w;
        wsum += w;
    }

    textureStore(dst, p, accum / max(wsum, 1.0e-6));
}
