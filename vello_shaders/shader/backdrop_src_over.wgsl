// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

struct BackdropSrcOverParams {
    frame_size: vec2<u32>,
    _pad0: vec2<u32>,
}

@group(0) @binding(0)
var<uniform> params: BackdropSrcOverParams;

@group(0) @binding(1)
var base_image: texture_2d<f32>;

@group(0) @binding(2)
var overlay_image: texture_2d<f32>;

@group(0) @binding(3)
var dst: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.frame_size.x || gid.y >= params.frame_size.y {
        return;
    }

    let p = vec2<i32>(i32(gid.x), i32(gid.y));
    let base = textureLoad(base_image, p, 0);
    let overlay = textureLoad(overlay_image, p, 0);
    textureStore(dst, p, overlay + base * (1.0 - overlay.a));
}
