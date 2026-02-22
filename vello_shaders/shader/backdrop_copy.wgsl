// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

struct BackdropCopyParams {
    frame_size: vec2<u32>,
    _pad0: vec2<u32>,
}

@group(0) @binding(0)
var<uniform> params: BackdropCopyParams;

@group(0) @binding(1)
var src: texture_2d<f32>;

@group(0) @binding(2)
var dst: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.frame_size.x || gid.y >= params.frame_size.y {
        return;
    }

    let p = vec2<i32>(i32(gid.x), i32(gid.y));
    textureStore(dst, p, textureLoad(src, p, 0));
}
