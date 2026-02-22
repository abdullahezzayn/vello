// Copyright 2024 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Snapshot tests using the test scenes from [`scenes`].

use std::sync::{Mutex, OnceLock};

use scenes::{ExampleScene, test_scenes};
use vello_tests::{TestParams, encode_test_scene, snapshot_test_sync};

/// Make sure the CPU and GPU renderers match on the test scenes
fn snapshot_test_scene(test_scene: ExampleScene, params: TestParams) {
    snapshot_test_scene_with_threshold(test_scene, params, 0.0095);
}

fn snapshot_test_scene_with_threshold(
    test_scene: ExampleScene,
    mut params: TestParams,
    mean_threshold: f32,
) {
    let scene = encode_test_scene(test_scene, &mut params);
    snapshot_test_sync(scene, &params)
        .unwrap()
        .assert_mean_less_than(mean_threshold);
}

fn backdrop_snapshot_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

fn snapshot_backdrop_scene_with_threshold(
    test_scene: ExampleScene,
    params: TestParams,
    mean_threshold: f32,
) {
    let _lock = backdrop_snapshot_lock()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    snapshot_test_scene_with_threshold(test_scene, params, mean_threshold);
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn snapshot_splash() {
    let test_scene = test_scenes::splash_with_tiger();
    let params = TestParams::new("splash", 300, 300);

    snapshot_test_scene(test_scene, params);
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn snapshot_funky_paths() {
    let test_scene = test_scenes::funky_paths();
    let params = TestParams::new("funky_paths", 600, 600);

    snapshot_test_scene(test_scene, params);
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn snapshot_stroke_styles() {
    let test_scene = test_scenes::stroke_styles();
    let params = TestParams::new("stroke_styles", 600, 425);

    snapshot_test_scene(test_scene, params);
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn snapshot_stroke_styles_non_uniform() {
    let test_scene = test_scenes::stroke_styles_non_uniform();
    let params = TestParams::new("stroke_styles_non_uniform", 600, 425);

    snapshot_test_scene(test_scene, params);
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn snapshot_stroke_styles_skew() {
    let test_scene = test_scenes::stroke_styles_skew();
    let params = TestParams::new("stroke_styles_skew", 600, 425);
    snapshot_test_scene(test_scene, params);
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn snapshot_tricky_strokes() {
    let test_scene = test_scenes::tricky_strokes();
    let params = TestParams::new("tricky_strokes", 600, 425);
    snapshot_test_scene(test_scene, params);
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn snapshot_fill_types() {
    let test_scene = test_scenes::fill_types();
    let params = TestParams::new("fill_types", 700, 350);
    snapshot_test_scene(test_scene, params);
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn snapshot_deep_blend() {
    let test_scene = test_scenes::deep_blend();
    let params = TestParams::new("deep_blend", 200, 200);
    snapshot_test_scene(test_scene, params);
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn snapshot_gradient_extend() {
    let test_scene = test_scenes::gradient_extend();
    let params = TestParams::new("gradient_extend", 200, 200);
    snapshot_test_scene(test_scene, params);
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn snapshot_many_clips() {
    let test_scene = test_scenes::many_clips();
    let params = TestParams::new("many_clips", 200, 200);
    snapshot_test_scene(test_scene, params);
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn snapshot_clip_test() {
    let test_scene = test_scenes::clip_test();
    let params = TestParams::new("clip_test", 512, 768);
    snapshot_test_scene(test_scene, params);
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn snapshot_blurred_rounded_rect() {
    let test_scene = test_scenes::blurred_rounded_rect();
    let params = TestParams::new("blurred_rounded_rect", 400, 400);
    snapshot_test_scene(test_scene, params);
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn snapshot_backdrop_blur_suite() {
    snapshot_backdrop_scene_with_threshold(
        test_scenes::backdrop_blur_basic(),
        TestParams::new("backdrop_blur_basic", 460, 310),
        0.0095,
    );
    snapshot_backdrop_scene_with_threshold(
        test_scenes::backdrop_blur_nested(),
        TestParams::new("backdrop_blur_nested", 460, 340),
        0.02,
    );
    snapshot_backdrop_scene_with_threshold(
        test_scenes::backdrop_blur_overlap_transform(),
        TestParams::new("backdrop_blur_overlap_transform", 460, 310),
        0.03,
    );
    snapshot_backdrop_scene_with_threshold(
        test_scenes::backdrop_blur_clipped_layer(),
        TestParams::new("backdrop_blur_clipped_layer", 460, 310),
        0.03,
    );
    snapshot_backdrop_scene_with_threshold(
        test_scenes::backdrop_blur_clipped_nested(),
        TestParams::new("backdrop_blur_clipped_nested", 460, 320),
        0.035,
    );
    snapshot_backdrop_scene_with_threshold(
        test_scenes::backdrop_blur_layered_nested(),
        TestParams::new("backdrop_blur_layered_nested", 460, 320),
        0.035,
    );
    snapshot_backdrop_scene_with_threshold(
        test_scenes::backdrop_blur_layered_nested_opaque(),
        TestParams::new("backdrop_blur_layered_nested_opaque", 460, 320),
        0.04,
    );
    snapshot_backdrop_scene_with_threshold(
        test_scenes::backdrop_blur_layered_nested_src_over_alpha_sparse(),
        TestParams::new("backdrop_blur_layered_nested_src_over_alpha_sparse", 460, 320),
        0.04,
    );
    snapshot_backdrop_scene_with_threshold(
        test_scenes::backdrop_blur_single_layer_active(),
        TestParams::new("backdrop_blur_single_layer_active", 460, 320),
        0.04,
    );
    snapshot_backdrop_scene_with_threshold(
        test_scenes::backdrop_blur_blend_multiply_layer(),
        TestParams::new("backdrop_blur_blend_multiply_layer", 460, 320),
        0.04,
    );
    snapshot_backdrop_scene_with_threshold(
        test_scenes::backdrop_blur_blend_multiply_sparse(),
        TestParams::new("backdrop_blur_blend_multiply_sparse", 460, 320),
        0.04,
    );
    snapshot_backdrop_scene_with_threshold(
        test_scenes::backdrop_blur_blend_multiply_post_pop_paint(),
        TestParams::new("backdrop_blur_blend_multiply_post_pop_paint", 460, 320),
        0.04,
    );
    snapshot_backdrop_scene_with_threshold(
        test_scenes::backdrop_blur_blend_multiply_transparent_span(),
        TestParams::new("backdrop_blur_blend_multiply_transparent_span", 460, 320),
        0.04,
    );
    snapshot_backdrop_scene_with_threshold(
        test_scenes::backdrop_blur_blend_multiply_closed_span(),
        TestParams::new("backdrop_blur_blend_multiply_closed_span", 460, 320),
        0.04,
    );
    snapshot_backdrop_scene_with_threshold(
        test_scenes::backdrop_blur_blend_multiply_zero_alpha_layer(),
        TestParams::new("backdrop_blur_blend_multiply_zero_alpha_layer", 460, 320),
        0.04,
    );
    snapshot_backdrop_scene_with_threshold(
        test_scenes::backdrop_blur_luminance_mask_layer(),
        TestParams::new("backdrop_blur_luminance_mask_layer", 460, 320),
        0.04,
    );
    snapshot_backdrop_scene_with_threshold(
        test_scenes::backdrop_blur_luminance_mask_post_pop_paint(),
        TestParams::new("backdrop_blur_luminance_mask_post_pop_paint", 460, 320),
        0.04,
    );
    snapshot_backdrop_scene_with_threshold(
        test_scenes::backdrop_blur_prefixed_blend_layer(),
        TestParams::new("backdrop_blur_prefixed_blend_layer", 460, 320),
        0.04,
    );
    snapshot_backdrop_scene_with_threshold(
        test_scenes::backdrop_blur_blend_multiply_tail_active_paint(),
        TestParams::new("backdrop_blur_blend_multiply_tail_active_paint", 460, 320),
        0.04,
    );
    snapshot_backdrop_scene_with_threshold(
        test_scenes::backdrop_blur_blend_multiply_tail_active_after_first_backdrop(),
        TestParams::new(
            "backdrop_blur_blend_multiply_tail_active_after_first_backdrop",
            460,
            320,
        ),
        0.04,
    );
    snapshot_backdrop_scene_with_threshold(
        test_scenes::backdrop_blur_blend_multiply_intermediate_active_after_first_backdrop(),
        TestParams::new(
            "backdrop_blur_blend_multiply_intermediate_active_after_first_backdrop",
            460,
            320,
        ),
        0.04,
    );
    snapshot_backdrop_scene_with_threshold(
        test_scenes::backdrop_blur_blend_multiply_intermediate_active_prefirst(),
        TestParams::new(
            "backdrop_blur_blend_multiply_intermediate_active_prefirst",
            460,
            320,
        ),
        0.04,
    );
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn snapshot_longpathdash_butt() {
    let test_scene = test_scenes::longpathdash_butt();
    let params = TestParams::new("longpathdash_butt", 440, 80);
    snapshot_test_scene(test_scene, params);
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn snapshot_image_sampling() {
    let test_scene = test_scenes::image_sampling();
    let params = TestParams::new("image_sampling", 400, 400);
    snapshot_test_scene(test_scene, params);
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn snapshot_image_extend_modes_bilinear() {
    let test_scene = test_scenes::image_extend_modes_bilinear();
    let params = TestParams::new("image_extend_modes_bilinear", 400, 400);
    snapshot_test_scene(test_scene, params);
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn snapshot_image_extend_modes_nearest_neighbor() {
    let test_scene = test_scenes::image_extend_modes_nearest_neighbor();
    let params = TestParams::new("image_extend_modes_nearest_neighbor", 400, 400);
    snapshot_test_scene(test_scene, params);
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn snapshot_luminance_mask() {
    let test_scene = test_scenes::luminance_mask();
    // This has been manually validated to match the example in
    // https://developer.mozilla.org/en-US/docs/Web/SVG/Reference/Attribute/mask-type
    let params = TestParams::new("luminance_mask", 55, 55);
    snapshot_test_scene(test_scene, params);
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn image_luminance_mask() {
    let test_scene = test_scenes::image_luminance_mask();
    let params = TestParams::new("image_luminance_mask", 350, 250);
    snapshot_test_scene(test_scene, params);
}
