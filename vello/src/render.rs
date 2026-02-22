// Copyright 2022 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Take an encoded scene and create a graph to render it

use std::sync::Arc;

use crate::recording::{BufferProxy, ImageFormat, ImageProxy, Recording, ResourceProxy};
use crate::shaders::FullShaders;
use crate::{AaConfig, RenderParams};
use peniko::{Blob, ImageAlphaType, ImageData};
#[cfg(feature = "wgpu")]
use peniko::{BlendMode, Color, Compose, Mix, kurbo::Rect};

#[cfg(feature = "wgpu")]
use crate::Scene;

#[cfg(feature = "wgpu")]
use vello_encoding::{
    DrawBeginClip, DrawColor, DrawMonoid, DrawTag, Monoid, PathMonoid, PathTag, Style, Transform,
};
use vello_encoding::{
    Encoding, Layout, Ramps, Resolver, WorkgroupSize, make_mask_lut, make_mask_lut_16,
};

/// State for a render in progress.
pub struct Render {
    fine_wg_count: Option<WorkgroupSize>,
    fine_resources: Option<FineResources>,
    mask_buf: Option<ResourceProxy>,

    #[cfg(feature = "debug_layers")]
    captured_buffers: Option<CapturedBuffers>,
}

#[cfg(feature = "debug_layers")]
impl Drop for Render {
    fn drop(&mut self) {
        if self.captured_buffers.is_some() {
            unreachable!("Render captured buffers without freeing them");
        }
    }
}

/// Resources produced by pipeline, needed for fine rasterization.
struct FineResources {
    aa_config: AaConfig,

    config_buf: ResourceProxy,
    bump_buf: ResourceProxy,
    tile_buf: ResourceProxy,
    segments_buf: ResourceProxy,
    ptcl_buf: ResourceProxy,
    gradient_image: ResourceProxy,
    info_bin_data_buf: ResourceProxy,
    image_atlas: ResourceProxy,
    blend_spill_buf: ResourceProxy,

    out_image: ImageProxy,
}

#[cfg(feature = "wgpu")]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct BackdropBlurParams {
    frame_size: [u32; 2],
    radius: u32,
    _pad0: u32,
    roi: [u32; 4],
    sigma: f32,
    _pad1: [u32; 3],
}

#[cfg(feature = "wgpu")]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct BackdropApplyParams {
    frame_size: [u32; 2],
    tint_rgba: u32,
    _pad0: u32,
    roi: [u32; 4],
    inv_row0: [f32; 4],
    inv_row1: [f32; 4],
    rect: [f32; 4],
    radius_soft: [f32; 4],
}

#[cfg(feature = "wgpu")]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct BackdropSrcOverParams {
    frame_size: [u32; 2],
    _pad0: [u32; 2],
}

#[cfg(feature = "wgpu")]
fn align_up_to(value: usize, multiple: usize) -> usize {
    if multiple == 0 {
        return value;
    }
    value.div_ceil(multiple) * multiple
}

#[cfg(feature = "wgpu")]
fn pack_path_tags(tags: &[PathTag]) -> PathMonoid {
    let mut monoid = PathMonoid::default();
    for chunk in tags.chunks(4) {
        let mut word = 0u32;
        for (i, tag) in chunk.iter().enumerate() {
            word |= (tag.0 as u32) << (i * 8);
        }
        monoid = monoid.combine(&PathMonoid::new(word));
    }
    monoid
}

#[cfg(feature = "wgpu")]
fn draw_prefix_monoid(tags: &[DrawTag]) -> DrawMonoid {
    tags.iter().fold(DrawMonoid::default(), |m, tag| {
        m.combine(&DrawMonoid::new(*tag))
    })
}

#[cfg(feature = "wgpu")]
fn build_packed_prefix(
    full_packed: &[u8],
    full_layout: Layout,
    draw_end: u32,
) -> (Layout, Vec<u8>) {
    let draw_tags = full_layout.draw_tags(full_packed);
    let draw_end = draw_end.min(draw_tags.len() as u32);
    let draw_tags_prefix = &draw_tags[..draw_end as usize];
    let draw_monoid = draw_prefix_monoid(draw_tags_prefix);
    let draw_data_words = draw_monoid.scene_offset as usize;
    let path_count = draw_monoid.path_ix as usize;
    let clip_count = draw_monoid.clip_ix;
    let bin_data_start: u32 = draw_tags_prefix.iter().map(|tag| tag.info_size()).sum();

    let full_path_tags = full_layout.path_tags(full_packed);
    let mut path_tag_end = 0usize;
    let mut seen_paths = 0usize;
    while path_tag_end < full_path_tags.len() && seen_paths < path_count {
        if full_path_tags[path_tag_end] == PathTag::PATH {
            seen_paths += 1;
        }
        path_tag_end += 1;
    }
    let path_tags_prefix = &full_path_tags[..path_tag_end];
    let path_tag_monoid = pack_path_tags(path_tags_prefix);
    let path_data_words = path_tag_monoid.pathseg_offset as usize;
    let transform_count = path_tag_monoid.trans_ix as usize;
    let style_words = path_tag_monoid.style_ix as usize;
    let style_words_per_style = size_of::<Style>() / size_of::<u32>();
    let style_count = style_words / style_words_per_style;

    let full_path_data = full_layout.path_data(full_packed);
    let full_draw_data = full_layout.draw_data(full_packed);
    let full_transforms = full_layout.transforms(full_packed);
    let full_styles = full_layout.styles(full_packed);

    let mut packed = Vec::new();
    let mut layout = Layout::default();
    layout.n_draw_objects = draw_end;
    layout.n_paths = draw_monoid.path_ix;
    layout.n_clips = clip_count;
    layout.bin_data_start = bin_data_start;

    layout.path_tag_base = 0;
    packed.extend_from_slice(bytemuck::cast_slice(path_tags_prefix));
    packed.resize(align_up_to(packed.len(), 4), 0);

    layout.path_data_base = (packed.len() / 4) as u32;
    packed.extend_from_slice(&full_path_data[..path_data_words * 4]);

    layout.draw_tag_base = (packed.len() / 4) as u32;
    packed.extend_from_slice(bytemuck::cast_slice(draw_tags_prefix));

    layout.draw_data_base = (packed.len() / 4) as u32;
    packed.extend_from_slice(bytemuck::cast_slice(&full_draw_data[..draw_data_words]));

    layout.transform_base = (packed.len() / 4) as u32;
    packed.extend_from_slice(bytemuck::cast_slice(&full_transforms[..transform_count]));

    layout.style_base = (packed.len() / 4) as u32;
    packed.extend_from_slice(bytemuck::cast_slice(&full_styles[..style_count]));

    (layout, packed)
}

#[cfg(feature = "wgpu")]
fn path_tag_end_for_path_count(path_tags: &[PathTag], path_count: usize) -> usize {
    if path_count == 0 {
        return 0;
    }
    let mut path_tag_end = 0usize;
    let mut seen_paths = 0usize;
    while path_tag_end < path_tags.len() && seen_paths < path_count {
        if path_tags[path_tag_end] == PathTag::PATH {
            seen_paths += 1;
        }
        path_tag_end += 1;
    }
    path_tag_end
}

#[cfg(feature = "wgpu")]
fn build_packed_contiguous_range(
    full_packed: &[u8],
    full_layout: Layout,
    draw_start: u32,
    draw_end: u32,
) -> (Layout, Vec<u8>) {
    let draw_tags = full_layout.draw_tags(full_packed);
    let draw_start = draw_start.min(draw_tags.len() as u32);
    let draw_end = draw_end.min(draw_tags.len() as u32);
    if draw_end <= draw_start {
        return (Layout::default(), Vec::new());
    }

    let draw_prefix_start = draw_prefix_monoid(&draw_tags[..draw_start as usize]);
    let draw_prefix_end = draw_prefix_monoid(&draw_tags[..draw_end as usize]);
    let draw_tags_range = &draw_tags[draw_start as usize..draw_end as usize];
    let draw_data_start_words = draw_prefix_start.scene_offset as usize;
    let draw_data_end_words = draw_prefix_end.scene_offset as usize;
    let clip_count = draw_prefix_end.clip_ix - draw_prefix_start.clip_ix;
    let bin_data_start: u32 = draw_tags_range.iter().map(|tag| tag.info_size()).sum();

    let path_start = draw_prefix_start.path_ix as usize;
    let path_end = draw_prefix_end.path_ix as usize;
    let full_path_tags = full_layout.path_tags(full_packed);
    let path_tag_start = path_tag_end_for_path_count(full_path_tags, path_start);
    let path_tag_end = path_tag_end_for_path_count(full_path_tags, path_end);
    let path_tags_range = &full_path_tags[path_tag_start..path_tag_end];

    let path_prefix_start = pack_path_tags(&full_path_tags[..path_tag_start]);
    let path_prefix_end = pack_path_tags(&full_path_tags[..path_tag_end]);
    let path_data_start_words = path_prefix_start.pathseg_offset as usize;
    let path_data_end_words = path_prefix_end.pathseg_offset as usize;
    let transform_range_start = path_prefix_start.trans_ix as usize;
    let transform_end = path_prefix_end.trans_ix as usize;
    let transform_seed_start = transform_range_start.saturating_sub(1);
    let style_words_per_style = size_of::<Style>() / size_of::<u32>();
    let style_words_range_start = path_prefix_start.style_ix as usize;
    let style_words_end = path_prefix_end.style_ix as usize;
    let style_words_seed_start = style_words_range_start.saturating_sub(style_words_per_style);
    let style_range_start = style_words_range_start / style_words_per_style;
    let style_seed_start = style_words_seed_start / style_words_per_style;
    let style_end = style_words_end / style_words_per_style;
    let needs_seed_state = path_tag_start > 0;
    let has_seed_state =
        needs_seed_state && transform_seed_start < transform_end && style_seed_start < style_end;
    let transform_start = if has_seed_state {
        transform_seed_start
    } else {
        transform_range_start
    };
    let style_start = if has_seed_state {
        style_seed_start
    } else {
        style_range_start
    };

    let full_path_data = full_layout.path_data(full_packed);
    let full_draw_data = full_layout.draw_data(full_packed);
    let full_transforms = full_layout.transforms(full_packed);
    let full_styles = full_layout.styles(full_packed);

    let mut packed = Vec::new();
    let mut layout = Layout::default();
    layout.n_draw_objects = draw_end - draw_start;
    layout.n_paths = draw_prefix_end.path_ix - draw_prefix_start.path_ix;
    layout.n_clips = clip_count;
    layout.bin_data_start = bin_data_start;

    // Path encoding keeps transform/style as state that carries across draw boundaries.
    // For segmented replay, seed the range with the active pre-range transform/style.
    let mut path_tags_segment = Vec::with_capacity(path_tags_range.len() + 2);
    if has_seed_state {
        path_tags_segment.push(PathTag::TRANSFORM);
        path_tags_segment.push(PathTag::STYLE);
    }
    path_tags_segment.extend_from_slice(path_tags_range);

    layout.path_tag_base = 0;
    packed.extend_from_slice(bytemuck::cast_slice(path_tags_segment.as_slice()));
    packed.resize(align_up_to(packed.len(), 4), 0);

    layout.path_data_base = (packed.len() / 4) as u32;
    packed.extend_from_slice(&full_path_data[path_data_start_words * 4..path_data_end_words * 4]);

    layout.draw_tag_base = (packed.len() / 4) as u32;
    packed.extend_from_slice(bytemuck::cast_slice(draw_tags_range));

    layout.draw_data_base = (packed.len() / 4) as u32;
    packed.extend_from_slice(bytemuck::cast_slice(
        &full_draw_data[draw_data_start_words..draw_data_end_words],
    ));

    layout.transform_base = (packed.len() / 4) as u32;
    packed.extend_from_slice(bytemuck::cast_slice(
        &full_transforms[transform_start..transform_end],
    ));

    layout.style_base = (packed.len() / 4) as u32;
    packed.extend_from_slice(bytemuck::cast_slice(&full_styles[style_start..style_end]));

    (layout, packed)
}

#[cfg(feature = "wgpu")]
fn merge_packed_parts(parts: &[(Layout, Vec<u8>)]) -> (Layout, Vec<u8>) {
    if parts.is_empty() {
        return (Layout::default(), Vec::new());
    }

    let mut path_tags: Vec<PathTag> = Vec::new();
    let mut path_data: Vec<u8> = Vec::new();
    let mut draw_tags: Vec<DrawTag> = Vec::new();
    let mut draw_data: Vec<u32> = Vec::new();
    let mut transforms: Vec<Transform> = Vec::new();
    let mut styles: Vec<Style> = Vec::new();
    let mut layout = Layout::default();

    for (part_layout, part_packed) in parts {
        if part_layout.n_draw_objects == 0 {
            continue;
        }
        path_tags.extend_from_slice(part_layout.path_tags(part_packed));
        path_data.extend_from_slice(part_layout.path_data(part_packed));
        draw_tags.extend_from_slice(part_layout.draw_tags(part_packed));
        draw_data.extend_from_slice(part_layout.draw_data(part_packed));
        transforms.extend_from_slice(part_layout.transforms(part_packed));
        styles.extend_from_slice(part_layout.styles(part_packed));
        layout.n_draw_objects = layout
            .n_draw_objects
            .saturating_add(part_layout.n_draw_objects);
        layout.n_paths = layout.n_paths.saturating_add(part_layout.n_paths);
        layout.n_clips = layout.n_clips.saturating_add(part_layout.n_clips);
        layout.bin_data_start = layout
            .bin_data_start
            .saturating_add(part_layout.bin_data_start);
    }

    let mut packed = Vec::new();
    layout.path_tag_base = 0;
    packed.extend_from_slice(bytemuck::cast_slice(path_tags.as_slice()));
    packed.resize(align_up_to(packed.len(), 4), 0);

    layout.path_data_base = (packed.len() / 4) as u32;
    packed.extend_from_slice(path_data.as_slice());

    layout.draw_tag_base = (packed.len() / 4) as u32;
    packed.extend_from_slice(bytemuck::cast_slice(draw_tags.as_slice()));

    layout.draw_data_base = (packed.len() / 4) as u32;
    packed.extend_from_slice(bytemuck::cast_slice(draw_data.as_slice()));

    layout.transform_base = (packed.len() / 4) as u32;
    packed.extend_from_slice(bytemuck::cast_slice(transforms.as_slice()));

    layout.style_base = (packed.len() / 4) as u32;
    packed.extend_from_slice(bytemuck::cast_slice(styles.as_slice()));

    (layout, packed)
}

#[cfg(feature = "wgpu")]
fn repack_streams(
    mut path_tags: Vec<PathTag>,
    path_data: Vec<u8>,
    draw_tags: Vec<DrawTag>,
    draw_data: Vec<u32>,
    transforms: Vec<Transform>,
    styles: Vec<Style>,
    n_clips: u32,
) -> (Layout, Vec<u8>) {
    if path_tags.len() < draw_tags.len() {
        path_tags.extend(std::iter::repeat_n(
            PathTag::PATH,
            draw_tags.len() - path_tags.len(),
        ));
    }
    let mut packed = Vec::new();
    let mut layout = Layout::default();
    layout.n_draw_objects = draw_tags.len() as u32;
    layout.n_paths = draw_tags.len() as u32;
    layout.n_clips = n_clips;
    layout.bin_data_start = draw_tags.iter().map(|tag| tag.info_size()).sum();

    layout.path_tag_base = 0;
    packed.extend_from_slice(bytemuck::cast_slice(path_tags.as_slice()));
    packed.resize(align_up_to(packed.len(), 4), 0);

    layout.path_data_base = (packed.len() / 4) as u32;
    packed.extend_from_slice(path_data.as_slice());

    layout.draw_tag_base = (packed.len() / 4) as u32;
    packed.extend_from_slice(bytemuck::cast_slice(draw_tags.as_slice()));

    layout.draw_data_base = (packed.len() / 4) as u32;
    packed.extend_from_slice(bytemuck::cast_slice(draw_data.as_slice()));

    layout.transform_base = (packed.len() / 4) as u32;
    packed.extend_from_slice(bytemuck::cast_slice(transforms.as_slice()));

    layout.style_base = (packed.len() / 4) as u32;
    packed.extend_from_slice(bytemuck::cast_slice(styles.as_slice()));

    (layout, packed)
}

#[cfg(feature = "wgpu")]
fn append_synthetic_end_clips(
    layout: Layout,
    packed: Vec<u8>,
    synthetic_end_clip_count: u32,
) -> (Layout, Vec<u8>) {
    if synthetic_end_clip_count == 0 {
        return (layout, packed);
    }
    let mut path_tags = layout.path_tags(&packed).to_vec();
    path_tags.extend(std::iter::repeat_n(
        PathTag::PATH,
        synthetic_end_clip_count as usize,
    ));
    let path_data = layout.path_data(&packed).to_vec();
    let mut draw_tags = layout.draw_tags(&packed).to_vec();
    draw_tags.extend(std::iter::repeat_n(
        DrawTag::END_CLIP,
        synthetic_end_clip_count as usize,
    ));
    let draw_data = layout.draw_data(&packed).to_vec();
    let transforms = layout.transforms(&packed).to_vec();
    let styles = layout.styles(&packed).to_vec();
    repack_streams(
        path_tags,
        path_data,
        draw_tags,
        draw_data,
        transforms,
        styles,
        layout.n_clips.saturating_add(synthetic_end_clip_count),
    )
}

#[cfg(feature = "wgpu")]
fn build_packed_range(
    full_packed: &[u8],
    full_layout: Layout,
    draw_start: u32,
    draw_end: u32,
    clip_seed_begin_draws: &[u32],
    synthetic_end_clip_count: u32,
) -> (Layout, Vec<u8>) {
    let (layout, packed) = if clip_seed_begin_draws.is_empty() {
        build_packed_contiguous_range(full_packed, full_layout, draw_start, draw_end)
    } else {
        let mut parts = Vec::with_capacity(clip_seed_begin_draws.len() + 1);
        for draw_ix in clip_seed_begin_draws {
            let start = (*draw_ix).min(full_layout.n_draw_objects);
            let end = start.saturating_add(1).min(full_layout.n_draw_objects);
            if end > start {
                parts.push(build_packed_contiguous_range(
                    full_packed,
                    full_layout,
                    start,
                    end,
                ));
            }
        }
        parts.push(build_packed_contiguous_range(
            full_packed,
            full_layout,
            draw_start,
            draw_end,
        ));
        merge_packed_parts(&parts)
    };
    append_synthetic_end_clips(layout, packed, synthetic_end_clip_count)
}

#[cfg(feature = "wgpu")]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct LayerSeedState {
    begin_draws: Vec<u32>,
    has_active_non_clip: bool,
    active_non_clip_src_over: bool,
    active_non_clip_replay_safe: bool,
    non_replay_safe_non_clip_begin_draws: Vec<u32>,
}

#[cfg(feature = "wgpu")]
fn layer_seed_state_from_stack(stack: &[(u32, bool, bool, bool, bool)]) -> LayerSeedState {
    LayerSeedState {
        begin_draws: stack.iter().map(|(ix, _, _, _, _)| *ix).collect(),
        has_active_non_clip: stack.iter().any(|(_, is_clip, _, _, _)| !*is_clip),
        active_non_clip_src_over: stack
            .iter()
            .filter_map(|(_, is_clip, is_src_over, _, _)| (!*is_clip).then_some(*is_src_over))
            .all(|is_src_over| is_src_over),
        active_non_clip_replay_safe: stack
            .iter()
            .filter_map(|(_, is_clip, _, is_replay_safe, _)| (!*is_clip).then_some(*is_replay_safe))
            .all(|is_replay_safe| is_replay_safe),
        non_replay_safe_non_clip_begin_draws: stack
            .iter()
            .filter_map(|(ix, is_clip, _, _, requires_paint_free)| {
                (!*is_clip && *requires_paint_free).then_some(*ix)
            })
            .collect(),
    }
}

#[cfg(feature = "wgpu")]
fn src_over_begin_clip_blend_mode() -> u32 {
    DrawBeginClip::new(BlendMode::new(Mix::Normal, Compose::SrcOver), 1.0).blend_mode
}

#[cfg(feature = "wgpu")]
fn begin_clip_src_over(draw_data: &[u32], draw_data_offset: usize) -> bool {
    if draw_data_offset.saturating_add(1) >= draw_data.len() {
        return false;
    }
    let blend_mode = draw_data[draw_data_offset];
    blend_mode == src_over_begin_clip_blend_mode()
}

#[cfg(feature = "wgpu")]
fn begin_clip_noop_zero_alpha_non_luminance(draw_data: &[u32], draw_data_offset: usize) -> bool {
    if draw_data_offset.saturating_add(1) >= draw_data.len() {
        return false;
    }
    let blend_mode = draw_data[draw_data_offset];
    if blend_mode == DrawBeginClip::LUMINANCE_MASK_BLEND_MODE {
        return false;
    }
    let alpha = f32::from_bits(draw_data[draw_data_offset + 1]);
    alpha.abs() <= f32::EPSILON
}

#[cfg(feature = "wgpu")]
fn begin_clip_replay_safe(draw_data: &[u32], draw_data_offset: usize) -> bool {
    if !begin_clip_src_over(draw_data, draw_data_offset) {
        return false;
    }
    let alpha = f32::from_bits(draw_data[draw_data_offset + 1]);
    (alpha - 1.0).abs() <= f32::EPSILON
}

#[cfg(feature = "wgpu")]
fn begin_clip_requires_active_span_paint_free(draw_data: &[u32], draw_data_offset: usize) -> bool {
    if draw_data_offset.saturating_add(1) >= draw_data.len() {
        return true;
    }
    if draw_data[draw_data_offset] == DrawBeginClip::CLIP_BLEND_MODE {
        return false;
    }
    !begin_clip_replay_safe(draw_data, draw_data_offset)
        && !begin_clip_noop_zero_alpha_non_luminance(draw_data, draw_data_offset)
}

#[cfg(feature = "wgpu")]
fn layer_seed_states(draw_tags: &[DrawTag], draw_data: &[u32]) -> Option<Vec<LayerSeedState>> {
    let mut scene_offset = 0usize;
    // Track active begin-clip stack while computing boundary state for every draw index.
    let mut stack: Vec<(u32, bool, bool, bool, bool)> = Vec::new();
    let mut states = Vec::with_capacity(draw_tags.len() + 1);
    states.push(layer_seed_state_from_stack(&stack));
    for (ix, tag) in draw_tags.iter().enumerate() {
        if *tag == DrawTag::BEGIN_CLIP {
            if scene_offset >= draw_data.len() {
                return None;
            }
            let is_clip = draw_data[scene_offset] == DrawBeginClip::CLIP_BLEND_MODE;
            let is_src_over = is_clip || begin_clip_src_over(draw_data, scene_offset);
            let is_replay_safe = is_clip || begin_clip_replay_safe(draw_data, scene_offset);
            let requires_paint_free =
                !is_clip && begin_clip_requires_active_span_paint_free(draw_data, scene_offset);
            stack.push((
                ix as u32,
                is_clip,
                is_src_over,
                is_replay_safe,
                requires_paint_free,
            ));
        } else if *tag == DrawTag::END_CLIP {
            stack.pop()?;
        }
        scene_offset = scene_offset.saturating_add(DrawMonoid::new(*tag).scene_offset as usize);
        states.push(layer_seed_state_from_stack(&stack));
    }
    Some(states)
}

#[cfg(feature = "wgpu")]
fn layer_seed_state_at(states: &[LayerSeedState], draw_ix: u32) -> Option<&LayerSeedState> {
    states.get(draw_ix as usize)
}

#[cfg(feature = "wgpu")]
fn post_first_backdrop_boundary_allows_non_replay_safe_painted(
    seed_state: &LayerSeedState,
    first_backdrop_draw_ix: u32,
    allow_multi_backdrop_non_clip_boundary_replay: bool,
    first_range_non_replay_safe_paint_free: bool,
) -> bool {
    allow_multi_backdrop_non_clip_boundary_replay
        && !seed_state.active_non_clip_src_over
        && !seed_state.non_replay_safe_non_clip_begin_draws.is_empty()
        && (seed_state
            .non_replay_safe_non_clip_begin_draws
            .iter()
            .all(|draw_ix| *draw_ix > first_backdrop_draw_ix)
            || first_range_non_replay_safe_paint_free)
}

#[cfg(feature = "wgpu")]
fn allow_prefirst_paint_free_non_replay_safe_boundary(ops_len: usize) -> bool {
    ops_len > 2
}

#[cfg(feature = "wgpu")]
fn first_range_supports_seeded_non_clip_layers(
    draw_tags: &[DrawTag],
    draw_data: &[u32],
    draw_offsets: &[usize],
    draw_end: u32,
    require_src_over_non_clip: bool,
    require_safe_non_clip: bool,
) -> bool {
    let end = (draw_end as usize).min(draw_tags.len());
    if end == 0 {
        return true;
    }
    let mut stack: Vec<(u32, bool, bool, bool)> = Vec::new();
    let mut non_clip_begins = Vec::new();
    for ix in 0..end {
        if draw_tags[ix] == DrawTag::BEGIN_CLIP {
            let dd = draw_offsets[ix];
            if dd >= draw_data.len() {
                return false;
            }
            let is_clip = draw_data[dd] == DrawBeginClip::CLIP_BLEND_MODE;
            let is_src_over = is_clip || begin_clip_src_over(draw_data, dd);
            let is_replay_safe = is_clip || begin_clip_replay_safe(draw_data, dd);
            stack.push((ix as u32, is_clip, is_src_over, is_replay_safe));
            if !is_clip {
                if require_src_over_non_clip && !is_src_over {
                    return false;
                }
                if require_safe_non_clip && !is_replay_safe {
                    return false;
                }
                non_clip_begins.push((ix as u32, is_src_over, is_replay_safe));
            }
        } else if draw_tags[ix] == DrawTag::END_CLIP {
            if stack.pop().is_none() {
                return false;
            }
        }
    }
    if non_clip_begins.is_empty() {
        return true;
    }
    let active_non_clip = stack
        .iter()
        .filter_map(|(ix, is_clip, _, _)| (!*is_clip).then_some(*ix))
        .collect::<std::collections::BTreeSet<_>>();
    non_clip_begins
        .into_iter()
        .all(|(ix, is_src_over, is_replay_safe)| {
            active_non_clip.contains(&ix) && (!require_safe_non_clip || is_replay_safe)
                && (!require_src_over_non_clip || is_src_over)
        })
}

#[cfg(feature = "wgpu")]
fn supports_segment_replay(ops: &[crate::scene::BackdropBlurOp], draw_tags: &[DrawTag]) -> bool {
    ops.iter()
        .all(|op| (op.draw_index as usize) < draw_tags.len())
}

#[cfg(feature = "wgpu")]
fn draw_data_offsets(draw_tags: &[DrawTag]) -> Vec<usize> {
    let mut offsets = Vec::with_capacity(draw_tags.len() + 1);
    offsets.push(0);
    let mut scene_offset = 0usize;
    for tag in draw_tags {
        scene_offset = scene_offset.saturating_add(DrawMonoid::new(*tag).scene_offset as usize);
        offsets.push(scene_offset);
    }
    offsets
}

#[cfg(feature = "wgpu")]
fn replay_range_supports_layers(
    draw_tags: &[DrawTag],
    draw_data: &[u32],
    draw_offsets: &[usize],
    draw_start: u32,
    draw_end: u32,
    allow_non_clip_src_over: bool,
    require_non_clip_replay_safe: bool,
) -> bool {
    let start = (draw_start as usize).min(draw_tags.len());
    let end = (draw_end as usize).min(draw_tags.len());
    if end <= start {
        return true;
    }
    let strict_non_clip = start == 0;
    let mut stack: Vec<(bool, bool)> = Vec::new();
    for ix in start..end {
        if draw_tags[ix] == DrawTag::BEGIN_CLIP {
            let dd = draw_offsets[ix];
            if dd >= draw_data.len() {
                return false;
            }
            let is_clip = draw_data[dd] == DrawBeginClip::CLIP_BLEND_MODE;
            if is_clip {
                stack.push((true, true));
            } else {
                let is_supported = allow_non_clip_src_over
                    && begin_clip_src_over(draw_data, dd)
                    && (!require_non_clip_replay_safe || begin_clip_replay_safe(draw_data, dd));
                if strict_non_clip && !is_supported {
                    return false;
                }
                stack.push((false, is_supported));
            }
        } else if draw_tags[ix] == DrawTag::END_CLIP {
            // End clips can pop layers that were already active before this range.
            // We only track begin-clips created in this range; underflow is benign here.
            if stack.pop().is_none() {
                continue;
            }
        }
    }
    stack
        .into_iter()
        .all(|(is_clip, is_supported)| is_clip || is_supported)
}

#[cfg(feature = "wgpu")]
fn draw_tag_is_effective_paint(
    draw_tags: &[DrawTag],
    draw_data: &[u32],
    draw_offsets: &[usize],
    draw_ix: usize,
) -> bool {
    if draw_ix >= draw_tags.len() || draw_ix >= draw_offsets.len() {
        return false;
    }
    let tag = draw_tags[draw_ix];
    let dd = draw_offsets[draw_ix];
    match tag {
        DrawTag::BEGIN_CLIP | DrawTag::END_CLIP | DrawTag::NOP => false,
        DrawTag::COLOR => {
            if dd >= draw_data.len() {
                return true;
            }
            let rgba = draw_data[dd];
            let alpha = (rgba >> 24) as u8;
            alpha != 0
        }
        DrawTag::BLUR_RECT | DrawTag::BACKDROP_BLUR_RECT => {
            if dd >= draw_data.len() {
                return true;
            }
            let rgba = draw_data[dd];
            let alpha = (rgba >> 24) as u8;
            alpha != 0
        }
        DrawTag::IMAGE => {
            if dd.saturating_add(2) >= draw_data.len() {
                return true;
            }
            let sample_alpha = draw_data[dd + 2];
            let alpha = (sample_alpha & 0xff) as u8;
            alpha != 0
        }
        _ => true,
    }
}

#[cfg(feature = "wgpu")]
fn replay_range_non_replay_safe_layers_paint_free(
    draw_tags: &[DrawTag],
    draw_data: &[u32],
    draw_offsets: &[usize],
    draw_start: u32,
    draw_end: u32,
    seed_state: Option<&LayerSeedState>,
) -> bool {
    let start = (draw_start as usize).min(draw_tags.len());
    let end = (draw_end as usize).min(draw_tags.len());
    if end <= start {
        return true;
    }
    let mut non_replay_safe_stack = Vec::new();
    if let Some(seed_state) = seed_state {
        let seeded_non_replay_safe = seed_state
            .non_replay_safe_non_clip_begin_draws
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>();
        non_replay_safe_stack.extend(
            seed_state
                .begin_draws
                .iter()
                .map(|ix| seeded_non_replay_safe.contains(ix)),
        );
    }
    let mut has_active_non_replay_safe = non_replay_safe_stack.iter().any(|is_non_replay| *is_non_replay);
    for ix in start..end {
        let tag = draw_tags[ix];
        if has_active_non_replay_safe && draw_tag_is_effective_paint(draw_tags, draw_data, draw_offsets, ix)
        {
            return false;
        }
        if tag == DrawTag::BEGIN_CLIP {
            let dd = draw_offsets[ix];
            if dd >= draw_data.len() {
                return false;
            }
            let is_clip = draw_data[dd] == DrawBeginClip::CLIP_BLEND_MODE;
            let is_non_replay_safe_non_clip =
                !is_clip && begin_clip_requires_active_span_paint_free(draw_data, dd);
            non_replay_safe_stack.push(is_non_replay_safe_non_clip);
            has_active_non_replay_safe |= is_non_replay_safe_non_clip;
        } else if tag == DrawTag::END_CLIP {
            let Some(popped) = non_replay_safe_stack.pop() else {
                return false;
            };
            if popped {
                has_active_non_replay_safe =
                    non_replay_safe_stack.iter().any(|is_non_replay| *is_non_replay);
            }
        }
    }
    true
}

#[cfg(feature = "wgpu")]
fn rect_to_roi(rect: Rect, width: u32, height: u32) -> Option<[u32; 4]> {
    let width_f = width as f64;
    let height_f = height as f64;
    let x0 = rect.x0.floor().clamp(0.0, width_f);
    let y0 = rect.y0.floor().clamp(0.0, height_f);
    let x1 = rect.x1.ceil().clamp(0.0, width_f);
    let y1 = rect.y1.ceil().clamp(0.0, height_f);
    if x1 <= x0 || y1 <= y0 {
        return None;
    }
    let x0 = x0 as u32;
    let y0 = y0 as u32;
    let x1 = x1 as u32;
    let y1 = y1 as u32;
    Some([x0, y0, x1 - x0, y1 - y0])
}

#[cfg(feature = "wgpu")]
fn solid_image_data(width: u32, height: u32, color: Color) -> ImageData {
    let mut data = vec![0u8; (width as usize) * (height as usize) * 4];
    let pixel = DrawColor::from(color).rgba.to_le_bytes();
    for chunk in data.chunks_exact_mut(4) {
        chunk.copy_from_slice(&pixel);
    }
    ImageData {
        data: Blob::new(Arc::new(data)),
        format: peniko::ImageFormat::Rgba8,
        width,
        height,
        alpha_type: ImageAlphaType::Alpha,
    }
}

#[cfg(all(test, feature = "wgpu"))]
mod replay_tests {
    use super::{
        append_synthetic_end_clips, draw_data_offsets, draw_tag_is_effective_paint,
        first_range_supports_seeded_non_clip_layers, layer_seed_states,
        post_first_backdrop_boundary_allows_non_replay_safe_painted,
        replay_range_non_replay_safe_layers_paint_free, replay_range_supports_layers, repack_streams,
        src_over_begin_clip_blend_mode, allow_prefirst_paint_free_non_replay_safe_boundary,
    };
    use vello_encoding::{DrawBeginClip, DrawTag, PathTag, Style, Transform};

    #[test]
    fn layer_seed_stack_ignores_closed_non_clip_layers() {
        let draw_tags = [DrawTag::BEGIN_CLIP, DrawTag::END_CLIP];
        let draw_data = [DrawBeginClip::CLIP_BLEND_MODE + 1, 0];
        let states = layer_seed_states(&draw_tags, &draw_data).unwrap();
        let seed = &states[2];
        assert_eq!(seed.begin_draws, Vec::<u32>::new());
        assert!(!seed.has_active_non_clip);
    }

    #[test]
    fn layer_seed_stack_includes_active_non_clip_layers() {
        let draw_tags = [DrawTag::BEGIN_CLIP];
        let draw_data = [DrawBeginClip::CLIP_BLEND_MODE + 1, 0.6f32.to_bits()];
        let states = layer_seed_states(&draw_tags, &draw_data).unwrap();
        let seed = &states[1];
        assert_eq!(seed.begin_draws, vec![0]);
        assert!(seed.has_active_non_clip);
        assert!(!seed.active_non_clip_src_over);
        assert!(!seed.active_non_clip_replay_safe);
        assert_eq!(seed.non_replay_safe_non_clip_begin_draws, vec![0]);
    }

    #[test]
    fn layer_seed_stack_marks_src_over_active_non_clip_layers() {
        let draw_tags = [DrawTag::BEGIN_CLIP];
        let draw_data = [src_over_begin_clip_blend_mode(), 0.72f32.to_bits()];
        let states = layer_seed_states(&draw_tags, &draw_data).unwrap();
        let seed = &states[1];
        assert_eq!(seed.begin_draws, vec![0]);
        assert!(seed.has_active_non_clip);
        assert!(seed.active_non_clip_src_over);
        assert!(!seed.active_non_clip_replay_safe);
        assert_eq!(seed.non_replay_safe_non_clip_begin_draws, vec![0]);
    }

    #[test]
    fn layer_seed_stack_treats_zero_alpha_src_over_as_non_contributing() {
        let draw_tags = [DrawTag::BEGIN_CLIP];
        let draw_data = [src_over_begin_clip_blend_mode(), 0.0f32.to_bits()];
        let states = layer_seed_states(&draw_tags, &draw_data).unwrap();
        let seed = &states[1];
        assert_eq!(seed.begin_draws, vec![0]);
        assert!(seed.has_active_non_clip);
        assert!(seed.active_non_clip_src_over);
        assert!(!seed.active_non_clip_replay_safe);
        assert_eq!(seed.non_replay_safe_non_clip_begin_draws, Vec::<u32>::new());
    }

    #[test]
    fn layer_seed_stack_treats_zero_alpha_non_luminance_as_non_contributing() {
        let draw_tags = [DrawTag::BEGIN_CLIP];
        let draw_data = [
            DrawBeginClip::new(
                super::BlendMode::new(super::Mix::Multiply, super::Compose::SrcOver),
                1.0,
            )
            .blend_mode,
            0.0f32.to_bits(),
        ];
        let states = layer_seed_states(&draw_tags, &draw_data).unwrap();
        let seed = &states[1];
        assert_eq!(seed.begin_draws, vec![0]);
        assert!(seed.has_active_non_clip);
        assert!(!seed.active_non_clip_src_over);
        assert!(!seed.active_non_clip_replay_safe);
        assert_eq!(seed.non_replay_safe_non_clip_begin_draws, Vec::<u32>::new());
    }

    #[test]
    fn layer_seed_stack_keeps_zero_alpha_luminance_mask_as_paint_sensitive() {
        let draw_tags = [DrawTag::BEGIN_CLIP];
        let draw_data = [DrawBeginClip::LUMINANCE_MASK_BLEND_MODE, 0.0f32.to_bits()];
        let states = layer_seed_states(&draw_tags, &draw_data).unwrap();
        let seed = &states[1];
        assert_eq!(seed.begin_draws, vec![0]);
        assert!(seed.has_active_non_clip);
        assert!(!seed.active_non_clip_src_over);
        assert!(!seed.active_non_clip_replay_safe);
        assert_eq!(seed.non_replay_safe_non_clip_begin_draws, vec![0]);
    }

    #[test]
    fn layer_seed_stack_marks_replay_safe_active_non_clip_layers() {
        let draw_tags = [DrawTag::BEGIN_CLIP];
        let draw_data = [src_over_begin_clip_blend_mode(), 1.0f32.to_bits()];
        let states = layer_seed_states(&draw_tags, &draw_data).unwrap();
        let seed = &states[1];
        assert_eq!(seed.begin_draws, vec![0]);
        assert!(seed.has_active_non_clip);
        assert!(seed.active_non_clip_src_over);
        assert!(seed.active_non_clip_replay_safe);
        assert_eq!(seed.non_replay_safe_non_clip_begin_draws, Vec::<u32>::new());
    }

    #[test]
    fn layer_seed_stack_returns_active_clip_begin_indices() {
        let draw_tags = [
            DrawTag::BEGIN_CLIP,
            DrawTag::BEGIN_CLIP,
            DrawTag::END_CLIP,
            DrawTag::BEGIN_CLIP,
        ];
        let draw_data = [
            DrawBeginClip::CLIP_BLEND_MODE,
            0,
            DrawBeginClip::CLIP_BLEND_MODE,
            0,
            DrawBeginClip::CLIP_BLEND_MODE,
            0,
        ];
        let states = layer_seed_states(&draw_tags, &draw_data).unwrap();
        let seed = &states[4];
        assert_eq!(seed.begin_draws, vec![0, 3]);
        assert!(!seed.has_active_non_clip);
    }

    #[test]
    fn first_range_supports_active_non_clip_only() {
        let draw_tags = [DrawTag::BEGIN_CLIP, DrawTag::BEGIN_CLIP];
        let draw_data = [
            DrawBeginClip::CLIP_BLEND_MODE + 1,
            0,
            DrawBeginClip::CLIP_BLEND_MODE,
            0,
        ];
        let offsets = draw_data_offsets(&draw_tags);
        assert!(first_range_supports_seeded_non_clip_layers(
            &draw_tags, &draw_data, &offsets, 2, false, false
        ));
        assert!(!first_range_supports_seeded_non_clip_layers(
            &draw_tags, &draw_data, &offsets, 2, true, false
        ));
        assert!(!first_range_supports_seeded_non_clip_layers(
            &draw_tags, &draw_data, &offsets, 2, true, true
        ));
    }

    #[test]
    fn first_range_rejects_closed_non_clip() {
        let draw_tags = [DrawTag::BEGIN_CLIP, DrawTag::END_CLIP];
        let draw_data = [DrawBeginClip::CLIP_BLEND_MODE + 1, 0];
        let offsets = draw_data_offsets(&draw_tags);
        assert!(!first_range_supports_seeded_non_clip_layers(
            &draw_tags, &draw_data, &offsets, 2, false, false
        ));
    }

    #[test]
    fn first_range_supports_src_over_active_non_clip() {
        let draw_tags = [DrawTag::BEGIN_CLIP, DrawTag::BEGIN_CLIP];
        let draw_data = [
            src_over_begin_clip_blend_mode(),
            0.68f32.to_bits(),
            DrawBeginClip::CLIP_BLEND_MODE,
            0,
        ];
        let offsets = draw_data_offsets(&draw_tags);
        assert!(first_range_supports_seeded_non_clip_layers(
            &draw_tags, &draw_data, &offsets, 2, true, false
        ));
        assert!(!first_range_supports_seeded_non_clip_layers(
            &draw_tags, &draw_data, &offsets, 2, true, true
        ));
    }

    #[test]
    fn first_range_supports_replay_safe_active_non_clip() {
        let draw_tags = [DrawTag::BEGIN_CLIP, DrawTag::BEGIN_CLIP];
        let draw_data = [
            src_over_begin_clip_blend_mode(),
            1.0f32.to_bits(),
            DrawBeginClip::CLIP_BLEND_MODE,
            0,
        ];
        let offsets = draw_data_offsets(&draw_tags);
        assert!(first_range_supports_seeded_non_clip_layers(
            &draw_tags, &draw_data, &offsets, 2, true, true
        ));
    }

    #[test]
    fn replay_range_supports_src_over_non_clip_layers() {
        let draw_tags = [DrawTag::BEGIN_CLIP];
        let draw_data = [src_over_begin_clip_blend_mode(), 0.7f32.to_bits()];
        let offsets = draw_data_offsets(&draw_tags);
        assert!(replay_range_supports_layers(
            &draw_tags, &draw_data, &offsets, 0, 1, true, false
        ));
        assert!(!replay_range_supports_layers(
            &draw_tags, &draw_data, &offsets, 0, 1, true, true
        ));
        assert!(!replay_range_supports_layers(
            &draw_tags, &draw_data, &offsets, 0, 1, false, false
        ));
    }

    #[test]
    fn post_first_backdrop_boundary_allows_non_replay_safe_painted_if_opened_after_first_backdrop()
    {
        let seed = super::LayerSeedState {
            begin_draws: vec![7, 12],
            has_active_non_clip: true,
            active_non_clip_src_over: false,
            active_non_clip_replay_safe: false,
            non_replay_safe_non_clip_begin_draws: vec![12],
        };
        assert!(post_first_backdrop_boundary_allows_non_replay_safe_painted(
            &seed, 8, true, false
        ));
        assert!(!post_first_backdrop_boundary_allows_non_replay_safe_painted(
            &seed, 12, true, false
        ));
        assert!(!post_first_backdrop_boundary_allows_non_replay_safe_painted(
            &seed, 8, false, false
        ));
    }

    #[test]
    fn post_first_backdrop_boundary_rejects_non_replay_safe_painted_if_opened_before_first_backdrop(
    ) {
        let seed = super::LayerSeedState {
            begin_draws: vec![4],
            has_active_non_clip: true,
            active_non_clip_src_over: false,
            active_non_clip_replay_safe: false,
            non_replay_safe_non_clip_begin_draws: vec![4],
        };
        assert!(!post_first_backdrop_boundary_allows_non_replay_safe_painted(
            &seed, 8, true, false
        ));
        assert!(post_first_backdrop_boundary_allows_non_replay_safe_painted(
            &seed, 8, true, true
        ));
    }

    #[test]
    fn post_first_backdrop_boundary_rejects_mixed_non_replay_safe_origins() {
        let seed = super::LayerSeedState {
            begin_draws: vec![4, 12],
            has_active_non_clip: true,
            active_non_clip_src_over: false,
            active_non_clip_replay_safe: false,
            non_replay_safe_non_clip_begin_draws: vec![4, 12],
        };
        assert!(!post_first_backdrop_boundary_allows_non_replay_safe_painted(
            &seed, 8, true, false
        ));
        assert!(post_first_backdrop_boundary_allows_non_replay_safe_painted(
            &seed, 8, true, true
        ));
    }

    #[test]
    fn post_first_backdrop_boundary_rejects_src_over_non_replay_safe_layers() {
        let seed = super::LayerSeedState {
            begin_draws: vec![4],
            has_active_non_clip: true,
            active_non_clip_src_over: true,
            active_non_clip_replay_safe: false,
            non_replay_safe_non_clip_begin_draws: vec![4],
        };
        assert!(!post_first_backdrop_boundary_allows_non_replay_safe_painted(
            &seed, 8, true, true
        ));
    }

    #[test]
    fn prefirst_paint_free_boundary_allowance_requires_three_backdrops() {
        assert!(!allow_prefirst_paint_free_non_replay_safe_boundary(0));
        assert!(!allow_prefirst_paint_free_non_replay_safe_boundary(1));
        assert!(!allow_prefirst_paint_free_non_replay_safe_boundary(2));
        assert!(allow_prefirst_paint_free_non_replay_safe_boundary(3));
        assert!(allow_prefirst_paint_free_non_replay_safe_boundary(4));
    }

    #[test]
    fn replay_range_supports_closed_non_replay_safe_non_clip_layers() {
        let draw_tags = [
            DrawTag::COLOR,
            DrawTag::BEGIN_CLIP,
            DrawTag::COLOR,
            DrawTag::END_CLIP,
        ];
        let draw_data = [
            0xff010203u32,
            DrawBeginClip::CLIP_BLEND_MODE + 1,
            0.84f32.to_bits(),
            0xff000000,
        ];
        let offsets = draw_data_offsets(&draw_tags);
        assert!(replay_range_supports_layers(
            &draw_tags, &draw_data, &offsets, 1, 4, false, false
        ));
        assert!(replay_range_supports_layers(
            &draw_tags, &draw_data, &offsets, 1, 4, true, true
        ));
        assert!(!replay_range_supports_layers(
            &draw_tags, &draw_data, &offsets, 0, 4, false, false
        ));
    }

    #[test]
    fn replay_range_rejects_active_non_replay_safe_non_clip_layers() {
        let draw_tags = [DrawTag::COLOR, DrawTag::BEGIN_CLIP, DrawTag::COLOR];
        let draw_data = [
            0xff010203u32,
            DrawBeginClip::CLIP_BLEND_MODE + 1,
            0.84f32.to_bits(),
            0xff000000,
        ];
        let offsets = draw_data_offsets(&draw_tags);
        assert!(!replay_range_supports_layers(
            &draw_tags, &draw_data, &offsets, 1, 3, false, false
        ));
        assert!(!replay_range_supports_layers(
            &draw_tags, &draw_data, &offsets, 1, 3, true, true
        ));
    }

    #[test]
    fn replay_range_seeded_non_replay_safe_paint_rules() {
        let draw_tags = [DrawTag::END_CLIP, DrawTag::COLOR];
        let draw_data = [0xff000000u32];
        let offsets = draw_data_offsets(&draw_tags);
        let seed = super::LayerSeedState {
            begin_draws: vec![8],
            has_active_non_clip: true,
            active_non_clip_src_over: false,
            active_non_clip_replay_safe: false,
            non_replay_safe_non_clip_begin_draws: vec![8],
        };
        assert!(replay_range_non_replay_safe_layers_paint_free(
            &draw_tags,
            &draw_data,
            &offsets,
            0,
            2,
            Some(&seed),
        ));

        let draw_tags = [DrawTag::COLOR, DrawTag::END_CLIP];
        let offsets = draw_data_offsets(&draw_tags);
        assert!(!replay_range_non_replay_safe_layers_paint_free(
            &draw_tags,
            &draw_data,
            &offsets,
            0,
            2,
            Some(&seed),
        ));

        let draw_tags = [DrawTag::BEGIN_CLIP, DrawTag::END_CLIP, DrawTag::COLOR];
        let draw_data = [DrawBeginClip::CLIP_BLEND_MODE + 1, 0.6f32.to_bits(), 0xff000000];
        let offsets = draw_data_offsets(&draw_tags);
        assert!(replay_range_non_replay_safe_layers_paint_free(
            &draw_tags,
            &draw_data,
            &offsets,
            0,
            3,
            None,
        ));

        let draw_tags = [DrawTag::BEGIN_CLIP, DrawTag::COLOR, DrawTag::END_CLIP];
        let offsets = draw_data_offsets(&draw_tags);
        assert!(!replay_range_non_replay_safe_layers_paint_free(
            &draw_tags,
            &draw_data,
            &offsets,
            0,
            3,
            None,
        ));
    }

    #[test]
    fn replay_range_luminance_mask_post_pop_paint_is_supported() {
        let draw_tags = [DrawTag::BEGIN_CLIP, DrawTag::END_CLIP, DrawTag::COLOR];
        let draw_data = [
            DrawBeginClip::LUMINANCE_MASK_BLEND_MODE,
            1.0f32.to_bits(),
            0xff000000,
        ];
        let offsets = draw_data_offsets(&draw_tags);
        assert!(replay_range_non_replay_safe_layers_paint_free(
            &draw_tags,
            &draw_data,
            &offsets,
            0,
            3,
            None,
        ));

        let draw_tags = [DrawTag::BEGIN_CLIP, DrawTag::COLOR, DrawTag::END_CLIP];
        let offsets = draw_data_offsets(&draw_tags);
        assert!(!replay_range_non_replay_safe_layers_paint_free(
            &draw_tags,
            &draw_data,
            &offsets,
            0,
            3,
            None,
        ));
    }

    #[test]
    fn replay_range_allows_strictly_transparent_paint_while_non_replay_safe_active() {
        let draw_tags = [DrawTag::BEGIN_CLIP, DrawTag::COLOR, DrawTag::END_CLIP];
        let transparent = 0x00ffffffu32;
        let draw_data = [DrawBeginClip::CLIP_BLEND_MODE + 1, 0.7f32.to_bits(), transparent];
        let offsets = draw_data_offsets(&draw_tags);
        assert!(!draw_tag_is_effective_paint(&draw_tags, &draw_data, &offsets, 0));
        assert!(!draw_tag_is_effective_paint(&draw_tags, &draw_data, &offsets, 1));
        assert!(replay_range_non_replay_safe_layers_paint_free(
            &draw_tags,
            &draw_data,
            &offsets,
            0,
            3,
            None,
        ));
    }

    #[test]
    fn replay_range_allows_paint_inside_zero_alpha_src_over_layer() {
        let draw_tags = [DrawTag::BEGIN_CLIP, DrawTag::COLOR, DrawTag::END_CLIP];
        let opaque = 0xff2a2a2au32;
        let draw_data = [src_over_begin_clip_blend_mode(), 0.0f32.to_bits(), opaque];
        let offsets = draw_data_offsets(&draw_tags);
        assert!(draw_tag_is_effective_paint(&draw_tags, &draw_data, &offsets, 1));
        assert!(replay_range_non_replay_safe_layers_paint_free(
            &draw_tags,
            &draw_data,
            &offsets,
            0,
            3,
            None,
        ));
    }

    #[test]
    fn replay_range_allows_paint_inside_zero_alpha_non_luminance_layer() {
        let draw_tags = [DrawTag::BEGIN_CLIP, DrawTag::COLOR, DrawTag::END_CLIP];
        let opaque = 0xff2a2a2au32;
        let draw_data = [
            DrawBeginClip::new(
                super::BlendMode::new(super::Mix::Multiply, super::Compose::SrcOver),
                1.0,
            )
            .blend_mode,
            0.0f32.to_bits(),
            opaque,
        ];
        let offsets = draw_data_offsets(&draw_tags);
        assert!(draw_tag_is_effective_paint(&draw_tags, &draw_data, &offsets, 1));
        assert!(replay_range_non_replay_safe_layers_paint_free(
            &draw_tags,
            &draw_data,
            &offsets,
            0,
            3,
            None,
        ));
    }

    #[test]
    fn replay_range_rejects_paint_inside_zero_alpha_luminance_mask_layer() {
        let draw_tags = [DrawTag::BEGIN_CLIP, DrawTag::COLOR, DrawTag::END_CLIP];
        let opaque = 0xff2a2a2au32;
        let draw_data = [
            DrawBeginClip::LUMINANCE_MASK_BLEND_MODE,
            0.0f32.to_bits(),
            opaque,
        ];
        let offsets = draw_data_offsets(&draw_tags);
        assert!(draw_tag_is_effective_paint(&draw_tags, &draw_data, &offsets, 1));
        assert!(!replay_range_non_replay_safe_layers_paint_free(
            &draw_tags,
            &draw_data,
            &offsets,
            0,
            3,
            None,
        ));
    }

    #[test]
    fn synthetic_end_clips_extend_draw_and_path_streams() {
        let (layout, packed) = repack_streams(
            vec![PathTag::PATH],
            Vec::new(),
            vec![DrawTag::BEGIN_CLIP],
            vec![DrawBeginClip::CLIP_BLEND_MODE, 0],
            vec![Transform::IDENTITY],
            vec![Style::default()],
            1,
        );
        let (extended_layout, extended_packed) = append_synthetic_end_clips(layout, packed, 1);
        assert_eq!(extended_layout.n_draw_objects, 2);
        assert_eq!(extended_layout.n_paths, 2);
        assert_eq!(extended_layout.n_clips, 2);
        assert!(extended_layout.draw_tags(&extended_packed) == [DrawTag::BEGIN_CLIP, DrawTag::END_CLIP]);
        assert!(extended_layout.path_tags(&extended_packed).len() >= 2);
        assert_eq!(
            extended_layout.draw_data(&extended_packed),
            &[DrawBeginClip::CLIP_BLEND_MODE, 0]
        );
    }
}

/// A collection of internal buffers that are used for debug visualization when the
/// `debug_layers` feature is enabled. The contents of these buffers remain GPU resident
/// and must be freed directly by the caller.
///
/// Some of these buffers are also scheduled for a download to allow their contents to be
/// processed for CPU-side validation. These buffers are documented as such.
#[cfg(feature = "debug_layers")]
pub struct CapturedBuffers {
    pub sizes: vello_encoding::BufferSizes,

    /// Buffers that remain GPU-only
    pub path_bboxes: BufferProxy,

    /// Buffers scheduled for download
    pub lines: BufferProxy,
}

#[cfg(feature = "debug_layers")]
impl CapturedBuffers {
    pub fn release_buffers(self, recording: &mut Recording) {
        recording.free_buffer(self.path_bboxes);
        recording.free_buffer(self.lines);
    }
}

#[cfg(feature = "wgpu")]
pub(crate) fn render_full(
    scene: &Scene,
    resolver: &mut Resolver,
    shaders: &FullShaders,
    params: &RenderParams,
) -> (Recording, ResourceProxy) {
    if !scene.backdrop_blur_ops().is_empty() {
        return render_scene_with_backdrop(scene, resolver, shaders, params);
    }
    render_encoding_full(scene.encoding(), resolver, shaders, params)
}

#[cfg(feature = "wgpu")]
/// Create a single recording with both coarse and fine render stages.
///
/// This function is not recommended when the scene can be complex, as it does not
/// implement robust dynamic memory.
pub(crate) fn render_encoding_full(
    encoding: &Encoding,
    resolver: &mut Resolver,
    shaders: &FullShaders,
    params: &RenderParams,
) -> (Recording, ResourceProxy) {
    let mut render = Render::new();
    let mut recording = render.render_encoding_coarse(encoding, resolver, shaders, params, false);
    let out_image = render.out_image();
    render.record_fine(shaders, &mut recording);
    (recording, out_image.into())
}

#[cfg(feature = "wgpu")]
fn render_scene_with_backdrop(
    scene: &Scene,
    resolver: &mut Resolver,
    shaders: &FullShaders,
    params: &RenderParams,
) -> (Recording, ResourceProxy) {
    let mut packed = vec![];
    let (layout, ramps, images) = resolver.resolve(scene.encoding(), &mut packed);

    let mut ops = scene
        .backdrop_blur_ops()
        .iter()
        .filter(|op| op.draw_index < layout.n_draw_objects)
        .cloned()
        .collect::<Vec<_>>();
    ops.sort_by_key(|op| op.draw_index);

    let draw_tags = layout.draw_tags(&packed);
    let draw_data = layout.draw_data(&packed);
    let draw_offsets = draw_data_offsets(draw_tags);
    let layer_seed_cache = match layer_seed_states(draw_tags, draw_data) {
        Some(states) => states,
        None => {
            return render_scene_with_backdrop_legacy(
                packed,
                layout,
                ramps,
                images.width,
                images.height,
                images.images,
                &ops,
                shaders,
                params,
            );
        }
    };
    let allow_single_backdrop_non_clip_boundary_replay = ops.len() == 1;
    let allow_multi_backdrop_non_clip_boundary_replay = ops.len() > 1;
    let allow_prefirst_paint_free_intermediate_non_replay_safe =
        allow_multi_backdrop_non_clip_boundary_replay
            && allow_prefirst_paint_free_non_replay_safe_boundary(ops.len());
    let boundary_supports_non_clip = |seed_state: &LayerSeedState,
                                      non_replay_safe_paint_free_while_active: bool,
                                      allow_non_replay_safe_painted: bool| {
        !seed_state.has_active_non_clip
            || allow_single_backdrop_non_clip_boundary_replay
            || (allow_multi_backdrop_non_clip_boundary_replay
                && (seed_state.active_non_clip_replay_safe
                    || non_replay_safe_paint_free_while_active
                    || allow_non_replay_safe_painted))
    };
    let first_end = ops
        .first()
        .map_or(layout.n_draw_objects, |op| op.draw_index);
    let first_range_non_replay_safe_paint_free = replay_range_non_replay_safe_layers_paint_free(
        draw_tags,
        draw_data,
        &draw_offsets,
        0,
        first_end,
        layer_seed_state_at(&layer_seed_cache, 0),
    );
    if !supports_segment_replay(&ops, draw_tags) {
        return render_scene_with_backdrop_legacy(
            packed,
            layout,
            ramps,
            images.width,
            images.height,
            images.images,
            &ops,
            shaders,
            params,
        );
    }
    // Segment replay supports root-level and clip-only nested boundaries by default.
    // For non-clip boundaries:
    // - single-backdrop scenes allow active non-clip carry-over
    // - multi-backdrop scenes require replay-safe non-clip begin-clip parameters when
    //   replayed ranges contain paint while non-replay-safe layers are active.
    //   Non-replay-safe layers are accepted when their active spans in replayed ranges
    //   remain paint-free.
    let first_range_supports_replay = replay_range_supports_layers(
        draw_tags,
        draw_data,
        &draw_offsets,
        0,
        first_end,
        allow_multi_backdrop_non_clip_boundary_replay,
        allow_multi_backdrop_non_clip_boundary_replay,
    ) || (allow_single_backdrop_non_clip_boundary_replay
        && first_range_supports_seeded_non_clip_layers(
            draw_tags,
            draw_data,
            &draw_offsets,
            first_end,
            false,
            false,
        ))
        || (allow_multi_backdrop_non_clip_boundary_replay
            && first_range_non_replay_safe_paint_free
            && first_range_supports_seeded_non_clip_layers(
                draw_tags,
                draw_data,
                &draw_offsets,
                first_end,
                false,
                false,
            ));
    if !first_range_supports_replay {
        return render_scene_with_backdrop_legacy(
            packed,
            layout,
            ramps,
            images.width,
            images.height,
            images.images,
            &ops,
            shaders,
            params,
        );
    }
    let mut replay_cursor = first_end;
    for op in &ops {
        let boundary_range_start = if op.draw_index == first_end {
            0
        } else {
            replay_cursor
        };
        let boundary_range_non_replay_safe_paint_free =
            replay_range_non_replay_safe_layers_paint_free(
                draw_tags,
                draw_data,
                &draw_offsets,
                boundary_range_start,
                op.draw_index,
                layer_seed_state_at(&layer_seed_cache, boundary_range_start),
            );
        let allow_op_boundary_non_replay_safe_painted = layer_seed_state_at(
            &layer_seed_cache,
            op.draw_index,
        )
        .is_some_and(|seed_state| {
            post_first_backdrop_boundary_allows_non_replay_safe_painted(
                seed_state,
                first_end,
                allow_multi_backdrop_non_clip_boundary_replay,
                first_range_non_replay_safe_paint_free
                    && allow_prefirst_paint_free_intermediate_non_replay_safe,
            )
        });
        if op.draw_index > 0 {
            match layer_seed_state_at(&layer_seed_cache, op.draw_index) {
                Some(seed_state)
                    if boundary_supports_non_clip(
                        seed_state,
                        boundary_range_non_replay_safe_paint_free,
                        allow_op_boundary_non_replay_safe_painted,
                    ) => {}
                _ => {
                    return render_scene_with_backdrop_legacy(
                        packed,
                        layout,
                        ramps,
                        images.width,
                        images.height,
                        images.images,
                        &ops,
                        shaders,
                        params,
                    );
                }
            }
        }
        if replay_cursor < op.draw_index {
            let range_supports_replay = replay_range_supports_layers(
                draw_tags,
                draw_data,
                &draw_offsets,
                replay_cursor,
                op.draw_index,
                allow_multi_backdrop_non_clip_boundary_replay,
                allow_multi_backdrop_non_clip_boundary_replay,
            ) || (allow_multi_backdrop_non_clip_boundary_replay
                && boundary_range_non_replay_safe_paint_free);
            if !range_supports_replay {
                return render_scene_with_backdrop_legacy(
                    packed,
                    layout,
                    ramps,
                    images.width,
                    images.height,
                    images.images,
                    &ops,
                    shaders,
                    params,
                );
            }
            if replay_cursor > 0 {
                let allow_cursor_boundary_non_replay_safe_painted = layer_seed_state_at(
                    &layer_seed_cache,
                    replay_cursor,
                )
                .is_some_and(|seed_state| {
                    post_first_backdrop_boundary_allows_non_replay_safe_painted(
                        seed_state,
                        first_end,
                        allow_multi_backdrop_non_clip_boundary_replay,
                        first_range_non_replay_safe_paint_free
                            && allow_prefirst_paint_free_intermediate_non_replay_safe,
                    )
                });
                match layer_seed_state_at(&layer_seed_cache, replay_cursor) {
                    Some(seed_state)
                        if boundary_supports_non_clip(
                            seed_state,
                            boundary_range_non_replay_safe_paint_free,
                            allow_cursor_boundary_non_replay_safe_painted,
                        ) => {}
                    _ => {
                        return render_scene_with_backdrop_legacy(
                            packed,
                            layout,
                            ramps,
                            images.width,
                            images.height,
                            images.images,
                            &ops,
                            shaders,
                            params,
                        );
                    }
                }
            }
        }
        replay_cursor = op.draw_index.saturating_add(1);
    }
    if replay_cursor < layout.n_draw_objects {
        let tail_range_non_replay_safe_paint_free =
            replay_range_non_replay_safe_layers_paint_free(
                draw_tags,
                draw_data,
                &draw_offsets,
                replay_cursor,
                layout.n_draw_objects,
                layer_seed_state_at(&layer_seed_cache, replay_cursor),
            );
        let tail_supports_replay = replay_range_supports_layers(
            draw_tags,
            draw_data,
            &draw_offsets,
            replay_cursor,
            layout.n_draw_objects,
            allow_multi_backdrop_non_clip_boundary_replay,
            allow_multi_backdrop_non_clip_boundary_replay,
        ) || (allow_multi_backdrop_non_clip_boundary_replay
            && tail_range_non_replay_safe_paint_free);
        if !tail_supports_replay {
            return render_scene_with_backdrop_legacy(
                packed,
                layout,
                ramps,
                images.width,
                images.height,
                images.images,
                &ops,
                shaders,
                params,
            );
        }
        if replay_cursor > 0 {
            let allow_tail_non_replay_safe_painted = layer_seed_state_at(&layer_seed_cache, replay_cursor)
                .is_some_and(|seed_state| {
                    post_first_backdrop_boundary_allows_non_replay_safe_painted(
                        seed_state,
                        first_end,
                        allow_multi_backdrop_non_clip_boundary_replay,
                        first_range_non_replay_safe_paint_free
                            && allow_prefirst_paint_free_intermediate_non_replay_safe,
                    )
                });
            match layer_seed_state_at(&layer_seed_cache, replay_cursor) {
                Some(seed_state)
                    if boundary_supports_non_clip(
                        seed_state,
                        tail_range_non_replay_safe_paint_free,
                        allow_tail_non_replay_safe_painted,
                    ) => {}
                _ => {
                    return render_scene_with_backdrop_legacy(
                        packed,
                        layout,
                        ramps,
                        images.width,
                        images.height,
                        images.images,
                        &ops,
                        shaders,
                        params,
                    );
                }
            }
        }
    }

    let mut recording = Recording::default();
    let mut scene_tmp = ImageProxy::new(params.width, params.height, ImageFormat::Rgba8);
    let blur_tmp = ImageProxy::new(params.width, params.height, ImageFormat::Rgba8);
    let blur_image = ImageProxy::new(params.width, params.height, ImageFormat::Rgba8);
    let src_over_params = BackdropSrcOverParams {
        frame_size: [params.width, params.height],
        _pad0: [0; 2],
    };
    let src_over_params_buf = recording.upload_uniform(
        "vello.backdrop_src_over_params",
        bytemuck::bytes_of(&src_over_params),
    );

    let first_synthetic_end_clips = layer_seed_state_at(&layer_seed_cache, first_end).map_or(
        0,
        |seed_state| {
            if seed_state.has_active_non_clip
                && boundary_supports_non_clip(
                    seed_state,
                    first_range_non_replay_safe_paint_free,
                    false,
                )
            {
                seed_state.begin_draws.len() as u32
            } else {
                0
            }
        },
    );
    let (first_layout, first_packed) = build_packed_range(
        &packed,
        layout,
        0,
        first_end,
        &[],
        first_synthetic_end_clips,
    );
    let mut scene_image = render_packed_segment(
        &mut recording,
        first_packed,
        first_layout,
        ramps,
        images.width,
        images.height,
        images.images,
        shaders,
        params,
        params.base_color,
    );

    let mut cursor = first_end;
    for op in &ops {
        if cursor < op.draw_index {
            let range_non_replay_safe_paint_free = replay_range_non_replay_safe_layers_paint_free(
                draw_tags,
                draw_data,
                &draw_offsets,
                cursor,
                op.draw_index,
                layer_seed_state_at(&layer_seed_cache, cursor),
            );
            let layer_seed_begin_draws: &[u32] = if cursor == 0 {
                &[]
            } else {
                layer_seed_state_at(&layer_seed_cache, cursor)
                    .map_or(&[], |seed_state| seed_state.begin_draws.as_slice())
            };
            let allow_op_boundary_non_replay_safe_painted = layer_seed_state_at(
                &layer_seed_cache,
                op.draw_index,
            )
            .is_some_and(|seed_state| {
                post_first_backdrop_boundary_allows_non_replay_safe_painted(
                    seed_state,
                    first_end,
                    allow_multi_backdrop_non_clip_boundary_replay,
                    first_range_non_replay_safe_paint_free
                        && allow_prefirst_paint_free_intermediate_non_replay_safe,
                )
            });
            let (segment_layout, segment_packed) = build_packed_range(
                &packed,
                layout,
                cursor,
                op.draw_index,
                layer_seed_begin_draws,
                layer_seed_state_at(&layer_seed_cache, op.draw_index).map_or(0, |seed_state| {
                    if seed_state.has_active_non_clip
                        && boundary_supports_non_clip(
                            seed_state,
                            range_non_replay_safe_paint_free,
                            allow_op_boundary_non_replay_safe_painted,
                        )
                    {
                        seed_state.begin_draws.len() as u32
                    } else {
                        0
                    }
                }),
            );
            let segment_image = render_packed_segment(
                &mut recording,
                segment_packed,
                segment_layout,
                ramps,
                images.width,
                images.height,
                images.images,
                shaders,
                params,
                Color::from_rgba8(0, 0, 0, 0),
            );
            compose_src_over(
                &mut recording,
                shaders,
                src_over_params_buf,
                params.width,
                params.height,
                scene_image,
                segment_image,
                scene_tmp,
            );
            recording.free_image(segment_image);
            std::mem::swap(&mut scene_image, &mut scene_tmp);
        }
        if apply_backdrop_op(
            &mut recording,
            shaders,
            params,
            src_over_params_buf,
            op,
            scene_image,
            blur_tmp,
            blur_image,
            scene_tmp,
        ) {
            std::mem::swap(&mut scene_image, &mut scene_tmp);
        }
        cursor = op.draw_index.saturating_add(1);
    }

    if cursor < layout.n_draw_objects {
        let layer_seed_begin_draws: &[u32] = if cursor == 0 {
            &[]
        } else {
            layer_seed_state_at(&layer_seed_cache, cursor)
                .map_or(&[], |seed_state| seed_state.begin_draws.as_slice())
        };
        let (tail_layout, tail_packed) = build_packed_range(
            &packed,
            layout,
            cursor,
            layout.n_draw_objects,
            layer_seed_begin_draws,
            0,
        );
        let tail_image = render_packed_segment(
            &mut recording,
            tail_packed,
            tail_layout,
            ramps,
            images.width,
            images.height,
            images.images,
            shaders,
            params,
            Color::from_rgba8(0, 0, 0, 0),
        );
        compose_src_over(
            &mut recording,
            shaders,
            src_over_params_buf,
            params.width,
            params.height,
            scene_image,
            tail_image,
            scene_tmp,
        );
        recording.free_image(tail_image);
        std::mem::swap(&mut scene_image, &mut scene_tmp);
    }

    let final_image = ImageProxy::new(params.width, params.height, ImageFormat::Rgba8);
    copy_image(
        &mut recording,
        shaders,
        src_over_params_buf,
        params.width,
        params.height,
        scene_image,
        final_image,
    );
    if scene_image.id != scene_tmp.id
        && scene_image.id != blur_tmp.id
        && scene_image.id != blur_image.id
    {
        recording.free_image(scene_image);
    }
    recording.free_buffer(src_over_params_buf);
    recording.free_image(blur_tmp);
    recording.free_image(blur_image);
    recording.free_image(scene_tmp);
    (recording, final_image.into())
}

#[cfg(feature = "wgpu")]
#[expect(
    clippy::too_many_arguments,
    reason = "This keeps callsites explicit while backdrop post-processing is phased in."
)]
fn render_packed_segment(
    recording: &mut Recording,
    packed: Vec<u8>,
    layout: Layout,
    ramps: Ramps<'_>,
    image_atlas_width: u32,
    image_atlas_height: u32,
    image_atlas_images: &[(ImageData, u32, u32)],
    shaders: &FullShaders,
    params: &RenderParams,
    base_color: Color,
) -> ImageProxy {
    if layout.n_draw_objects == 0 {
        let image = solid_image_data(params.width, params.height, base_color);
        return recording.upload_image(
            params.width,
            params.height,
            ImageFormat::Rgba8,
            image.data.data(),
        );
    }

    let mut pass = Render::new();
    let segment_params = RenderParams {
        base_color,
        width: params.width,
        height: params.height,
        antialiasing_method: params.antialiasing_method,
    };
    let mut segment_recording = pass.render_packed_coarse(
        packed,
        layout,
        ramps,
        image_atlas_width,
        image_atlas_height,
        image_atlas_images,
        shaders,
        &segment_params,
        false,
    );
    let segment_image = pass.out_image();
    segment_recording.write_image(
        segment_image,
        0,
        0,
        solid_image_data(params.width, params.height, base_color),
    );
    pass.record_fine(shaders, &mut segment_recording);
    recording.commands.append(&mut segment_recording.commands);
    segment_image
}

#[cfg(feature = "wgpu")]
#[expect(
    clippy::too_many_arguments,
    reason = "Compute composition callsites are kept explicit while backdrop support lands."
)]
fn compose_src_over(
    recording: &mut Recording,
    shaders: &FullShaders,
    params_buf: BufferProxy,
    width: u32,
    height: u32,
    base_image: ImageProxy,
    overlay_image: ImageProxy,
    dst_image: ImageProxy,
) {
    let workgroups = (width.div_ceil(8), height.div_ceil(8), 1);
    recording.dispatch(
        shaders.backdrop_src_over,
        workgroups,
        vec![
            ResourceProxy::from(params_buf),
            ResourceProxy::from(base_image),
            ResourceProxy::from(overlay_image),
            ResourceProxy::from(dst_image),
        ],
    );
}

#[cfg(feature = "wgpu")]
fn copy_image(
    recording: &mut Recording,
    shaders: &FullShaders,
    params_buf: BufferProxy,
    width: u32,
    height: u32,
    src_image: ImageProxy,
    dst_image: ImageProxy,
) {
    let workgroups = (width.div_ceil(8), height.div_ceil(8), 1);
    recording.dispatch(
        shaders.backdrop_copy,
        workgroups,
        vec![
            ResourceProxy::from(params_buf),
            ResourceProxy::from(src_image),
            ResourceProxy::from(dst_image),
        ],
    );
}

#[cfg(feature = "wgpu")]
#[expect(
    clippy::too_many_arguments,
    reason = "Keeping backdrop uniforms local makes execution order clear."
)]
fn apply_backdrop_op(
    recording: &mut Recording,
    shaders: &FullShaders,
    params: &RenderParams,
    copy_params_buf: BufferProxy,
    op: &crate::scene::BackdropBlurOp,
    scene_image: ImageProxy,
    blur_tmp: ImageProxy,
    blur_image: ImageProxy,
    scene_tmp: ImageProxy,
) -> bool {
    let backdrop_bounds = op.transform.transform_rect_bbox(op.rect);
    let sigma = op.style.sigma.max(0.0);
    let blur_roi = match rect_to_roi(
        backdrop_bounds.inflate((3.0 * sigma) + 2.0, (3.0 * sigma) + 2.0),
        params.width,
        params.height,
    ) {
        Some(roi) => roi,
        None => return false,
    };
    let apply_roi = match rect_to_roi(
        backdrop_bounds.inflate(2.0, 2.0),
        params.width,
        params.height,
    ) {
        Some(roi) => roi,
        None => return false,
    };

    let sigma_f32 = sigma as f32;
    let radius = (3.0 * sigma_f32).ceil() as u32;
    let blur_params = BackdropBlurParams {
        frame_size: [params.width, params.height],
        radius,
        _pad0: 0,
        roi: blur_roi,
        sigma: sigma_f32.max(1.0e-4),
        _pad1: [0; 3],
    };
    let blur_params_buf = recording.upload_uniform(
        "vello.backdrop_blur_params",
        bytemuck::bytes_of(&blur_params),
    );
    let blur_workgroups = (blur_roi[2].div_ceil(8), blur_roi[3].div_ceil(8), 1);
    recording.dispatch(
        shaders.backdrop_blur_h,
        blur_workgroups,
        vec![
            ResourceProxy::from(blur_params_buf),
            ResourceProxy::from(scene_image),
            ResourceProxy::from(blur_tmp),
        ],
    );
    recording.dispatch(
        shaders.backdrop_blur_v,
        blur_workgroups,
        vec![
            ResourceProxy::from(blur_params_buf),
            ResourceProxy::from(blur_tmp),
            ResourceProxy::from(blur_image),
        ],
    );
    recording.free_buffer(blur_params_buf);

    let inv = op.transform.inverse();
    let [m00, m10, m01, m11, m02, m12] = inv.as_coeffs();
    let center = op.rect.center();
    let half_w = 0.5 * op.rect.width();
    let half_h = 0.5 * op.rect.height();
    let radius = op.radius.max(0.0).min(half_w.min(half_h));
    let tint_rgba = DrawColor::from(op.style.tint).rgba;
    let apply_params = BackdropApplyParams {
        frame_size: [params.width, params.height],
        tint_rgba,
        _pad0: 0,
        roi: apply_roi,
        inv_row0: [m00 as f32, m01 as f32, m02 as f32, 0.0],
        inv_row1: [m10 as f32, m11 as f32, m12 as f32, 0.0],
        rect: [
            center.x as f32,
            center.y as f32,
            half_w as f32,
            half_h as f32,
        ],
        radius_soft: [radius as f32, 1.0, 0.0, 0.0],
    };
    let apply_params_buf = recording.upload_uniform(
        "vello.backdrop_apply_params",
        bytemuck::bytes_of(&apply_params),
    );
    let apply_workgroups = (apply_roi[2].div_ceil(8), apply_roi[3].div_ceil(8), 1);
    copy_image(
        recording,
        shaders,
        copy_params_buf,
        params.width,
        params.height,
        scene_image,
        scene_tmp,
    );
    recording.dispatch(
        shaders.backdrop_apply,
        apply_workgroups,
        vec![
            ResourceProxy::from(apply_params_buf),
            ResourceProxy::from(scene_image),
            ResourceProxy::from(blur_image),
            ResourceProxy::from(scene_tmp),
        ],
    );
    recording.free_buffer(apply_params_buf);
    true
}

#[cfg(feature = "wgpu")]
#[expect(
    clippy::too_many_arguments,
    reason = "Legacy backdrop route is preserved while segment replay compatibility broadens."
)]
fn render_scene_with_backdrop_legacy(
    packed: Vec<u8>,
    layout: Layout,
    ramps: Ramps<'_>,
    image_atlas_width: u32,
    image_atlas_height: u32,
    image_atlas_images: &[(ImageData, u32, u32)],
    ops: &[crate::scene::BackdropBlurOp],
    shaders: &FullShaders,
    params: &RenderParams,
) -> (Recording, ResourceProxy) {
    let mut recording = Recording::default();

    // Render full scene first (phase-1 fallback path still active in the draw tag stream),
    // then replace each backdrop region with a true sampled blur from prefix content.
    let mut full_pass = Render::new();
    let mut full_recording = full_pass.render_packed_coarse(
        packed.clone(),
        layout,
        ramps,
        image_atlas_width,
        image_atlas_height,
        image_atlas_images,
        shaders,
        params,
        false,
    );
    let mut scene_image = full_pass.out_image();
    full_pass.record_fine(shaders, &mut full_recording);
    recording.commands.append(&mut full_recording.commands);

    let blur_tmp = ImageProxy::new(params.width, params.height, ImageFormat::Rgba8);
    let blur_image = ImageProxy::new(params.width, params.height, ImageFormat::Rgba8);
    let mut scene_tmp = ImageProxy::new(params.width, params.height, ImageFormat::Rgba8);
    let src_over_params = BackdropSrcOverParams {
        frame_size: [params.width, params.height],
        _pad0: [0; 2],
    };
    let src_over_params_buf = recording.upload_uniform(
        "vello.backdrop_src_over_params",
        bytemuck::bytes_of(&src_over_params),
    );

    for op in ops {
        let backdrop_bounds = op.transform.transform_rect_bbox(op.rect);
        let sigma = op.style.sigma.max(0.0);
        let blur_roi = match rect_to_roi(
            backdrop_bounds.inflate((3.0 * sigma) + 2.0, (3.0 * sigma) + 2.0),
            params.width,
            params.height,
        ) {
            Some(roi) => roi,
            None => continue,
        };
        let apply_roi = match rect_to_roi(
            backdrop_bounds.inflate(2.0, 2.0),
            params.width,
            params.height,
        ) {
            Some(roi) => roi,
            None => continue,
        };
        let (prefix_layout, prefix_packed) = build_packed_prefix(&packed, layout, op.draw_index);
        let mut prefix_pass = Render::new();
        let mut prefix_recording = prefix_pass.render_packed_coarse(
            prefix_packed,
            prefix_layout,
            ramps,
            image_atlas_width,
            image_atlas_height,
            image_atlas_images,
            shaders,
            params,
            false,
        );
        let prefix_image = prefix_pass.out_image();
        prefix_pass.record_fine(shaders, &mut prefix_recording);
        recording.commands.append(&mut prefix_recording.commands);

        let sigma_f32 = sigma as f32;
        let radius = (3.0 * sigma_f32).ceil() as u32;
        let blur_params = BackdropBlurParams {
            frame_size: [params.width, params.height],
            radius,
            _pad0: 0,
            roi: blur_roi,
            sigma: sigma_f32.max(1.0e-4),
            _pad1: [0; 3],
        };
        let blur_params_buf = recording.upload_uniform(
            "vello.backdrop_blur_params",
            bytemuck::bytes_of(&blur_params),
        );
        let blur_workgroups = (blur_roi[2].div_ceil(8), blur_roi[3].div_ceil(8), 1);
        recording.dispatch(
            shaders.backdrop_blur_h,
            blur_workgroups,
            vec![
                ResourceProxy::from(blur_params_buf),
                ResourceProxy::from(prefix_image),
                ResourceProxy::from(blur_tmp),
            ],
        );
        recording.dispatch(
            shaders.backdrop_blur_v,
            blur_workgroups,
            vec![
                ResourceProxy::from(blur_params_buf),
                ResourceProxy::from(blur_tmp),
                ResourceProxy::from(blur_image),
            ],
        );
        recording.free_buffer(blur_params_buf);

        let inv = op.transform.inverse();
        let [m00, m10, m01, m11, m02, m12] = inv.as_coeffs();
        let center = op.rect.center();
        let half_w = 0.5 * op.rect.width();
        let half_h = 0.5 * op.rect.height();
        let radius = op.radius.max(0.0).min(half_w.min(half_h));
        let tint_rgba = DrawColor::from(op.style.tint).rgba;
        let apply_params = BackdropApplyParams {
            frame_size: [params.width, params.height],
            tint_rgba,
            _pad0: 0,
            roi: apply_roi,
            inv_row0: [m00 as f32, m01 as f32, m02 as f32, 0.0],
            inv_row1: [m10 as f32, m11 as f32, m12 as f32, 0.0],
            rect: [
                center.x as f32,
                center.y as f32,
                half_w as f32,
                half_h as f32,
            ],
            radius_soft: [radius as f32, 1.0, 0.0, 0.0],
        };
        let apply_params_buf = recording.upload_uniform(
            "vello.backdrop_apply_params",
            bytemuck::bytes_of(&apply_params),
        );
        let apply_workgroups = (apply_roi[2].div_ceil(8), apply_roi[3].div_ceil(8), 1);
        copy_image(
            &mut recording,
            shaders,
            src_over_params_buf,
            params.width,
            params.height,
            scene_image,
            scene_tmp,
        );
        recording.dispatch(
            shaders.backdrop_apply,
            apply_workgroups,
            vec![
                ResourceProxy::from(apply_params_buf),
                ResourceProxy::from(scene_image),
                ResourceProxy::from(blur_image),
                ResourceProxy::from(scene_tmp),
            ],
        );
        recording.free_buffer(apply_params_buf);
        recording.free_image(prefix_image);
        std::mem::swap(&mut scene_image, &mut scene_tmp);
    }

    let final_image = ImageProxy::new(params.width, params.height, ImageFormat::Rgba8);
    copy_image(
        &mut recording,
        shaders,
        src_over_params_buf,
        params.width,
        params.height,
        scene_image,
        final_image,
    );
    if scene_image.id != scene_tmp.id
        && scene_image.id != blur_tmp.id
        && scene_image.id != blur_image.id
    {
        recording.free_image(scene_image);
    }
    recording.free_buffer(src_over_params_buf);
    recording.free_image(blur_tmp);
    recording.free_image(blur_image);
    recording.free_image(scene_tmp);
    (recording, final_image.into())
}

impl Default for Render {
    fn default() -> Self {
        Self::new()
    }
}

impl Render {
    pub fn new() -> Self {
        Self {
            fine_wg_count: None,
            fine_resources: None,
            mask_buf: None,
            #[cfg(feature = "debug_layers")]
            captured_buffers: None,
        }
    }

    /// Prepare a recording for the coarse rasterization phase.
    ///
    /// The `robust` parameter controls whether we're preparing for readback
    /// of the atomic bump buffer, for robust dynamic memory.
    pub fn render_encoding_coarse(
        &mut self,
        encoding: &Encoding,
        resolver: &mut Resolver,
        shaders: &FullShaders,
        params: &RenderParams,
        robust: bool,
    ) -> Recording {
        let mut packed = vec![];
        let (layout, ramps, images) = resolver.resolve(encoding, &mut packed);
        self.render_packed_coarse(
            packed,
            layout,
            ramps,
            images.width,
            images.height,
            images.images,
            shaders,
            params,
            robust,
        )
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "This keeps callsites explicit while backdrop post-processing is phased in."
    )]
    pub fn render_packed_coarse(
        &mut self,
        mut packed: Vec<u8>,
        layout: Layout,
        ramps: Ramps<'_>,
        image_atlas_width: u32,
        image_atlas_height: u32,
        image_atlas_images: &[(ImageData, u32, u32)],
        shaders: &FullShaders,
        params: &RenderParams,
        robust: bool,
    ) -> Recording {
        use vello_encoding::RenderConfig;
        let mut recording = Recording::default();
        let gradient_image = if ramps.height == 0 {
            ResourceProxy::new_image(1, 1, ImageFormat::Rgba8)
        } else {
            let data: &[u8] = bytemuck::cast_slice(ramps.data);
            ResourceProxy::Image(recording.upload_image(
                ramps.width,
                ramps.height,
                ImageFormat::Rgba8,
                data,
            ))
        };
        let image_atlas = if image_atlas_images.is_empty() {
            ImageProxy::new(1, 1, ImageFormat::Rgba8)
        } else {
            ImageProxy::new(image_atlas_width, image_atlas_height, ImageFormat::Rgba8)
        };
        for image in image_atlas_images {
            recording.write_image(image_atlas, image.1, image.2, image.0.clone());
        }
        let cpu_config =
            RenderConfig::new(&layout, params.width, params.height, &params.base_color);
        // HACK: The coarse workgroup counts is the number of active bins.
        if (cpu_config.workgroup_counts.coarse.0
            * cpu_config.workgroup_counts.coarse.1
            * cpu_config.workgroup_counts.coarse.2)
            > 256
        {
            log::warn!(
                "Trying to paint too large image. {}x{}.\n\
                See https://github.com/linebender/vello/issues/680 for details",
                params.width,
                params.height
            );
        }
        let buffer_sizes = &cpu_config.buffer_sizes;
        let wg_counts = &cpu_config.workgroup_counts;

        if packed.is_empty() {
            // HACK: wgpu doesn't allow empty buffers, so we make sure that the scene buffer we upload
            // can contain at least one array item.
            // The values passed here should never be read, because the scene size in config
            // is zero.
            packed.resize(size_of::<u32>(), u8::MAX);
        }
        let scene_buf = ResourceProxy::Buffer(recording.upload("vello.scene", packed));
        let config_buf = ResourceProxy::Buffer(
            recording.upload_uniform("vello.config", bytemuck::bytes_of(&cpu_config.gpu)),
        );
        let info_bin_data_buf = ResourceProxy::new_buf(
            buffer_sizes.bin_data.size_in_bytes() as u64,
            "vello.info_bin_data_buf",
        );
        let tile_buf =
            ResourceProxy::new_buf(buffer_sizes.tiles.size_in_bytes().into(), "vello.tile_buf");
        let segments_buf = ResourceProxy::new_buf(
            buffer_sizes.segments.size_in_bytes().into(),
            "vello.segments_buf",
        );
        let ptcl_buf =
            ResourceProxy::new_buf(buffer_sizes.ptcl.size_in_bytes().into(), "vello.ptcl_buf");
        let reduced_buf = ResourceProxy::new_buf(
            buffer_sizes.path_reduced.size_in_bytes().into(),
            "vello.reduced_buf",
        );
        // TODO: really only need pathtag_wgs - 1
        recording.dispatch(
            shaders.pathtag_reduce,
            wg_counts.path_reduce,
            [config_buf, scene_buf, reduced_buf],
        );
        let mut pathtag_parent = reduced_buf;
        let mut large_pathtag_bufs = None;
        let use_large_path_scan = wg_counts.use_large_path_scan && !shaders.pathtag_is_cpu;
        if use_large_path_scan {
            let reduced2_buf = ResourceProxy::new_buf(
                buffer_sizes.path_reduced2.size_in_bytes().into(),
                "vello.reduced2_buf",
            );
            recording.dispatch(
                shaders.pathtag_reduce2,
                wg_counts.path_reduce2,
                [reduced_buf, reduced2_buf],
            );
            let reduced_scan_buf = ResourceProxy::new_buf(
                buffer_sizes.path_reduced_scan.size_in_bytes().into(),
                "reduced_scan_buf",
            );
            recording.dispatch(
                shaders.pathtag_scan1,
                wg_counts.path_scan1,
                [reduced_buf, reduced2_buf, reduced_scan_buf],
            );
            pathtag_parent = reduced_scan_buf;
            large_pathtag_bufs = Some((reduced2_buf, reduced_scan_buf));
        }

        let tagmonoid_buf = ResourceProxy::new_buf(
            buffer_sizes.path_monoids.size_in_bytes().into(),
            "vello.tagmonoid_buf",
        );
        let pathtag_scan = if use_large_path_scan {
            shaders.pathtag_scan_large
        } else {
            shaders.pathtag_scan
        };
        recording.dispatch(
            pathtag_scan,
            wg_counts.path_scan,
            [config_buf, scene_buf, pathtag_parent, tagmonoid_buf],
        );
        recording.free_resource(reduced_buf);
        if let Some((reduced2, reduced_scan)) = large_pathtag_bufs {
            recording.free_resource(reduced2);
            recording.free_resource(reduced_scan);
        }
        let path_bbox_buf = ResourceProxy::new_buf(
            buffer_sizes.path_bboxes.size_in_bytes().into(),
            "vello.path_bbox_buf",
        );
        recording.dispatch(
            shaders.bbox_clear,
            wg_counts.bbox_clear,
            [config_buf, path_bbox_buf],
        );
        let bump_buf = BufferProxy::new(
            buffer_sizes.bump_alloc.size_in_bytes().into(),
            "vello.bump_buf",
        );
        recording.clear_all(bump_buf);
        let bump_buf = ResourceProxy::Buffer(bump_buf);
        let lines_buf =
            ResourceProxy::new_buf(buffer_sizes.lines.size_in_bytes().into(), "vello.lines_buf");
        recording.dispatch(
            shaders.flatten,
            wg_counts.flatten,
            [
                config_buf,
                scene_buf,
                tagmonoid_buf,
                path_bbox_buf,
                bump_buf,
                lines_buf,
            ],
        );
        let draw_reduced_buf = ResourceProxy::new_buf(
            buffer_sizes.draw_reduced.size_in_bytes().into(),
            "vello.draw_reduced_buf",
        );
        recording.dispatch(
            shaders.draw_reduce,
            wg_counts.draw_reduce,
            [config_buf, scene_buf, draw_reduced_buf],
        );
        let draw_monoid_buf = ResourceProxy::new_buf(
            buffer_sizes.draw_monoids.size_in_bytes().into(),
            "vello.draw_monoid_buf",
        );
        let clip_inp_buf = ResourceProxy::new_buf(
            buffer_sizes.clip_inps.size_in_bytes().into(),
            "vello.clip_inp_buf",
        );
        recording.dispatch(
            shaders.draw_leaf,
            wg_counts.draw_leaf,
            [
                config_buf,
                scene_buf,
                draw_reduced_buf,
                path_bbox_buf,
                draw_monoid_buf,
                info_bin_data_buf,
                clip_inp_buf,
            ],
        );
        recording.free_resource(draw_reduced_buf);
        let clip_el_buf = ResourceProxy::new_buf(
            buffer_sizes.clip_els.size_in_bytes().into(),
            "vello.clip_el_buf",
        );
        let clip_bic_buf = ResourceProxy::new_buf(
            buffer_sizes.clip_bics.size_in_bytes().into(),
            "vello.clip_bic_buf",
        );
        if wg_counts.clip_reduce.0 > 0 {
            recording.dispatch(
                shaders.clip_reduce,
                wg_counts.clip_reduce,
                [clip_inp_buf, path_bbox_buf, clip_bic_buf, clip_el_buf],
            );
        }
        let clip_bbox_buf = ResourceProxy::new_buf(
            buffer_sizes.clip_bboxes.size_in_bytes().into(),
            "vello.clip_bbox_buf",
        );
        if wg_counts.clip_leaf.0 > 0 {
            recording.dispatch(
                shaders.clip_leaf,
                wg_counts.clip_leaf,
                [
                    config_buf,
                    clip_inp_buf,
                    path_bbox_buf,
                    clip_bic_buf,
                    clip_el_buf,
                    draw_monoid_buf,
                    clip_bbox_buf,
                ],
            );
        }
        recording.free_resource(clip_inp_buf);
        recording.free_resource(clip_bic_buf);
        recording.free_resource(clip_el_buf);
        let draw_bbox_buf = ResourceProxy::new_buf(
            buffer_sizes.draw_bboxes.size_in_bytes().into(),
            "vello.draw_bbox_buf",
        );
        let bin_header_buf = ResourceProxy::new_buf(
            buffer_sizes.bin_headers.size_in_bytes().into(),
            "vello.bin_header_buf",
        );
        recording.dispatch(
            shaders.binning,
            wg_counts.binning,
            [
                config_buf,
                draw_monoid_buf,
                path_bbox_buf,
                clip_bbox_buf,
                draw_bbox_buf,
                bump_buf,
                info_bin_data_buf,
                bin_header_buf,
            ],
        );
        recording.free_resource(draw_monoid_buf);
        recording.free_resource(clip_bbox_buf);
        // Note: this only needs to be rounded up because of the workaround to store the tile_offset
        // in storage rather than workgroup memory.
        let path_buf =
            ResourceProxy::new_buf(buffer_sizes.paths.size_in_bytes().into(), "vello.path_buf");
        recording.dispatch(
            shaders.tile_alloc,
            wg_counts.tile_alloc,
            [
                config_buf,
                scene_buf,
                draw_bbox_buf,
                bump_buf,
                path_buf,
                tile_buf,
            ],
        );
        recording.free_resource(draw_bbox_buf);
        recording.free_resource(tagmonoid_buf);
        let indirect_count_buf = BufferProxy::new(
            buffer_sizes.indirect_count.size_in_bytes().into(),
            "vello.indirect_count",
        );
        recording.dispatch(
            shaders.path_count_setup,
            wg_counts.path_count_setup,
            [bump_buf, indirect_count_buf.into()],
        );
        let seg_counts_buf = ResourceProxy::new_buf(
            buffer_sizes.seg_counts.size_in_bytes().into(),
            "vello.seg_counts_buf",
        );
        recording.dispatch_indirect(
            shaders.path_count,
            indirect_count_buf,
            0,
            [
                config_buf,
                bump_buf,
                lines_buf,
                path_buf,
                tile_buf,
                seg_counts_buf,
            ],
        );
        recording.dispatch(
            shaders.backdrop,
            wg_counts.backdrop,
            [config_buf, bump_buf, path_buf, tile_buf],
        );
        recording.dispatch(
            shaders.coarse,
            wg_counts.coarse,
            [
                config_buf,
                scene_buf,
                draw_monoid_buf,
                bin_header_buf,
                info_bin_data_buf,
                path_buf,
                tile_buf,
                bump_buf,
                ptcl_buf,
            ],
        );
        recording.dispatch(
            shaders.path_tiling_setup,
            wg_counts.path_tiling_setup,
            [bump_buf, indirect_count_buf.into(), ptcl_buf],
        );
        recording.dispatch_indirect(
            shaders.path_tiling,
            indirect_count_buf,
            0,
            [
                bump_buf,
                seg_counts_buf,
                lines_buf,
                path_buf,
                tile_buf,
                segments_buf,
            ],
        );
        recording.free_buffer(indirect_count_buf);
        recording.free_resource(seg_counts_buf);
        recording.free_resource(scene_buf);
        recording.free_resource(draw_monoid_buf);
        recording.free_resource(bin_header_buf);
        recording.free_resource(path_buf);
        let out_image = ImageProxy::new(params.width, params.height, ImageFormat::Rgba8);
        let blend_spill_buf = BufferProxy::new(
            buffer_sizes.blend_spill.size_in_bytes().into(),
            "vello.blend_spill",
        );
        self.fine_wg_count = Some(wg_counts.fine);
        self.fine_resources = Some(FineResources {
            aa_config: params.antialiasing_method,
            config_buf,
            bump_buf,
            tile_buf,
            segments_buf,
            ptcl_buf,
            gradient_image,
            info_bin_data_buf,
            blend_spill_buf: ResourceProxy::Buffer(blend_spill_buf),
            image_atlas: ResourceProxy::Image(image_atlas),
            out_image,
        });
        if robust {
            recording.download(*bump_buf.as_buf().unwrap());
        }
        recording.free_resource(bump_buf);

        #[cfg(feature = "debug_layers")]
        {
            if robust {
                let path_bboxes = *path_bbox_buf.as_buf().unwrap();
                let lines = *lines_buf.as_buf().unwrap();
                recording.download(lines);

                self.captured_buffers = Some(CapturedBuffers {
                    sizes: cpu_config.buffer_sizes,
                    path_bboxes,
                    lines,
                });
            } else {
                recording.free_resource(path_bbox_buf);
                recording.free_resource(lines_buf);
            }
        }
        #[cfg(not(feature = "debug_layers"))]
        {
            recording.free_resource(path_bbox_buf);
            recording.free_resource(lines_buf);
        }

        recording
    }

    /// Run fine rasterization assuming the coarse phase succeeded.
    pub fn record_fine(&mut self, shaders: &FullShaders, recording: &mut Recording) {
        let fine_wg_count = self.fine_wg_count.take().unwrap();
        let fine = self.fine_resources.take().unwrap();
        match fine.aa_config {
            AaConfig::Area => {
                recording.dispatch(
                    shaders
                        .fine_area
                        .expect("shaders not configured to support AA mode: area"),
                    fine_wg_count,
                    [
                        fine.config_buf,
                        fine.segments_buf,
                        fine.ptcl_buf,
                        fine.info_bin_data_buf,
                        fine.blend_spill_buf,
                        ResourceProxy::Image(fine.out_image),
                        fine.gradient_image,
                        fine.image_atlas,
                    ],
                );
            }
            _ => {
                if self.mask_buf.is_none() {
                    let mask_lut = match fine.aa_config {
                        AaConfig::Msaa16 => make_mask_lut_16(),
                        AaConfig::Msaa8 => make_mask_lut(),
                        _ => unreachable!(),
                    };
                    let buf = recording.upload("vello.mask_lut", mask_lut);
                    self.mask_buf = Some(buf.into());
                }
                let fine_shader = match fine.aa_config {
                    AaConfig::Msaa16 => shaders
                        .fine_msaa16
                        .expect("shaders not configured to support AA mode: msaa16"),
                    AaConfig::Msaa8 => shaders
                        .fine_msaa8
                        .expect("shaders not configured to support AA mode: msaa8"),
                    _ => unreachable!(),
                };
                recording.dispatch(
                    fine_shader,
                    fine_wg_count,
                    [
                        fine.config_buf,
                        fine.segments_buf,
                        fine.ptcl_buf,
                        fine.info_bin_data_buf,
                        fine.blend_spill_buf,
                        ResourceProxy::Image(fine.out_image),
                        fine.gradient_image,
                        fine.image_atlas,
                        self.mask_buf.unwrap(),
                    ],
                );
            }
        }
        recording.free_resource(fine.config_buf);
        recording.free_resource(fine.tile_buf);
        recording.free_resource(fine.segments_buf);
        recording.free_resource(fine.ptcl_buf);
        recording.free_resource(fine.gradient_image);
        recording.free_resource(fine.image_atlas);
        recording.free_resource(fine.info_bin_data_buf);
        recording.free_resource(fine.blend_spill_buf);
        // TODO: make mask buf persistent
        if let Some(mask_buf) = self.mask_buf.take() {
            recording.free_resource(mask_buf);
        }
    }

    /// Get the output image.
    ///
    /// This is going away, as the caller will add the output image to the bind
    /// map.
    pub fn out_image(&self) -> ImageProxy {
        self.fine_resources.as_ref().unwrap().out_image
    }

    pub fn bump_buf(&self) -> BufferProxy {
        *self
            .fine_resources
            .as_ref()
            .unwrap()
            .bump_buf
            .as_buf()
            .unwrap()
    }

    #[cfg(feature = "debug_layers")]
    pub fn take_captured_buffers(&mut self) -> Option<CapturedBuffers> {
        self.captured_buffers.take()
    }
}
