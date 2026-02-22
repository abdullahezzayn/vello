# True Backdrop Blur Design for Xilem/Masonry (`backdrop_blur` migration)

This document proposes a true backdrop blur pipeline for Vello that can power iOS/macOS-style "liquid glass" in Masonry/Xilem.

## Problem

Your current widget in `xilem_fork` (`masonry/src/widgets/backdrop_blur.rs` on branch `backdrop_blur`) uses:

- `Scene::draw_blurred_rounded_rect_in(...)`

That only blurs the blur primitive itself. It does not sample and blur already-rendered content behind the widget, so it cannot produce true system-style frosted glass.

## Target behavior

For each backdrop widget:

1. Capture the already rendered content behind the widget (not including foreground content above it).
2. Blur only a bounded region (shape bounds + blur spread).
3. Optionally apply saturation/brightness tweaks typical of Apple materials.
4. Composite the processed image back through the widget clip/mask.
5. Continue normal scene rendering (child content, overlays, borders, etc).

## API proposal

Add a first-class backdrop command to Vello scene API.

```rust
pub struct BackdropBlurStyle {
    pub sigma: f32,                  // Gaussian sigma in device px
    pub saturation: f32,             // 1.0 = unchanged, e.g. 1.6..2.0 for glass
    pub luminosity: f32,             // additive luminance bias (-1..1 style)
    pub tint: peniko::Color,         // translucent material tint
    pub edge_mode: BackdropEdgeMode, // Duplicate recommended
}

pub enum BackdropEdgeMode {
    Duplicate,
    Mirror,
    ClampToTransparent,
}

impl Scene {
    pub fn draw_backdrop_blur_in(
        &mut self,
        shape: &impl Shape,
        transform: Affine,
        style: &BackdropBlurStyle,
    );
}
```

For Xilem/Masonry, keep the existing view/widget surface, but switch internals from `draw_blurred_rounded_rect_in` to `draw_backdrop_blur_in`.

## Render architecture

Use ordered backdrop operations during render encoding playback.

### 1) Detect backdrop commands

Extend encoding with:

- new draw tag (for example `DrawTag::BACKDROP_BLUR`)
- draw payload containing style + shape metadata

### 2) Execute as ordered subpasses

At render time (in draw order):

1. Render normal commands up to next backdrop op into a working color target.
2. Compute inflated ROI from clip bounds and sigma:
   - `inflate = ceil(3.0 * sigma) + 1`
3. Copy ROI from working target into scratch texture.
4. Blur scratch texture:
   - separable gaussian, or
   - downsample + blur + upsample pyramid for large sigma.
5. Apply optional color matrix for saturation/luminosity.
6. Composite blurred ROI back into working target masked by backdrop shape.
7. Continue with remaining commands.

This guarantees each backdrop samples exactly what has already been painted behind it.

## Why this matches liquid glass

Apple-like material is not only blur. It is generally:

- strong background blur
- slight saturation increase
- subtle luminosity adjustment
- translucent tint layer
- thin high-contrast edge highlight

The proposed style block supports these directly.

Recommended defaults for "regular glass":

- `sigma`: `18.0`
- `saturation`: `1.7`
- `luminosity`: `0.06`
- `tint`: `rgba(255,255,255,0.16)`
- `edge_mode`: `Duplicate`

## Integration points in this repo

Core files likely involved:

- `vello/src/scene.rs` (public scene API)
- `vello_encoding/src/draw.rs` (new draw tag + payload struct)
- `vello_encoding/src/encoding.rs` (encode method for backdrop command)
- `vello/src/render.rs` (ordered subpass orchestration and ROI management)
- `vello/src/shaders.rs` + new WGSL compute shaders for copy/blur/composite

Potential shader additions:

- `vello_shaders/shader/backdrop_copy.wgsl`
- `vello_shaders/shader/backdrop_blur_h.wgsl`
- `vello_shaders/shader/backdrop_blur_v.wgsl`
- `vello_shaders/shader/backdrop_color_matrix.wgsl`
- `vello_shaders/shader/backdrop_composite.wgsl`

## Performance constraints

To keep UI smooth:

- Blur only ROI, never full frame unless required.
- Quantize sigma to a small set of kernel configs for pipeline reuse.
- Reuse scratch textures via pooled allocator.
- For large sigma, use pyramid blur to avoid large kernel cost.
- Collapse adjacent backdrop ops when shape/style/transform permit merge.

## Correctness notes

- Perform blur and color transforms in linear color space.
- Keep alpha premultiplied throughout.
- Use shape mask at composite time (not before blur) for natural edge bleed.
- For rounded rects, clip mask should be anti-aliased and match Masonry corner radius exactly.

## Migration from your current branch

Current branch (`xilem_fork/backdrop_blur`) is already good as API scaffolding.

Minimal Masonry/Xilem migration:

1. Keep `BackdropBlur` widget/view API.
2. Replace its pre-paint call from:
   - `draw_blurred_rounded_rect_in(...)`
   to:
   - `draw_backdrop_blur_in(...)`
3. Keep existing background/border painting on top.
4. Keep child paint order unchanged.

This preserves your public API while upgrading implementation to true backdrop.

## Phased implementation plan

1. Add encoding/API surface for backdrop command.
2. Add single-ROI blur path (no pyramid) for correctness first.
3. Add color-matrix stage (saturation/luminosity).
4. Add ROI pyramid optimization for large sigma.
5. Add nested-backdrop tests and screenshot baselines in Xilem/Masonry integration.

## Implementation status

- Completed: Phase 1 (new `Scene` backdrop API + dedicated draw tag + renderer fallback path).
- Completed: Phase 2 prototype (prefix-based backdrop sampling + separable blur + rounded-rect masked composite in the renderer).
- Completed: Phase 3 optimization (ROI-restricted blur and apply dispatches).
- Completed: Phase 4 partial ordering semantics (segment replay + `src-over` composition for root-level single-backdrop cases).
- Completed: Phase 5 snapshot-test coverage in `vello_tests` (basic, nested, overlap/transform, and clipped-layer backdrop scenes).
- Completed: Phase 6 root-level multi-backdrop segment replay (state-seeded packed range replay across multiple backdrop ops with deterministic ordering).
- Completed: Phase 7 non-root stress coverage in `vello_tests` (nested backdrop sequences under clip and blend-layer scopes).
- Completed: Phase 8 clip-stack carry-over replay (non-root replay now supports clip-only nesting by seeding active begin-clip draws at segment boundaries).
- Completed: Phase 9 compatibility gating + fallback coverage (segment replay now rejects ranges that require non-clip layer carry-over, including standard src-over layers, multiply-blend layers, and luminance-mask layers; snapshot coverage includes multiply-blend and luminance-mask backdrop scenes).
- Completed: Phase 10 fallback coverage expansion (added a prefixed blend-layer backdrop scene to ensure compatibility gating remains stable when non-clip layers appear before backdrop commands).
- Completed: Phase 11 active-stack clip-seed correctness (clip-stack seeding now only rejects non-clip layers that are active at the segment boundary; closed non-clip layers before the boundary no longer invalidate carry-over analysis, with dedicated unit tests in the renderer).
- Completed: Phase 12 non-clip boundary analysis scaffolding (renderer now has explicit layer-stack seed extraction for active begin-clip boundaries and dedicated tests for active/closed non-clip layer classification; replay compatibility behavior remains gated to legacy for non-clip range semantics).
- Completed: Phase 13 explicit active-non-clip boundary gating (segment replay now uses structured layer-seed state at each non-zero boundary and routes any active non-clip boundary to legacy fallback deterministically, with preserved snapshot parity).
- Completed: Phase 14 per-scene layer-seed cache reuse (renderer now computes boundary seed state once per scene and reuses cached boundary lookups for replay gating and clip-seed packed-range assembly, removing repeated draw-tag scans while keeping fallback and snapshot behavior unchanged).
- Completed: Phase 15 synthetic boundary-closure scaffolding (segment packed-range assembly can now append synthetic `END_CLIP` commands from cached boundary state, with dedicated renderer tests for stream/count correctness; compatibility gating remains in place so clip-only replay behavior stays snapshot-stable while non-clip carry-over semantics are finalized).
- Completed: Phase 16 single-backdrop active-layer carry-over gating (first-range replay now accepts active non-clip layers only for single-backdrop scenes when all encountered non-clip begins remain active at the boundary, reusing synthetic boundary closure while preserving existing fallback for multi-backdrop non-clip cases and snapshot parity).
- Completed: Phase 17 snapshot coverage for single-backdrop active non-clip replay (added a dedicated test scene with one backdrop inside an active non-clip layer and snapshot baseline coverage in `vello_tests`, validating Phase 16 behavior stays locked while broader multi-backdrop non-clip carry-over remains gated).
- Completed: Phase 18 strict multi-backdrop non-clip carry-over gating (multi-backdrop replay now accepts active non-clip boundaries only when begin-clip parameters are replay-equivalent (`Normal+SrcOver`, alpha `1.0`), with added renderer tests for replay-safe classification and snapshot coverage via a new layered opaque multi-backdrop scene).
- Completed: Phase 19 conservative multi-backdrop non-opaque `src-over` gating (multi-backdrop replay now accepts active non-clip `Normal+SrcOver` boundaries with non-opaque alpha only when the replayed range before each synthetic closure is paint-free, preventing repeated-alpha regressions while broadening compatibility; added renderer tests for src-over classification/range gating and snapshot coverage via `backdrop_blur_layered_nested_src_over_alpha_sparse`).
- Completed: Phase 20 paint-free multi-backdrop non-clip parameter broadening (multi-backdrop replay now accepts active non-clip boundaries for any layer parameters when the replayed range before synthetic closure is paint-free, while still requiring replay-safe parameters for painted ranges; snapshot coverage includes `backdrop_blur_blend_multiply_sparse`).
- Completed: Phase 21 active-span paint gating refinement (multi-backdrop replay paint-freeness checks now apply only while non-replay-safe non-clip layers are active in each replayed range, allowing post-pop painted tails without forcing fallback; renderer tests cover seeded/non-seeded non-replay-safe span behavior and snapshot coverage includes `backdrop_blur_blend_multiply_post_pop_paint`).
- Completed: Phase 22 luminance-mask active-span coverage (added renderer tests validating luminance-mask non-replay-safe span behavior, and snapshot coverage via `backdrop_blur_luminance_mask_post_pop_paint` to ensure post-pop painted tails under luminance-mask boundaries preserve replay parity).
- Completed: Phase 23 transparent-no-op span elision (active-span gating now treats strictly transparent color/image/blur draws as non-contributing while non-replay-safe layers are active, reducing unnecessary fallback without changing rendered output; added renderer tests and snapshot coverage via `backdrop_blur_blend_multiply_transparent_span`).
- Completed: Phase 24 zero-alpha layer span elision (active-span gating now treats `Normal+SrcOver` non-clip layers with group alpha `0.0` as non-contributing for paint-freeness checks, so visible paint inside those spans no longer forces fallback; added renderer tests for zero-alpha classification and replay-range acceptance).
- Completed: Phase 25 broadened zero-alpha non-luminance span elision (active-span gating now treats all non-luminance non-clip layers with group alpha `0.0` as non-contributing for paint-freeness checks while preserving luminance-mask strictness; added renderer tests for non-`src-over`/luminance behavior and snapshot coverage via `backdrop_blur_blend_multiply_zero_alpha_layer`).
- Completed: Phase 26 closed-span non-replay-safe range support (multi-backdrop range compatibility now accepts non-replay-safe non-clip layers with visible paint when those layers begin and end within the same replayed range; unsupported non-clip layers still active at range end continue to force fallback, with renderer tests and snapshot coverage via `backdrop_blur_blend_multiply_closed_span`).
- Completed: Phase 27 active-tail fallback guardrail coverage (added multi-backdrop snapshot coverage via `backdrop_blur_blend_multiply_tail_active_paint` to lock current conservative behavior where visible tail paint inside active non-replay-safe layers still routes to fallback, preventing parity regressions while broader carry-over support is designed).
- Completed: Phase 28 scoped tail carry-over for post-first-backdrop layers (tail boundary analysis now permits painted active non-replay-safe carry-over only when all active non-replay-safe layer begins occur strictly after the first backdrop boundary, reducing risk from pre-existing layer content; added renderer helper tests and snapshot coverage via `backdrop_blur_blend_multiply_tail_active_after_first_backdrop`).
- Completed: Phase 29 scoped intermediate carry-over for post-first-backdrop layers (the post-first-backdrop boundary allowance now applies at intermediate multi-backdrop boundaries in addition to tail boundaries, so painted ranges inside active non-replay-safe layers can replay when those layers were introduced after the first backdrop; added snapshot coverage via `backdrop_blur_blend_multiply_intermediate_active_after_first_backdrop`).
- Completed: Phase 30 pre-first intermediate fallback guardrail coverage (added explicit replay helper tests for mixed non-replay-safe origins and snapshot coverage via `backdrop_blur_blend_multiply_intermediate_active_prefirst`, locking current conservative behavior for painted intermediate spans when non-replay-safe layers originate before the first backdrop).
- Completed: Phase 31 scoped pre-first paint-free intermediate carry-over (intermediate boundary allowances now also accept non-`src-over` active non-replay-safe layers that originate before the first backdrop when the entire pre-first replay span is paint-free, with replay helper coverage tightened to keep `src-over` non-replay-safe layers excluded; validated against existing pre-first/intermediate backdrop snapshot coverage).
- Completed: Phase 32 conservative pre-first tail scope codification (explicitly retained the 3+-backdrop scope for pre-first paint-free non-replay-safe tail allowances after validating that two-backdrop pre-first tails still regress snapshot parity).
- Remaining: broaden multi-backdrop non-clip carry-over for painted pre-first-origin spans in two-backdrop tails and other non-`src-over`/luminance-sensitive cases while preserving snapshot parity, and add equivalent screenshot coverage in Xilem/Masonry integration.

## Test matrix

- Single backdrop over image/text background.
- Nested backdrops.
- Scaled/rotated transform.
- Rounded-rect + non-rect clip paths.
- Multiple overlapping backdrops with different sigma.
- Animation stress (moving content behind static backdrop, and vice versa).
- HiDPI correctness (scale factors 1.0, 1.5, 2.0, 3.0).
