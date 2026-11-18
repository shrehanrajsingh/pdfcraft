//! TeX math for AI answers, laid out and rasterized by RaTeX (KaTeX-compatible).

use std::collections::HashMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;

use egui::{Color32, ColorImage, FontId, RichText, TextureHandle, TextureOptions, Vec2};

const MAX_LEN: usize = 4096;
const MAX_PIXELS: usize = 2_000_000;
const CACHE_MAX: usize = 64;

#[derive(Clone)]
struct Entry {
    texture: TextureHandle,
    size: Vec2,
}

#[derive(Clone, Default)]
struct Cache {
    images: HashMap<u64, Entry>,
}

/// Draw one TeX island. Failed or oversized input shows the source, never panics.
pub(super) fn show(ui: &mut egui::Ui, latex: &str, display: bool) {
    let latex = latex.trim();
    if latex.is_empty() {
        return;
    }
    let color = ui.visuals().text_color();
    let dpr = ui.ctx().pixels_per_point().clamp(1.0, 3.0);
    let key = cache_key(latex, display, color, dpr);
    let cache_id = egui::Id::unique("pdfcraft-ratex-cache");
    if let Some(entry) = ui.ctx().data_mut(|d| d.get_temp_mut_or_default::<Cache>(cache_id).images.get(&key).cloned()) {
        paint(ui, &entry, display);
        return;
    }
    match rasterize(latex, display, color, dpr) {
        Ok(image) => {
            let size = Vec2::new(image.width() as f32 / dpr, image.height() as f32 / dpr);
            let texture = ui.ctx().load_texture(format!("pdfcraft-ratex-{key:016x}"), image.as_ref().clone(), TextureOptions::LINEAR);
            let entry = Entry { texture, size };
            ui.ctx().data_mut(|d| {
                let cache = d.get_temp_mut_or_default::<Cache>(cache_id);
                if cache.images.len() >= CACHE_MAX {
                    cache.images.clear();
                }
                cache.images.insert(key, entry.clone());
            });
            paint(ui, &entry, display);
        }
        Err(_) => {
            let size = if display { 13.0 } else { 12.0 };
            ui.add(egui::Label::new(RichText::new(latex).font(FontId::monospace(size)).color(color)).wrap());
        }
    }
}

fn paint(ui: &mut egui::Ui, entry: &Entry, _display: bool) {
    // Fit the answer column, not the gap left on the current line. A formula wider than
    // that column shrinks; a shorter one keeps its size and the line can wrap before it.
    let column = ui.max_rect().width();
    let column = if column.is_finite() && column > 1.0 { column } else { ui.available_width() };
    ui.add(egui::Image::new((entry.texture.id(), fitted_size(entry.size, column))));
}

/// Width of a formula inside the answer column. A wider image keeps its shape and fits the pane.
fn fitted_size(natural: Vec2, available: f32) -> Vec2 {
    if available.is_finite() && available > 1.0 && natural.x > available && natural.x.is_finite() && natural.x > 0.0 {
        natural * (available / natural.x)
    } else {
        natural
    }
}

fn cache_key(latex: &str, display: bool, color: Color32, dpr: f32) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    latex.hash(&mut hasher);
    display.hash(&mut hasher);
    color.hash(&mut hasher);
    dpr.to_bits().hash(&mut hasher);
    hasher.finish()
}

fn rasterize(latex: &str, display: bool, color: Color32, dpr: f32) -> Result<Arc<ColorImage>, String> {
    if latex.len() > MAX_LEN {
        return Err("formula too long".into());
    }
    match catch_unwind(AssertUnwindSafe(|| rasterize_inner(latex, display, color, dpr))) {
        Ok(result) => result,
        Err(_) => Err("math renderer panicked".into()),
    }
}

fn rasterize_inner(latex: &str, display: bool, color: Color32, dpr: f32) -> Result<Arc<ColorImage>, String> {
    let nodes = ratex_parser::parse(latex).map_err(|err| err.to_string())?;
    let style = if display { ratex_types::math_style::MathStyle::Display } else { ratex_types::math_style::MathStyle::Text };
    let options = ratex_layout::LayoutOptions::default().with_style(style).with_color(ink(color));
    let layout = ratex_layout::layout(&nodes, &options);
    ratex_layout::validate_layout(&layout)?;
    let display_list = ratex_layout::to_display_list(&layout);
    let png = ratex_render::render_to_png(
        &display_list,
        &ratex_render::RenderOptions {
            font_size: if display { 18.0 } else { 14.0 },
            padding: if display { 4.0 } else { 1.0 },
            background_color: ratex_types::color::Color::new(0.0, 0.0, 0.0, 0.0),
            font_dir: String::new(),
            device_pixel_ratio: dpr,
        },
    )?;
    png_to_image(&png)
}

fn ink(color: Color32) -> ratex_types::color::Color {
    ratex_types::color::Color::new(
        f32::from(color.r()) / 255.0,
        f32::from(color.g()) / 255.0,
        f32::from(color.b()) / 255.0,
        f32::from(color.a()) / 255.0,
    )
}

fn png_to_image(bytes: &[u8]) -> Result<Arc<ColorImage>, String> {
    let decoded = image::load_from_memory_with_format(bytes, image::ImageFormat::Png).map_err(|err| err.to_string())?.into_rgba8();
    let width = decoded.width() as usize;
    let height = decoded.height() as usize;
    if width == 0 || height == 0 {
        return Err("empty math raster".into());
    }
    if width.saturating_mul(height) > MAX_PIXELS {
        return Err("math raster too large".into());
    }
    Ok(Arc::new(ColorImage::from_rgba_unmultiplied([width, height], decoded.as_raw())))
}

#[cfg(test)]
mod tests {
    use super::{MAX_LEN, rasterize};
    use egui::Color32;

    #[test]
    fn ratex_renders_frac_to_a_png_raster() {
        let image = rasterize(r"\frac{9}{3.5}", true, Color32::BLACK, 1.0).expect("RaTeX should layout a fraction");
        assert!(image.width() > 8 && image.height() > 8, "{}x{}", image.width(), image.height());
        let opaque = image.pixels.iter().filter(|p| p.a() > 8).count();
        assert!(opaque > 16, "expected inked pixels, got {opaque}");
    }

    #[test]
    fn a_wide_formula_stays_inside_a_narrow_column() {
        let ctx = egui::Context::default();
        let input = egui::RawInput { screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(220.0, 400.0))), ..Default::default() };
        let mut width = 0.0f32;
        let output = ctx.run_ui(input, |ui| {
            super::show(ui, r"f(x+1,y)+f(x-1,y)+f(x,y+1)+f(x,y-1)-4f(x,y)", false);
            width = ui.min_rect().width();
        });
        output.drop_without_applying_deltas();
        assert!(width <= 221.0, "formula width {width} exceeds the 220 column");
    }

    #[test]
    fn a_wide_formula_fits_the_answer_column() {
        let fitted = super::fitted_size(egui::vec2(640.0, 32.0), 280.0);
        assert!((fitted.x - 280.0).abs() < 0.01, "{}", fitted.x);
        assert!((fitted.y - 14.0).abs() < 0.01, "{}", fitted.y);
        assert_eq!(super::fitted_size(egui::vec2(100.0, 20.0), 280.0), egui::vec2(100.0, 20.0));
        assert_eq!(super::fitted_size(egui::vec2(640.0, 32.0), f32::INFINITY), egui::vec2(640.0, 32.0));
    }

    #[test]
    fn ratex_rejects_oversized_source() {
        let long = "a".repeat(MAX_LEN + 1);
        assert!(rasterize(&long, false, Color32::BLACK, 1.0).is_err());
    }
}
