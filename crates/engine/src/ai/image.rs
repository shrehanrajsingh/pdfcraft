//! Session-only pictures attached to an AI turn. Encoded bytes are bounded before a request.

use super::{MAX_CHAT_IMAGE_BYTES, MAX_CHAT_IMAGE_SIDE, MAX_CHAT_IMAGES, MAX_CHAT_IMAGES_PER_TURN, Turn};

/// Encoded picture sent with a user turn. Never persisted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChatImage {
    kind: ChatImageKind,
    bytes: Vec<u8>,
    width: u32,
    height: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChatImageKind {
    Png,
    Jpeg,
}

impl ChatImageKind {
    pub fn mime(self) -> &'static str {
        match self {
            Self::Png => "image/png",
            Self::Jpeg => "image/jpeg",
        }
    }
}

impl ChatImage {
    pub fn kind(&self) -> ChatImageKind {
        self.kind
    }

    pub fn mime(&self) -> &'static str {
        self.kind.mime()
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    /// Clipboard or decoded pixels, downscaled and stored as PNG.
    pub fn from_rgba(width: usize, height: usize, rgba: &[u8]) -> Result<Self, String> {
        let (width, height, pixels) = fit_rgba(width, height, rgba)?;
        encode_png(width, height, &pixels)
    }

    /// A PNG or JPEG file, or any other still image `craft-image-codecs` can decode, re-encoded
    /// as PNG when the original cannot be sent as-is.
    pub fn from_encoded(bytes: Vec<u8>) -> Result<Self, String> {
        if bytes.is_empty() || bytes.len() > MAX_CHAT_IMAGE_BYTES.saturating_mul(2) {
            return Err(format!("Each image must be a readable file of at most {} bytes.", MAX_CHAT_IMAGE_BYTES.saturating_mul(2)));
        }
        let format = craft_image_codecs::detect(&bytes).ok_or_else(|| "Attach a PNG, JPEG, GIF, TIFF or BMP image.".to_string())?;
        if let Some(image) = keep_encoded(format, &bytes) {
            return image;
        }
        decode_and_encode(&bytes)
    }

    /// RGBA preview of the stored picture, for a thumbnail. Bounded by the same decode limits.
    pub fn rgba8(&self) -> Result<(u32, u32, Vec<u8>), String> {
        let decoded = decode_limited(self.bytes()).map_err(|e| e.to_string())?;
        let (w, h) = (decoded.width(), decoded.height());
        Ok((w, h, decoded.to_rgba8()))
    }
}

/// OpenAI-style chat message. Images belong on user turns only.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: String,
    pub text: String,
    pub images: Vec<ChatImage>,
}

pub(super) fn validate_images(turns: &[Turn], snapshot: &[ChatImage]) -> Result<(), String> {
    check_batch(snapshot, "this conversation")?;
    let mut total = snapshot.len();
    for turn in turns {
        if turn.role != super::Role::User && !turn.images.is_empty() {
            return Err("Only user messages can include images.".into());
        }
        check_batch(&turn.images, "one message")?;
        total = total.saturating_add(turn.images.len());
    }
    if total > MAX_CHAT_IMAGES {
        return Err(format!("This conversation already has {MAX_CHAT_IMAGES} images. Start a new conversation."));
    }
    Ok(())
}

fn check_batch(images: &[ChatImage], where_: &str) -> Result<(), String> {
    if images.len() > MAX_CHAT_IMAGES_PER_TURN {
        return Err(format!("Attach at most {MAX_CHAT_IMAGES_PER_TURN} images in {where_}."));
    }
    for image in images {
        if image.bytes.len() > MAX_CHAT_IMAGE_BYTES {
            return Err(format!("Each image must be at most {MAX_CHAT_IMAGE_BYTES} bytes."));
        }
        if image.width == 0 || image.height == 0 || image.width > MAX_CHAT_IMAGE_SIDE || image.height > MAX_CHAT_IMAGE_SIDE {
            return Err(format!("Each image must be between 1 and {MAX_CHAT_IMAGE_SIDE} pixels on a side."));
        }
    }
    Ok(())
}

fn keep_encoded(format: craft_image_codecs::Format, bytes: &[u8]) -> Option<Result<ChatImage, String>> {
    let kind = match format {
        craft_image_codecs::Format::Png => ChatImageKind::Png,
        craft_image_codecs::Format::Jpeg => ChatImageKind::Jpeg,
        _ => return None,
    };
    if bytes.len() > MAX_CHAT_IMAGE_BYTES {
        return None;
    }
    let (width, height) = crate::image_dimensions(bytes)?;
    if width == 0 || height == 0 || width > MAX_CHAT_IMAGE_SIDE || height > MAX_CHAT_IMAGE_SIDE {
        return None;
    }
    Some(Ok(ChatImage { kind, bytes: bytes.to_vec(), width, height }))
}

fn decode_and_encode(bytes: &[u8]) -> Result<ChatImage, String> {
    let decoded = decode_limited(bytes)?;
    let rgba = decoded.to_rgba8();
    let (w, h) = (decoded.width() as usize, decoded.height() as usize);
    let (width, height, pixels) = fit_rgba(w, h, &rgba)?;
    encode_png(width, height, &pixels)
}

fn decode_limited(bytes: &[u8]) -> Result<craft_image_codecs::Image, String> {
    let format = craft_image_codecs::detect(bytes).ok_or_else(|| "Attach a PNG, JPEG, GIF, TIFF or BMP image.".to_string())?;
    let opts = craft_image_codecs::DecodeOptions {
        keep_orientation: false,
        limits: craft_image_codecs::Limits {
            max_width: MAX_CHAT_IMAGE_SIDE.saturating_mul(4).max(MAX_CHAT_IMAGE_SIDE),
            max_height: MAX_CHAT_IMAGE_SIDE.saturating_mul(4).max(MAX_CHAT_IMAGE_SIDE),
            max_pixels: u64::from(MAX_CHAT_IMAGE_SIDE).saturating_mul(u64::from(MAX_CHAT_IMAGE_SIDE)).saturating_mul(4),
            max_alloc: u64::from(MAX_CHAT_IMAGE_SIDE).saturating_mul(u64::from(MAX_CHAT_IMAGE_SIDE)).saturating_mul(16),
        },
    };
    craft_image_codecs::decode_as_with(format, bytes, &opts).map_err(|e| match e {
        craft_image_codecs::CodecError::LimitExceeded(_) => "That image is too large to send.".into(),
        other => format!("Could not read the image ({other})."),
    })
}

fn fit_rgba(width: usize, height: usize, rgba: &[u8]) -> Result<(u32, u32, Vec<u8>), String> {
    let expected = width.checked_mul(height).and_then(|p| p.checked_mul(4));
    if width == 0 || height == 0 || expected != Some(rgba.len()) {
        return Err("The image is empty or malformed.".into());
    }
    let max = MAX_CHAT_IMAGE_SIDE as usize;
    if width > max.saturating_mul(4) || height > max.saturating_mul(4) {
        return Err("That image is too large to send.".into());
    }
    let long = width.max(height);
    if long <= max {
        let w = u32::try_from(width).map_err(|_| "The image is too large to send.")?;
        let h = u32::try_from(height).map_err(|_| "The image is too large to send.")?;
        return Ok((w, h, rgba.to_vec()));
    }
    let nw = (width.saturating_mul(max) / long).max(1);
    let nh = (height.saturating_mul(max) / long).max(1);
    let mut out = vec![0u8; nw.saturating_mul(nh).saturating_mul(4)];
    for y in 0..nh {
        let sy = y.saturating_mul(height) / nh;
        for x in 0..nw {
            let sx = x.saturating_mul(width) / nw;
            let src = sy.saturating_mul(width).saturating_add(sx).saturating_mul(4);
            let dst = y.saturating_mul(nw).saturating_add(x).saturating_mul(4);
            if let (Some(chunk), Some(dest)) = (rgba.get(src..src.saturating_add(4)), out.get_mut(dst..dst.saturating_add(4))) {
                dest.copy_from_slice(chunk);
            }
        }
    }
    let w = u32::try_from(nw).map_err(|_| "The image is too large to send.")?;
    let h = u32::try_from(nh).map_err(|_| "The image is too large to send.")?;
    Ok((w, h, out))
}

fn encode_png(width: u32, height: u32, rgba: &[u8]) -> Result<ChatImage, String> {
    let img = craft_image_codecs::Image::from_u8(width, height, craft_image_codecs::ChannelLayout::Rgba, rgba.to_vec())
        .map_err(|e| format!("Could not encode the image ({e})."))?;
    let bytes = craft_image_codecs::encode(&img, craft_image_codecs::Format::Png, &craft_image_codecs::EncodeOptions::default())
        .map_err(|e| format!("Could not encode the image ({e})."))?;
    if bytes.len() > MAX_CHAT_IMAGE_BYTES {
        return Err(format!("Each image must be at most {MAX_CHAT_IMAGE_BYTES} bytes after encoding."));
    }
    Ok(ChatImage { kind: ChatImageKind::Png, bytes, width, height })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rgb_png(w: u32, h: u32) -> Vec<u8> {
        let n = (w as usize).saturating_mul(h as usize).saturating_mul(4);
        let img = craft_image_codecs::Image::from_u8(w, h, craft_image_codecs::ChannelLayout::Rgba, vec![200u8; n]).unwrap();
        craft_image_codecs::encode(&img, craft_image_codecs::Format::Png, &craft_image_codecs::EncodeOptions::default()).unwrap()
    }

    #[test]
    fn chat_image_accepts_png_and_rejects_junk() {
        let png = rgb_png(4, 3);
        let image = ChatImage::from_encoded(png.clone()).unwrap();
        assert_eq!(image.mime(), "image/png");
        assert_eq!((image.width(), image.height()), (4, 3));
        assert_eq!(image.bytes(), png);
        assert!(ChatImage::from_encoded(b"not an image".to_vec()).is_err());
        assert!(ChatImage::from_encoded(Vec::new()).is_err());
        assert!(ChatImage::from_rgba(0, 1, &[]).is_err());
        assert!(ChatImage::from_rgba(1, 1, &[1, 2, 3]).is_err());
    }

    #[test]
    fn chat_image_from_rgba_downscales_and_encodes() {
        let side = MAX_CHAT_IMAGE_SIDE as usize + 8;
        let rgba = [10u8, 20, 30, 255].repeat(side.saturating_mul(2));
        let image = ChatImage::from_rgba(side, 2, &rgba).unwrap();
        assert!(image.width() <= MAX_CHAT_IMAGE_SIDE);
        assert_eq!(image.height(), 1);
        assert!(image.bytes().starts_with(b"\x89PNG"));
        assert!(image.bytes().len() <= MAX_CHAT_IMAGE_BYTES);
    }

    #[test]
    fn validate_images_bounds_turns_and_roles() {
        let png = ChatImage::from_encoded(rgb_png(2, 2)).unwrap();
        let user = Turn { role: super::super::Role::User, text: "see this".into(), images: vec![png.clone()], citations: Vec::new() };
        validate_images(std::slice::from_ref(&user), &[]).unwrap();
        let assistant = Turn { role: super::super::Role::Assistant, text: "ok".into(), images: vec![png.clone()], citations: Vec::new() };
        assert!(validate_images(&[assistant], &[]).unwrap_err().contains("user"));
        let many = vec![png.clone(); MAX_CHAT_IMAGES_PER_TURN + 1];
        assert!(validate_images(&[], &many).is_err());
    }
}
