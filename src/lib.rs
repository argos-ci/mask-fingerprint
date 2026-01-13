#![deny(clippy::all)]

use napi::bindgen_prelude::{Buffer, Either};
use napi_derive::napi;
use std::io::{BufReader, Cursor};

#[derive(Clone, Copy, Debug)]
struct RedThreshold {
  r_min: u8,
  g_max: u8,
  b_max: u8,
  a_min: u8,
}

#[derive(Clone, Debug)]
struct EqualityFingerprintOptions {
  red_threshold: RedThreshold,
  dilate_radius: u8, // 0 or 1
  grid_size: usize,  // 8, 16, 32
  density_thresholds: Vec<f32>,
  pad_to_square: bool,
}

impl Default for EqualityFingerprintOptions {
  fn default() -> Self {
    Self {
      red_threshold: RedThreshold {
        r_min: 200,
        g_max: 90,
        b_max: 90,
        a_min: 16,
      },
      dilate_radius: 1,
      grid_size: 16,
      density_thresholds: vec![0.002, 0.02, 0.08],
      pad_to_square: true,
    }
  }
}

/* ------------------------------ Node.js options ------------------------------ */

#[napi(object)]
pub struct JsRedThreshold {
  pub r_min: Option<u32>,
  pub g_max: Option<u32>,
  pub b_max: Option<u32>,
  pub a_min: Option<u32>,
}

#[napi(object)]
pub struct JsEqualityFingerprintOptions {
  pub red_threshold: Option<JsRedThreshold>,
  pub dilate_radius: Option<u32>, // 0 or 1
  pub grid_size: Option<u32>,     // 8, 16, 32
  pub density_thresholds: Option<Vec<f64>>,
  pub pad_to_square: Option<bool>,
}

fn build_options(
  js: Option<JsEqualityFingerprintOptions>,
) -> napi::Result<EqualityFingerprintOptions> {
  let mut opts = EqualityFingerprintOptions::default();

  let Some(js) = js else {
    return Ok(opts);
  };

  if let Some(v) = js.dilate_radius {
    if v > 1 {
      return Err(napi::Error::from_reason("dilateRadius must be 0 or 1"));
    }
    opts.dilate_radius = v as u8;
  }

  if let Some(v) = js.grid_size {
    if v != 8 && v != 16 && v != 32 {
      return Err(napi::Error::from_reason("gridSize must be 8, 16, or 32"));
    }
    opts.grid_size = v as usize;
  }

  if let Some(v) = js.pad_to_square {
    opts.pad_to_square = v;
  }

  if let Some(th) = js.density_thresholds {
    if th.is_empty() {
      return Err(napi::Error::from_reason(
        "densityThresholds must not be empty",
      ));
    }

    let mut out = Vec::with_capacity(th.len());
    let mut prev = f32::NEG_INFINITY;

    for value in th {
      if !(0.0..=1.0).contains(&value) {
        return Err(napi::Error::from_reason(
          "densityThresholds values must be between 0 and 1",
        ));
      }
      let v = value as f32;
      if v < prev {
        return Err(napi::Error::from_reason(
          "densityThresholds must be sorted ascending",
        ));
      }
      prev = v;
      out.push(v);
    }

    opts.density_thresholds = out;
  }

  if let Some(rt) = js.red_threshold {
    if let Some(v) = rt.r_min {
      opts.red_threshold.r_min = clamp_u32_to_u8(v);
    }
    if let Some(v) = rt.g_max {
      opts.red_threshold.g_max = clamp_u32_to_u8(v);
    }
    if let Some(v) = rt.b_max {
      opts.red_threshold.b_max = clamp_u32_to_u8(v);
    }
    if let Some(v) = rt.a_min {
      opts.red_threshold.a_min = clamp_u32_to_u8(v);
    }
  }

  Ok(opts)
}

fn clamp_u32_to_u8(v: u32) -> u8 {
  if v > 255 {
    255
  } else {
    v as u8
  }
}

/* ---------------------------------- NAPI ---------------------------------- */

#[napi]
pub fn fingerprint_diff(
  png_input: Either<String, Buffer>,
  options: Option<JsEqualityFingerprintOptions>,
) -> napi::Result<String> {
  enum InputBytes {
    File(Vec<u8>),
    Buffer(Buffer),
  }

  let input = match png_input {
    Either::A(path) => {
      let bytes = std::fs::read(&path).map_err(|e| {
        napi::Error::from_reason(format!("Failed to read PNG file '{path}': {e}"))
      })?;
      InputBytes::File(bytes)
    }
    Either::B(buffer) => InputBytes::Buffer(buffer),
  };

  let png_bytes: &[u8] = match &input {
    InputBytes::File(bytes) => bytes.as_slice(),
    InputBytes::Buffer(buffer) => buffer.as_ref(),
  };

  let (rgba, width, height) = decode_png_to_rgba(png_bytes)?;
  let opts = build_options(options)?;
  Ok(fingerprint_rgba_for_equality(&rgba, width, height, &opts))
}

/* -------------------------------- PNG decode -------------------------------- */

fn decode_png_to_rgba(png_bytes: &[u8]) -> napi::Result<(Vec<u8>, usize, usize)> {
  let cursor = Cursor::new(png_bytes);
  let reader = BufReader::new(cursor);

  let mut decoder = png::Decoder::new(reader);

  decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::ALPHA);

  let mut reader = decoder
    .read_info()
    .map_err(|e| napi::Error::from_reason(format!("PNG decode error: {e}")))?;

  let buffer_size = reader
    .output_buffer_size()
    .ok_or_else(|| napi::Error::from_reason("PNG output buffer size unknown"))?;

  let mut buf = vec![0u8; buffer_size];

  let info = reader
    .next_frame(&mut buf)
    .map_err(|e| napi::Error::from_reason(format!("PNG decode error: {e}")))?;

  let width = info.width as usize;
  let height = info.height as usize;
  if info.color_type == png::ColorType::Rgba && info.bit_depth == png::BitDepth::Eight {
    buf.truncate(info.buffer_size());
    return Ok((buf, width, height));
  }

  let bytes = &buf[..info.buffer_size()];
  let rgba = match (info.color_type, info.bit_depth) {
    (png::ColorType::Rgb, png::BitDepth::Eight) => rgb8_to_rgba8(bytes, width, height),
    (png::ColorType::Grayscale, png::BitDepth::Eight) => gray8_to_rgba8(bytes, width, height),
    (png::ColorType::GrayscaleAlpha, png::BitDepth::Eight) => {
      gray_alpha8_to_rgba8(bytes, width, height)
    }
    _ => {
      return Err(napi::Error::from_reason(format!(
        "Unsupported PNG format after transform: {:?} {:?}",
        info.color_type, info.bit_depth
      )))
    }
  };

  Ok((rgba, width, height))
}

fn rgb8_to_rgba8(src: &[u8], width: usize, height: usize) -> Vec<u8> {
  let n = width * height;
  let mut out = vec![0u8; n * 4];
  for (dst, src_px) in out.chunks_exact_mut(4).zip(src.chunks_exact(3)) {
    dst[0] = src_px[0];
    dst[1] = src_px[1];
    dst[2] = src_px[2];
    dst[3] = 255;
  }
  out
}

fn gray8_to_rgba8(src: &[u8], width: usize, height: usize) -> Vec<u8> {
  let n = width * height;
  let mut out = vec![0u8; n * 4];
  for (dst, &v) in out.chunks_exact_mut(4).zip(src.iter()) {
    dst[0] = v;
    dst[1] = v;
    dst[2] = v;
    dst[3] = 255;
  }
  out
}

fn gray_alpha8_to_rgba8(src: &[u8], width: usize, height: usize) -> Vec<u8> {
  let n = width * height;
  let mut out = vec![0u8; n * 4];
  for (dst, src_px) in out.chunks_exact_mut(4).zip(src.chunks_exact(2)) {
    let v = src_px[0];
    dst[0] = v;
    dst[1] = v;
    dst[2] = v;
    dst[3] = src_px[1];
  }
  out
}

/* ---------------------------- fingerprint internals ---------------------------- */

#[derive(Clone, Copy, Debug)]
struct Bbox {
  x: usize,
  y: usize,
  width: usize,
  height: usize,
}

#[derive(Clone, Debug)]
struct BinaryImage {
  data: Vec<u8>, // 0 or 1
  width: usize,
  height: usize,
}

fn fingerprint_rgba_for_equality(
  rgba: &[u8],
  width: usize,
  height: usize,
  options: &EqualityFingerprintOptions,
) -> String {
  let mask = extract_red_mask(rgba, width, height, options.red_threshold);

  let mask2 = if options.dilate_radius == 1 {
    dilate_radius1(&mask, width, height)
  } else {
    mask
  };

  let bbox = match compute_bbox(&mask2, width, height) {
    Some(b) => b,
    None => return "empty".to_string(),
  };

  let cropped = crop_mask(&mask2, width, &bbox);

  let normalized = if options.pad_to_square {
    pad_binary_to_square(&cropped.data, cropped.width, cropped.height)
  } else {
    cropped
  };

  let h = hash_grid_density(
    &normalized.data,
    normalized.width,
    normalized.height,
    options.grid_size,
    options.grid_size,
    &options.density_thresholds,
  );

  let t_joined = options
    .density_thresholds
    .iter()
    .map(|v| format!("{v}"))
    .collect::<Vec<_>>()
    .join(",");

  format!(
    "v1:g{}:d{}:t{}:{:016x}",
    options.grid_size, options.dilate_radius, t_joined, h
  )
}

fn extract_red_mask(rgba: &[u8], width: usize, height: usize, t: RedThreshold) -> Vec<u8> {
  let n = width * height;
  let mut out = vec![0u8; n];

  for (dst, px) in out.iter_mut().zip(rgba.chunks_exact(4)) {
    let r = px[0];
    let g = px[1];
    let b = px[2];
    let a = px[3];

    *dst = if r >= t.r_min && g <= t.g_max && b <= t.b_max && a >= t.a_min {
      1
    } else {
      0
    };
  }

  out
}

fn dilate_radius1(mask: &[u8], width: usize, height: usize) -> Vec<u8> {
  let mut out = vec![0u8; mask.len()];

  for y in 0..height {
    let y0 = y.saturating_sub(1);
    let y1 = (y + 1).min(height - 1);

    for x in 0..width {
      let x0 = x.saturating_sub(1);
      let x1 = (x + 1).min(width - 1);

      let mut v = 0u8;

      'outer: for yy in y0..=y1 {
        let row = yy * width;
        for xx in x0..=x1 {
          if mask[row + xx] == 1 {
            v = 1;
            break 'outer;
          }
        }
      }

      out[y * width + x] = v;
    }
  }

  out
}

fn compute_bbox(mask: &[u8], width: usize, height: usize) -> Option<Bbox> {
  let mut min_x = width;
  let mut min_y = height;
  let mut max_x: isize = -1;
  let mut max_y: isize = -1;

  for y in 0..height {
    let row = y * width;
    for x in 0..width {
      if mask[row + x] == 1 {
        if x < min_x {
          min_x = x;
        }
        if y < min_y {
          min_y = y;
        }
        if (x as isize) > max_x {
          max_x = x as isize;
        }
        if (y as isize) > max_y {
          max_y = y as isize;
        }
      }
    }
  }

  if max_x < min_x as isize || max_y < min_y as isize {
    return None;
  }

  Some(Bbox {
    x: min_x,
    y: min_y,
    width: (max_x as usize) - min_x + 1,
    height: (max_y as usize) - min_y + 1,
  })
}

fn crop_mask(mask: &[u8], src_width: usize, bbox: &Bbox) -> BinaryImage {
  let mut out = vec![0u8; bbox.width * bbox.height];

  for yy in 0..bbox.height {
    let src_row = (bbox.y + yy) * src_width;
    let dst_row = yy * bbox.width;

    for xx in 0..bbox.width {
      out[dst_row + xx] = mask[src_row + (bbox.x + xx)];
    }
  }

  BinaryImage {
    data: out,
    width: bbox.width,
    height: bbox.height,
  }
}

fn pad_binary_to_square(src: &[u8], src_w: usize, src_h: usize) -> BinaryImage {
  let side = src_w.max(src_h);
  let mut out = vec![0u8; side * side];

  let off_x = (side - src_w) / 2;
  let off_y = (side - src_h) / 2;

  for y in 0..src_h {
    let dst_row = (y + off_y) * side;
    let src_row = y * src_w;

    for x in 0..src_w {
      out[dst_row + (x + off_x)] = src[src_row + x];
    }
  }

  BinaryImage {
    data: out,
    width: side,
    height: side,
  }
}

fn hash_grid_density(
  src: &[u8],
  src_w: usize,
  src_h: usize,
  grid_w: usize,
  grid_h: usize,
  thresholds: &[f32],
) -> u64 {
  let integral = build_integral_image(src, src_w, src_h);
  let mut hash: u64 = 0xcbf29ce484222325;
  let prime: u64 = 0x00000100000001b3;
  // Stream packed 2-bit quantized values into the hash to avoid extra buffers.
  let mut packed_byte: u8 = 0;
  let mut byte_shift: u8 = 0;

  for gy in 0..grid_h {
    let y0 = (gy * src_h) / grid_h;
    let y1 = ((gy + 1) * src_h) / grid_h;

    for gx in 0..grid_w {
      let x0 = (gx * src_w) / grid_w;
      let x1 = ((gx + 1) * src_w) / grid_w;

      let area = ((x1.saturating_sub(x0)) * (y1.saturating_sub(y0))).max(1);
      let sum = rect_sum(&integral, src_w, x0, y0, x1, y1);
      let density = (sum as f32) / (area as f32);

      let v = quantize_density(density, thresholds) & 3;
      packed_byte |= v << byte_shift;
      if byte_shift == 6 {
        hash ^= packed_byte as u64;
        hash = hash.wrapping_mul(prime);
        packed_byte = 0;
        byte_shift = 0;
      } else {
        byte_shift += 2;
      }
    }
  }

  if byte_shift != 0 {
    hash ^= packed_byte as u64;
    hash = hash.wrapping_mul(prime);
  }

  hash
}

fn quantize_density(density: f32, thresholds: &[f32]) -> u8 {
  for (i, t) in thresholds.iter().enumerate() {
    if density < *t {
      return i as u8;
    }
  }
  thresholds.len() as u8
}

fn build_integral_image(src: &[u8], src_w: usize, src_h: usize) -> Vec<u32> {
  let mut out = vec![0u32; src_w * src_h];

  for y in 0..src_h {
    let mut row_sum: u32 = 0;
    let row = y * src_w;

    for x in 0..src_w {
      row_sum += src[row + x] as u32;
      let above = if y > 0 { out[(y - 1) * src_w + x] } else { 0 };
      out[row + x] = above + row_sum;
    }
  }

  out
}

// Sum on rectangle [x0,x1) x [y0,y1)
fn rect_sum(integral: &[u32], w: usize, x0: usize, y0: usize, x1: usize, y1: usize) -> u32 {
  let xa = x0 as isize - 1;
  let ya = y0 as isize - 1;
  let xb = x1 as isize - 1;
  let yb = y1 as isize - 1;

  let a = get_integral(integral, w, xa, ya);
  let b = get_integral(integral, w, xb, ya);
  let c = get_integral(integral, w, xa, yb);
  let d = get_integral(integral, w, xb, yb);

  d.wrapping_sub(b).wrapping_sub(c).wrapping_add(a)
}

fn get_integral(integral: &[u32], w: usize, x: isize, y: isize) -> u32 {
  if x < 0 || y < 0 {
    0
  } else {
    integral[y as usize * w + x as usize]
  }
}
