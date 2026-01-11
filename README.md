# mask-fingerprint

Fast, deterministic **visual diff fingerprinting** for PNG images, exposed as a Node.js native module via N-API.

This library takes a PNG diff image (for example red-pixel visual diffs) and produces a **stable equality fingerprint**.  
Two diffs that are visually very close will produce **the same fingerprint**, enabling:

- `fingerprintA === fingerprintB`
- database indexing
- grouping and deduplication
- joins without custom comparison logic

The core algorithm is written in Rust for performance and exposed to Node.js.

## What problem this solves

Classic perceptual hashes require distance comparisons, which do not work well with SQL indexes or strict equality.

This project instead:

- normalizes the diff mask
- quantizes spatial density
- hashes a coarse representation

The result is **tolerant equality**, not approximate similarity.

## How it works

1. Decode PNG into RGBA
2. Extract a binary mask of red pixels
3. Optional dilation to absorb small pixel noise
4. Crop to the bounding box
5. Optional square padding
6. Split into a small grid (8×8, 16×16, or 32×32)
7. Compute per-cell red density
8. Quantize densities into bins
9. Pack bits and hash with FNV-1a 64-bit
10. Return a deterministic string key

Small local differences usually do not change the final fingerprint.

## Installation

### Requirements

- Node.js 18 or newer
- Rust stable
- Cargo

### Install dependencies

```bash
pnpm install
```

or

```bash
npm install
```

This builds the native module.

## Usage

```ts
import fs from 'node:fs'
import { fingerprint_diff_for_equality } from 'mask-fingerprint'

const png = fs.readFileSync('diff.png')

const fingerprint = fingerprint_diff_for_equality(png, {
  gridSize: 16,
  dilateRadius: 1,
  padToSquare: true,
  densityThresholds: [0.002, 0.02, 0.08],
  redThreshold: {
    rMin: 200,
    gMax: 90,
    bMax: 90,
    aMin: 16,
  },
})

console.log(fingerprint)
// v1:g16:d1:t0.002,0.02,0.08:3fa1c2e9a4b8d210
```

The returned value is designed to be stored in a database, indexed, and compared using strict equality.

## API

### `fingerprint_diff_for_equality(buffer, options?)`

#### Parameters

**buffer**

- Node.js `Buffer`
- Raw PNG file contents

**options** (optional)

```ts
{
  gridSize?: 8 | 16 | 32        // default: 16
  dilateRadius?: 0 | 1          // default: 1
  padToSquare?: boolean         // default: true

  densityThresholds?: number[]  // default: [0.002, 0.02, 0.08]
                                // sorted, values between 0 and 1

  redThreshold?: {
    rMin?: number               // default: 200
    gMax?: number               // default: 90
    bMax?: number               // default: 90
    aMin?: number               // default: 16
  }
}
```

#### Returns

```ts
string
```

A deterministic fingerprint string suitable for equality comparison and indexing.

## Supported PNG formats

- RGBA
- RGB
- Grayscale
- Grayscale with alpha
- Indexed or palette PNGs

Palette and low bit-depth images are automatically expanded.

## Performance characteristics

- Linear time in number of pixels
- No heavy floating-point operations
- Minimal allocations in hot paths
- Suitable for batch processing

Typical workloads handle thousands of diffs per second.

## When to use this

Good fit:

- Visual regression testing
- Screenshot diff deduplication
- CI artifact clustering
- Database-backed visual change tracking

Not a good fit:

- General image similarity search
- Large geometric transformations
- Rotation or strong scale invariance

This library is intentionally optimized for **diff masks**, not arbitrary images.

## Development

```bash
cargo check
cargo test
```

Rebuild the native module:

```bash
pnpm build
```

## License

MIT
