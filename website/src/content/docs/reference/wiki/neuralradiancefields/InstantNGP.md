---
title: "InstantNGP<T>"
description: "Implements Instant Neural Graphics Primitives (Instant-NGP) for fast NeRF training and rendering."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralRadianceFields.Models`

Implements Instant Neural Graphics Primitives (Instant-NGP) for fast NeRF training and rendering.

## For Beginners

Instant-NGP is a dramatically faster version of NeRF, making it practical for real-time use.

## How It Works

Speed comparison:

- Original NeRF: Hours to train, seconds to render each image
- Instant-NGP: Minutes to train, milliseconds to render each image
- Speedup: ~100× faster training, ~1000× faster rendering

Key innovations:

1. Multiresolution hash encoding: Replace expensive positional encoding
2. Tiny MLP: Much smaller network (2-4 layers vs 8 layers)
3. CUDA optimization: Highly optimized GPU kernels
4. Occupancy grids: Skip empty space efficiently

Multiresolution hash encoding explained:

- Traditional NeRF: Encode position with sin/cos functions (expensive)
- Instant-NGP: Look up features from a hash table (very fast)

How it works:

1. Multiple levels of resolution (coarse to fine)
2. At each level: Hash 3D position to table index
3. Look up learned features at that index
4. Interpolate features from nearby grid points
5. Concatenate features from all levels
6. Feed to small MLP for final color/density

Example with 3 levels:

- Level 0: Coarse grid (16³ cells) for large-scale features
- Level 1: Medium grid (64³ cells) for mid-scale features
- Level 2: Fine grid (256³ cells) for fine details
- Total table size: Much smaller than full 256³ grid
- Hash function maps 3D position to table index

Why hash tables are fast:

- No expensive trigonometric operations
- Direct memory lookup (O(1) time)
- Cache-friendly access patterns
- Parallelizes extremely well on GPU

Why hash collisions are okay:

- Multiple positions may hash to same index
- Network learns to handle collisions
- Collisions mostly affect similar regions
- Final quality is still excellent

Tiny MLP architecture:

- Original NeRF: 8 layers × 256 units = ~1M parameters
- Instant-NGP: 2 layers × 64 units = ~10K parameters
- Most representation power is in hash table, not MLP
- MLP just needs to combine hash features

Occupancy grids:

- Discretize space into voxel grid
- Mark which voxels contain geometry (occupancy)
- Skip sampling in empty voxels
- Huge speedup: Don't waste time on empty space

Example:

- Room scene: Most space is empty air
- Occupancy grid: Mark only voxels with walls/furniture
- Rendering: Skip ~90% of samples (the empty ones)
- Result: ~10× faster rendering

Training process:

1. Initialize hash tables randomly
2. Initialize occupancy grid (start assuming all occupied)
3. For each training iteration:
- Sample rays from training images
- Use occupancy grid to skip empty space
- Query hash tables + tiny MLP
- Compute rendering loss
- Backprop to update hash tables and MLP
- Periodically update occupancy grid
4. Converges in minutes instead of hours

Applications:

- Interactive 3D scanning: Scan object, view it seconds later
- Real-time novel view synthesis: Move camera and render instantly
- AR/VR: Low latency is critical for immersion
- Robotics: Build 3D maps in real-time
- Game development: Capture real objects for games

Limitations:

- Requires good GPU (CUDA implementation is fastest)
- Hash table size is a trade-off:
- Larger: Better quality, more memory
- Smaller: Faster, lower quality
- Still requires multiple images from different views
- Per-scene optimization (not a general model)

Comparison with NeRF:

Feature | NeRF | Instant-NGP
-------------------|-----------|-------------
Training time | 1-2 days | 5-10 minutes
Rendering speed | 30s/image | 30ms/image
Model size | ~5MB | ~50MB (with hash tables)
Quality | Excellent | Excellent
Memory usage | Low | Medium
GPU requirement | Any | CUDA (for best performance)

Reference: "Instant Neural Graphics Primitives with a Multiresolution Hash Encoding"
by Müller et al., ACM Transactions on Graphics 2022

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `InstantNGP(Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Double,Double,Double,Int32,Vector<>,Vector<>,ILossFunction<>)` | Initializes a new instance of the InstantNGP class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ForwardForTraining(Tensor<>)` | Tape-aware forward pass used by `TrainWithTape`. |
| `GetOptions` |  |
| `MultiresolutionHashEncoding(Tensor<>)` | Performs multiresolution hash encoding using vectorized Engine operations. |
| `NormalizePositionsToUnit(Tensor<>)` | Normalizes positions from world space to [0, 1] range based on scene bounds. |

