# PR #447 3D AI Implementation Plan

This plan defines the complete scope for production-ready 3D AI support in
AiDotNet and the phased roadmap to deliver it. Each phase is a checklist of
concrete deliverables; no placeholders or stubs are acceptable.

## Goals
- Exhaustive 3D support across point clouds, triangle meshes, voxel grids, and
  implicit fields, with real-world performance and accuracy.
- First-class integration with existing performance stack: AutoDiff, JIT,
  CPU/GPU acceleration via IEngine, and numerically stable defaults.
- Full configurability: every model and pipeline component exposes options with
  industry-standard defaults set via NumOps.FromDouble() in constructors.
- Verified via integration tests, benchmarks, and documented example workflows.

## Phase 0 - Audit Baseline (Complete the gap analysis, lock scope)
- [x] Audit issue #399 and PR #447 for required features and omissions.
- [x] Identify missing 3D primitives, model families, metrics, and benchmarks.
- [ ] Finalize acceptance criteria per feature family (models, datasets, tests).
- [ ] Establish minimal performance targets (CPU/GPU throughput baselines).

## Phase 1 - Core 3D Data Structures and IO
- [x] Triangle mesh data structure with validated topology and attributes.
- [x] Voxel grid data structure with world-space metadata (origin, voxel size).
- [x] Point cloud conversion utilities (mesh -> points, voxels -> points).
- [ ] IO utilities for mesh and point data (OBJ, PLY, STL) with tests.
- [ ] Dataset adapters for ModelNet40, ShapeNetCore, ScanNet (IO + metadata).

## Phase 2 - 3D Preprocessing and Geometry Ops
- [ ] Standardization and normalization (center, scale, unit sphere).
- [ ] Sampling utilities (uniform, farthest point, Poisson disk).
- [ ] Neighbor search (kNN, radius) with CPU/GPU acceleration.
- [ ] Mesh ops (normal computation, adjacency, subdivision, simplification).
- [ ] Voxelization from mesh/points, occupancy grid generation.
- [ ] Quality metrics for geometry (Chamfer, EMD, F-score, IoU for voxels).

## Phase 3 - Model Families (Production-ready)
- [ ] Point-based: PointNet, PointNet++, DGCNN (classification + segmentation).
- [ ] Voxel-based: 3D CNN (VoxNet-style), 3D U-Net (segmentation).
- [ ] Mesh-based: MeshCNN (edge conv + pooling), mesh GNN variants.
- [ ] Implicit: NeRF, Instant-NGP with correct occupancy grid; SDF (DeepSDF/NeuS).
- [ ] Explicit: Gaussian Splatting with training, densification, and rendering.

## Phase 4 - Training Pipelines, Metrics, and Benchmarks
- [ ] End-to-end training runners for each model family.
- [ ] Metrics: PSNR, SSIM, LPIPS (rendering); mIoU/accuracy for segmentation.
- [ ] Benchmarks: ModelNet40, ShapeNet, ScanNet, NeRF synthetic/real datasets.
- [ ] Performance validation: CPU vs GPU profiling and memory usage.

## Phase 5 - Integration, Examples, and Documentation
- [ ] Vision model integration hooks (ViT adapters for 3D embeddings).
- [ ] Example pipelines (classification, segmentation, detection, rendering).
- [ ] Updated docs with configuration tables and reproducible benchmarks.
- [ ] Migration notes and API stability review.

## Definition of Done (per phase)
- All tasks in the phase are implemented (no TODOs or stubs).
- Integration tests cover core paths and edge cases.
- Defaults use NumOps.FromDouble() and are fully configurable via options.
- Performance paths respect AutoDiff, JIT, and IEngine where applicable.
