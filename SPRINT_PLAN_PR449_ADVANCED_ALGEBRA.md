# Implementation Plan: Advanced Algebra + Geometry (PR #449)

## Scope (must include in this PR)
- Octonions + octonion-based neural layers/models.
- Multivectors / Clifford algebras for geometric algebra transformers and Clifford neural networks.
- Hyperbolic manifolds and Riemannian optimization for hyperbolic models.
- Simplicial complexes and simplicial neural networks.
- Lie groups and Lie algebras with equivariant layers.
- Sparse tensors with dense interop and accelerated ops.

## Global Constraints (codebase standards)
- Use INumericOperations only; no INumber usage.
- No generic constraints on numeric types.
- net471 compatibility for core library; AiDotNet.Serving remains net8 only.
- All training paths must have AutoDiff + IEngine acceleration; finite-difference only as fallback.
- Industry-standard defaults via NumOps.FromDouble(...) (never default(T)).

## Non-Negotiable GPU Requirements
- Every new operation must have IEngine + GpuEngine implementations (no CPU-only paths).
- GPU kernels cover forward and backward paths; CPU fallback only for unsupported hardware or types.
- All new layers/models route compute through Engine to guarantee GPU acceleration.

## Architecture Targets
- Algebraic primitives live in `src/AiDotNet.Tensors/LinearAlgebra`.
- Numeric ops in `src/AiDotNet.Tensors/NumericOperations`.
- Engine ops in `src/AiDotNet.Tensors/Engines` (CPU + GPU parity for all new ops).
- Autodiff entry points in `src/Autodiff/TensorOperations.cs`.
- JIT coverage in `src/JitCompiler` (IR ops + codegen + GPU codegen).
- Layers/models in `src/NeuralNetworks` with parity to existing patterns.
- Benchmark suites in `src/Benchmarking` and domain-specific benchmark modules (integrated with BenchmarkSuite).
- Tests in `tests/AiDotNet.Tests` (UnitTests + IntegrationTests + Benchmarks).

## Benchmark + Dataset Coverage (all standard suites)
- Extend `BenchmarkSuite`, `BenchmarkSuiteRegistry`, `BenchmarkRunner`, and `BenchmarkingOptions` to include new Algebra/Geometry suites.
- All benchmarks consume datasets via `IDataLoader` and `DataLoaders`, with new loaders under `src/Data` (GraphDataLoaderBase/InputOutputDataLoaderBase derivatives).
- Dataset adapters include caching, deterministic sampling for CI, and synthetic fallback fixtures.
- Target benchmark suites (exhaustive coverage):
  - Octonion and hypercomplex: CIFAR-10/100, ImageNet (optional large), hyperspectral (Indian Pines, Pavia University, Salinas), audio (ESC-50, UrbanSound8K), multi-sensor fusion sets.
  - Clifford/Geometric Algebra: ModelNet40, ShapeNetCore, ScanNet, FAUST, S3DIS, KITTI 3D; physics field datasets (Navier-Stokes, Maxwell, elasticity).
  - Hyperbolic: WordNet subtrees, HyperLex, tree-structured classification, Cora/CiteSeer/PubMed, OGB graph benchmarks.
  - Simplicial: coauthorship/citation with higher-order relations, simplicial complex datasets, brain connectomes, synthetic TDA suites.
  - Lie groups: SO(3)/SE(3) pose benchmarks (ModelNet rotations, ShapeNet part, KITTI odometry, TUM RGB-D, EuRoC, Human3.6M).
  - Sparse tensors: MovieLens, Criteo, Amazon reviews, OGB sparse graphs, bag-of-words NLP corpora.

## Plan by Sprint (commit after each sprint)

### Sprint A: Algebraic Primitives (types + core ops)
**Goal:** Establish foundational numeric types with correct algebraic behavior.
- Octonion<T> struct:
  - Addition/subtraction, multiplication (non-associative), conjugate, norm, inverse.
  - Exponential/log (component-wise + polar handling).
  - JSON/CSV-friendly formatting and parsing helpers.
- Multivector<T> + Clifford signature:
  - Basis blade indexing, grade decomposition, reverse, involution, dual.
  - Geometric product, outer (wedge), inner products.
  - CliffordAlgebra signature (p, q, r) and basis cache.
- Hyperbolic manifold primitives:
  - Poincare ball + hyperboloid models (exp/log, mobius add, distance, parallel transport).
  - Curvature parameterization with safe defaults (c = 1.0).
- Simplicial complex primitives:
  - Simplex and complex containers, boundary operators, incidence matrices.
  - Hodge Laplacians (0-, 1-, and 2-forms).
- Lie groups/algebras:
  - Interfaces for LieGroup and LieAlgebra (exp/log, adjoint, composition).
  - Implement SO(3), SE(3), SU(2) with stable numerics.
- SparseTensor<T> core:
  - COO/CSR/CSC storage, coalesce, transpose, to/from dense.

**Deliverables:**
- New types + algebraic unit tests (identities, invariants, norms, round-trips).
- NumericOperations stubs for Octonion and Multivector with correct scalar behavior.

### Sprint B: NumericOperations + Engine Integration
**Goal:** Make new types usable across existing tensor APIs.
- INumericOperations implementations:
  - OctonionOperations<T>, MultivectorOperations<T>.
  - Ensure FromDouble/Zero/One consistent with algebra semantics.
- IEngine support (CPU + GPU parity):
  - Elementwise add/sub/mul/div for Octonion and Multivector tensors.
  - Sparse tensor ops: SpMM, SpMV, sparse-dense add/mul, gather/scatter.
  - Hyperbolic ops (exp/log/mobius add/distance) as engine kernels.
- GPU strategy (mandatory):
  - Provide GPU kernels for octonion, multivector/clifford, hyperbolic, simplicial, lie group, and sparse tensor ops.
  - CPU fallback only when GPU is unavailable; enforce CPU/GPU parity tests.

**Deliverables:**
- Engine tests for correctness and determinism.
- Baseline perf benchmarks for dense vs sparse ops.

### Sprint C: AutoDiff + JIT Coverage
**Goal:** Ensure training paths are fully autodiff and JIT-friendly.
- Autodiff nodes:
  - OctonionMatMul, OctonionMultiply, OctonionLinear.
  - GeometricProduct, WedgeProduct, InnerProduct for multivectors.
  - Hyperbolic ops (exp/log/mobius add/distance) with gradients.
  - Sparse ops (SpMM, sparse gather/scatter) with gradients.
- JIT:
  - IR op types + codegen for above ops (CPU + GPU).
  - Ensure net471-compatible codegen paths.

**Deliverables:**
- Gradient checks (finite-difference comparison) for each new op.
- JIT unit tests mirroring Autodiff coverage.

## Model Coverage Matrix (exhaustive)
- Octonion:
  - Layers: OctonionLinear, OctonionConv1D/2D/3D, OctonionBatchNorm, OctonionLayerNorm, OctonionDropout.
  - Models: OctonionMLP, OctonionResNet, OctonionUNet, OctonionRNN/LSTM/GRU, OctonionTransformer, OctonionAutoencoder, OctonionVAE, OctonionGAN, OctonionGNN.
- Clifford / Geometric Algebra:
  - Layers: MultivectorLinear, CliffordConv1D/2D/3D, CliffordBatchNorm, CliffordLayerNorm, GeometricProductLayer, WedgeProductLayer, InnerProductLayer.
  - Models: CliffordMLP, CliffordResNet, CliffordUNet, CliffordRNN, GeometricAlgebraTransformer, CliffordAutoencoder, CliffordVAE, CliffordGAN, CliffordGNN.
- Hyperbolic:
  - Layers: HyperbolicEmbedding, HyperbolicLinear, HyperbolicLayerNorm, HyperbolicAttention, HyperbolicPooling.
  - Models: HyperbolicMLP, HyperbolicGNN (HGCN/HGNN), HyperbolicTransformer, HyperbolicRNN, HyperbolicAutoencoder, HyperbolicVAE, HyperbolicContrastiveModel.
  - Optimizers: RSGD, RAdam, RiemannianAdaGrad with projection/retraction.
- Simplicial:
  - Layers: SimplicialConvolution, HodgeLaplacianLayer, SimplicialAttention, SimplicialPooling.
  - Models: SimplicialGNN, SimplicialTransformer, SimplicialAutoencoder, SimplicialUNet.
- Lie Groups / Lie Algebras:
  - Layers: SO(3)/SE(3)/SU(2) equivariant layers, LieGroupNormalization, LieGroupAttention, LieGroupLinear.
  - Models: LieConvNet, SE3Transformer, LieGroupRNN, LieGroupGNN, PoseTransformer.
- Sparse Tensors:
  - Layers: SparseLinear, SparseConv1D/2D/3D, SparseBatchNorm, SparseAttention, SparsePooling.
  - Models: SparseMLP, SparseTransformer, SparseMoE, SparseGNN, SparseAutoencoder.

### Sprint D: Model Layers + Architectures
**Goal:** Implement all models listed in the Model Coverage Matrix with full AutoDiff + Engine + JIT support.

**Deliverables:**
- End-to-end training smoke tests for each model family.
- Example configs in existing model builder patterns.

### Sprint E: Tests, Benchmarks, and Docs
**Goal:** Production-grade validation and documentation.
- Unit tests:
  - Algebraic identities (octonion non-associativity, Clifford grades).
  - Hyperbolic geodesic distance and exp/log round-trips.
  - Lie group exp/log consistency, adjoint invariants.
  - Simplicial boundary and Laplacian properties.
  - Sparse tensor correctness vs dense for small sizes.
- Integration tests:
  - Train small models to convergence on synthetic tasks.
  - Verify AutoDiff + IEngine paths are used.
- Benchmarks:
  - Sparse vs dense ops; hyperbolic ops; Clifford ops; octonion ops; simplicial and lie group ops.
  - Full benchmark suites integrated into `BenchmarkingOptions` + `BenchmarkRunner` and the existing benchmark registry.
  - All datasets integrated with `IDataLoader`/`DataLoaders` and domain loader bases (GraphDataLoaderBase/InputOutputDataLoaderBase).
- Docs:
  - API usage pages and example code for each feature.

**Deliverables:**
- Benchmark tests committed (CI-friendly, deterministic).
- Benchmark suites registered in BenchmarkSuiteRegistry with dataset loader integration.
- Updated docs in `docs/` with usage examples.

## Verification Matrix (production readiness)
- Algebra: identity tests + randomized property checks.
- Autodiff: gradient checks for each new operation.
- Engine: CPU correctness; GPU parity for every new op.
- JIT: compile + run with parity to non-JIT outputs (CPU + GPU).
- Training: at least one working model per feature set.
- net471: all tests and build succeed for core library.
- Datasets: loader parity tests and benchmark suite smoke runs.

## Risks and Mitigations
- Non-associativity (octonions): enforce left-to-right multiply order and test explicitly.
- Clifford dimensionality explosion: use basis caching and sparse storage for high-grade.
- Hyperbolic stability: clamp norms and use safe epsilon in exp/log.
- Sparse tensor performance: avoid densification; add coalesce paths.

## Decisions Locked for This PR
- GPU acceleration required for every new op/model via IEngine + GpuEngine.
- Model coverage must include the full exhaustive list in the Model Coverage Matrix.
- Benchmarks and datasets are comprehensive and integrated with the existing benchmark system and dataloaders.
