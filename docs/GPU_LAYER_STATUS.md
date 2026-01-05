# GPU Layer ForwardGpu Implementation Status

This document tracks which neural network layers have GPU-optimized `ForwardGpu` implementations.

## Legend

| Status | Meaning |
|--------|---------|
| Done | `SupportsGpuExecution => true` and `ForwardGpu` implemented |
| Pending | Implementation planned but not yet started |
| Blocked | Implementation blocked by missing GPU kernel or infrastructure |
| N/A | Layer does not benefit from GPU (trivial operation or CPU-bound) |

---

## Priority 1: Core Layers (High Impact)

These layers are used in most neural networks and have high GPU acceleration potential.

| Layer | Status | Notes |
|-------|--------|-------|
| DenseLayer | Done | Uses FusedGemmBiasRelu, any-rank tensor support |
| ConvolutionalLayer | Done | Uses FusedConv2DGpu with MapActivationToFused |
| BatchNormalizationLayer | Pending | Needs BatchNorm GPU kernel |
| MaxPoolingLayer | Done | Basic GPU support |
| AveragePoolingLayer | Done | Basic GPU support |
| DropoutLayer | Done | Training mode check, pass-through inference |
| FlattenLayer | Done | Zero-copy reshape via CreateView |
| ReshapeLayer | Done | Zero-copy reshape via CreateView |
| ActivationLayer | Pending | Uses existing activation kernels |

---

## Priority 2: Attention & Transformer Layers (High Impact for LLMs)

| Layer | Status | Notes |
|-------|--------|-------|
| MultiHeadAttentionLayer | Pending | Needs attention kernel (Q*K^T, softmax, V matmul) |
| SelfAttentionLayer | Pending | Similar to MultiHeadAttention |
| CrossAttentionLayer | Pending | Similar to MultiHeadAttention |
| TransformerEncoderLayer | Pending | Composition of attention + FFN |
| TransformerDecoderLayer | Pending | Composition of attention + FFN |
| PositionalEncodingLayer | Pending | Simple addition, may not need GPU |
| FeedForwardLayer | Pending | Linear + activation, similar to Dense |
| EmbeddingLayer | Pending | Lookup table, may benefit from GPU |

---

## Priority 3: Normalization Layers

| Layer | Status | Notes |
|-------|--------|-------|
| LayerNormalizationLayer | Pending | Needs LayerNorm GPU kernel |
| GroupNormalizationLayer | Pending | Needs GroupNorm GPU kernel |
| InstanceNormalizationLayer | Pending | Needs InstanceNorm GPU kernel |
| SpectralNormalizationLayer | Pending | Needs SVD on GPU |

---

## Priority 4: Convolutional Variants

| Layer | Status | Notes |
|-------|--------|-------|
| DeconvolutionalLayer | Pending | Needs transposed conv kernel |
| DepthwiseSeparableConvolutionalLayer | Pending | Needs depthwise conv kernel |
| DilatedConvolutionalLayer | Pending | Needs dilated conv kernel |
| DeformableConvolutionalLayer | Blocked | Complex kernel, low priority |
| Conv3DLayer | Pending | Needs 3D convolution kernel |
| SeparableConvolutionalLayer | Pending | Composition of depthwise + pointwise |
| SubpixelConvolutionalLayer | Pending | Pixel shuffle operation |
| LocallyConnectedLayer | Pending | Non-shared weights |

---

## Priority 5: Recurrent Layers

| Layer | Status | Notes |
|-------|--------|-------|
| LSTMLayer | Pending | Needs LSTM gates kernel |
| GRULayer | Pending | Needs GRU gates kernel |
| RecurrentLayer | Pending | Basic recurrence |
| BidirectionalLayer | Pending | Wrapper, depends on inner layer |
| ConvLSTMLayer | Blocked | Complex kernel combination |

---

## Priority 6: Pooling & Spatial Layers

| Layer | Status | Notes |
|-------|--------|-------|
| GlobalPoolingLayer | Pending | Reduction kernel |
| AdaptiveAveragePoolingLayer | Pending | Variable-size pooling |
| MaxPool3DLayer | Pending | 3D max pooling kernel |
| PoolingLayer | Done | Generic pooling base |
| UpsamplingLayer | Pending | Interpolation kernel |
| Upsample3DLayer | Pending | 3D interpolation |
| CroppingLayer | N/A | Simple tensor slice |
| PaddingLayer | N/A | Simple tensor padding |

---

## Priority 7: Graph Neural Network Layers

| Layer | Status | Notes |
|-------|--------|-------|
| GraphConvolutionalLayer | Blocked | Needs sparse matrix support |
| GraphAttentionLayer | Blocked | Sparse attention |
| GraphSAGELayer | Blocked | Neighborhood sampling |
| GraphIsomorphismLayer | Blocked | Graph isomorphism |
| GraphTransformerLayer | Blocked | Graph + transformer |
| MessagePassingLayer | Blocked | Generic message passing |
| DirectionalGraphLayer | Blocked | Directed graphs |
| HeterogeneousGraphLayer | Blocked | Multiple node/edge types |
| EdgeConditionalConvolutionalLayer | Blocked | Edge features |
| DiffusionConvLayer | Blocked | Diffusion on graphs |
| MeshEdgeConvLayer | Blocked | Mesh processing |
| MeshPoolLayer | Blocked | Mesh pooling |
| PrincipalNeighbourhoodAggregationLayer | Blocked | PNA aggregation |
| SpiralConvLayer | Blocked | Spiral convolution |

---

## Priority 8: Capsule Network Layers

| Layer | Status | Notes |
|-------|--------|-------|
| CapsuleLayer | Blocked | Dynamic routing |
| PrimaryCapsuleLayer | Blocked | Initial capsule |
| DigitCapsuleLayer | Blocked | Output capsules |

---

## Priority 9: Specialized Layers

| Layer | Status | Notes |
|-------|--------|-------|
| HighwayLayer | Pending | Gate + Dense |
| ResidualLayer | Pending | Skip connection wrapper |
| GatedLinearUnitLayer | Pending | GLU gating |
| SqueezeAndExcitationLayer | Pending | Channel attention |
| MixtureOfExpertsLayer | Blocked | Expert routing |
| ExpertLayer | Pending | Individual expert |
| DenseBlockLayer | Pending | DenseNet block |
| RRDBLayer | Pending | RRDB block |
| TransitionLayer | Pending | DenseNet transition |
| RepParameterizationLayer | Pending | Re-parameterization |

---

## Priority 10: Memory & Attention Variants

| Layer | Status | Notes |
|-------|--------|-------|
| AttentionLayer | Pending | Basic attention |
| MemoryReadLayer | Blocked | NTM memory |
| MemoryWriteLayer | Blocked | NTM memory |
| ContinuumMemorySystemLayer | Blocked | Continuous memory |
| SpatialTransformerLayer | Blocked | Spatial attention |

---

## Priority 11: Sequence Layers

| Layer | Status | Notes |
|-------|--------|-------|
| SequenceLastLayer | N/A | Simple index |
| TimeDistributedLayer | Pending | Apply layer across time |
| TimeEmbeddingLayer | Pending | Diffusion time embedding |
| PatchEmbeddingLayer | Pending | ViT patch embedding |

---

## Priority 12: Simple Operations (Low Priority)

| Layer | Status | Notes |
|-------|--------|-------|
| AddLayer | N/A | Element-wise add |
| MultiplyLayer | N/A | Element-wise multiply |
| ConcatenateLayer | N/A | Tensor concatenation |
| SplitLayer | N/A | Tensor split |
| InputLayer | N/A | Identity |
| LambdaLayer | N/A | User-defined |
| MaskingLayer | N/A | Mask application |
| GaussianNoiseLayer | N/A | Random noise |

---

## Priority 13: Experimental/Specialized

| Layer | Status | Notes |
|-------|--------|-------|
| QuantumLayer | Blocked | Quantum simulation |
| SpikingLayer | Blocked | Spiking neural networks |
| RBFLayer | Pending | Radial basis function |
| RBMLayer | Blocked | Restricted Boltzmann machine |
| ReservoirLayer | Blocked | Echo state network |
| ReadoutLayer | Pending | Linear readout |
| AnomalyDetectorLayer | Pending | Anomaly detection |
| ConditionalRandomFieldLayer | Blocked | CRF layer |
| HyperbolicLinearLayer | Blocked | Hyperbolic geometry |
| OctonionLinearLayer | Blocked | Octonion algebra |
| SpatialPoolerLayer | Blocked | HTM spatial pooler |
| TemporalMemoryLayer | Blocked | HTM temporal memory |
| SynapticPlasticityLayer | Blocked | Synaptic plasticity |
| MeasurementLayer | Blocked | Quantum measurement |
| SpyNetLayer | Blocked | Optical flow |
| PixelShuffleLayer | Pending | Pixel shuffle upsampling |
| DecoderLayer | Pending | Generic decoder |
| ReconstructionLayer | Pending | VAE reconstruction |
| LogVarianceLayer | Pending | VAE log variance |
| MeanLayer | Pending | VAE mean |
| FullyConnectedLayer | Pending | Alias for Dense |

---

## Summary Statistics

| Category | Done | Pending | Blocked | N/A | Total |
|----------|------|---------|---------|-----|-------|
| Core Layers | 8 | 1 | 0 | 0 | 9 |
| Attention/Transformer | 0 | 8 | 0 | 0 | 8 |
| Normalization | 0 | 4 | 0 | 0 | 4 |
| Conv Variants | 0 | 7 | 1 | 0 | 8 |
| Recurrent | 0 | 4 | 1 | 0 | 5 |
| Pooling/Spatial | 1 | 5 | 0 | 2 | 8 |
| Graph NN | 0 | 0 | 14 | 0 | 14 |
| Capsule | 0 | 0 | 3 | 0 | 3 |
| Specialized | 0 | 9 | 1 | 0 | 10 |
| Memory/Attention | 0 | 1 | 4 | 0 | 5 |
| Sequence | 0 | 4 | 0 | 1 | 5 |
| Simple Ops | 0 | 0 | 0 | 8 | 8 |
| Experimental | 0 | 8 | 13 | 0 | 21 |
| **Total** | **9** | **51** | **37** | **11** | **108** |

---

## Next Steps

1. **BatchNormalizationLayer** - Create BatchNorm GPU kernel (mean, variance, normalize, scale/shift)
2. **MultiHeadAttentionLayer** - Create attention GPU kernel (QKV matmul, softmax, output projection)
3. **LayerNormalizationLayer** - Create LayerNorm GPU kernel
4. **ActivationLayer** - Wire up existing activation kernels

---

## Blockers

| Blocker | Affected Layers | Resolution |
|---------|-----------------|------------|
| Sparse matrix GPU support | All Graph NN layers | Implement CSR/COO sparse kernels |
| Dynamic routing | Capsule layers | Implement dynamic routing kernel |
| Complex memory addressing | Memory layers | Implement content-based addressing |
| SVD on GPU | SpectralNormalization | Implement iterative power method |

---

## Last Updated

2025-01-05 - Initial comprehensive layer list created
