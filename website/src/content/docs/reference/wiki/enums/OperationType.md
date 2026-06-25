---
title: "OperationType"
description: "Represents different operation types in computation graphs for JIT compilation and automatic differentiation."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Represents different operation types in computation graphs for JIT compilation and automatic differentiation.

## For Beginners

Operation types identify mathematical operations performed on tensors in neural networks.

When building a computation graph, each operation (like adding two tensors or applying an activation function)
needs to be identified so that:

1. The JIT compiler can optimize the code
2. The automatic differentiation system can compute gradients correctly
3. The system can analyze and transform the computation graph

This enum provides type-safe identification of operations, preventing typos and enabling better tooling support.

## Fields

| Field | Summary |
|:-----|:--------|
| `Abs` | Element-wise absolute value - \|x\| for each element. |
| `Activation` | Generic activation function application. |
| `AdaptivePooling` | Adaptive pooling. |
| `Add` | Element-wise addition of two tensors. |
| `AffineGrid` | Affine grid generation for spatial transformers. |
| `And` | Logical AND. |
| `AnomalyScore` | Anomaly score computation. |
| `Attention` | Generic attention mechanism operation. |
| `AveragePooling` | Average pooling operation. |
| `AvgPool2D` | 2D average pooling. |
| `BatchNorm` | Batch normalization. |
| `BatchNormalization` | Batch normalization. |
| `BentIdentity` | Bent Identity - (sqrt(x² + 1) - 1) / 2 + x, smooth alternative to ReLU. |
| `Broadcast` | Broadcast operation - expands tensor dimensions to match target shape. |
| `CELU` | Continuously Differentiable ELU - max(0, x) + min(0, α * (exp(x/α) - 1)). |
| `CRFForward` | CRF forward algorithm for sequence labeling. |
| `Cast` | Type cast operation. |
| `CliffordInnerProduct` | Inner (contraction) product of multivectors. |
| `Clip` | Clip values to range. |
| `ComplexMatMul` | Complex matrix multiplication for quantum operations. |
| `ComplexMultiply` | Element-wise complex multiplication. |
| `Concat` | Concatenate multiple tensors along an axis. |
| `Constant` | Constant node - represents a constant value that doesn't require gradients. |
| `Conv2D` | 2D convolution operation. |
| `ConvTranspose2D` | 2D transposed convolution (deconvolution). |
| `Convolution` | General convolution operation. |
| `Convolution2D` | 2D convolution operation. |
| `Convolution3D` | 3D convolution operation. |
| `Crop` | Crop tensor by removing border elements. |
| `CrossAttention` | Cross-attention operation. |
| `Custom` | Custom user-defined operation for extensibility. |
| `Deconvolution` | Deconvolution (transposed convolution) operation. |
| `DeformableConv2D` | 2D deformable convolution with learnable offsets and optional modulation. |
| `Dense` | Dense (fully connected) layer. |
| `DepthwiseConv2D` | 2D depthwise convolution. |
| `DepthwiseConvolution` | Depthwise convolution operation. |
| `DilatedConv2D` | 2D dilated (atrous) convolution. |
| `DilatedConvolution` | Dilated convolution operation. |
| `Divide` | Element-wise division of two tensors. |
| `DropPath` | DropPath regularization. |
| `Dropout` | Dropout regularization operation - randomly zeros elements during training. |
| `ELU` | Exponential Linear Unit - ELU(x) = x if x > 0, alpha * (exp(x) - 1) otherwise. |
| `Embedding` | Embedding lookup operation. |
| `Equal` | Element-wise equality. |
| `Exp` | Element-wise exponential function - e^x for each element. |
| `Expand` | Expand tensor dimensions. |
| `FakeQuantization` | Fake quantization operation with Straight-Through Estimator (STE) for differentiable quantization. |
| `Flatten` | Flatten tensor to 1D. |
| `FullyConnected` | Fully connected layer. |
| `FusedAddReLU` | Fused addition + ReLU. |
| `FusedConvBatchNorm` | Fused convolution + batch normalization. |
| `FusedConvBatchNormReLU` | Fused Conv + BatchNorm + ReLU. |
| `FusedLayerNormAttention` | Fused LayerNorm + Attention. |
| `FusedLinearReLU` | Fused linear layer with ReLU (MatMul + Add + ReLU). |
| `FusedMatMulAdd` | Fused matrix multiplication + addition (MatMul + Add). |
| `FusedMatMulBias` | Fused MatMul + Bias. |
| `FusedMatMulBiasGELU` | Fused MatMul + Bias + GELU. |
| `FusedMatMulBiasReLU` | Fused MatMul + Bias + ReLU. |
| `FusedMultiHeadAttention` | Fused MultiHead Attention. |
| `GELU` | Gaussian Error Linear Unit - x * Φ(x) where Φ is standard normal CDF. |
| `GRU` | GRU recurrent layer. |
| `GRUCell` | GRU cell operation for recurrent networks. |
| `Gather` | Gather operation - selects elements from a tensor using indices. |
| `Gaussian` | Gaussian activation - exp(-x²), bell-shaped response curve. |
| `Gemm` | General Matrix Multiplication. |
| `GeometricProduct` | Geometric product of multivectors. |
| `GlobalAveragePooling` | Global average pooling. |
| `GlobalMaxPooling` | Global max pooling. |
| `GradeProject` | Grade projection of multivectors. |
| `GraphConv` | Graph convolutional operation for GNNs. |
| `Greater` | Element-wise greater than. |
| `GreaterOrEqual` | Element-wise greater or equal. |
| `GridSample` | Grid sampling for spatial transformers. |
| `GroupNormalization` | Group normalization. |
| `GumbelSoftmax` | Gumbel-Softmax for differentiable discrete sampling (used in stochastic layers). |
| `HardSigmoid` | Hard Sigmoid - piecewise linear approximation of sigmoid: clip((x + 1) / 2, 0, 1). |
| `HardTanh` | Hard Tanh - piecewise linear approximation of tanh: clip(x, -1, 1). |
| `HierarchicalSoftmax` | Hierarchical Softmax - tree-based efficient softmax for large vocabularies. |
| `HyperboloidDistance` | Hyperboloid distance metric. |
| `ISRU` | Inverse Square Root Unit - x / sqrt(1 + alpha * x²). |
| `Input` | Input node - represents a variable or parameter in the computation graph. |
| `InstanceNormalization` | Instance normalization. |
| `LSTM` | LSTM recurrent layer. |
| `LSTMCell` | LSTM cell operation for recurrent networks. |
| `LayerNorm` | Layer normalization. |
| `LayerNormalization` | Layer normalization. |
| `LeakyReLU` | Leaky Rectified Linear Unit - max(alpha * x, x) where alpha is typically 0.01. |
| `LeakyStateUpdate` | Leaky state update for reservoir/echo state networks. |
| `Less` | Element-wise less than. |
| `LessOrEqual` | Element-wise less or equal. |
| `LiSHT` | Linearly Scaled Hyperbolic Tangent - x * tanh(x). |
| `LocallyConnectedConv2D` | 2D locally connected convolution. |
| `Log` | Element-wise natural logarithm. |
| `LogSoftmax` | Log-Softmax - log(softmax(x)), numerically stable for cross-entropy loss. |
| `LogSoftmin` | Log-Softmin - log(softmin(x)) = log(softmax(-x)). |
| `MatMul` | Matrix multiplication (not element-wise). |
| `MaxPool2D` | 2D max pooling. |
| `MaxPool3D` | 3D max pooling. |
| `MaxPooling` | Max pooling operation. |
| `Maxout` | Maxout activation - maximum over multiple linear pieces. |
| `Mean` | Mean operation (reduces all dimensions). |
| `Mish` | Mish activation - x * tanh(softplus(x)). |
| `MobiusAdd` | Mobius addition in Poincare ball model. |
| `MultiHeadAttention` | Multi-head attention operation. |
| `Multiply` | Element-wise multiplication (Hadamard product) of two tensors. |
| `MultivectorAdd` | Multivector addition. |
| `MultivectorReverse` | Multivector reverse operation. |
| `Negate` | Element-wise negation - multiplies each element by -1. |
| `Norm` | L2 norm computation along an axis - sqrt(sum(x²)). |
| `Not` | Logical NOT. |
| `OctonionAdd` | Octonion addition. |
| `OctonionConjugate` | Octonion conjugation. |
| `OctonionMatMul` | Octonion matrix multiplication for neural networks. |
| `OctonionMultiply` | Octonion multiplication (non-associative). |
| `Or` | Logical OR. |
| `Output` | Output node in computation graph. |
| `PReLU` | Parametric Rectified Linear Unit - max(0, x) + alpha * min(0, x) where alpha is learned. |
| `Pad` | Pad tensor with values. |
| `Permute` | Permute tensor dimensions (general transpose). |
| `PixelShuffle` | Pixel shuffle operation for upsampling. |
| `PoincareDistance` | Poincare ball distance metric. |
| `PoincareExpMap` | Poincare exponential map. |
| `PoincareLogMap` | Poincare logarithmic map. |
| `PositionalEncoding` | Positional encoding for transformers. |
| `Power` | Element-wise power operation - raises each element to a specified exponent. |
| `RBFKernel` | RBF (Radial Basis Function) kernel operation. |
| `RNN` | Basic RNN layer. |
| `RReLU` | Randomized Leaky ReLU - LeakyReLU with random alpha during training. |
| `ReLU` | Rectified Linear Unit - max(0, x). |
| `ReduceLogVariance` | Log-variance reduction along specified axes. |
| `ReduceMax` | Maximum value reduction along specified axes. |
| `ReduceMean` | Mean reduction along specified axes. |
| `ReduceMin` | Minimum value reduction. |
| `ReduceSum` | Sum reduction along specified axes. |
| `Reshape` | Reshape tensor to new dimensions. |
| `SELU` | Scaled Exponential Linear Unit - self-normalizing activation with fixed lambda and alpha. |
| `SQRBF` | Square Radial Basis Function - smooth bell-shaped activation. |
| `ScaledDotProductAttention` | Scaled dot-product attention. |
| `ScaledTanh` | Scaled Tanh - parameterized tanh with adjustable steepness β. |
| `Scatter` | Scatter values to indices. |
| `Se3Exp` | SE(3) exponential map. |
| `Se3Log` | SE(3) logarithmic map. |
| `SelfAttention` | Self-attention operation. |
| `Sigmoid` | Sigmoid activation - 1 / (1 + e^(-x)). |
| `Sign` | Sign function with surrogate gradient for training - returns -1, 0, or 1. |
| `Slice` | Slice tensor along an axis - extract a portion with optional stride. |
| `So3Exp` | SO(3) exponential map. |
| `So3Log` | SO(3) logarithmic map. |
| `SoftKNN` | Soft K-Nearest Neighbors operation for differentiable instance-based learning. |
| `SoftLocallyWeighted` | Soft locally-weighted regression operation for differentiable instance-based learning. |
| `SoftPlus` | SoftPlus activation - ln(1 + e^x), smooth approximation of ReLU. |
| `SoftSign` | SoftSign activation - x / (1 + \|x\|), alternative to tanh with polynomial tails. |
| `SoftSplit` | Soft split operation for differentiable decision trees. |
| `Softmax` | Softmax activation - converts logits to probability distribution. |
| `Softmin` | Softmin - softmax(-x), assigns higher probability to lower values. |
| `SpMM` | Sparse matrix-matrix multiplication. |
| `SpMV` | Sparse matrix-vector multiplication. |
| `SparseGather` | Sparse gather operation. |
| `SparseScatter` | Sparse scatter operation. |
| `Sparsemax` | Sparsemax - projects onto probability simplex, can produce sparse outputs. |
| `SphericalSoftmax` | Spherical Softmax - L2 normalization followed by softmax. |
| `Split` | Split tensor along an axis into multiple tensors. |
| `Sqrt` | Element-wise square root. |
| `Square` | Element-wise square - x² for each element. |
| `Squash` | Squashing activation for capsule networks - s(v) = \|\|v\|\|² / (1 + \|\|v\|\|²) * (v / \|\|v\|\|). |
| `Squeeze` | Remove dimensions of size 1. |
| `Stack` | Stack tensors along new axis. |
| `StraightThroughThreshold` | Straight-through threshold for HTM-style sparse activations. |
| `Subtract` | Element-wise subtraction of two tensors. |
| `SurrogateSpike` | Surrogate spike function for spiking neural networks with gradient estimation. |
| `Swish` | Swish/SiLU activation - x * sigmoid(x). |
| `Tanh` | Hyperbolic tangent activation. |
| `TaylorSoftmax` | Taylor Softmax - softmax using Taylor series approximation of exp. |
| `ThresholdedReLU` | Thresholded Rectified Linear Unit - x if x > threshold, 0 otherwise. |
| `TopKSoftmax` | Top-K softmax for mixture-of-experts routing. |
| `Transpose` | Matrix transpose - swaps rows and columns. |
| `Unknown` | Unknown operation type. |
| `Unsqueeze` | Add dimension of size 1. |
| `Upsample` | Upsample tensor by repeating elements. |
| `Upsample3D` | 3D upsampling operation for volumetric data. |
| `WedgeProduct` | Wedge (outer) product of multivectors. |
| `Xor` | Logical XOR. |

