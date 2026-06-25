---
title: "TensorOperations<T>"
description: "Provides computation graph operations on `ComputationNode` for legacy autodiff."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Autodiff`

Provides computation graph operations on `ComputationNode` for legacy autodiff.

## How It Works

TensorOperations builds a `ComputationNode` graph with backward functions
for reverse-mode differentiation. Each operation creates a node that remembers its inputs
and how to propagate gradients.

**Note:** Tape recording for these operations is handled by the engine layer
(via `AiDotNet.Tensors.Engines.Autodiff.GradientTape`). This class no longer
records to a tape directly.

## Methods

| Method | Summary |
|:-----|:--------|
| `Abs(ComputationNode<>)` | Computes the absolute value of each element in a computation node. |
| `Add(ComputationNode<>,ComputationNode<>)` | Performs element-wise addition of two computation nodes. |
| `AffineGrid(ComputationNode<>,Int32,Int32)` | Generates a sampling grid for spatial transformer networks using affine transformation matrices. |
| `AnomalyScore(ComputationNode<>,ComputationNode<>)` | Anomaly score computation using reconstruction error or density estimation. |
| `ApplyActivation(ComputationNode<>,IActivationFunction<>)` | Applies a generic activation function (scalar or element-wise) with automatic differentiation. |
| `Atanh(Double)` | Computes arctanh (inverse hyperbolic tangent) with numerical stability. |
| `AvgPool2D(ComputationNode<>,Int32[],Int32[])` | Performs 2D average pooling on a 4D tensor (batch, channels, height, width). |
| `BatchMatrixMultiply(ComputationNode<>,ComputationNode<>)` | Performs batched matrix multiplication of two 3D computation nodes. |
| `BatchNorm(ComputationNode<>,ComputationNode<>,ComputationNode<>,Tensor<>,Tensor<>,Boolean,Double)` | Applies batch normalization to a computation node. |
| `BentIdentity(ComputationNode<>)` | Applies the Bent Identity activation function element-wise. |
| `Broadcast(ComputationNode<>,Int32[])` | Broadcasts a 1D tensor to a 2D tensor by tiling along the batch dimension. |
| `BroadcastAdd(Tensor<>,Tensor<>,INumericOperations<>)` | Performs broadcasting addition of two tensors with different shapes. |
| `BroadcastAddHelper(Tensor<>,Tensor<>,INumericOperations<>)` | Helper method that broadcasts the smaller tensor to match the larger one. |
| `BroadcastMultiply(Tensor<>,Tensor<>,INumericOperations<>)` | Performs broadcasting multiplication of two tensors with different shapes. |
| `CELU(ComputationNode<>,Double)` | Applies the CELU (Continuously Differentiable ELU) activation function element-wise. |
| `CRFForward(ComputationNode<>,ComputationNode<>,ComputationNode<>,ComputationNode<>)` | CRF forward algorithm for sequence labeling. |
| `ComplexMatMul(ComputationNode<>,ComputationNode<>,String)` | Performs complex matrix multiplication on tensors representing complex numbers as [real, imag] pairs. |
| `ComplexMultiply(ComputationNode<>,ComputationNode<>,String)` | Performs element-wise complex multiplication. |
| `ComputeFlatIndex(Int32[],Int32[])` | Computes flat index from multi-dimensional indices for N-dimensional tensors. |
| `Concat(List<ComputationNode<>>,Int32)` | Concatenates multiple computation nodes along a specified axis. |
| `Constant(Tensor<>,String)` | Creates a constant computation node from a tensor value. |
| `Conv2D(ComputationNode<>,ComputationNode<>,ComputationNode<>,Int32[],Int32[])` | Performs 2D convolution on a 4D tensor (batch, channels, height, width). |
| `Conv3D(ComputationNode<>,ComputationNode<>,ComputationNode<>,Int32[],Int32[])` | Performs 3D convolution on a 5D tensor (batch, channels, depth, height, width). |
| `ConvTranspose2D(ComputationNode<>,ComputationNode<>,ComputationNode<>,Int32[],Int32[],Int32[])` | Performs 2D transposed convolution (deconvolution) on a 4D tensor. |
| `CopyPaddedDataRecursive(Tensor<>,Tensor<>,Int32[],Int32[],Int32[],Int32)` | Helper method to recursively copy data from source to padded destination tensor. |
| `Crop(ComputationNode<>,Int32[])` | Crops a tensor by removing elements from the edges. |
| `DeformableConv2D(ComputationNode<>,ComputationNode<>,ComputationNode<>,ComputationNode<>,ComputationNode<>,Int32[],Int32[],Int32[])` | Performs 2D deformable convolution with learnable offsets and optional modulation mask. |
| `DepthwiseConv2D(ComputationNode<>,ComputationNode<>,ComputationNode<>,Int32[],Int32[])` | Performs depthwise 2D convolution where each input channel is convolved with its own set of filters. |
| `DilatedConv2D(ComputationNode<>,ComputationNode<>,ComputationNode<>,Int32[],Int32[],Int32[])` | Performs dilated (atrous) 2D convolution operation. |
| `Divide(ComputationNode<>,ComputationNode<>)` | Performs element-wise division of two computation nodes. |
| `ELU(ComputationNode<>,Double)` | Applies the Exponential Linear Unit (ELU) activation function to a computation node. |
| `ElementwiseMultiply(ComputationNode<>,ComputationNode<>)` | Performs element-wise multiplication of two computation nodes. |
| `EmbeddingLookup(ComputationNode<>,ComputationNode<>)` | Performs embedding lookup operation. |
| `Exp(ComputationNode<>)` | Computes the exponential function (e^x) for a computation node. |
| `ExtractPaddedDataRecursive(Tensor<>,Tensor<>,Int32[],Int32[],Int32[],Int32)` | Helper method to recursively extract data from padded source to unpadded destination tensor. |
| `FakeQuantize(ComputationNode<>,Int32,,,Boolean)` | Performs fake quantization with Straight-Through Estimator (STE) for differentiable quantization. |
| `GELU(ComputationNode<>)` | Applies the Gaussian Error Linear Unit (GELU) activation function. |
| `GRUCell(ComputationNode<>,ComputationNode<>,ComputationNode<>,ComputationNode<>,ComputationNode<>)` | GRU cell forward pass. |
| `Gaussian(ComputationNode<>)` | Applies the Gaussian activation function element-wise: f(x) = exp(-x²). |
| `GraphConv(ComputationNode<>,ComputationNode<>,ComputationNode<>,ComputationNode<>)` | Performs graph convolution operation for graph neural networks. |
| `GridSample(ComputationNode<>,ComputationNode<>)` | Samples input using bilinear interpolation at grid locations for spatial transformer networks. |
| `GroupNorm(ComputationNode<>,Int32,ComputationNode<>,ComputationNode<>,Double)` | Applies group normalization to a computation node. |
| `GumbelSoftmax(ComputationNode<>,Double,Boolean)` | Applies Gumbel-Softmax for differentiable discrete sampling approximation. |
| `HardSigmoid(ComputationNode<>)` | Applies the Hard Sigmoid activation function element-wise: f(x) = clip((x + 3) / 6, 0, 1). |
| `HardTanh(ComputationNode<>)` | Applies the Hard Tanh activation function element-wise: f(x) = clip(x, -1, 1). |
| `HierarchicalSoftmax(ComputationNode<>,ComputationNode<>,Int32)` | Applies the Hierarchical Softmax activation function for efficient large-vocabulary classification. |
| `HyperbolicLinear(ComputationNode<>,ComputationNode<>,ComputationNode<>,Double)` | Hyperbolic linear transformation in the Poincare ball model. |
| `ISRU(ComputationNode<>,Double)` | Applies the Inverse Square Root Unit (ISRU) activation function. |
| `LSTMCell(ComputationNode<>,ComputationNode<>,ComputationNode<>,ComputationNode<>,ComputationNode<>,ComputationNode<>)` | LSTM cell forward pass. |
| `LayerNorm(ComputationNode<>,Int32[],ComputationNode<>,ComputationNode<>,Double)` | Applies layer normalization to a computation node. |
| `LeakyReLU(ComputationNode<>,Double)` | Applies the Leaky Rectified Linear Unit (LeakyReLU) activation function. |
| `LeakyStateUpdate(ComputationNode<>,ComputationNode<>,ComputationNode<>,Double)` | Leaky state update for reservoir/echo state networks. |
| `LiSHT(ComputationNode<>)` | Applies the LiSHT (Linearly Scaled Hyperbolic Tangent) activation function element-wise. |
| `LocallyConnectedConv2D(ComputationNode<>,ComputationNode<>,ComputationNode<>,Int32[])` | Performs locally connected 2D convolution where weights are NOT shared across spatial locations. |
| `Log(ComputationNode<>)` | Computes the natural logarithm for a computation node. |
| `LogSoftmax(ComputationNode<>,Int32)` | Applies the Log-Softmax function for numerically stable cross-entropy loss computation. |
| `LogSoftmin(ComputationNode<>,Int32)` | Applies the Log-Softmin function for numerically stable computation. |
| `MatrixMultiply(ComputationNode<>,ComputationNode<>)` | Performs matrix multiplication on two computation nodes. |
| `MatrixVectorMultiply(ComputationNode<>,ComputationNode<>)` | Performs a matrix-vector multiplication (2D x 1D) by reshaping the vector into a column matrix. |
| `MaxPool2D(ComputationNode<>,Int32[],Int32[])` | Performs 2D max pooling on a 4D tensor (batch, channels, height, width). |
| `MaxPool3D(ComputationNode<>,Int32[],Int32[])` | Performs 3D max pooling on a 5D tensor (batch, channels, depth, height, width). |
| `Maxout(ComputationNode<>,Int32)` | Applies the Maxout activation function which takes maximum over groups of inputs. |
| `Mean(ComputationNode<>)` | Computes the mean of elements in a computation node. |
| `Mish(ComputationNode<>)` | Applies the Mish activation function. |
| `MobiusAdd(ComputationNode<>,ComputationNode<>,Double)` | Mobius addition in the Poincare ball model. |
| `MultiHeadAttention(ComputationNode<>,ComputationNode<>,ComputationNode<>,Int32,ComputationNode<>,ComputationNode<>,ComputationNode<>,ComputationNode<>)` | Applies multi-head attention mechanism. |
| `Negate(ComputationNode<>)` | Negates a computation node (computes -a). |
| `Norm(ComputationNode<>,Int32,Boolean,Double)` | Computes the L2 norm along a specified axis. |
| `OctonionConjugateComponents(INumericOperations<>,[])` | Computes the conjugate of an octonion. |
| `OctonionMatMul(ComputationNode<>,ComputationNode<>,ComputationNode<>)` | Performs octonion matrix multiplication for OctonionLinearLayer. |
| `OctonionMultiplyComponents(INumericOperations<>,[],[])` | Multiplies two octonions represented as 8-component arrays. |
| `PReLU(ComputationNode<>,Double)` | Applies the Parametric Rectified Linear Unit (PReLU) activation function. |
| `Pad(ComputationNode<>,Int32[0:,0:],)` | Pads a tensor with a constant value along specified dimensions. |
| `Pad(ComputationNode<>,Int32[])` | Pads a tensor with zeros along specified dimensions. |
| `Permute(ComputationNode<>,Int32[])` | Permutes the dimensions of a computation node (general transpose). |
| `PixelShuffle(ComputationNode<>,Int32)` | Performs pixel shuffle (depth-to-space) operation for sub-pixel convolution. |
| `PoincareDistance(ComputationNode<>,ComputationNode<>,Double)` | Computes the Poincare ball distance between two points. |
| `PoincareExpMap(ComputationNode<>,ComputationNode<>,Double)` | Poincare ball exponential map from tangent space at a point. |
| `PoincareLogMap(ComputationNode<>,ComputationNode<>,Double)` | Poincare ball logarithmic map to tangent space at a point. |
| `PoincareProject(ComputationNode<>,Double,Double)` | Projects a point onto the Poincare ball to ensure it stays inside the unit ball. |
| `Power(ComputationNode<>,Double)` | Raises a computation node to a power. |
| `RBFKernel(ComputationNode<>,ComputationNode<>,ComputationNode<>)` | Computes Gaussian Radial Basis Function (RBF) kernel activations. |
| `RReLU(ComputationNode<>,Double,Double,Boolean,Nullable<Int32>)` | Applies the Randomized Leaky ReLU (RReLU) activation function. |
| `ReLU(ComputationNode<>)` | Computes the ReLU (Rectified Linear Unit) activation for a computation node. |
| `ReduceGradient(Tensor<>,Int32[])` | Reduces gradient to match the original shape by summing across broadcasted dimensions. |
| `ReduceLogVariance(ComputationNode<>,Int32,Double)` | Computes the natural logarithm of variance along the specified axis. |
| `ReduceMax(ComputationNode<>,Int32[],Boolean)` | Reduces a tensor by computing the maximum value along specified axes. |
| `ReduceMean(ComputationNode<>,Int32[],Boolean)` | Reduces a tensor by computing the mean value along specified axes. |
| `Reshape(ComputationNode<>,Int32[])` | Reshapes a computation node to a new shape. |
| `SELU(ComputationNode<>)` | Applies the SELU (Scaled Exponential Linear Unit) activation function element-wise. |
| `SQRBF(ComputationNode<>,Double)` | Applies the Squared Radial Basis Function (SQRBF) activation. |
| `ScaledDotProductAttention(ComputationNode<>,ComputationNode<>,ComputationNode<>,ComputationNode<>)` | Computes scaled dot-product attention: softmax(Q @ K^T / sqrt(d_k)) @ V. |
| `ScaledTanh(ComputationNode<>,Double)` | Applies the Scaled Tanh activation function element-wise. |
| `Sigmoid(ComputationNode<>)` | Computes the sigmoid function for a computation node. |
| `SinusoidalTimeEmbedding(ComputationNode<>,Int32)` | Creates sinusoidal time embeddings for diffusion models. |
| `Slice(ComputationNode<>,Int32,Int32,Int32,Int32)` | Extracts a slice from a tensor along a specified axis. |
| `SoftKNN(ComputationNode<>,ComputationNode<>,ComputationNode<>,)` | Performs a soft K-Nearest Neighbors operation for differentiable instance-based learning. |
| `SoftLocallyWeighted(ComputationNode<>,ComputationNode<>,ComputationNode<>,)` | Performs soft locally-weighted regression for differentiable instance-based learning. |
| `SoftPlus(ComputationNode<>)` | Applies the SoftPlus activation function element-wise: f(x) = ln(1 + e^x). |
| `SoftSign(ComputationNode<>)` | Applies the SoftSign activation function element-wise: f(x) = x / (1 + \|x\|). |
| `SoftSplit(ComputationNode<>,ComputationNode<>,ComputationNode<>,Int32,,)` | Performs a soft split operation for differentiable decision trees. |
| `Softmax(ComputationNode<>,Int32)` | Computes the softmax function for a computation node along a specified axis. |
| `Softmin(ComputationNode<>,Int32)` | Applies the Softmin function, which assigns higher probability to lower values. |
| `Sparsemax(ComputationNode<>,Int32)` | Applies the Sparsemax activation function which projects onto the probability simplex. |
| `SphericalSoftmax(ComputationNode<>,Int32)` | Applies the Spherical Softmax activation function. |
| `Split(ComputationNode<>,Int32,Int32)` | Splits a tensor along a specified axis into multiple tensors. |
| `Sqrt(ComputationNode<>)` | Computes the square root for a computation node. |
| `Square(ComputationNode<>)` | Computes the element-wise square of the input (x²). |
| `Squash(ComputationNode<>,Double)` | Computes the squashing function used in capsule networks: s(x) = \|\|x\|\|² / (1 + \|\|x\|\|²) * (x / \|\|x\|\|). |
| `StraightThroughThreshold(ComputationNode<>,Double)` | Applies a straight-through threshold for HTM-style sparse activations. |
| `Subtract(ComputationNode<>,ComputationNode<>)` | Performs element-wise subtraction of two computation nodes. |
| `Sum(ComputationNode<>,Int32[],Boolean)` | Sums elements of a computation node along specified axes. |
| `SurrogateSpike(ComputationNode<>,Double,Double)` | Applies a surrogate spike function for spiking neural network JIT compilation. |
| `Swish(ComputationNode<>)` | Applies the Swish (SiLU) activation function. |
| `Tanh(ComputationNode<>)` | Computes the hyperbolic tangent (tanh) for a computation node. |
| `TaylorSoftmax(ComputationNode<>,Int32,Int32)` | Applies the Taylor Softmax activation function using Taylor series approximation. |
| `ThresholdedReLU(ComputationNode<>,Double)` | Applies the Thresholded Rectified Linear Unit activation function. |
| `TopKSoftmax(ComputationNode<>,Int32)` | Differentiable Top-K selection for mixture-of-experts routing. |
| `Transpose(ComputationNode<>)` | Transposes a 2D computation node (matrix). |
| `Upsample(ComputationNode<>,Int32)` | Upsamples a tensor using nearest neighbor interpolation. |
| `Upsample3D(ComputationNode<>,Int32,Int32,Int32)` | Performs 3D upsampling (nearest neighbor) on a 5D tensor. |
| `Variable(Tensor<>,String,Boolean)` | Creates a computation node from a tensor value. |

