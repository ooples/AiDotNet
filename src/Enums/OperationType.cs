namespace AiDotNet.Enums;

/// <summary>
/// Represents different operation types in computation graphs for JIT compilation and automatic differentiation.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Operation types identify mathematical operations performed on tensors in neural networks.
///
/// When building a computation graph, each operation (like adding two tensors or applying an activation function)
/// needs to be identified so that:
/// 1. The JIT compiler can optimize the code
/// 2. The automatic differentiation system can compute gradients correctly
/// 3. The system can analyze and transform the computation graph
///
/// This enum provides type-safe identification of operations, preventing typos and enabling better tooling support.
/// </para>
/// </remarks>
public enum OperationType
{
    /// <summary>
    /// Input node - represents a variable or parameter in the computation graph.
    /// </summary>
    Input,

    /// <summary>
    /// Constant node - represents a constant value that doesn't require gradients.
    /// </summary>
    Constant,

    // Arithmetic Operations

    /// <summary>
    /// Element-wise addition of two tensors.
    /// </summary>
    Add,

    /// <summary>
    /// Element-wise subtraction of two tensors.
    /// </summary>
    Subtract,

    /// <summary>
    /// Element-wise multiplication (Hadamard product) of two tensors.
    /// </summary>
    Multiply,

    /// <summary>
    /// Element-wise division of two tensors.
    /// </summary>
    Divide,

    /// <summary>
    /// Element-wise power operation - raises each element to a specified exponent.
    /// </summary>
    Power,

    /// <summary>
    /// Element-wise negation - multiplies each element by -1.
    /// </summary>
    Negate,

    // Mathematical Functions

    /// <summary>
    /// Element-wise exponential function - e^x for each element.
    /// </summary>
    Exp,

    /// <summary>
    /// Element-wise natural logarithm.
    /// </summary>
    Log,

    /// <summary>
    /// Element-wise square root.
    /// </summary>
    Sqrt,

    /// <summary>
    /// Element-wise square - x² for each element.
    /// </summary>
    Square,

    // Matrix Operations

    /// <summary>
    /// Matrix multiplication (not element-wise).
    /// </summary>
    MatMul,

    /// <summary>
    /// Matrix transpose - swaps rows and columns.
    /// </summary>
    Transpose,

    // Activation Functions

    /// <summary>
    /// Rectified Linear Unit - max(0, x).
    /// </summary>
    ReLU,

    /// <summary>
    /// Sigmoid activation - 1 / (1 + e^(-x)).
    /// </summary>
    Sigmoid,

    /// <summary>
    /// Hyperbolic tangent activation.
    /// </summary>
    Tanh,

    /// <summary>
    /// Softmax activation - converts logits to probability distribution.
    /// </summary>
    Softmax,

    /// <summary>
    /// Generic activation function application.
    /// </summary>
    Activation,

    /// <summary>
    /// Squashing activation for capsule networks - s(v) = ||v||² / (1 + ||v||²) * (v / ||v||).
    /// </summary>
    Squash,

    // Reduction Operations

    /// <summary>
    /// Sum reduction along specified axes.
    /// </summary>
    ReduceSum,

    /// <summary>
    /// Mean reduction along specified axes.
    /// </summary>
    ReduceMean,

    /// <summary>
    /// Maximum value reduction along specified axes.
    /// </summary>
    ReduceMax,

    /// <summary>
    /// Log-variance reduction along specified axes.
    /// </summary>
    ReduceLogVariance,

    /// <summary>
    /// Mean operation (reduces all dimensions).
    /// </summary>
    Mean,

    // Shape Operations

    /// <summary>
    /// Reshape tensor to new dimensions.
    /// </summary>
    Reshape,

    /// <summary>
    /// Concatenate multiple tensors along an axis.
    /// </summary>
    Concat,

    /// <summary>
    /// Pad tensor with values.
    /// </summary>
    Pad,

    /// <summary>
    /// Crop tensor by removing border elements.
    /// </summary>
    Crop,

    /// <summary>
    /// Split tensor along an axis into multiple tensors.
    /// </summary>
    Split,

    /// <summary>
    /// Upsample tensor by repeating elements.
    /// </summary>
    Upsample,

    /// <summary>
    /// Pixel shuffle operation for upsampling.
    /// </summary>
    PixelShuffle,

    // Convolutional Operations

    /// <summary>
    /// 2D convolution operation.
    /// </summary>
    Conv2D,

    /// <summary>
    /// 2D transposed convolution (deconvolution).
    /// </summary>
    ConvTranspose2D,

    /// <summary>
    /// 2D dilated (atrous) convolution.
    /// </summary>
    DilatedConv2D,

    /// <summary>
    /// 2D depthwise convolution.
    /// </summary>
    DepthwiseConv2D,

    /// <summary>
    /// 2D locally connected convolution.
    /// </summary>
    LocallyConnectedConv2D,

    // Pooling Operations

    /// <summary>
    /// 2D max pooling.
    /// </summary>
    MaxPool2D,

    /// <summary>
    /// 2D average pooling.
    /// </summary>
    AvgPool2D,

    // Normalization Operations

    /// <summary>
    /// Layer normalization.
    /// </summary>
    LayerNorm,

    /// <summary>
    /// Batch normalization.
    /// </summary>
    BatchNorm,

    // Advanced Operations

    /// <summary>
    /// RBF (Radial Basis Function) kernel operation.
    /// </summary>
    RBFKernel,

    /// <summary>
    /// Affine grid generation for spatial transformers.
    /// </summary>
    AffineGrid,

    /// <summary>
    /// Grid sampling for spatial transformers.
    /// </summary>
    GridSample,

    /// <summary>
    /// Graph convolutional operation for GNNs.
    /// </summary>
    GraphConv,

    /// <summary>
    /// Embedding lookup operation.
    /// </summary>
    Embedding,

    /// <summary>
    /// Scaled dot-product attention.
    /// </summary>
    ScaledDotProductAttention,

    /// <summary>
    /// Multi-head attention operation.
    /// </summary>
    MultiHeadAttention,

    /// <summary>
    /// GRU cell operation for recurrent networks.
    /// </summary>
    GRUCell,

    // Fused Operations (for JIT optimization)

    /// <summary>
    /// Fused matrix multiplication + addition (MatMul + Add).
    /// </summary>
    FusedMatMulAdd,

    /// <summary>
    /// Fused linear layer with ReLU (MatMul + Add + ReLU).
    /// </summary>
    FusedLinearReLU,

    /// <summary>
    /// Fused convolution + batch normalization.
    /// </summary>
    FusedConvBatchNorm,

    /// <summary>
    /// Fused addition + ReLU.
    /// </summary>
    FusedAddReLU
}
