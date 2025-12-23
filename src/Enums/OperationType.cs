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

    /// <summary>
    /// Element-wise absolute value - |x| for each element.
    /// </summary>
    Abs,

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

    /// <summary>
    /// L2 norm computation along an axis - sqrt(sum(x²)).
    /// </summary>
    Norm,

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
    /// Exponential Linear Unit - ELU(x) = x if x > 0, alpha * (exp(x) - 1) otherwise.
    /// </summary>
    ELU,

    /// <summary>
    /// Leaky Rectified Linear Unit - max(alpha * x, x) where alpha is typically 0.01.
    /// </summary>
    LeakyReLU,

    /// <summary>
    /// Gaussian Error Linear Unit - x * Φ(x) where Φ is standard normal CDF.
    /// </summary>
    GELU,

    /// <summary>
    /// Swish/SiLU activation - x * sigmoid(x).
    /// </summary>
    Swish,

    /// <summary>
    /// Mish activation - x * tanh(softplus(x)).
    /// </summary>
    Mish,

    /// <summary>
    /// SoftPlus activation - ln(1 + e^x), smooth approximation of ReLU.
    /// </summary>
    SoftPlus,

    /// <summary>
    /// Scaled Exponential Linear Unit - self-normalizing activation with fixed lambda and alpha.
    /// </summary>
    SELU,

    /// <summary>
    /// Hard Sigmoid - piecewise linear approximation of sigmoid: clip((x + 1) / 2, 0, 1).
    /// </summary>
    HardSigmoid,

    /// <summary>
    /// Hard Tanh - piecewise linear approximation of tanh: clip(x, -1, 1).
    /// </summary>
    HardTanh,

    /// <summary>
    /// SoftSign activation - x / (1 + |x|), alternative to tanh with polynomial tails.
    /// </summary>
    SoftSign,

    /// <summary>
    /// Continuously Differentiable ELU - max(0, x) + min(0, α * (exp(x/α) - 1)).
    /// </summary>
    CELU,

    /// <summary>
    /// Linearly Scaled Hyperbolic Tangent - x * tanh(x).
    /// </summary>
    LiSHT,

    /// <summary>
    /// Bent Identity - (sqrt(x² + 1) - 1) / 2 + x, smooth alternative to ReLU.
    /// </summary>
    BentIdentity,

    /// <summary>
    /// Gaussian activation - exp(-x²), bell-shaped response curve.
    /// </summary>
    Gaussian,

    /// <summary>
    /// Scaled Tanh - parameterized tanh with adjustable steepness β.
    /// </summary>
    ScaledTanh,

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
    /// Permute tensor dimensions (general transpose).
    /// </summary>
    Permute,

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
    /// Slice tensor along an axis - extract a portion with optional stride.
    /// </summary>
    Slice,

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
    /// 3D max pooling.
    /// </summary>
    MaxPool3D,

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

    /// <summary>
    /// LSTM cell operation for recurrent networks.
    /// </summary>
    LSTMCell,

    // Complex Number Operations

    /// <summary>
    /// Complex matrix multiplication for quantum operations.
    /// </summary>
    ComplexMatMul,

    /// <summary>
    /// Element-wise complex multiplication.
    /// </summary>
    ComplexMultiply,

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
    FusedAddReLU,

    // Differentiable Approximations for Dynamic Layers

    /// <summary>
    /// Gumbel-Softmax for differentiable discrete sampling (used in stochastic layers).
    /// </summary>
    GumbelSoftmax,

    /// <summary>
    /// Surrogate spike function for spiking neural networks with gradient estimation.
    /// </summary>
    SurrogateSpike,

    /// <summary>
    /// Straight-through threshold for HTM-style sparse activations.
    /// </summary>
    StraightThroughThreshold,

    /// <summary>
    /// Top-K softmax for mixture-of-experts routing.
    /// </summary>
    TopKSoftmax,

    /// <summary>
    /// Leaky state update for reservoir/echo state networks.
    /// </summary>
    LeakyStateUpdate,

    /// <summary>
    /// CRF forward algorithm for sequence labeling.
    /// </summary>
    CRFForward,

    /// <summary>
    /// Anomaly score computation.
    /// </summary>
    AnomalyScore,

    // Additional Activation Functions

    /// <summary>
    /// Parametric Rectified Linear Unit - max(0, x) + alpha * min(0, x) where alpha is learned.
    /// </summary>
    PReLU,

    /// <summary>
    /// Thresholded Rectified Linear Unit - x if x > threshold, 0 otherwise.
    /// </summary>
    ThresholdedReLU,

    /// <summary>
    /// Inverse Square Root Unit - x / sqrt(1 + alpha * x²).
    /// </summary>
    ISRU,

    /// <summary>
    /// Sign function with surrogate gradient for training - returns -1, 0, or 1.
    /// </summary>
    Sign,

    /// <summary>
    /// Log-Softmax - log(softmax(x)), numerically stable for cross-entropy loss.
    /// </summary>
    LogSoftmax,

    /// <summary>
    /// Softmin - softmax(-x), assigns higher probability to lower values.
    /// </summary>
    Softmin,

    /// <summary>
    /// Log-Softmin - log(softmin(x)) = log(softmax(-x)).
    /// </summary>
    LogSoftmin,

    /// <summary>
    /// Square Radial Basis Function - smooth bell-shaped activation.
    /// </summary>
    SQRBF,

    /// <summary>
    /// Maxout activation - maximum over multiple linear pieces.
    /// </summary>
    Maxout,

    /// <summary>
    /// Randomized Leaky ReLU - LeakyReLU with random alpha during training.
    /// </summary>
    RReLU,

    /// <summary>
    /// Spherical Softmax - L2 normalization followed by softmax.
    /// </summary>
    SphericalSoftmax,

    /// <summary>
    /// Taylor Softmax - softmax using Taylor series approximation of exp.
    /// </summary>
    TaylorSoftmax,

    /// <summary>
    /// Sparsemax - projects onto probability simplex, can produce sparse outputs.
    /// </summary>
    Sparsemax,

    /// <summary>
    /// Hierarchical Softmax - tree-based efficient softmax for large vocabularies.
    /// </summary>
    HierarchicalSoftmax,

    // Differentiable Approximation Operations

    /// <summary>
    /// Soft split operation for differentiable decision trees.
    /// Uses sigmoid gating: p_left = σ((threshold - x[feature]) / temperature)
    /// output = p_left * left_value + (1 - p_left) * right_value
    /// </summary>
    SoftSplit,

    /// <summary>
    /// Soft K-Nearest Neighbors operation for differentiable instance-based learning.
    /// Uses attention-weighted contributions from all support vectors instead of hard k-selection.
    /// weights = softmax(-distances / temperature), output = Σ weights * labels
    /// </summary>
    SoftKNN,

    /// <summary>
    /// Soft locally-weighted regression operation for differentiable instance-based learning.
    /// Uses attention-weighted linear combination of training targets based on distance.
    /// weights = softmax(-||x - X_train||² / bandwidth), output = weights @ y_train
    /// </summary>
    SoftLocallyWeighted,

    /// <summary>
    /// Fake quantization operation with Straight-Through Estimator (STE) for differentiable quantization.
    /// Forward: quantized = round(x / scale) * scale
    /// Backward: gradient passes through unchanged (STE)
    /// </summary>
    FakeQuantization,

    /// <summary>
    /// Custom user-defined operation for extensibility.
    /// </summary>
    Custom,

    /// <summary>
    /// Dropout regularization operation - randomly zeros elements during training.
    /// </summary>
    Dropout,

    /// <summary>
    /// Gather operation - selects elements from a tensor using indices.
    /// </summary>
    Gather,

    /// <summary>
    /// Broadcast operation - expands tensor dimensions to match target shape.
    /// </summary>
    Broadcast,

    /// <summary>
    /// Generic attention mechanism operation.
    /// </summary>
    Attention,

    // InferenceOptimization Operations

    /// <summary>
    /// Output node in computation graph.
    /// </summary>
    Output,

    /// <summary>
    /// General convolution operation.
    /// </summary>
    Convolution,

    /// <summary>
    /// 2D convolution operation.
    /// </summary>
    Convolution2D,

    /// <summary>
    /// 3D convolution operation.
    /// </summary>
    Convolution3D,

    /// <summary>
    /// Depthwise convolution operation.
    /// </summary>
    DepthwiseConvolution,

    /// <summary>
    /// Dilated convolution operation.
    /// </summary>
    DilatedConvolution,

    /// <summary>
    /// Deconvolution (transposed convolution) operation.
    /// </summary>
    Deconvolution,

    /// <summary>
    /// Batch normalization.
    /// </summary>
    BatchNormalization,

    /// <summary>
    /// Layer normalization.
    /// </summary>
    LayerNormalization,

    /// <summary>
    /// Instance normalization.
    /// </summary>
    InstanceNormalization,

    /// <summary>
    /// Group normalization.
    /// </summary>
    GroupNormalization,

    /// <summary>
    /// Max pooling operation.
    /// </summary>
    MaxPooling,

    /// <summary>
    /// Average pooling operation.
    /// </summary>
    AveragePooling,

    /// <summary>
    /// Global average pooling.
    /// </summary>
    GlobalAveragePooling,

    /// <summary>
    /// Global max pooling.
    /// </summary>
    GlobalMaxPooling,

    /// <summary>
    /// Adaptive pooling.
    /// </summary>
    AdaptivePooling,

    /// <summary>
    /// Dense (fully connected) layer.
    /// </summary>
    Dense,

    /// <summary>
    /// Fully connected layer.
    /// </summary>
    FullyConnected,

    /// <summary>
    /// General Matrix Multiplication.
    /// </summary>
    Gemm,

    /// <summary>
    /// Minimum value reduction.
    /// </summary>
    ReduceMin,

    /// <summary>
    /// Self-attention operation.
    /// </summary>
    SelfAttention,

    /// <summary>
    /// Cross-attention operation.
    /// </summary>
    CrossAttention,

    /// <summary>
    /// LSTM recurrent layer.
    /// </summary>
    LSTM,

    /// <summary>
    /// GRU recurrent layer.
    /// </summary>
    GRU,

    /// <summary>
    /// Basic RNN layer.
    /// </summary>
    RNN,

    /// <summary>
    /// Flatten tensor to 1D.
    /// </summary>
    Flatten,

    /// <summary>
    /// Remove dimensions of size 1.
    /// </summary>
    Squeeze,

    /// <summary>
    /// Add dimension of size 1.
    /// </summary>
    Unsqueeze,

    /// <summary>
    /// Expand tensor dimensions.
    /// </summary>
    Expand,

    /// <summary>
    /// DropPath regularization.
    /// </summary>
    DropPath,

    /// <summary>
    /// Positional encoding for transformers.
    /// </summary>
    PositionalEncoding,

    /// <summary>
    /// Stack tensors along new axis.
    /// </summary>
    Stack,

    /// <summary>
    /// Element-wise equality.
    /// </summary>
    Equal,

    /// <summary>
    /// Element-wise greater than.
    /// </summary>
    Greater,

    /// <summary>
    /// Element-wise less than.
    /// </summary>
    Less,

    /// <summary>
    /// Element-wise greater or equal.
    /// </summary>
    GreaterOrEqual,

    /// <summary>
    /// Element-wise less or equal.
    /// </summary>
    LessOrEqual,

    /// <summary>
    /// Logical AND.
    /// </summary>
    And,

    /// <summary>
    /// Logical OR.
    /// </summary>
    Or,

    /// <summary>
    /// Logical NOT.
    /// </summary>
    Not,

    /// <summary>
    /// Logical XOR.
    /// </summary>
    Xor,

    /// <summary>
    /// Type cast operation.
    /// </summary>
    Cast,

    /// <summary>
    /// Clip values to range.
    /// </summary>
    Clip,

    /// <summary>
    /// Scatter values to indices.
    /// </summary>
    Scatter,

    // Fused Operations for InferenceOptimization

    /// <summary>
    /// Fused Conv + BatchNorm + ReLU.
    /// </summary>
    FusedConvBatchNormReLU,

    /// <summary>
    /// Fused MatMul + Bias.
    /// </summary>
    FusedMatMulBias,

    /// <summary>
    /// Fused MatMul + Bias + ReLU.
    /// </summary>
    FusedMatMulBiasReLU,

    /// <summary>
    /// Fused MatMul + Bias + GELU.
    /// </summary>
    FusedMatMulBiasGELU,

    /// <summary>
    /// Fused MultiHead Attention.
    /// </summary>
    FusedMultiHeadAttention,

    /// <summary>
    /// Fused LayerNorm + Attention.
    /// </summary>
    FusedLayerNormAttention,

    /// <summary>
    /// Unknown operation type.
    /// </summary>
    Unknown
}
