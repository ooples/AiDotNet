namespace AiDotNet.Enums;

/// <summary>
/// Represents the type of operation in a computation graph.
/// Used for graph optimization and operator fusion.
/// </summary>
public enum OperationType
{
    // Input/Output Operations
    Input,
    Output,
    Constant,

    // Convolution Operations
    Convolution,
    Convolution2D,
    Convolution3D,
    DepthwiseConvolution,
    DilatedConvolution,
    Deconvolution,

    // Normalization Operations
    BatchNormalization,
    LayerNormalization,
    InstanceNormalization,
    GroupNormalization,

    // Activation Operations
    ReLU,
    LeakyReLU,
    PReLU,
    ELU,
    SELU,
    GELU,
    Sigmoid,
    Tanh,
    Softmax,
    Swish,
    Mish,

    // Pooling Operations
    MaxPooling,
    AveragePooling,
    GlobalAveragePooling,
    GlobalMaxPooling,
    AdaptivePooling,

    // Linear/Dense Operations
    MatMul,
    Dense,
    FullyConnected,
    Gemm, // General Matrix Multiplication

    // Elementwise Operations
    Add,
    Subtract,
    Multiply,
    Divide,
    Power,
    Sqrt,
    Exp,
    Log,

    // Reduction Operations
    ReduceSum,
    ReduceMean,
    ReduceMax,
    ReduceMin,

    // Attention Operations
    Attention,
    MultiHeadAttention,
    SelfAttention,
    CrossAttention,

    // Recurrent Operations
    LSTM,
    GRU,
    RNN,

    // Reshaping Operations
    Reshape,
    Flatten,
    Transpose,
    Permute,
    Squeeze,
    Unsqueeze,
    Expand,

    // Dropout/Regularization
    Dropout,
    DropPath,

    // Embedding Operations
    Embedding,
    PositionalEncoding,

    // Concatenation/Split
    Concat,
    Split,
    Stack,

    // Comparison Operations
    Equal,
    Greater,
    Less,
    GreaterOrEqual,
    LessOrEqual,

    // Logical Operations
    And,
    Or,
    Not,
    Xor,

    // Special Operations
    Cast,
    Clip,
    Pad,
    Slice,
    Gather,
    Scatter,

    // Fused Operations (Result of operator fusion)
    FusedConvBatchNormReLU,
    FusedConvBatchNorm,
    FusedMatMulBias,
    FusedMatMulBiasReLU,
    FusedMatMulBiasGELU,
    FusedMultiHeadAttention,
    FusedLayerNormAttention,

    // Custom/Unknown
    Custom,
    Unknown
}
