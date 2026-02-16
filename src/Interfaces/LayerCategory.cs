namespace AiDotNet.Interfaces;

/// <summary>
/// Classification of neural network layer types for automated per-layer decisions.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> Neural networks are made of different types of layers, each with a
/// specific role. This enum categorizes layers so that tools like quantizers, pruners, and
/// pipeline schedulers can make smart per-layer decisions automatically.
///
/// For example, a quantizer might keep attention layers at 8-bit precision (they're sensitive
/// to quantization) while reducing dense layers to 4-bit (they're more tolerant).
/// </remarks>
public enum LayerCategory
{
    /// <summary>Dense/fully-connected layers (e.g., FullyConnectedLayer, DenseLayer).</summary>
    Dense,

    /// <summary>Convolutional layers (1D, 2D, 3D, depthwise, separable, dilated, deformable).</summary>
    Convolution,

    /// <summary>Self-attention, cross-attention, multi-head attention layers.</summary>
    Attention,

    /// <summary>Normalization layers (BatchNorm, LayerNorm, GroupNorm, InstanceNorm, RMSNorm).</summary>
    Normalization,

    /// <summary>Activation function layers (ReLU, GELU, SiLU, Sigmoid, Tanh, etc.).</summary>
    Activation,

    /// <summary>Pooling layers (MaxPool, AveragePool, AdaptivePool, GlobalPool).</summary>
    Pooling,

    /// <summary>Embedding layers (token, positional, patch, time).</summary>
    Embedding,

    /// <summary>Recurrent layers (LSTM, GRU, RNN, bidirectional).</summary>
    Recurrent,

    /// <summary>Regularization layers (Dropout, GaussianNoise).</summary>
    Regularization,

    /// <summary>Residual/skip connection layers.</summary>
    Residual,

    /// <summary>Feed-forward / MLP block layers.</summary>
    FeedForward,

    /// <summary>Graph neural network layers (GCN, GAT, GraphSAGE, GIN, message passing).</summary>
    Graph,

    /// <summary>Reshape, flatten, split, concatenate, and other structural layers.</summary>
    Structural,

    /// <summary>Input layers.</summary>
    Input,

    /// <summary>Custom, specialized, or unclassified layers.</summary>
    Other
}
