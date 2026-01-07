namespace AiDotNet.Tensors.Engines.Gpu;

/// <summary>
/// Defines the role of a GPU-resident tensor for memory management and optimization decisions.
/// </summary>
public enum GpuTensorRole
{
    /// <summary>
    /// General-purpose tensor with no specific role.
    /// </summary>
    General = 0,

    /// <summary>
    /// Input tensor that will be uploaded from CPU once and reused.
    /// </summary>
    Input = 1,

    /// <summary>
    /// Weight tensor that persists across forward passes.
    /// These are candidates for permanent GPU residency.
    /// </summary>
    Weight = 2,

    /// <summary>
    /// Bias tensor that persists across forward passes.
    /// These are candidates for permanent GPU residency.
    /// </summary>
    Bias = 3,

    /// <summary>
    /// Activation tensor produced during forward pass.
    /// May be kept for backward pass in training mode.
    /// </summary>
    Activation = 4,

    /// <summary>
    /// Intermediate tensor that is temporary within an operation.
    /// Can be released immediately after use.
    /// </summary>
    Intermediate = 5,

    /// <summary>
    /// Gradient tensor produced during backward pass.
    /// </summary>
    Gradient = 6,

    /// <summary>
    /// Output tensor that will eventually be downloaded to CPU.
    /// </summary>
    Output = 7,

    /// <summary>
    /// Normalization statistics (mean, variance, etc.).
    /// May be kept for inference or updated during training.
    /// </summary>
    Statistics = 8,

    /// <summary>
    /// Attention cache for transformer models (KV cache).
    /// These can be large and benefit from persistent GPU residency.
    /// </summary>
    AttentionCache = 9,

    /// <summary>
    /// Optimizer state tensor (momentum, Adam m/v buffers, etc.).
    /// Persists across training steps and benefits from permanent GPU residency.
    /// </summary>
    OptimizerState = 10
}
