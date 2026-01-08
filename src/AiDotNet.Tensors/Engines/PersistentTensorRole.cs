// Copyright (c) AiDotNet. All rights reserved.
// Persistent tensor role enum for GPU memory management hints.

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// Specifies the role of a persistent tensor for GPU memory management hints.
/// </summary>
/// <remarks>
/// <para><b>Purpose:</b></para>
/// <para>
/// When a tensor is registered as persistent, the engine may choose to keep it
/// resident in GPU memory across multiple operations, avoiding repeated CPU-GPU transfers.
/// </para>
/// <para><b>Performance Impact:</b></para>
/// <para>
/// Without persistence: Transfer weights (e.g., 285MB) every forward pass.
/// With persistence: Transfer weights once, only activations per pass.
/// Expected speedup: 100-1000x for weight-heavy layers.
/// </para>
/// </remarks>
public enum PersistentTensorRole
{
    /// <summary>
    /// Layer weights that change only during training updates.
    /// These are the primary candidates for GPU persistence.
    /// </summary>
    Weights,

    /// <summary>
    /// Layer biases that change only during training updates.
    /// </summary>
    Biases,

    /// <summary>
    /// Normalization parameters (gamma/beta for BatchNorm, LayerNorm, etc.).
    /// </summary>
    NormalizationParams,

    /// <summary>
    /// Embedding lookup tables.
    /// These can be very large and benefit significantly from GPU persistence.
    /// </summary>
    Embeddings,

    /// <summary>
    /// Attention key/value caches for inference.
    /// Used in autoregressive generation to cache previous attention states.
    /// </summary>
    AttentionCache,

    /// <summary>
    /// Optimizer state tensors (velocity, momentum, etc.).
    /// </summary>
    OptimizerState,

    /// <summary>
    /// Constant tensors that never change (e.g., precomputed frequencies, positional encodings).
    /// </summary>
    Constant,

    /// <summary>
    /// Other persistent tensors not fitting above categories.
    /// </summary>
    Other
}
