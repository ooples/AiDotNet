namespace AiDotNet.Interfaces;

/// <summary>
/// Metadata about a single layer within a layered model, including its position
/// in the flat parameter vector and computational characteristics.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> When you have a neural network with many layers, you often need
/// to know details about each one: how many parameters it has, what type it is, where its
/// parameters sit in the overall parameter vector, and how expensive it is to compute.
///
/// This class packages all that information together so that tools like pipeline partitioners,
/// quantizers, and pruners can make smart per-layer decisions.
///
/// For example, a pipeline partitioner can use <see cref="EstimatedFlops"/> to balance
/// computational load across GPUs, or a quantizer can use <see cref="Category"/> to apply
/// different bit-widths to attention vs dense layers.
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class LayerInfo<T>
{
    /// <summary>
    /// Layer index within the model (0-based).
    /// </summary>
    public int Index { get; init; }

    /// <summary>
    /// Human-readable layer name (e.g., "SelfAttentionLayer_0", "FullyConnectedLayer_3").
    /// </summary>
    public string Name { get; init; } = string.Empty;

    /// <summary>
    /// Layer type classification for automated decisions.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This tells you what kind of layer this is (dense, convolution,
    /// attention, etc.) without having to check the concrete type. Tools can use this to
    /// apply different strategies to different layer types.
    /// </remarks>
    public LayerCategory Category { get; init; }

    /// <summary>
    /// Reference to the actual layer instance.
    /// </summary>
    public required ILayer<T> Layer { get; init; }

    /// <summary>
    /// Start index of this layer's parameters in the flat parameter vector
    /// returned by <c>GetParameters()</c>.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> When you call <c>model.GetParameters()</c>, you get one big
    /// vector with all parameters concatenated. This offset tells you where this layer's
    /// parameters begin in that vector, so you can slice out just this layer's weights.
    /// </remarks>
    public int ParameterOffset { get; init; }

    /// <summary>
    /// Number of trainable parameters in this layer.
    /// </summary>
    public int ParameterCount { get; init; }

    /// <summary>
    /// Input shape expected by this layer.
    /// </summary>
    public int[] InputShape { get; init; } = Array.Empty<int>();

    /// <summary>
    /// Output shape produced by this layer.
    /// </summary>
    public int[] OutputShape { get; init; } = Array.Empty<int>();

    /// <summary>
    /// Whether this layer has trainable parameters.
    /// </summary>
    public bool IsTrainable { get; init; }

    /// <summary>
    /// Estimated computational cost (FLOPs) for a single forward pass.
    /// Used by pipeline partitioners for load balancing.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> FLOPs (Floating-Point Operations) measure how much
    /// computation this layer requires. A dense layer with 1000x1000 weights needs
    /// about 2 million FLOPs per forward pass. Pipeline schedulers use this to
    /// distribute layers evenly across GPUs so no single GPU is a bottleneck.
    /// </remarks>
    public long EstimatedFlops { get; init; }

    /// <summary>
    /// Estimated activation memory (bytes) needed during forward pass.
    /// Used by activation checkpointing for selective recomputation decisions.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> During training, each layer saves its output (activation)
    /// so the backward pass can compute gradients. Attention layers typically need much
    /// more memory than normalization layers. Selective checkpointing uses this to decide
    /// which layers to recompute vs which to keep in memory.
    /// </remarks>
    public long EstimatedActivationMemory { get; init; }
}
