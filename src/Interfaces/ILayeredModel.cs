namespace AiDotNet.Interfaces;

/// <summary>
/// Provides layer-level access to a neural network's architecture and parameters.
/// </summary>
/// <remarks>
/// <para>
/// This interface exposes individual layers with their metadata, enabling per-layer operations
/// across the AiDotNet stack: pipeline parallelism, quantization, pruning, LoRA, meta-learning,
/// activation checkpointing, model export, and knowledge distillation.
/// </para>
/// <para>
/// <b>For Beginners:</b> Neural networks are made up of layers stacked on top of each other.
/// Most model interfaces only let you access all parameters as one big list. This interface
/// lets you inspect and manipulate individual layers - their shapes, weights, types, and
/// connections. This enables advanced techniques like:
/// </para>
/// <list type="bullet">
/// <item><description><b>Pipeline parallelism:</b> splitting the model across GPUs at layer boundaries</description></item>
/// <item><description><b>Mixed-precision quantization:</b> using different bit-widths for different layers</description></item>
/// <item><description><b>LoRA:</b> automatically finding which layers to adapt</description></item>
/// <item><description><b>Pruning:</b> removing less important neurons from specific layers</description></item>
/// <item><description><b>Selective checkpointing:</b> only saving expensive layers' activations</description></item>
/// </list>
/// <para>
/// <b>Reference:</b> This design is inspired by PyTorch's <c>nn.Module</c> hierarchy,
/// Megatron-LM's pipeline partition API, and Flax NNX's module introspection.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public interface ILayeredModel<T>
{
    /// <summary>
    /// Gets the ordered list of layers in this model.
    /// </summary>
    IReadOnlyList<ILayer<T>> Layers { get; }

    /// <summary>
    /// Gets the number of layers in this model.
    /// </summary>
    int LayerCount { get; }

    /// <summary>
    /// Gets metadata for a specific layer including its parameter offset
    /// within the flat parameter vector, enabling layer-aware slicing.
    /// </summary>
    /// <param name="layerIndex">Zero-based index of the layer.</param>
    /// <returns>Metadata about the layer at the specified index.</returns>
    /// <exception cref="ArgumentOutOfRangeException">
    /// Thrown when <paramref name="layerIndex"/> is less than 0 or greater than or equal to <see cref="LayerCount"/>.
    /// </exception>
    LayerInfo<T> GetLayerInfo(int layerIndex);

    /// <summary>
    /// Gets metadata for all layers. Includes parameter offsets, types,
    /// shapes, names, and cost estimates for each layer.
    /// </summary>
    /// <returns>An ordered list of layer metadata.</returns>
    IReadOnlyList<LayerInfo<T>> GetAllLayerInfo();

    /// <summary>
    /// Validates that a partition point between layers is valid
    /// (output shape of layer at <paramref name="afterLayerIndex"/> is compatible with
    /// input shape of the next layer).
    /// </summary>
    /// <param name="afterLayerIndex">The index of the layer after which to partition.
    /// Must be between 0 and <see cref="LayerCount"/> - 2.</param>
    /// <returns>True if the partition point is valid; false otherwise.</returns>
    bool ValidatePartitionPoint(int afterLayerIndex);
}
