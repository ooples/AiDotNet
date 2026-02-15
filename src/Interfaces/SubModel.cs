namespace AiDotNet.Interfaces;

/// <summary>
/// Represents a contiguous sub-model extracted from a larger <see cref="ILayeredModel{T}"/>.
/// Contains a slice of layers that can be independently forwarded through.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> When you split a neural network across GPUs for pipeline
/// parallelism, each GPU gets a sub-model - a consecutive sequence of layers from the
/// original network. This class represents that slice.</para>
///
/// <para>A sub-model is not a full neural network - it doesn't own the parameters or
/// support training directly. It provides read-only access to the layers and enables
/// sequential forward passes through them.</para>
///
/// <para><b>Reference:</b> Inspired by PyTorch's <c>split_module()</c> which returns
/// <c>nn.Sequential</c> modules for each pipeline stage.</para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class SubModel<T>
{
    private readonly IReadOnlyList<ILayer<T>> _layers;

    /// <summary>
    /// Gets the ordered list of layers in this sub-model.
    /// </summary>
    public IReadOnlyList<ILayer<T>> Layers => _layers;

    /// <summary>
    /// Gets the number of layers in this sub-model.
    /// </summary>
    public int LayerCount => _layers.Count;

    /// <summary>
    /// Gets the start index of this sub-model within the parent model.
    /// </summary>
    public int StartIndex { get; }

    /// <summary>
    /// Gets the end index (inclusive) of this sub-model within the parent model.
    /// </summary>
    public int EndIndex { get; }

    /// <summary>
    /// Gets the total parameter count across all layers in this sub-model.
    /// </summary>
    public int ParameterCount { get; }

    /// <summary>
    /// Gets the total estimated FLOPs across all layers in this sub-model.
    /// </summary>
    public long TotalEstimatedFlops { get; }

    /// <summary>
    /// Gets the layer metadata for all layers in this sub-model.
    /// </summary>
    public IReadOnlyList<LayerInfo<T>> LayerInfos { get; }

    /// <summary>
    /// Creates a new sub-model from the specified layers and metadata.
    /// </summary>
    /// <param name="layers">The layers included in this sub-model.</param>
    /// <param name="layerInfos">The metadata for each layer.</param>
    /// <param name="startIndex">The start index within the parent model.</param>
    /// <param name="endIndex">The end index (inclusive) within the parent model.</param>
    public SubModel(IReadOnlyList<ILayer<T>> layers, IReadOnlyList<LayerInfo<T>> layerInfos,
        int startIndex, int endIndex)
    {
        if (layers is null)
        {
            throw new ArgumentNullException(nameof(layers));
        }
        if (layerInfos is null)
        {
            throw new ArgumentNullException(nameof(layerInfos));
        }
        if (layers.Count == 0)
        {
            throw new ArgumentException("Sub-model must contain at least one layer.", nameof(layers));
        }

        _layers = layers;
        LayerInfos = layerInfos;
        StartIndex = startIndex;
        EndIndex = endIndex;

        int totalParams = 0;
        long totalFlops = 0;
        for (int i = 0; i < layerInfos.Count; i++)
        {
            totalParams += layerInfos[i].ParameterCount;
            totalFlops += layerInfos[i].EstimatedFlops;
        }
        ParameterCount = totalParams;
        TotalEstimatedFlops = totalFlops;
    }

    /// <summary>
    /// Performs a sequential forward pass through all layers in this sub-model.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <returns>The output tensor after passing through all layers.</returns>
    public Tensor<T> Forward(Tensor<T> input)
    {
        if (input is null)
        {
            throw new ArgumentNullException(nameof(input));
        }

        var current = input;
        for (int i = 0; i < _layers.Count; i++)
        {
            current = _layers[i].Forward(current);
        }
        return current;
    }

    /// <summary>
    /// Gets the input shape expected by the first layer in this sub-model.
    /// </summary>
    public int[] GetInputShape() => _layers[0].GetInputShape();

    /// <summary>
    /// Gets the output shape produced by the last layer in this sub-model.
    /// </summary>
    public int[] GetOutputShape() => _layers[_layers.Count - 1].GetOutputShape();
}
