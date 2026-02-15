namespace AiDotNet.Interfaces;

/// <summary>
/// Represents a contiguous sub-model extracted from a larger <see cref="ILayeredModel{T}"/>.
/// Contains a slice of layers that can be independently forwarded through and further partitioned.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> When you split a neural network across GPUs for pipeline
/// parallelism, each GPU gets a sub-model - a consecutive sequence of layers from the
/// original network. This class represents that slice.</para>
///
/// <para>A sub-model implements <see cref="ILayeredModel{T}"/> itself, so you can extract
/// sub-models of sub-models, enabling hierarchical partitioning (e.g., virtual pipeline stages
/// in Megatron-LM).</para>
///
/// <para><b>Reference:</b> Inspired by PyTorch's <c>split_module()</c> which returns
/// <c>nn.Sequential</c> modules for each pipeline stage.</para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class SubModel<T> : ILayeredModel<T>
{
    private readonly IReadOnlyList<ILayer<T>> _layers;
    private readonly IReadOnlyList<LayerInfo<T>> _layerInfos;

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
    public IReadOnlyList<LayerInfo<T>> LayerInfos => _layerInfos;

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
        if (layers.Count != layerInfos.Count)
        {
            throw new ArgumentException(
                $"Layer count ({layers.Count}) must match layer info count ({layerInfos.Count}).",
                nameof(layerInfos));
        }
        if (startIndex < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(startIndex), "Start index cannot be negative.");
        }
        if (endIndex < startIndex)
        {
            throw new ArgumentOutOfRangeException(nameof(endIndex),
                $"End index ({endIndex}) must be >= start index ({startIndex}).");
        }

        _layers = layers;
        _layerInfos = layerInfos;
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

    /// <inheritdoc/>
    public LayerInfo<T> GetLayerInfo(int layerIndex)
    {
        if (layerIndex < 0 || layerIndex >= _layerInfos.Count)
        {
            throw new ArgumentOutOfRangeException(nameof(layerIndex),
                $"Layer index must be between 0 and {_layerInfos.Count - 1}.");
        }

        return _layerInfos[layerIndex];
    }

    /// <inheritdoc/>
    public IReadOnlyList<LayerInfo<T>> GetAllLayerInfo() => _layerInfos;

    /// <inheritdoc/>
    public bool ValidatePartitionPoint(int afterLayerIndex)
    {
        if (afterLayerIndex < 0 || afterLayerIndex >= _layers.Count - 1)
        {
            return false;
        }

        var currentOutput = _layers[afterLayerIndex].GetOutputShape();
        var nextInput = _layers[afterLayerIndex + 1].GetInputShape();

        if (currentOutput.Length != nextInput.Length)
        {
            return false;
        }

        for (int i = 0; i < currentOutput.Length; i++)
        {
            if (currentOutput[i] != nextInput[i])
            {
                return false;
            }
        }

        return true;
    }

    /// <inheritdoc/>
    public SubModel<T> ExtractSubModel(int startLayer, int endLayer)
    {
        if (startLayer < 0 || startLayer >= _layers.Count)
        {
            throw new ArgumentOutOfRangeException(nameof(startLayer),
                $"Start layer must be between 0 and {_layers.Count - 1}.");
        }
        if (endLayer < startLayer || endLayer >= _layers.Count)
        {
            throw new ArgumentOutOfRangeException(nameof(endLayer),
                $"End layer must be between {startLayer} and {_layers.Count - 1}.");
        }

        int count = endLayer - startLayer + 1;
        var subLayers = new List<ILayer<T>>(count);
        var subInfos = new List<LayerInfo<T>>(count);

        int localOffset = 0;
        for (int i = startLayer; i <= endLayer; i++)
        {
            subLayers.Add(_layers[i]);
            var original = _layerInfos[i];
            subInfos.Add(new LayerInfo<T>
            {
                Index = i - startLayer,
                Name = original.Name,
                Category = original.Category,
                Layer = original.Layer,
                ParameterOffset = localOffset,
                ParameterCount = original.ParameterCount,
                InputShape = original.InputShape,
                OutputShape = original.OutputShape,
                IsTrainable = original.IsTrainable,
                EstimatedFlops = original.EstimatedFlops,
                EstimatedActivationMemory = original.EstimatedActivationMemory
            });
            localOffset += original.ParameterCount;
        }

        return new SubModel<T>(subLayers, subInfos,
            StartIndex + startLayer, StartIndex + endLayer);
    }
}
