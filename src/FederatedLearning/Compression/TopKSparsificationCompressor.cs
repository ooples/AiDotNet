using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.FederatedLearning.Compression;

/// <summary>
/// Implements Top-k Sparsification — send only the k largest gradient elements.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Most gradient values are small and contribute little to learning.
/// Top-k sparsification identifies the k largest values (by magnitude) and only sends those,
/// zeroing out the rest. With error feedback (accumulating the "residual" for next round),
/// this converges to the same solution as full gradient but with much less communication.</para>
///
/// <para>Algorithm:</para>
/// <code>
/// 1. accumulated_error += gradient
/// 2. top_k = indices of k largest |accumulated_error| values
/// 3. send = accumulated_error[top_k]  (sparse)
/// 4. accumulated_error[top_k] = 0     (reset sent values)
/// </code>
///
/// <para>Reference: Aji, A. &amp; Heafield, K. (2017). "Sparse Communication for Distributed
/// Gradient Descent." EMNLP 2017.</para>
/// </remarks>
/// <typeparam name="T">The numeric type for model parameters.</typeparam>
[ComponentType(ComponentType.FederatedAggregator)]
[PipelineStage(PipelineStage.Training)]
public class TopKSparsificationCompressor<T> : Infrastructure.FederatedLearningComponentBase<T>
{
    private readonly double _compressionRatio;
    private readonly bool _useErrorFeedback;
    private Dictionary<int, Dictionary<string, T[]>>? _errorAccumulators;

    /// <summary>
    /// Creates a new Top-k sparsification compressor.
    /// </summary>
    /// <param name="compressionRatio">Fraction of elements to keep (k/d). Default: 0.01 (top 1%).</param>
    /// <param name="useErrorFeedback">Whether to accumulate residual errors. Default: true.</param>
    public TopKSparsificationCompressor(double compressionRatio = 0.01, bool useErrorFeedback = true)
    {
        if (compressionRatio <= 0 || compressionRatio > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(compressionRatio), "Compression ratio must be in (0, 1].");
        }

        _compressionRatio = compressionRatio;
        _useErrorFeedback = useErrorFeedback;
    }

    /// <summary>
    /// Compresses a gradient using top-k sparsification with optional error feedback.
    /// </summary>
    /// <param name="gradient">Full gradient.</param>
    /// <param name="clientId">Client ID for error accumulator tracking.</param>
    /// <returns>Sparse gradient (non-top-k elements are zero).</returns>
    public Dictionary<string, T[]> Compress(Dictionary<string, T[]> gradient, int clientId)
    {
        Guard.NotNull(gradient);
        var selection = SelectTopK(gradient, clientId);

        var compressed = new Dictionary<string, T[]>(gradient.Count);
        for (int l = 0; l < selection.Layers.Length; l++)
        {
            var result = new T[selection.Lengths[l]];
            Array.Copy(selection.Selected, selection.Offsets[l], result, 0, result.Length);
            compressed[selection.Layers[l]] = result;
        }

        return compressed;
    }

    /// <summary>
    /// Compresses a gradient to a sparse representation (index-value pairs), saving bandwidth.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Instead of sending a full array with zeros for non-top-k elements,
    /// a sparse representation only sends the (index, value) pairs for the top-k entries. For 1%
    /// sparsification this reduces communication by ~50x (indices + values for 1% vs full array).</para>
    /// </remarks>
    /// <param name="gradient">Full gradient dictionary.</param>
    /// <param name="clientId">Client ID for error accumulator tracking.</param>
    /// <returns>Sparse gradient: layerName → list of (index, value) pairs.</returns>
    public Dictionary<string, List<(int Index, T Value)>> CompressSparse(
        Dictionary<string, T[]> gradient, int clientId)
    {
        Guard.NotNull(gradient);
        var selection = SelectTopK(gradient, clientId);

        var sparse = new Dictionary<string, List<(int Index, T Value)>>();
        int layer = 0;
        foreach (int flatIndex in selection.SortedTopIndices)
        {
            while (flatIndex >= selection.Offsets[layer] + selection.Lengths[layer])
            {
                layer++;
            }

            string layerName = selection.Layers[layer];
            if (!sparse.TryGetValue(layerName, out var entries))
            {
                entries = new List<(int Index, T Value)>();
                sparse[layerName] = entries;
            }

            entries.Add((flatIndex - selection.Offsets[layer], selection.Selected[flatIndex]));
        }

        return sparse;
    }

    /// <summary>
    /// Adds the gradient to the client's error accumulator (when error feedback is on), keeps the k entries of
    /// largest magnitude across all layers, and leaves the unsent remainder in the accumulator.
    /// </summary>
    private TopKSelection SelectTopK(Dictionary<string, T[]> gradient, int clientId)
    {
        _errorAccumulators ??= new Dictionary<int, Dictionary<string, T[]>>();
        if (!_errorAccumulators.TryGetValue(clientId, out var errorAcc))
        {
            errorAcc = new Dictionary<string, T[]>();
            _errorAccumulators[clientId] = errorAcc;
        }

        var layers = gradient.Keys.ToArray();
        var offsets = new int[layers.Length];
        var lengths = new int[layers.Length];
        int totalElements = 0;
        for (int l = 0; l < layers.Length; l++)
        {
            offsets[l] = totalElements;
            lengths[l] = gradient[layers[l]].Length;
            totalElements += lengths[l];
        }

        // Candidates: the accumulated error (error feedback) or the raw gradient.
        var candidates = new T[totalElements];
        for (int l = 0; l < layers.Length; l++)
        {
            var values = gradient[layers[l]];
            if (!errorAcc.TryGetValue(layers[l], out var err))
            {
                err = new T[values.Length];
                Vector<T>.Wrap(err).Fill(NumOps.Zero);
                errorAcc[layers[l]] = err;
            }

            var source = values;
            if (_useErrorFeedback)
            {
                Engine.Add(Vector<T>.Wrap(err), Vector<T>.Wrap(values)).AsSpan().CopyTo(err);
                source = err;
            }

            Array.Copy(source, 0, candidates, offsets[l], values.Length);
        }

        int k = Math.Min(Math.Max(1, (int)(totalElements * _compressionRatio)), totalElements);
        var selected = new T[totalElements];
        var sortedTop = new int[k];
        if (k > 0)
        {
            var flat = Vector<T>.Wrap(candidates);
            Engine.TensorTopK(Engine.Abs(flat).AsTensor(), k, 0, out var topIndices);
            sortedTop = topIndices.ToArray();
            Array.Sort(sortedTop);

            var mask = new int[totalElements];
            foreach (int index in sortedTop)
            {
                mask[index] = 1;
            }

            var isTop = Vector<int>.Wrap(mask).AsTensor().Cast<T>();
            var zeros = new Tensor<T>(new[] { totalElements });
            Engine.TensorWhere(isTop, flat.AsTensor(), zeros).ToArray().CopyTo(selected, 0);

            if (_useErrorFeedback)
            {
                // Sent values leave the accumulator; the rest carries over to the next round.
                var remainder = Engine.TensorWhere(isTop, zeros, flat.AsTensor()).ToArray();
                for (int l = 0; l < layers.Length; l++)
                {
                    Array.Copy(remainder, offsets[l], errorAcc[layers[l]], 0, lengths[l]);
                }
            }
        }

        return new TopKSelection(layers, offsets, lengths, selected, sortedTop);
    }

    private sealed class TopKSelection
    {
        public TopKSelection(string[] layers, int[] offsets, int[] lengths, T[] selected, int[] sortedTopIndices)
        {
            Layers = layers;
            Offsets = offsets;
            Lengths = lengths;
            Selected = selected;
            SortedTopIndices = sortedTopIndices;
        }

        public string[] Layers { get; }

        public int[] Offsets { get; }

        public int[] Lengths { get; }

        public T[] Selected { get; }

        public int[] SortedTopIndices { get; }
    }

    /// <summary>
    /// Decompresses a sparse representation back to a dense gradient dictionary.
    /// </summary>
    /// <param name="sparse">Sparse gradient from CompressSparse.</param>
    /// <param name="layerSizes">Expected size of each layer: layerName → array length.</param>
    /// <returns>Dense gradient dictionary with zeros for non-top-k elements.</returns>
    public Dictionary<string, T[]> Decompress(
        Dictionary<string, List<(int Index, T Value)>> sparse,
        Dictionary<string, int> layerSizes)
    {
        Guard.NotNull(sparse);
        Guard.NotNull(layerSizes);
        var dense = new Dictionary<string, T[]>(layerSizes.Count);
        foreach (var (layerName, size) in layerSizes)
        {
            var result = new T[size];
            if (sparse.TryGetValue(layerName, out var entries))
            {
                foreach (var (index, value) in entries)
                {
                    if (index < 0 || index >= size)
                    {
                        throw new ArgumentOutOfRangeException(nameof(sparse),
                            $"Layer '{layerName}' contains out-of-bounds index {index} (size={size}).");
                    }

                    result[index] = value;
                }
            }

            dense[layerName] = result;
        }

        return dense;
    }

    /// <summary>
    /// Aggregates sparse gradients from multiple clients by summing overlapping entries.
    /// </summary>
    /// <param name="clientSparseGradients">Sparse gradients from each client.</param>
    /// <param name="layerSizes">Expected layer sizes for decompression.</param>
    /// <returns>Aggregated dense gradient (averaged across clients).</returns>
    public Dictionary<string, T[]> AggregateSparse(
        Dictionary<int, Dictionary<string, List<(int Index, T Value)>>> clientSparseGradients,
        Dictionary<string, int> layerSizes)
    {
        Guard.NotNull(clientSparseGradients);
        Guard.NotNull(layerSizes);
        int numClients = clientSparseGradients.Count;
        if (numClients == 0)
        {
            throw new ArgumentException("No client gradients to aggregate.", nameof(clientSparseGradients));
        }

        var aggregated = new Dictionary<string, Tensor<T>>(layerSizes.Count);
        foreach (var (layerName, size) in layerSizes)
        {
            aggregated[layerName] = new Tensor<T>(new[] { size });
        }

        foreach (var (clientId, sparseGrad) in clientSparseGradients)
        {
            foreach (var (layerName, entries) in sparseGrad)
            {
                if (!aggregated.TryGetValue(layerName, out var agg))
                {
                    throw new ArgumentException(
                        $"Client {clientId} sent unknown layer '{layerName}' not present in layerSizes.",
                        nameof(clientSparseGradients));
                }

                int size = layerSizes[layerName];
                var indices = new int[entries.Count];
                var values = new T[entries.Count];
                for (int e = 0; e < entries.Count; e++)
                {
                    var (index, value) = entries[e];
                    if (index < 0 || index >= size)
                    {
                        throw new ArgumentOutOfRangeException(nameof(clientSparseGradients),
                            $"Client {clientId} layer '{layerName}' contains out-of-bounds index {index} (size={size}).");
                    }

                    indices[e] = index;
                    values[e] = value;
                }

                if (entries.Count > 0)
                {
                    aggregated[layerName] = Engine.TensorScatterAdd(
                        agg, Vector<int>.Wrap(indices).AsTensor(), Vector<T>.Wrap(values).AsTensor(), 0);
                }
            }
        }

        // Average and convert to T[].
        var result = new Dictionary<string, T[]>(layerSizes.Count);
        var invClients = NumOps.FromDouble(1.0 / numClients);
        foreach (var (layerName, agg) in aggregated)
        {
            result[layerName] = Engine.TensorMultiplyScalar(agg, invClients).ToArray();
        }

        return result;
    }

    /// <summary>Gets the compression ratio (fraction of elements kept).</summary>
    public double CompressionRatio => _compressionRatio;

    /// <summary>Gets whether error feedback is enabled.</summary>
    public bool UseErrorFeedback => _useErrorFeedback;
}
