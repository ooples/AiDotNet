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
public class TopKSparsificationCompressor<T> : Infrastructure.FederatedLearningComponentBase<T>
{
    private readonly double _compressionRatio;
    private readonly bool _useErrorFeedback;
    private Dictionary<int, Dictionary<string, double[]>>? _errorAccumulators;

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
        _errorAccumulators ??= new Dictionary<int, Dictionary<string, double[]>>();
        if (!_errorAccumulators.ContainsKey(clientId))
        {
            _errorAccumulators[clientId] = new Dictionary<string, double[]>();
        }

        var errorAcc = _errorAccumulators[clientId];
        var compressed = new Dictionary<string, T[]>(gradient.Count);

        // Flatten to find global top-k.
        int totalElements = gradient.Values.Sum(v => v.Length);
        int k = Math.Max(1, (int)(totalElements * _compressionRatio));

        var allValues = new List<(string Layer, int Index, double AbsValue)>(totalElements);

        foreach (var kvp in gradient)
        {
            if (!errorAcc.ContainsKey(kvp.Key))
            {
                errorAcc[kvp.Key] = new double[kvp.Value.Length];
            }

            var err = errorAcc[kvp.Key];
            for (int i = 0; i < kvp.Value.Length; i++)
            {
                double val = NumOps.ToDouble(kvp.Value[i]);
                if (_useErrorFeedback)
                {
                    err[i] += val;
                    allValues.Add((kvp.Key, i, Math.Abs(err[i])));
                }
                else
                {
                    allValues.Add((kvp.Key, i, Math.Abs(val)));
                }
            }
        }

        // Find threshold for top-k.
        allValues.Sort((a, b) => b.AbsValue.CompareTo(a.AbsValue));
        double threshold = k < allValues.Count ? allValues[k - 1].AbsValue : 0;

        // Build sparse output.
        var topKSet = new HashSet<(string, int)>();
        for (int i = 0; i < Math.Min(k, allValues.Count); i++)
        {
            topKSet.Add((allValues[i].Layer, allValues[i].Index));
        }

        foreach (var kvp in gradient)
        {
            var result = new T[kvp.Value.Length];
            var err = errorAcc[kvp.Key];

            for (int i = 0; i < result.Length; i++)
            {
                if (topKSet.Contains((kvp.Key, i)))
                {
                    result[i] = _useErrorFeedback
                        ? NumOps.FromDouble(err[i])
                        : kvp.Value[i];

                    if (_useErrorFeedback)
                    {
                        err[i] = 0; // Reset sent values.
                    }
                }
                else
                {
                    result[i] = NumOps.Zero;
                }
            }

            compressed[kvp.Key] = result;
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
        _errorAccumulators ??= new Dictionary<int, Dictionary<string, double[]>>();
        if (!_errorAccumulators.ContainsKey(clientId))
        {
            _errorAccumulators[clientId] = new Dictionary<string, double[]>();
        }

        var errorAcc = _errorAccumulators[clientId];

        // Accumulate error feedback.
        int totalElements = gradient.Values.Sum(v => v.Length);
        int k = Math.Max(1, (int)(totalElements * _compressionRatio));

        var allValues = new List<(string Layer, int Index, double AbsValue, double Value)>(totalElements);

        foreach (var kvp in gradient)
        {
            if (!errorAcc.ContainsKey(kvp.Key))
            {
                errorAcc[kvp.Key] = new double[kvp.Value.Length];
            }

            var err = errorAcc[kvp.Key];
            for (int i = 0; i < kvp.Value.Length; i++)
            {
                double val = NumOps.ToDouble(kvp.Value[i]);
                if (_useErrorFeedback)
                {
                    err[i] += val;
                    allValues.Add((kvp.Key, i, Math.Abs(err[i]), err[i]));
                }
                else
                {
                    allValues.Add((kvp.Key, i, Math.Abs(val), val));
                }
            }
        }

        allValues.Sort((a, b) => b.AbsValue.CompareTo(a.AbsValue));

        var topKSet = new HashSet<(string, int)>();
        for (int i = 0; i < Math.Min(k, allValues.Count); i++)
        {
            topKSet.Add((allValues[i].Layer, allValues[i].Index));
        }

        var sparse = new Dictionary<string, List<(int Index, T Value)>>();
        foreach (var kvp in gradient)
        {
            var entries = new List<(int Index, T Value)>();
            var err = errorAcc[kvp.Key];

            for (int i = 0; i < kvp.Value.Length; i++)
            {
                if (topKSet.Contains((kvp.Key, i)))
                {
                    double val = _useErrorFeedback ? err[i] : NumOps.ToDouble(kvp.Value[i]);
                    entries.Add((i, NumOps.FromDouble(val)));

                    if (_useErrorFeedback)
                    {
                        err[i] = 0;
                    }
                }
            }

            if (entries.Count > 0)
            {
                sparse[kvp.Key] = entries;
            }
        }

        return sparse;
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

        var aggregated = new Dictionary<string, double[]>(layerSizes.Count);
        foreach (var (layerName, size) in layerSizes)
        {
            aggregated[layerName] = new double[size];
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

                foreach (var (index, value) in entries)
                {
                    if (index < 0 || index >= agg.Length)
                    {
                        throw new ArgumentOutOfRangeException(nameof(clientSparseGradients),
                            $"Client {clientId} layer '{layerName}' contains out-of-bounds index {index} (size={agg.Length}).");
                    }

                    agg[index] += NumOps.ToDouble(value);
                }
            }
        }

        // Average and convert to T[].
        var result = new Dictionary<string, T[]>(layerSizes.Count);
        double invClients = 1.0 / numClients;
        foreach (var (layerName, agg) in aggregated)
        {
            var values = new T[agg.Length];
            for (int i = 0; i < agg.Length; i++)
            {
                values[i] = NumOps.FromDouble(agg[i] * invClients);
            }

            result[layerName] = values;
        }

        return result;
    }

    /// <summary>Gets the compression ratio (fraction of elements kept).</summary>
    public double CompressionRatio => _compressionRatio;

    /// <summary>Gets whether error feedback is enabled.</summary>
    public bool UseErrorFeedback => _useErrorFeedback;
}
