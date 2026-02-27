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

    /// <summary>Gets the compression ratio (fraction of elements kept).</summary>
    public double CompressionRatio => _compressionRatio;

    /// <summary>Gets whether error feedback is enabled.</summary>
    public bool UseErrorFeedback => _useErrorFeedback;
}
