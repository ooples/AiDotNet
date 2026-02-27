namespace AiDotNet.FederatedLearning.Compression;

/// <summary>
/// Implements SignSGD — 1-bit gradient compression for federated learning.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> SignSGD is the most aggressive gradient compression possible:
/// instead of sending each gradient value (32 bits), it sends only the sign (+1 or -1, i.e. 1 bit).
/// This gives 32x compression. Surprisingly, this works well because the direction of the gradient
/// is often more important than its magnitude. The server aggregates by majority vote.</para>
///
/// <para>Algorithm:</para>
/// <code>
/// Client: compressed = sign(gradient)   // +1 or -1
/// Server: aggregated = sign(sum(compressed_k))  // majority vote
/// Update: params -= lr * aggregated
/// </code>
///
/// <para>Reference: Bernstein, J., et al. (2018). "signSGD: Compressed Optimisation for
/// Non-Convex Problems." ICML 2018.</para>
/// </remarks>
/// <typeparam name="T">The numeric type for model parameters.</typeparam>
public class SignSGDCompressor<T> : Infrastructure.FederatedLearningComponentBase<T>
{
    private readonly double _learningRate;
    private readonly bool _useMajorityVote;

    /// <summary>
    /// Creates a new SignSGD compressor.
    /// </summary>
    /// <param name="learningRate">Server-side learning rate for sign updates. Default: 0.01.</param>
    /// <param name="useMajorityVote">If true, server takes majority vote of signs. Default: true.</param>
    public SignSGDCompressor(double learningRate = 0.01, bool useMajorityVote = true)
    {
        if (learningRate <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(learningRate), "Learning rate must be positive.");
        }

        _learningRate = learningRate;
        _useMajorityVote = useMajorityVote;
    }

    /// <summary>
    /// Compresses a gradient to its signs.
    /// </summary>
    /// <param name="gradient">The gradient parameter dictionary.</param>
    /// <returns>Sign-compressed dictionary (values are +1, -1, or 0).</returns>
    public Dictionary<string, T[]> Compress(Dictionary<string, T[]> gradient)
    {
        var compressed = new Dictionary<string, T[]>(gradient.Count);
        foreach (var kvp in gradient)
        {
            var signs = new T[kvp.Value.Length];
            for (int i = 0; i < signs.Length; i++)
            {
                double v = NumOps.ToDouble(kvp.Value[i]);
                signs[i] = NumOps.FromDouble(v > 0 ? 1.0 : v < 0 ? -1.0 : 0.0);
            }

            compressed[kvp.Key] = signs;
        }

        return compressed;
    }

    /// <summary>
    /// Aggregates sign-compressed gradients from multiple clients via majority vote.
    /// </summary>
    /// <param name="clientSigns">Client sign-compressed gradients.</param>
    /// <returns>Aggregated sign gradient.</returns>
    public Dictionary<string, T[]> AggregateVote(Dictionary<int, Dictionary<string, T[]>> clientSigns)
    {
        if (clientSigns.Count == 0)
        {
            throw new ArgumentException("No client signs to aggregate.", nameof(clientSigns));
        }

        var reference = clientSigns.First().Value;
        var result = new Dictionary<string, T[]>(reference.Count);

        foreach (var layerName in reference.Keys)
        {
            int len = reference[layerName].Length;
            var aggregated = new T[len];

            for (int i = 0; i < len; i++)
            {
                double sum = 0;
                foreach (var client in clientSigns.Values)
                {
                    sum += NumOps.ToDouble(client[layerName][i]);
                }

                if (_useMajorityVote)
                {
                    aggregated[i] = NumOps.FromDouble(sum > 0 ? _learningRate : sum < 0 ? -_learningRate : 0);
                }
                else
                {
                    aggregated[i] = NumOps.FromDouble(_learningRate * sum / clientSigns.Count);
                }
            }

            result[layerName] = aggregated;
        }

        return result;
    }

    /// <summary>Gets the compression ratio (always 32x for float32).</summary>
    public double CompressionRatio => 32.0;

    /// <summary>Gets the server learning rate.</summary>
    public double LearningRate => _learningRate;
}
