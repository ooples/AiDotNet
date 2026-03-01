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
internal class SignSGDCompressor<T> : Infrastructure.FederatedLearningComponentBase<T>
{
    private readonly double _learningRate;
    private readonly bool _useMajorityVote;
    private readonly double _momentum;
    private Dictionary<int, Dictionary<string, double[]>>? _errorAccumulators;
    private Dictionary<int, Dictionary<string, double[]>>? _momentumBuffers;

    /// <summary>
    /// Creates a new SignSGD compressor.
    /// </summary>
    /// <param name="learningRate">Server-side learning rate for sign updates. Default: 0.01.</param>
    /// <param name="useMajorityVote">If true, server takes majority vote of signs. Default: true.</param>
    /// <param name="momentum">Momentum factor for SIGNUM variant. 0 = no momentum. Default: 0.9.</param>
    public SignSGDCompressor(double learningRate = 0.01, bool useMajorityVote = true, double momentum = 0.9)
    {
        if (learningRate <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(learningRate), "Learning rate must be positive.");
        }

        if (momentum < 0 || momentum >= 1)
        {
            throw new ArgumentOutOfRangeException(nameof(momentum), "Momentum must be in [0, 1).");
        }

        _learningRate = learningRate;
        _useMajorityVote = useMajorityVote;
        _momentum = momentum;
    }

    /// <summary>
    /// Compresses a gradient to its signs with error feedback and optional momentum (SIGNUM).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Plain SignSGD discards the gradient magnitude, which loses information.
    /// Error feedback accumulates the discarded residual and adds it to the next gradient, ensuring
    /// nothing is permanently lost. SIGNUM (momentum variant) applies momentum before taking the sign,
    /// which smooths noise and improves convergence — it's the equivalent of Adam but for 1-bit compression.</para>
    /// </remarks>
    /// <param name="gradient">The gradient parameter dictionary.</param>
    /// <param name="clientId">Client ID for error accumulator tracking. Default: 0.</param>
    /// <returns>Sign-compressed dictionary (values are +1, -1, or 0).</returns>
    public Dictionary<string, T[]> Compress(Dictionary<string, T[]> gradient, int clientId = 0)
    {
        Guard.NotNull(gradient);
        if (gradient.Count == 0)
        {
            return new Dictionary<string, T[]>();
        }

        _errorAccumulators ??= new Dictionary<int, Dictionary<string, double[]>>();
        _momentumBuffers ??= new Dictionary<int, Dictionary<string, double[]>>();

        if (!_errorAccumulators.ContainsKey(clientId))
        {
            _errorAccumulators[clientId] = new Dictionary<string, double[]>();
        }

        if (!_momentumBuffers.ContainsKey(clientId))
        {
            _momentumBuffers[clientId] = new Dictionary<string, double[]>();
        }

        var errorAcc = _errorAccumulators[clientId];
        var momBuf = _momentumBuffers[clientId];
        var compressed = new Dictionary<string, T[]>(gradient.Count);

        foreach (var kvp in gradient)
        {
            if (!errorAcc.ContainsKey(kvp.Key))
            {
                errorAcc[kvp.Key] = new double[kvp.Value.Length];
            }

            if (!momBuf.ContainsKey(kvp.Key))
            {
                momBuf[kvp.Key] = new double[kvp.Value.Length];
            }

            var err = errorAcc[kvp.Key];
            var mom = momBuf[kvp.Key];
            var signs = new T[kvp.Value.Length];

            for (int i = 0; i < signs.Length; i++)
            {
                double grad = NumOps.ToDouble(kvp.Value[i]) + err[i]; // Add error feedback.

                // SIGNUM: apply momentum before sign.
                mom[i] = _momentum * mom[i] + (1 - _momentum) * grad;
                double signInput = _momentum > 0 ? mom[i] : grad;

                double sign = signInput > 0 ? 1.0 : signInput < 0 ? -1.0 : 0.0;
                signs[i] = NumOps.FromDouble(sign);

                // Error feedback: residual = original - reconstructed
                err[i] = grad - sign; // sign is the "reconstruction" for sign compression.
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
        Guard.NotNull(clientSigns);
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
                    if (!client.TryGetValue(layerName, out var clientLayer))
                    {
                        continue; // Skip clients missing this layer.
                    }

                    if (clientLayer.Length != len)
                    {
                        throw new ArgumentException(
                            $"Client layer '{layerName}' length {clientLayer.Length} does not match reference length {len}.");
                    }

                    sum += NumOps.ToDouble(clientLayer[i]);
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

    /// <summary>Gets the compression ratio (source bits per element / 1 bit for sign).</summary>
    public double CompressionRatio => System.Runtime.InteropServices.Marshal.SizeOf<T>() * 8;

    /// <summary>Gets the server learning rate.</summary>
    public double LearningRate => _learningRate;
}
