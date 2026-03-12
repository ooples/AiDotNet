using AiDotNet.FederatedLearning.Infrastructure;
using AiDotNet.Models.Options;
using AiDotNet.Tensors;

namespace AiDotNet.FederatedLearning.Compression;

/// <summary>
/// Adaptive compressor: dynamically adjusts compression ratio per client based on bandwidth,
/// gradient importance, and staleness.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> In a real federation, clients have different network speeds.
/// A phone on 5G can send more data than one on 3G. Rather than using the same compression
/// for everyone, the adaptive compressor gives faster clients less compression (better accuracy)
/// and slower clients more compression (saves bandwidth). It also prioritizes clients whose
/// gradients carry more information.</para>
///
/// <para><b>How it works:</b></para>
/// <list type="number">
/// <item><description><b>Bandwidth estimation:</b> Track each client's recent round-trip times to estimate
/// their available bandwidth. Faster clients get higher compression ratios (less compression).</description></item>
/// <item><description><b>Gradient importance:</b> Clients with larger gradient norms (more to contribute)
/// get less compression to preserve their valuable information.</description></item>
/// <item><description><b>Staleness:</b> Clients that haven't communicated recently get priority (less compression)
/// to keep their contribution current.</description></item>
/// <item><description><b>Top-K sparsification:</b> The selected compression ratio determines what fraction
/// of gradient elements to keep (top-k by absolute value).</description></item>
/// </list>
/// </remarks>
/// <typeparam name="T">The numeric type used for model parameters.</typeparam>
public class AdaptiveCompressor<T> : FederatedLearningComponentBase<T>
{
    private readonly AdvancedCompressionOptions _options;
    private readonly Dictionary<int, List<double>> _clientBandwidthHistory = new();
    private readonly Dictionary<int, int> _clientLastRound = new();

    /// <summary>
    /// Initializes a new instance of <see cref="AdaptiveCompressor{T}"/>.
    /// </summary>
    /// <param name="options">Advanced compression configuration.</param>
    public AdaptiveCompressor(AdvancedCompressionOptions options)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
    }

    /// <summary>
    /// Compresses a gradient using an adaptively determined top-k ratio.
    /// </summary>
    /// <param name="gradient">The gradient tensor to compress.</param>
    /// <param name="clientId">Client ID for adaptive ratio determination.</param>
    /// <param name="currentRound">Current training round number.</param>
    /// <param name="roundTripMs">Optional measured round-trip time in milliseconds for bandwidth estimation.</param>
    /// <returns>Compressed sparse gradient and the actual compression ratio used.</returns>
    public (Tensor<T> Compressed, double ActualRatio) Compress(
        Tensor<T> gradient, int clientId, int currentRound, double roundTripMs = 0)
    {
        if (gradient is null) throw new ArgumentNullException(nameof(gradient));

        // Update bandwidth history
        if (roundTripMs > 0)
        {
            RecordBandwidth(clientId, roundTripMs);
        }

        int size = gradient.Shape[0];

        // Compute adaptive ratio
        double ratio = ComputeAdaptiveRatio(clientId, gradient, currentRound);
        int topK = Math.Max(1, (int)(size * ratio));
        topK = Math.Min(topK, size);

        // Apply top-k sparsification
        var compressed = ApplyTopK(gradient, size, topK);

        // Update staleness tracker
        _clientLastRound[clientId] = currentRound;

        return (compressed, ratio);
    }

    /// <summary>
    /// Records a bandwidth measurement for a client.
    /// </summary>
    /// <param name="clientId">Client ID.</param>
    /// <param name="roundTripMs">Round-trip time in milliseconds.</param>
    public void RecordBandwidth(int clientId, double roundTripMs)
    {
        if (!_clientBandwidthHistory.ContainsKey(clientId))
        {
            _clientBandwidthHistory[clientId] = new List<double>();
        }

        var history = _clientBandwidthHistory[clientId];
        history.Add(roundTripMs);

        // Keep only the last N measurements
        while (history.Count > _options.AdaptiveBandwidthWindow)
        {
            history.RemoveAt(0);
        }
    }

    /// <summary>
    /// Gets the current adaptive compression ratio for a client.
    /// </summary>
    /// <param name="clientId">Client ID.</param>
    /// <param name="gradient">Current gradient (for importance estimation).</param>
    /// <param name="currentRound">Current round number (for staleness).</param>
    /// <returns>Compression ratio in [MinRatio, MaxRatio].</returns>
    public double ComputeAdaptiveRatio(int clientId, Tensor<T> gradient, int currentRound)
    {
        double bandwidthFactor = ComputeBandwidthFactor(clientId);
        double importanceFactor = ComputeImportanceFactor(gradient);
        double stalenessFactor = ComputeStalenessFactor(clientId, currentRound);

        // Combine factors: higher = less compression (send more)
        // bandwidth: fast client = high, slow = low
        // importance: important gradient = high
        // staleness: stale client = high (prioritize catching up)
        double combinedScore = bandwidthFactor * 0.4 + importanceFactor * 0.3 + stalenessFactor * 0.3;

        // Map to ratio range
        double ratio = _options.AdaptiveMinRatio +
            combinedScore * (_options.AdaptiveMaxRatio - _options.AdaptiveMinRatio);

        return Math.Max(_options.AdaptiveMinRatio, Math.Min(_options.AdaptiveMaxRatio, ratio));
    }

    private double ComputeBandwidthFactor(int clientId)
    {
        if (!_clientBandwidthHistory.ContainsKey(clientId) || _clientBandwidthHistory[clientId].Count == 0)
        {
            return 0.5; // Unknown bandwidth, use middle
        }

        var history = _clientBandwidthHistory[clientId];

        // Average recent round-trip time
        double avgRtt = 0;
        foreach (double rtt in history)
        {
            avgRtt += rtt;
        }
        avgRtt /= history.Count;

        // Lower RTT = higher bandwidth = can send more
        // Map: 10ms = 1.0 (excellent), 1000ms = 0.0 (poor)
        double factor = 1.0 - Math.Min(1.0, Math.Max(0, Math.Log10(avgRtt / 10.0)));
        return Math.Max(0, Math.Min(1.0, factor));
    }

    private double ComputeImportanceFactor(Tensor<T> gradient)
    {
        // Importance = L2 norm of gradient (larger gradients carry more information)
        int size = gradient.Shape[0];
        double sumSq = 0;

        for (int i = 0; i < size; i++)
        {
            double val = NumOps.ToDouble(gradient[i]);
            sumSq += val * val;
        }

        double norm = Math.Sqrt(sumSq);

        // Map norm to [0, 1] using sigmoid-like scaling
        // norm of 1.0 maps to ~0.5; higher norms approach 1.0
        return 2.0 / (1.0 + Math.Exp(-norm)) - 1.0;
    }

    private double ComputeStalenessFactor(int clientId, int currentRound)
    {
        if (!_clientLastRound.ContainsKey(clientId))
        {
            return 1.0; // New client, maximize their priority
        }

        int staleness = currentRound - _clientLastRound[clientId];

        // More stale = higher factor (prioritize catching up)
        // 0 rounds stale = 0.0; 10+ rounds stale = 1.0
        return Math.Min(1.0, staleness / 10.0);
    }

    private Tensor<T> ApplyTopK(Tensor<T> gradient, int size, int topK)
    {
        var result = new Tensor<T>(new[] { size });

        // Find the top-k absolute values
        var indices = new int[size];
        var absValues = new double[size];

        for (int i = 0; i < size; i++)
        {
            indices[i] = i;
            absValues[i] = Math.Abs(NumOps.ToDouble(gradient[i]));
        }

        // Partial sort to find top-k (selection using nth_element-like approach)
        // For simplicity, full sort (production would use quickselect)
        Array.Sort(absValues, indices);

        // Keep only top-k (last k elements after ascending sort)
        int threshold = size - topK;
        for (int i = threshold; i < size; i++)
        {
            result[indices[i]] = gradient[indices[i]];
        }

        return result;
    }
}
