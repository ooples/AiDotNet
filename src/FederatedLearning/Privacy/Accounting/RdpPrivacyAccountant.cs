namespace AiDotNet.FederatedLearning.Privacy.Accounting;

/// <summary>
/// A simple RĂ©nyi Differential Privacy (RDP) accountant for repeated Gaussian mechanisms.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> RDP accounting often gives tighter (less pessimistic) privacy reporting
/// than basic composition. This implementation uses a conservative Gaussian-mechanism RDP bound
/// and converts it back to an (epsilon, delta) style report.
/// </remarks>
public sealed class RdpPrivacyAccountant : PrivacyAccountantBase
{
    private readonly double _clipNorm;
    private readonly List<double> _orders;
    private readonly double[] _rdpTotals;
    private double _deltaTotal;

    public RdpPrivacyAccountant(double clipNorm = 1.0, IEnumerable<double>? orders = null)
    {
        if (clipNorm <= 0.0)
        {
            throw new ArgumentOutOfRangeException(nameof(clipNorm), "Clip norm must be positive.");
        }

        _clipNorm = clipNorm;
        _orders = orders?.ToList() ?? new List<double> { 2, 3, 4, 5, 8, 16, 32, 64, 128 };
        if (_orders.Count == 0 || _orders.Any(a => a <= 1.0))
        {
            throw new ArgumentException("RDP orders must be > 1 and non-empty.", nameof(orders));
        }

        _rdpTotals = new double[_orders.Count];
        _deltaTotal = 0.0;
    }

    public override void AddRound(double epsilon, double delta, double samplingRate)
    {
        if (epsilon <= 0.0)
        {
            throw new ArgumentOutOfRangeException(nameof(epsilon), "Epsilon must be positive.");
        }

        if (delta <= 0.0 || delta >= 1.0)
        {
            throw new ArgumentOutOfRangeException(nameof(delta), "Delta must be in (0, 1).");
        }

        if (samplingRate <= 0.0 || samplingRate > 1.0)
        {
            throw new ArgumentOutOfRangeException(nameof(samplingRate), "Sampling rate must be in (0, 1].");
        }

        // Convert per-round (epsilon, delta) for a Gaussian mechanism with L2 sensitivity = clipNorm
        // into an equivalent noise sigma, then apply a conservative (non-subsampled) RDP bound:
        // RDP(α) = α / (2σ^2).
        //
        // Note: This ignores privacy amplification by subsampling; using samplingRate only for validation
        // keeps the report conservative.
        double sensitivity = _clipNorm;
        double sigma = (sensitivity / epsilon) * Math.Sqrt(2.0 * Math.Log(1.25 / delta));

        double twoSigmaSq = 2.0 * sigma * sigma;
        for (int i = 0; i < _orders.Count; i++)
        {
            double alpha = _orders[i];
            _rdpTotals[i] += alpha / twoSigmaSq;
        }

        _deltaTotal += delta;
    }

    public override double GetTotalEpsilonConsumed()
    {
        // Default to a conservative report at delta_total if no explicit delta requested.
        var delta = Math.Min(Math.Max(_deltaTotal, 1e-12), 0.999999999999);
        return GetEpsilonAtDelta(delta);
    }

    public override double GetTotalDeltaConsumed() => _deltaTotal;

    public override double GetEpsilonAtDelta(double targetDelta)
    {
        if (targetDelta <= 0.0 || targetDelta >= 1.0)
        {
            throw new ArgumentOutOfRangeException(nameof(targetDelta), "Target delta must be in (0, 1).");
        }

        // Convert RDP to (ε, δ): ε = min_α (RDP(α) + log(1/δ)/(α-1)).
        double best = double.PositiveInfinity;
        double logTerm = Math.Log(1.0 / targetDelta);
        for (int i = 0; i < _orders.Count; i++)
        {
            double alpha = _orders[i];
            double eps = _rdpTotals[i] + logTerm / (alpha - 1.0);
            if (eps < best)
            {
                best = eps;
            }
        }

        return double.IsPositiveInfinity(best) ? 0.0 : best;
    }

    public override string GetAccountantName() => "RDP";
}

