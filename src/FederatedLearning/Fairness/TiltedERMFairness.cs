namespace AiDotNet.FederatedLearning.Fairness;

/// <summary>
/// Implements TERM (Tilted Empirical Risk Minimization) for fairness in FL.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> TERM smoothly interpolates between average and worst-case
/// optimization using a tilt parameter t. When t=0, it's standard average loss. When t>0,
/// it up-weights high-loss clients (moving toward worst-case). When t&lt;0, it focuses on
/// easy clients (useful for outlier robustness). This gives a smooth, differentiable fairness
/// objective that's easier to optimize than the hard minimax of AFL.</para>
///
/// <para>Objective:</para>
/// <code>
/// TERM_t(w) = (1/t) * log((1/n) * sum(exp(t * L_k(w))))
/// t > 0: focus on high-loss clients (fairness)
/// t = 0: standard ERM (average loss)
/// t &lt; 0: focus on low-loss clients (robustness to outliers)
/// </code>
///
/// <para>Reference: Li, T., et al. (2021). "Tilted Empirical Risk Minimization." ICLR 2021.</para>
/// </remarks>
/// <typeparam name="T">The numeric type for model parameters.</typeparam>
public class TiltedERMFairness<T> : Infrastructure.FederatedLearningComponentBase<T>
{
    private readonly double _tilt;

    /// <summary>
    /// Creates a new TERM fairness instance.
    /// </summary>
    /// <param name="tilt">Tilt parameter. Positive = fairness, negative = robustness. Default: 1.0.</param>
    public TiltedERMFairness(double tilt = 1.0)
    {
        _tilt = tilt;
    }

    /// <summary>
    /// Computes TERM aggregation weights based on client losses.
    /// </summary>
    /// <param name="clientLosses">Current loss for each client.</param>
    /// <returns>TERM-adjusted weights.</returns>
    public Dictionary<int, double> ComputeWeights(Dictionary<int, double> clientLosses)
    {
        if (clientLosses.Count == 0)
        {
            throw new ArgumentException("Client losses cannot be empty.", nameof(clientLosses));
        }

        if (Math.Abs(_tilt) < 1e-10)
        {
            // t ≈ 0: uniform weights (standard ERM).
            return clientLosses.ToDictionary(kvp => kvp.Key, _ => 1.0 / clientLosses.Count);
        }

        // Softmax with temperature = 1/t.
        double maxLoss = clientLosses.Values.Max();
        var weights = new Dictionary<int, double>();
        double total = 0;

        foreach (var (clientId, loss) in clientLosses)
        {
            double w = Math.Exp(_tilt * (loss - maxLoss));
            weights[clientId] = w;
            total += w;
        }

        if (total > 0)
        {
            foreach (var key in weights.Keys.ToArray())
            {
                weights[key] /= total;
            }
        }

        return weights;
    }

    /// <summary>
    /// Computes the TERM objective value.
    /// </summary>
    /// <param name="clientLosses">Client losses.</param>
    /// <returns>TERM objective value.</returns>
    public double ComputeObjective(Dictionary<int, double> clientLosses)
    {
        if (Math.Abs(_tilt) < 1e-10)
        {
            return clientLosses.Values.Average();
        }

        double maxLoss = clientLosses.Values.Max();
        double sum = 0;
        foreach (var loss in clientLosses.Values)
        {
            sum += Math.Exp(_tilt * (loss - maxLoss));
        }

        return maxLoss + (1.0 / _tilt) * Math.Log(sum / clientLosses.Count);
    }

    /// <summary>Gets the tilt parameter.</summary>
    public double Tilt => _tilt;
}
