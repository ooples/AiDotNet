namespace AiDotNet.FederatedLearning.Fairness;

/// <summary>
/// Implements q-FFL (q-Fair Federated Learning) — parameterized fairness via power-mean.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> q-FFL provides a tunable knob for fairness. The parameter q
/// controls how much we care about the worst-off clients: q=0 gives standard FedAvg (optimize
/// average loss), q=1 weights clients proportionally to their loss, and q→∞ gives minimax
/// fairness (optimize worst-case). This lets you smoothly trade off average performance for
/// fairness.</para>
///
/// <para>Objective:</para>
/// <code>
/// min_w (1/(q+1)) * sum(L_k(w)^(q+1))
/// Aggregation weight: w_k = L_k(w)^q / sum(L_j(w)^q)
/// </code>
///
/// <para>Reference: Li, T., et al. (2020). "Fair Resource Allocation in Federated Learning."
/// ICLR 2020.</para>
/// </remarks>
/// <typeparam name="T">The numeric type for model parameters.</typeparam>
public class QFairFederatedLearning<T> : Infrastructure.FederatedLearningComponentBase<T>
{
    private readonly double _q;

    /// <summary>
    /// Creates a new q-FFL instance.
    /// </summary>
    /// <param name="q">Fairness parameter. q=0: FedAvg, q→∞: minimax. Default: 1.0.</param>
    public QFairFederatedLearning(double q = 1.0)
    {
        if (q < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(q), "q must be non-negative.");
        }

        _q = q;
    }

    /// <summary>
    /// Computes q-fair aggregation weights based on client losses.
    /// </summary>
    /// <param name="clientLosses">Current loss for each client.</param>
    /// <returns>Fairness-adjusted weights proportional to L^q.</returns>
    public Dictionary<int, double> ComputeWeights(Dictionary<int, double> clientLosses)
    {
        Guard.NotNull(clientLosses);
        if (clientLosses.Count == 0)
        {
            throw new ArgumentException("Client losses cannot be empty.", nameof(clientLosses));
        }

        var weights = new Dictionary<int, double>();
        double total = 0;

        foreach (var (clientId, loss) in clientLosses)
        {
            if (loss < 0 || double.IsNaN(loss) || double.IsInfinity(loss))
            {
                throw new ArgumentException($"Client {clientId} has invalid loss: {loss}. Must be non-negative and finite.");
            }

            double w = Math.Pow(Math.Max(loss, 1e-10), _q);
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
    /// Computes the q-fair objective value.
    /// </summary>
    /// <param name="clientLosses">Client losses.</param>
    /// <returns>q-fair objective value.</returns>
    public double ComputeObjective(Dictionary<int, double> clientLosses)
    {
        Guard.NotNull(clientLosses);
        if (clientLosses.Count == 0)
        {
            throw new ArgumentException("Client losses cannot be empty.", nameof(clientLosses));
        }

        double sum = 0;
        foreach (var loss in clientLosses.Values)
        {
            sum += Math.Pow(Math.Max(loss, 1e-10), _q + 1);
        }

        return sum / (_q + 1);
    }

    /// <summary>Gets the fairness parameter q.</summary>
    public double Q => _q;
}
