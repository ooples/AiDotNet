namespace AiDotNet.FederatedLearning.Fairness;

/// <summary>
/// Implements FedFair — multi-objective optimization balancing accuracy, fairness, and efficiency.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Real FL deployments must balance multiple goals: high accuracy
/// (model quality), fairness (no client left behind), and efficiency (fast convergence, low
/// communication). FedFair treats these as a multi-objective optimization problem and finds
/// Pareto-optimal aggregation weights that don't sacrifice one goal unnecessarily for another.
/// The user specifies preference weights for each objective.</para>
///
/// <para>Objectives:</para>
/// <list type="bullet">
/// <item><b>Accuracy</b>: minimize average loss (standard FL)</item>
/// <item><b>Fairness</b>: minimize variance of losses across clients</item>
/// <item><b>Efficiency</b>: prefer clients with faster convergence</item>
/// </list>
///
/// <para>Scalarization: w = alpha_acc * w_acc + alpha_fair * w_fair + alpha_eff * w_eff</para>
///
/// <para>Reference: FedFair: Multi-Objective Federated Learning (2024).</para>
/// </remarks>
/// <typeparam name="T">The numeric type for model parameters.</typeparam>
public class FedFairOptimizer<T> : Infrastructure.FederatedLearningComponentBase<T>
{
    private readonly double _accuracyWeight;
    private readonly double _fairnessWeight;
    private readonly double _efficiencyWeight;

    /// <summary>
    /// Creates a new FedFair optimizer.
    /// </summary>
    /// <param name="accuracyWeight">Preference weight for accuracy. Default: 0.5.</param>
    /// <param name="fairnessWeight">Preference weight for fairness. Default: 0.3.</param>
    /// <param name="efficiencyWeight">Preference weight for efficiency. Default: 0.2.</param>
    public FedFairOptimizer(
        double accuracyWeight = 0.5,
        double fairnessWeight = 0.3,
        double efficiencyWeight = 0.2)
    {
        if (accuracyWeight < 0 || fairnessWeight < 0 || efficiencyWeight < 0)
        {
            throw new ArgumentException("All preference weights must be non-negative.");
        }

        double total = accuracyWeight + fairnessWeight + efficiencyWeight;
        _accuracyWeight = accuracyWeight / total;
        _fairnessWeight = fairnessWeight / total;
        _efficiencyWeight = efficiencyWeight / total;
    }

    /// <summary>
    /// Computes multi-objective aggregation weights.
    /// </summary>
    /// <param name="clientLosses">Current loss for each client.</param>
    /// <param name="clientSampleCounts">Number of training samples per client.</param>
    /// <param name="clientLatencies">Communication latency per client (lower = more efficient).</param>
    /// <returns>Pareto-balanced aggregation weights.</returns>
    public Dictionary<int, double> ComputeWeights(
        Dictionary<int, double> clientLosses,
        Dictionary<int, int> clientSampleCounts,
        Dictionary<int, double>? clientLatencies = null)
    {
        if (clientLosses.Count == 0)
        {
            throw new ArgumentException("Client losses cannot be empty.", nameof(clientLosses));
        }

        var clientIds = clientLosses.Keys.ToList();
        int n = clientIds.Count;

        // Accuracy weights: proportional to sample count (standard FedAvg).
        double totalSamples = clientSampleCounts.Values.Sum();
        var accWeights = clientIds.ToDictionary(
            id => id,
            id => totalSamples > 0 ? clientSampleCounts.GetValueOrDefault(id, 1) / totalSamples : 1.0 / n);

        // Fairness weights: higher weight for higher-loss clients (TERM-inspired).
        double maxLoss = clientLosses.Values.Max();
        double fairTotal = 0;
        var fairWeights = new Dictionary<int, double>();
        foreach (var id in clientIds)
        {
            double w = Math.Exp(clientLosses[id] - maxLoss);
            fairWeights[id] = w;
            fairTotal += w;
        }

        foreach (var id in clientIds)
        {
            fairWeights[id] = fairTotal > 0 ? fairWeights[id] / fairTotal : 1.0 / n;
        }

        // Efficiency weights: inversely proportional to latency.
        var effWeights = new Dictionary<int, double>();
        if (clientLatencies != null && clientLatencies.Count > 0)
        {
            double effTotal = 0;
            foreach (var id in clientIds)
            {
                double latency = clientLatencies.GetValueOrDefault(id, 1.0);
                double w = 1.0 / Math.Max(latency, 1e-6);
                effWeights[id] = w;
                effTotal += w;
            }

            foreach (var id in clientIds)
            {
                effWeights[id] = effTotal > 0 ? effWeights[id] / effTotal : 1.0 / n;
            }
        }
        else
        {
            foreach (var id in clientIds)
            {
                effWeights[id] = 1.0 / n;
            }
        }

        // Scalarize: weighted combination of the three objective weights.
        var finalWeights = new Dictionary<int, double>();
        double finalTotal = 0;
        foreach (var id in clientIds)
        {
            double w = _accuracyWeight * accWeights[id]
                     + _fairnessWeight * fairWeights[id]
                     + _efficiencyWeight * effWeights[id];
            finalWeights[id] = w;
            finalTotal += w;
        }

        foreach (var id in clientIds)
        {
            finalWeights[id] = finalTotal > 0 ? finalWeights[id] / finalTotal : 1.0 / n;
        }

        return finalWeights;
    }

    /// <summary>Gets the accuracy preference weight.</summary>
    public double AccuracyWeight => _accuracyWeight;

    /// <summary>Gets the fairness preference weight.</summary>
    public double FairnessWeight => _fairnessWeight;

    /// <summary>Gets the efficiency preference weight.</summary>
    public double EfficiencyWeight => _efficiencyWeight;
}
