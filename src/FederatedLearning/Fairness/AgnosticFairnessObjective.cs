namespace AiDotNet.FederatedLearning.Fairness;

/// <summary>
/// Implements AFL (Agnostic Federated Learning) — minimax fairness optimization.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Standard FL minimizes the average loss across all clients.
/// This can leave some clients with terrible performance (e.g., a hospital with rare diseases).
/// AFL instead optimizes for the worst-performing client — it's "agnostic" to which client
/// distribution the model will be tested on. This ensures no client is left behind, at the
/// cost of slightly lower average performance.</para>
///
/// <para>Objective:</para>
/// <code>
/// min_w max_lambda sum(lambda_k * L_k(w))
/// s.t. lambda in simplex (lambda_k >= 0, sum = 1)
/// </code>
/// <para>The lambda weights dynamically increase for clients with high loss.</para>
///
/// <para>Reference: Mohri, M., et al. (2019). "Agnostic Federated Learning." ICML 2019.</para>
/// </remarks>
/// <typeparam name="T">The numeric type for model parameters.</typeparam>
public class AgnosticFairnessObjective<T> : Infrastructure.FederatedLearningComponentBase<T>
{
    private readonly double _lambdaLearningRate;
    private Dictionary<int, double>? _lambdas;

    /// <summary>
    /// Creates a new AFL objective.
    /// </summary>
    /// <param name="lambdaLearningRate">Learning rate for lambda updates. Default: 0.1.</param>
    public AgnosticFairnessObjective(double lambdaLearningRate = 0.1)
    {
        if (lambdaLearningRate <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(lambdaLearningRate), "Learning rate must be positive.");
        }

        _lambdaLearningRate = lambdaLearningRate;
    }

    /// <summary>
    /// Computes AFL aggregation weights based on client losses.
    /// </summary>
    /// <param name="clientLosses">Current loss for each client.</param>
    /// <returns>Fairness-adjusted aggregation weights.</returns>
    public Dictionary<int, double> ComputeWeights(Dictionary<int, double> clientLosses)
    {
        if (clientLosses.Count == 0)
        {
            throw new ArgumentException("Client losses cannot be empty.", nameof(clientLosses));
        }

        // Initialize lambdas uniformly.
        _lambdas ??= clientLosses.ToDictionary(kvp => kvp.Key, _ => 1.0 / clientLosses.Count);

        // Update lambdas: increase for clients with higher loss.
        foreach (var (clientId, loss) in clientLosses)
        {
            if (_lambdas.ContainsKey(clientId))
            {
                _lambdas[clientId] *= Math.Exp(_lambdaLearningRate * loss);
            }
            else
            {
                _lambdas[clientId] = 1.0;
            }
        }

        // Project back to simplex (normalize).
        double total = _lambdas.Values.Sum();
        var weights = new Dictionary<int, double>();
        foreach (var (clientId, lambda) in _lambdas)
        {
            weights[clientId] = total > 0 ? lambda / total : 1.0 / _lambdas.Count;
        }

        return weights;
    }

    /// <summary>Gets the lambda learning rate.</summary>
    public double LambdaLearningRate => _lambdaLearningRate;
}
