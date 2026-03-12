namespace AiDotNet.FederatedLearning.ContinualLearning;

/// <summary>
/// Federated Elastic Weight Consolidation (EWC) — prevents forgetting by penalizing changes
/// to parameters that are important for previously learned tasks.
/// </summary>
/// <remarks>
/// <para>
/// EWC (Kirkpatrick et al., 2017) uses the Fisher information matrix to estimate which
/// parameters are important for each task. In federated EWC, the Fisher information is
/// computed locally at each client and aggregated at the server to form a global estimate
/// of parameter importance across all tasks and clients.
/// </para>
/// <para>
/// <b>For Beginners:</b> Some model parameters are critical for recognizing cats, while
/// others are critical for recognizing dogs. EWC identifies which parameters matter for which
/// tasks and penalizes changing the important ones. In federated EWC, each client reports
/// which parameters are important for their data, and the server keeps a global importance map.
/// </para>
/// <para>
/// Reference: Kirkpatrick et al. (2017), "Overcoming Catastrophic Forgetting in Neural Networks".
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class FederatedEWC<T> : Infrastructure.FederatedLearningComponentBase<T>, IFederatedContinualLearningStrategy<T>
{
    private readonly int _fisherSamples;

    /// <summary>
    /// Creates a new Federated EWC strategy.
    /// </summary>
    /// <param name="fisherSamples">Number of data samples to estimate Fisher information. Default: 200.</param>
    public FederatedEWC(int fisherSamples = 200)
    {
        _fisherSamples = fisherSamples;
    }

    /// <inheritdoc/>
    public Vector<T> ComputeImportance(Vector<T> modelParameters, Matrix<T> taskData)
    {
        int d = modelParameters.Length;
        int n = Math.Min(_fisherSamples, taskData.Rows);

        // Approximate Fisher information: average squared gradient per parameter
        var fisher = new double[d];

        for (int i = 0; i < n; i++)
        {
            // Compute pseudo-gradient: how much each parameter contributes to prediction
            for (int j = 0; j < d; j++)
            {
                double paramVal = NumOps.ToDouble(modelParameters[j]);
                double dataContrib = 0;

                // Cross-correlate parameter with data features
                int featIdx = j % taskData.Columns;
                double dataVal = NumOps.ToDouble(taskData[i, featIdx]);
                dataContrib = paramVal * dataVal;

                // Squared gradient approximation
                fisher[j] += dataContrib * dataContrib;
            }
        }

        // Normalize
        var importance = new T[d];
        for (int j = 0; j < d; j++)
        {
            importance[j] = NumOps.FromDouble(fisher[j] / n);
        }

        return new Vector<T>(importance);
    }

    /// <inheritdoc/>
    public T ComputeRegularizationPenalty(Vector<T> currentParameters, Vector<T> referenceParameters,
        Vector<T> importanceWeights, double regularizationStrength)
    {
        int d = currentParameters.Length;
        double penalty = 0;

        for (int i = 0; i < d; i++)
        {
            double diff = NumOps.ToDouble(currentParameters[i]) - NumOps.ToDouble(referenceParameters[i]);
            double importance = NumOps.ToDouble(importanceWeights[i]);
            penalty += importance * diff * diff;
        }

        return NumOps.FromDouble(0.5 * regularizationStrength * penalty);
    }

    /// <inheritdoc/>
    public Vector<T> ProjectGradient(Vector<T> gradient, Vector<T> importanceWeights)
    {
        // EWC doesn't project gradients; it uses regularization instead
        // Return gradient unchanged — the penalty handles forgetting prevention
        return gradient;
    }

    /// <inheritdoc/>
    public Vector<T> AggregateImportance(Dictionary<int, Vector<T>> clientImportances,
        Dictionary<int, double>? clientWeights)
    {
        if (clientImportances.Count == 0)
            throw new ArgumentException("No client importances provided.", nameof(clientImportances));

        int d = clientImportances.Values.First().Length;
        var aggregated = new double[d];
        double totalWeight = 0;

        foreach (var (clientId, importance) in clientImportances)
        {
            double w = clientWeights?.GetValueOrDefault(clientId, 1.0) ?? 1.0;
            totalWeight += w;

            for (int i = 0; i < d; i++)
            {
                aggregated[i] += NumOps.ToDouble(importance[i]) * w;
            }
        }

        var result = new T[d];
        for (int i = 0; i < d; i++)
        {
            result[i] = NumOps.FromDouble(aggregated[i] / totalWeight);
        }

        return new Vector<T>(result);
    }
}
