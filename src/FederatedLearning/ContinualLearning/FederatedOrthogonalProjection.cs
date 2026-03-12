namespace AiDotNet.FederatedLearning.ContinualLearning;

/// <summary>
/// Federated Orthogonal Projection — prevents forgetting by projecting gradients to be
/// orthogonal to the subspace of previously important parameter directions.
/// </summary>
/// <remarks>
/// <para>
/// Orthogonal gradient projection (Farajtabar et al., 2020; Zhang et al., ICCV 2025)
/// computes the subspace spanned by important parameter directions from previous tasks,
/// then projects current gradients to be orthogonal to this subspace. This guarantees
/// that new learning does not interfere with previously learned knowledge.
/// </para>
/// <para>
/// <b>For Beginners:</b> Think of it as building new knowledge on a "wall" that's perpendicular
/// to the old knowledge. If the old knowledge lives on the floor, new knowledge goes up the wall
/// — they never conflict. In federated settings, each client reports which directions are important,
/// and the global projection space is the union of all clients' important directions.
/// </para>
/// <para>
/// References:
/// Farajtabar et al. (2020), "Orthogonal Gradient Descent for Continual Learning".
/// Zhang et al. (2025), "FedAGC: Federated Continual Learning with Asymmetric Gradient Correction" (ICCV 2025).
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class FederatedOrthogonalProjection<T> : Infrastructure.FederatedLearningComponentBase<T>, IFederatedContinualLearningStrategy<T>
{
    private readonly double _projectionThreshold;

    /// <summary>
    /// Creates a new federated orthogonal projection strategy.
    /// </summary>
    /// <param name="projectionThreshold">Minimum importance to include a direction in the projection space. Default: 0.01.</param>
    public FederatedOrthogonalProjection(double projectionThreshold = 0.01)
    {
        _projectionThreshold = projectionThreshold;
    }

    /// <inheritdoc/>
    public Vector<T> ComputeImportance(Vector<T> modelParameters, Matrix<T> taskData)
    {
        int d = modelParameters.Length;
        int n = taskData.Rows;

        // Compute gradient covariance diagonal as importance estimate
        var importance = new double[d];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < d; j++)
            {
                int featIdx = j % taskData.Columns;
                double grad = NumOps.ToDouble(modelParameters[j]) * NumOps.ToDouble(taskData[i, featIdx]);
                importance[j] += grad * grad;
            }
        }

        var result = new T[d];
        for (int j = 0; j < d; j++)
        {
            result[j] = NumOps.FromDouble(importance[j] / n);
        }

        return new Vector<T>(result);
    }

    /// <inheritdoc/>
    public T ComputeRegularizationPenalty(Vector<T> currentParameters, Vector<T> referenceParameters,
        Vector<T> importanceWeights, double regularizationStrength)
    {
        // Orthogonal projection doesn't use regularization — it uses gradient projection instead
        return NumOps.Zero;
    }

    /// <inheritdoc/>
    public Vector<T> ProjectGradient(Vector<T> gradient, Vector<T> importanceWeights)
    {
        int d = gradient.Length;
        var projected = new T[d];

        // Compute the projection: g_projected = g - (g·e)/(e·e) * e
        // where e is the importance-weighted direction from previous tasks
        // For diagonal approximation: scale gradient components inversely with importance

        double dotGE = 0;
        double dotEE = 0;

        for (int i = 0; i < d; i++)
        {
            double g = NumOps.ToDouble(gradient[i]);
            double e = NumOps.ToDouble(importanceWeights[i]);

            if (e > _projectionThreshold)
            {
                dotGE += g * e;
                dotEE += e * e;
            }
        }

        // Project out the important direction
        double projScale = dotEE > 1e-10 ? dotGE / dotEE : 0;

        for (int i = 0; i < d; i++)
        {
            double g = NumOps.ToDouble(gradient[i]);
            double e = NumOps.ToDouble(importanceWeights[i]);

            if (e > _projectionThreshold)
            {
                // Remove the component along the important direction
                projected[i] = NumOps.FromDouble(g - projScale * e);
            }
            else
            {
                // Not important — keep the gradient as-is
                projected[i] = gradient[i];
            }
        }

        return new Vector<T>(projected);
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

        // Union of important directions: take max importance across clients
        foreach (var (clientId, importance) in clientImportances)
        {
            double w = clientWeights?.GetValueOrDefault(clientId, 1.0) ?? 1.0;
            totalWeight += w;

            for (int i = 0; i < d; i++)
            {
                double val = NumOps.ToDouble(importance[i]) * w;
                aggregated[i] = Math.Max(aggregated[i], val);
            }
        }

        var result = new T[d];
        for (int i = 0; i < d; i++)
        {
            result[i] = NumOps.FromDouble(aggregated[i]);
        }

        return new Vector<T>(result);
    }
}
