namespace AiDotNet.FederatedLearning.Decentralized;

/// <summary>
/// Implements DeTAG — Decentralized gradient Tracking for exact convergence.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Basic decentralized averaging has a problem: it converges to
/// a "consensus" point that may not be the true global minimum because each client only sees
/// their local gradient. DeTAG (Decentralized gradient Tracking) fixes this by having each
/// client track the difference between their local gradient and the global gradient estimate.
/// This correction term ensures exact convergence even with heterogeneous data.</para>
///
/// <para>Gradient tracking update:</para>
/// <code>
/// y_k = sum(W_kj * y_j) + grad_new_k - grad_old_k  // track gradient change
/// x_k = sum(W_kj * x_j) - lr * y_k                  // parameter update
/// </code>
///
/// <para>Reference: Li, H., et al. (2023). "DeTAG: Decentralized Tracking-based
/// Asynchronous Gradient methods." 2023.</para>
/// </remarks>
/// <typeparam name="T">The numeric type for model parameters.</typeparam>
public class DeTAGProtocol<T> : Infrastructure.FederatedLearningComponentBase<T>
{
    private readonly double _learningRate;
    private Dictionary<int, double[]>? _gradientTrackers;

    /// <summary>
    /// Creates a new DeTAG protocol.
    /// </summary>
    /// <param name="learningRate">Step size. Default: 0.01.</param>
    public DeTAGProtocol(double learningRate = 0.01)
    {
        if (learningRate <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(learningRate), "Learning rate must be positive.");
        }

        _learningRate = learningRate;
    }

    /// <summary>
    /// Performs one DeTAG update step for a client, including neighbor-averaged gradient tracking.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Each client maintains a "gradient tracker" that estimates
    /// the global gradient. The tracker is updated by: (1) averaging with neighbors' trackers
    /// (consensus step), then (2) correcting with the local gradient change (tracking step).
    /// This two-step process ensures that the tracker converges to the true global gradient,
    /// enabling exact convergence even with decentralized heterogeneous data.</para>
    ///
    /// <para>Update rules:</para>
    /// <code>
    /// y_k = sum(W_kj * y_j) + grad_new_k - grad_old_k  // tracker: consensus + correction
    /// x_k = sum(W_kj * x_j) - lr * y_k                  // params: consensus - lr * tracker
    /// </code>
    /// </remarks>
    /// <param name="clientId">This client's ID.</param>
    /// <param name="currentParams">Current model parameters (flattened).</param>
    /// <param name="newGradient">Gradient computed on current data.</param>
    /// <param name="previousGradient">Gradient from previous round.</param>
    /// <param name="neighborParams">Neighboring clients' parameters.</param>
    /// <param name="neighborTrackers">Neighboring clients' gradient trackers (from their last step).</param>
    /// <param name="mixingWeights">Mixing weights for neighbors (doubly-stochastic matrix row).</param>
    /// <returns>Updated parameters.</returns>
    public T[] Step(
        int clientId,
        T[] currentParams,
        T[] newGradient,
        T[] previousGradient,
        Dictionary<int, T[]> neighborParams,
        Dictionary<int, double[]>? neighborTrackers,
        Dictionary<int, double> mixingWeights)
    {
        int d = currentParams.Length;
        _gradientTrackers ??= new Dictionary<int, double[]>();

        if (!_gradientTrackers.ContainsKey(clientId))
        {
            _gradientTrackers[clientId] = new double[d];
            for (int i = 0; i < d; i++)
            {
                _gradientTrackers[clientId][i] = NumOps.ToDouble(newGradient[i]);
            }
        }

        var tracker = _gradientTrackers[clientId];

        // Step 1: Gradient tracker update — y_k = sum(W_kj * y_j) + grad_new - grad_old.
        // First compute the consensus average of neighbor trackers.
        double totalWeight = 0;
        foreach (var w in mixingWeights.Values)
        {
            totalWeight += w;
        }

        var newTracker = new double[d];

        if (neighborTrackers != null && neighborTrackers.Count > 0)
        {
            // Average neighbors' gradient trackers: sum(W_kj * y_j).
            for (int i = 0; i < d; i++)
            {
                double trackerAvg = 0;
                double trackerWeightSum = 0;

                foreach (var (neighborId, w) in mixingWeights)
                {
                    if (neighborTrackers.TryGetValue(neighborId, out var nt) && i < nt.Length)
                    {
                        trackerAvg += (w / totalWeight) * nt[i];
                        trackerWeightSum += w;
                    }
                }

                // Include self-weight for the consensus (if not in neighbor list).
                double selfWeight = 1.0 - (trackerWeightSum / totalWeight);
                trackerAvg += selfWeight * tracker[i];

                // Add gradient correction: + grad_new - grad_old.
                newTracker[i] = trackerAvg + NumOps.ToDouble(newGradient[i]) - NumOps.ToDouble(previousGradient[i]);
            }
        }
        else
        {
            // Fallback: no neighbor trackers available, use local correction only.
            for (int i = 0; i < d; i++)
            {
                newTracker[i] = tracker[i] + NumOps.ToDouble(newGradient[i]) - NumOps.ToDouble(previousGradient[i]);
            }
        }

        // Store updated tracker.
        _gradientTrackers[clientId] = newTracker;

        // Step 2: Parameter update — x_k = sum(W_kj * x_j) - lr * y_k.
        var result = new T[d];
        for (int i = 0; i < d; i++)
        {
            double paramAvg = 0;
            foreach (var (neighborId, w) in mixingWeights)
            {
                if (neighborParams.TryGetValue(neighborId, out var np) && i < np.Length)
                {
                    paramAvg += (w / totalWeight) * NumOps.ToDouble(np[i]);
                }
            }

            // Include self-weight for consensus.
            double selfW = 0;
            foreach (var w in mixingWeights.Values)
            {
                selfW += w;
            }

            selfW = 1.0 - (selfW / totalWeight);
            paramAvg += selfW * NumOps.ToDouble(currentParams[i]);

            result[i] = NumOps.FromDouble(paramAvg - _learningRate * newTracker[i]);
        }

        return result;
    }

    /// <summary>
    /// Gets the current gradient tracker for a client (for sharing with neighbors).
    /// </summary>
    /// <param name="clientId">The client ID.</param>
    /// <returns>The gradient tracker array, or null if not initialized.</returns>
    public double[]? GetTracker(int clientId)
    {
        if (_gradientTrackers != null && _gradientTrackers.TryGetValue(clientId, out var tracker))
        {
            return (double[])tracker.Clone();
        }

        return null;
    }

    /// <summary>Gets the learning rate.</summary>
    public double LearningRate => _learningRate;
}
