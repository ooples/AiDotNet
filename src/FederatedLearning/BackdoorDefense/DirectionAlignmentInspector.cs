namespace AiDotNet.FederatedLearning.BackdoorDefense;

/// <summary>
/// Direction Alignment Inspector — detects backdoor attacks via gradient direction analysis.
/// </summary>
/// <remarks>
/// <para>
/// This detector identifies backdoor attacks by analyzing the alignment between each client's
/// update direction and the "honest" consensus direction. Backdoor updates tend to have
/// anomalous gradient directions that point away from the clean learning direction, particularly
/// in specific parameter subspaces corresponding to the trigger-target mapping.
/// </para>
/// <para>
/// <b>For Beginners:</b> In normal training, all clients push the model in roughly the same
/// direction. A backdoor attacker pushes in a different direction for certain parts of the model
/// (the parts that encode the trigger). By checking whether each client's "push direction"
/// aligns with the majority, we can identify suspicious clients.
/// </para>
/// <para>
/// Reference: Xu et al. (2025), "Detecting Backdoor Attacks in Federated Learning via Direction
/// Alignment Inspection" (CVPR 2025).
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class DirectionAlignmentInspector<T> : Infrastructure.FederatedLearningComponentBase<T>, IBackdoorDetector<T>
{
    private readonly int _numSubspaces;
    private readonly double _alignmentThreshold;

    /// <inheritdoc/>
    public string DetectorName => "DirectionAlignmentInspector";

    /// <summary>
    /// Creates a new Direction Alignment Inspector.
    /// </summary>
    /// <param name="numSubspaces">Number of parameter subspaces to analyze. Default: 10.</param>
    /// <param name="alignmentThreshold">Cosine similarity threshold below which an update is suspicious. Default: 0.3.</param>
    public DirectionAlignmentInspector(int numSubspaces = 10, double alignmentThreshold = 0.3)
    {
        _numSubspaces = numSubspaces;
        _alignmentThreshold = alignmentThreshold;
    }

    /// <inheritdoc/>
    public Dictionary<int, double> DetectSuspiciousUpdates(Dictionary<int, Vector<T>> clientUpdates, Vector<T> globalModel)
    {
        if (clientUpdates.Count < 2)
            return clientUpdates.ToDictionary(kv => kv.Key, _ => 0.0);

        int d = clientUpdates.Values.First().Length;
        int subspaceSize = Math.Max(1, d / _numSubspaces);

        // Step 1: Compute honest consensus direction (median of all updates)
        var consensusDirection = ComputeConsensusDirection(clientUpdates, d);

        // Step 2: For each client, compute per-subspace alignment scores
        var suspicionScores = new Dictionary<int, double>();

        foreach (var (clientId, update) in clientUpdates)
        {
            double totalSuspicion = 0;
            int numChecked = 0;

            for (int s = 0; s < _numSubspaces; s++)
            {
                int start = s * subspaceSize;
                int end = Math.Min(start + subspaceSize, d);
                if (start >= d) break;

                // Cosine similarity in this subspace
                double dotProduct = 0, normA = 0, normB = 0;
                for (int i = start; i < end; i++)
                {
                    double a = NumOps.ToDouble(update[i]);
                    double b = consensusDirection[i];
                    dotProduct += a * b;
                    normA += a * a;
                    normB += b * b;
                }

                double cosineSim = (normA > 1e-10 && normB > 1e-10)
                    ? dotProduct / (Math.Sqrt(normA) * Math.Sqrt(normB))
                    : 0;

                // Low alignment in any subspace increases suspicion
                if (cosineSim < _alignmentThreshold)
                {
                    totalSuspicion += 1.0 - cosineSim;
                }
                numChecked++;
            }

            // Normalize suspicion to [0, 1]
            suspicionScores[clientId] = numChecked > 0 ? Math.Min(1.0, totalSuspicion / numChecked) : 0;
        }

        return suspicionScores;
    }

    /// <inheritdoc/>
    public Dictionary<int, Vector<T>> FilterMaliciousUpdates(Dictionary<int, Vector<T>> clientUpdates,
        Vector<T> globalModel, double suspicionThreshold)
    {
        var scores = DetectSuspiciousUpdates(clientUpdates, globalModel);
        var filtered = new Dictionary<int, Vector<T>>();

        foreach (var (clientId, update) in clientUpdates)
        {
            if (scores.GetValueOrDefault(clientId, 0) < suspicionThreshold)
            {
                filtered[clientId] = update;
            }
        }

        // If all clients filtered out, keep the least suspicious one
        if (filtered.Count == 0 && clientUpdates.Count > 0)
        {
            var leastSuspicious = scores.OrderBy(kv => kv.Value).First().Key;
            filtered[leastSuspicious] = clientUpdates[leastSuspicious];
        }

        return filtered;
    }

    private double[] ComputeConsensusDirection(Dictionary<int, Vector<T>> clientUpdates, int d)
    {
        // Coordinate-wise median as robust consensus
        var allValues = new double[clientUpdates.Count][];
        int idx = 0;
        foreach (var update in clientUpdates.Values)
        {
            allValues[idx] = new double[d];
            for (int i = 0; i < d; i++)
            {
                allValues[idx][i] = NumOps.ToDouble(update[i]);
            }
            idx++;
        }

        var consensus = new double[d];
        for (int i = 0; i < d; i++)
        {
            var values = new double[clientUpdates.Count];
            for (int j = 0; j < clientUpdates.Count; j++)
            {
                values[j] = allValues[j][i];
            }
            Array.Sort(values);
            consensus[i] = values[values.Length / 2]; // Median
        }

        return consensus;
    }
}
