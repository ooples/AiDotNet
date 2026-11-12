using AiDotNet.Attributes;
using AiDotNet.Enums;

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
[ComponentType(ComponentType.FederatedAggregator)]
[PipelineStage(PipelineStage.Training)]
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

        // Step 1: Compute honest consensus direction (coordinate-wise median of all updates)
        var clientIds = clientUpdates.Keys.ToArray();
        int n = clientIds.Length;
        var stacked = Engine.TensorStack(clientUpdates.Values.Select(u => u.AsTensor()).ToArray(), 0);
        var consensusDirection = ComputeConsensusDirection(stacked, n, d);

        // Step 2: Per-subspace cosine alignment of every client with the consensus, batched as [n, S, size].
        // Every analyzed subspace has exactly subspaceSize coordinates.
        int numChecked = Math.Min(_numSubspaces, d / subspaceSize);
        int covered = numChecked * subspaceSize;
        var updates = Engine.TensorSlice(stacked, new[] { 0, 0 }, new[] { n, covered })
            .Reshape(n, numChecked, subspaceSize);
        var consensus = Engine.TensorBroadcastTo(
            Engine.TensorSlice(consensusDirection, new[] { 0 }, new[] { covered }).Reshape(1, numChecked, subspaceSize),
            new[] { n, numChecked, subspaceSize });

        var axes = new[] { 2 };
        var dots = Engine.ReduceSum(Engine.TensorMultiply(updates, consensus), axes, false);
        var normsA = Engine.ReduceSum(Engine.TensorSquare(updates), axes, false);
        var normsB = Engine.ReduceSum(Engine.TensorSquare(consensus), axes, false);

        // cos = dot / (||a|| ||b||) when both squared norms exceed 1e-10, otherwise 0.
        var eps = NumOps.FromDouble(1e-10);
        var zeros = new Tensor<T>(new[] { n, numChecked });
        var valid = Engine.TensorMultiply(Engine.TensorGreaterThan(normsA, eps), Engine.TensorGreaterThan(normsB, eps));
        var cosine = Engine.TensorWhere(
            valid,
            Engine.TensorDivide(dots, Engine.TensorSqrt(Engine.TensorMultiply(normsA, normsB))),
            zeros);

        // Low alignment in any subspace increases suspicion by (1 - cos).
        var lowAlignment = Engine.TensorGreaterThan(Engine.TensorNegate(cosine), NumOps.FromDouble(-_alignmentThreshold));
        var suspicion = Engine.TensorWhere(lowAlignment, Engine.ScalarMinusTensor(NumOps.One, cosine), zeros);
        var totals = Engine.ReduceSum(suspicion, new[] { 1 }, false).ToArray();

        // Normalize suspicion to [0, 1]
        var suspicionScores = new Dictionary<int, double>();
        for (int k = 0; k < n; k++)
        {
            suspicionScores[clientIds[k]] = numChecked > 0 ? Math.Min(1.0, NumOps.ToDouble(totals[k]) / numChecked) : 0;
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

    /// <summary>
    /// Coordinate-wise median of the stacked client updates ([n, d]); for an even client count the
    /// median is the mean of the two middle order statistics.
    /// </summary>
    private Tensor<T> ComputeConsensusDirection(Tensor<T> stacked, int n, int d)
    {
        var sorted = Engine.TensorSort(stacked, 0, false).Values;
        int lo = (n - 1) / 2;
        int count = n % 2 == 0 ? 2 : 1;
        return Engine.ReduceMean(Engine.TensorSlice(sorted, new[] { lo, 0 }, new[] { count, d }), new[] { 0 }, false);
    }
}
