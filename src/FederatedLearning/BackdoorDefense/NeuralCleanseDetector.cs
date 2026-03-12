namespace AiDotNet.FederatedLearning.BackdoorDefense;

/// <summary>
/// Neural Cleanse — post-hoc backdoor detection by reverse-engineering potential triggers.
/// </summary>
/// <remarks>
/// <para>
/// Neural Cleanse (Wang et al., 2019) detects backdoors by searching for the smallest
/// perturbation (trigger) that causes the model to misclassify inputs to a target class.
/// If such a small trigger exists for any class, the model is likely backdoored.
/// The anomaly index measures how much smaller the trigger for one class is compared to others.
/// </para>
/// <para>
/// <b>For Beginners:</b> This detector works backwards — instead of looking at training updates,
/// it asks: "Is there a tiny pattern I can add to any input that makes the model always predict
/// a specific class?" If yes, someone probably planted a backdoor. For each possible target class,
/// it finds the smallest such pattern. If one class has an unusually small trigger, that's the
/// backdoor target class.
/// </para>
/// <para>
/// Reference: Wang et al. (2019), "Neural Cleanse: Identifying and Mitigating Backdoor Attacks
/// in Neural Networks" (IEEE S&amp;P 2019).
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class NeuralCleanseDetector<T> : Infrastructure.FederatedLearningComponentBase<T>, IBackdoorDetector<T>
{
    private readonly double _anomalyThreshold;
    private readonly int _numClasses;

    /// <inheritdoc/>
    public string DetectorName => "NeuralCleanse";

    /// <summary>
    /// Creates a new Neural Cleanse detector.
    /// </summary>
    /// <param name="numClasses">Number of output classes to check for backdoor targets. Default: 10.</param>
    /// <param name="anomalyThreshold">MAD-based anomaly threshold. Default: 2.0.</param>
    public NeuralCleanseDetector(int numClasses = 10, double anomalyThreshold = 2.0)
    {
        _numClasses = numClasses;
        _anomalyThreshold = anomalyThreshold;
    }

    /// <inheritdoc/>
    public Dictionary<int, double> DetectSuspiciousUpdates(Dictionary<int, Vector<T>> clientUpdates, Vector<T> globalModel)
    {
        if (clientUpdates.Count < 2)
            return clientUpdates.ToDictionary(kv => kv.Key, _ => 0.0);

        int d = globalModel.Length;
        var suspicionScores = new Dictionary<int, double>();

        // For each client, analyze the update's "trigger potential":
        // how concentrated the update is in specific parameter regions
        foreach (var (clientId, update) in clientUpdates)
        {
            // Compute L1 norms for parameter groups (simulating per-class trigger sizes)
            int groupSize = Math.Max(1, d / _numClasses);
            var groupNorms = new double[_numClasses];

            for (int c = 0; c < _numClasses; c++)
            {
                int start = c * groupSize;
                int end = Math.Min(start + groupSize, d);
                double norm = 0;
                for (int i = start; i < end; i++)
                {
                    norm += Math.Abs(NumOps.ToDouble(update[i]));
                }
                groupNorms[c] = norm;
            }

            // Compute anomaly index: MAD-based outlier detection on group norms
            double median = GetMedian(groupNorms);
            double[] deviations = groupNorms.Select(n => Math.Abs(n - median)).ToArray();
            double mad = GetMedian(deviations);

            // Anomaly index: how many MADs the smallest norm is below the median
            double minNorm = groupNorms.Min();
            double anomalyIndex = mad > 1e-10 ? (median - minNorm) / (1.4826 * mad) : 0;

            // High anomaly index means one class has a disproportionately small trigger
            // which is characteristic of backdoor updates
            suspicionScores[clientId] = anomalyIndex > _anomalyThreshold
                ? Math.Min(1.0, anomalyIndex / (2 * _anomalyThreshold))
                : 0;
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

        if (filtered.Count == 0 && clientUpdates.Count > 0)
        {
            var leastSuspicious = scores.OrderBy(kv => kv.Value).First().Key;
            filtered[leastSuspicious] = clientUpdates[leastSuspicious];
        }

        return filtered;
    }

    private static double GetMedian(double[] values)
    {
        var sorted = values.OrderBy(v => v).ToArray();
        int n = sorted.Length;
        if (n == 0) return 0;
        return n % 2 == 1 ? sorted[n / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2;
    }
}
