using AiDotNet.FederatedLearning.Infrastructure;
using AiDotNet.Models.Options;
using AiDotNet.Tensors;

namespace AiDotNet.FederatedLearning.DriftDetection;

/// <summary>
/// Statistical drift detector: uses Page-Hinkley, ADWIN, or DDM tests on client metrics.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This detector monitors each client's loss or accuracy over rounds
/// and applies proven statistical tests to detect when the distribution has shifted.</para>
///
/// <para><b>Methods:</b></para>
/// <list type="bullet">
/// <item><description><b>Page-Hinkley:</b> Tracks cumulative deviation from the running mean.
/// When deviation exceeds a threshold, drift is declared. Good for detecting mean shifts.</description></item>
/// <item><description><b>ADWIN:</b> Maintains a variable-length window and tests if sub-windows
/// have different distributions. Self-adapting to drift speed. Best general-purpose method.</description></item>
/// <item><description><b>DDM:</b> Monitors error rate and standard deviation. Warning at 2-sigma
/// increase, drift at 3-sigma. Simple and effective for classification tasks.</description></item>
/// </list>
/// </remarks>
/// <typeparam name="T">The numeric type used for model parameters.</typeparam>
public class StatisticalDriftDetector<T> : FederatedLearningComponentBase<T>, IFederatedDriftDetector<T>
{
    private readonly FederatedDriftOptions _options;
    private readonly Dictionary<int, List<double>> _clientMetricHistory = new();
    private readonly Dictionary<int, int> _clientDriftStartRound = new();

    /// <inheritdoc/>
    public string MethodName => _options.Method.ToString();

    /// <summary>
    /// Initializes a new instance of <see cref="StatisticalDriftDetector{T}"/>.
    /// </summary>
    /// <param name="options">Drift detection configuration.</param>
    public StatisticalDriftDetector(FederatedDriftOptions options)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
    }

    /// <inheritdoc/>
    public DriftReport DetectDrift(
        int round,
        Dictionary<int, Tensor<T>> clientModels,
        Tensor<T> globalModel,
        Dictionary<int, double> clientMetrics)
    {
        if (clientMetrics is null) throw new ArgumentNullException(nameof(clientMetrics));

        var report = new DriftReport
        {
            Round = round,
            Method = _options.Method
        };

        int driftingClients = 0;
        double totalDriftScore = 0;

        foreach (var kvp in clientMetrics)
        {
            int clientId = kvp.Key;
            double metric = kvp.Value;

            // Record metric history
            if (!_clientMetricHistory.ContainsKey(clientId))
            {
                _clientMetricHistory[clientId] = new List<double>();
            }

            _clientMetricHistory[clientId].Add(metric);

            // Trim to lookback window
            var history = _clientMetricHistory[clientId];
            while (history.Count > _options.LookbackWindowRounds)
            {
                history.RemoveAt(0);
            }

            // Run drift test
            var result = RunDriftTest(clientId, history, round);
            report.ClientResults.Add(result);

            totalDriftScore += result.DriftScore;
            if (result.DriftType != DriftType.None && result.DriftType != DriftType.Warning)
            {
                driftingClients++;
            }
        }

        int totalClients = clientMetrics.Count;
        report.DriftingClientFraction = totalClients > 0 ? (double)driftingClients / totalClients : 0;
        report.AverageDriftScore = totalClients > 0 ? totalDriftScore / totalClients : 0;
        report.GlobalDriftDetected = report.DriftingClientFraction >= _options.GlobalDriftThreshold;

        report.Summary = $"Round {round}: {driftingClients}/{totalClients} clients drifting " +
                        $"({report.DriftingClientFraction:P0}), avg drift score: {report.AverageDriftScore:F4}. " +
                        $"Global drift: {(report.GlobalDriftDetected ? "YES" : "no")}.";

        return report;
    }

    /// <inheritdoc/>
    public Dictionary<int, double> GetAdaptiveWeights(
        Dictionary<int, double> originalWeights,
        DriftReport driftReport)
    {
        if (originalWeights is null) throw new ArgumentNullException(nameof(originalWeights));
        if (driftReport is null || !_options.AdaptAggregationWeights)
        {
            return new Dictionary<int, double>(originalWeights);
        }

        var adjustedWeights = new Dictionary<int, double>();

        foreach (var kvp in originalWeights)
        {
            int clientId = kvp.Key;
            double originalWeight = kvp.Value;

            // Find this client's drift result
            double multiplier = 1.0;
            foreach (var result in driftReport.ClientResults)
            {
                if (result.ClientId == clientId)
                {
                    multiplier = result.SuggestedWeightMultiplier;
                    break;
                }
            }

            adjustedWeights[clientId] = originalWeight * multiplier;
        }

        // Renormalize
        double totalWeight = 0;
        foreach (double w in adjustedWeights.Values)
        {
            totalWeight += w;
        }

        if (totalWeight > 1e-12)
        {
            var keys = new List<int>(adjustedWeights.Keys);
            foreach (int key in keys)
            {
                adjustedWeights[key] /= totalWeight;
            }
        }

        return adjustedWeights;
    }

    /// <inheritdoc/>
    public void Reset()
    {
        _clientMetricHistory.Clear();
        _clientDriftStartRound.Clear();
    }

    private ClientDriftResult RunDriftTest(int clientId, List<double> history, int round)
    {
        var result = new ClientDriftResult { ClientId = clientId };

        if (history.Count < 3)
        {
            return result; // Not enough data
        }

        double driftScore = _options.Method switch
        {
            FederatedDriftMethod.PageHinkley => PageHinkleyTest(history),
            FederatedDriftMethod.ADWIN => AdwinTest(history),
            FederatedDriftMethod.DDM => DdmTest(history),
            _ => PageHinkleyTest(history) // Default
        };

        result.DriftScore = Math.Min(1.0, Math.Max(0, driftScore));

        // Classify drift
        if (driftScore >= _options.SensitivityThreshold * 3)
        {
            result.DriftType = ClassifyDriftType(history);
            result.RecommendedAction = DriftAction.ReduceWeight;
            result.SuggestedWeightMultiplier = Math.Max(_options.MinDriftWeight, 1.0 - driftScore);

            if (!_clientDriftStartRound.ContainsKey(clientId))
            {
                _clientDriftStartRound[clientId] = round;
            }

            result.DriftStartRound = _clientDriftStartRound[clientId];

            if (driftScore > 0.8)
            {
                result.RecommendedAction = _options.TriggerSelectiveRetraining
                    ? DriftAction.SelectiveRetrain
                    : DriftAction.TemporaryExclude;
            }
        }
        else if (driftScore >= _options.SensitivityThreshold)
        {
            result.DriftType = DriftType.Warning;
            result.RecommendedAction = DriftAction.Monitor;
            result.SuggestedWeightMultiplier = 1.0 - driftScore * 0.3;
        }
        else
        {
            // No drift - clear previous drift start
            _clientDriftStartRound.Remove(clientId);
        }

        return result;
    }

    private double PageHinkleyTest(List<double> history)
    {
        // Page-Hinkley: detect change in the mean
        // Accumulate deviations from the running mean; drift when cumulative sum exceeds threshold
        int n = history.Count;
        double runningMean = 0;
        double cumulativeSum = 0;
        double minCumulativeSum = double.MaxValue;
        double maxDeviation = 0;

        for (int i = 0; i < n; i++)
        {
            runningMean = runningMean + (history[i] - runningMean) / (i + 1);
            cumulativeSum += history[i] - runningMean - _options.SensitivityThreshold;
            minCumulativeSum = Math.Min(minCumulativeSum, cumulativeSum);
            maxDeviation = Math.Max(maxDeviation, cumulativeSum - minCumulativeSum);
        }

        // Normalize by window size for comparable scores
        double threshold = _options.SensitivityThreshold * n;
        return threshold > 0 ? maxDeviation / threshold : 0;
    }

    private double AdwinTest(List<double> history)
    {
        // ADWIN-inspired: compare statistics of two halves of the window
        // Find the split point that maximizes the difference in means
        int n = history.Count;
        if (n < 4) return 0;

        double totalSum = 0;
        foreach (double v in history)
        {
            totalSum += v;
        }

        double maxDiff = 0;
        double leftSum = 0;

        for (int split = 2; split < n - 1; split++)
        {
            leftSum += history[split - 1];
            double leftMean = leftSum / split;
            double rightMean = (totalSum - leftSum) / (n - split);

            // Hoeffding-like bound
            double epsilon = Math.Sqrt(Math.Log(2.0 / _options.SensitivityThreshold) /
                                       (2.0 * Math.Min(split, n - split)));

            double diff = Math.Abs(leftMean - rightMean);
            if (diff > epsilon)
            {
                maxDiff = Math.Max(maxDiff, diff / Math.Max(1e-12, epsilon));
            }
        }

        return Math.Min(1.0, maxDiff);
    }

    private double DdmTest(List<double> history)
    {
        // DDM: track error rate mean and std, detect increases
        int n = history.Count;
        if (n < 5) return 0;

        // Use first half as reference, second half as current
        int refEnd = n / 2;
        double refMean = 0, refVar = 0;

        for (int i = 0; i < refEnd; i++)
        {
            refMean += history[i];
        }
        refMean /= refEnd;

        for (int i = 0; i < refEnd; i++)
        {
            double diff = history[i] - refMean;
            refVar += diff * diff;
        }
        refVar /= refEnd;
        double refStd = Math.Sqrt(refVar);

        // Current statistics
        double curMean = 0;
        for (int i = refEnd; i < n; i++)
        {
            curMean += history[i];
        }
        curMean /= (n - refEnd);

        // DDM: drift if current mean > ref_mean + 3 * ref_std
        double deviation = (curMean - refMean) / Math.Max(1e-12, refStd);

        // Map to [0, 1]: 2-sigma = warning (0.5), 3-sigma = drift (1.0)
        return Math.Max(0, Math.Min(1.0, (deviation - 1.0) / 2.0));
    }

    private static DriftType ClassifyDriftType(List<double> history)
    {
        if (history.Count < 4) return DriftType.Sudden;

        int n = history.Count;
        int halfPoint = n / 2;

        // Check for sudden vs gradual by looking at the transition speed
        // Compute running mean of first and second halves
        double firstHalfMean = 0, secondHalfMean = 0;
        for (int i = 0; i < halfPoint; i++)
        {
            firstHalfMean += history[i];
        }
        firstHalfMean /= halfPoint;

        for (int i = halfPoint; i < n; i++)
        {
            secondHalfMean += history[i];
        }
        secondHalfMean /= (n - halfPoint);

        // Check if the change happened abruptly (max single-step change)
        double maxStep = 0;
        for (int i = 1; i < n; i++)
        {
            maxStep = Math.Max(maxStep, Math.Abs(history[i] - history[i - 1]));
        }

        double totalChange = Math.Abs(secondHalfMean - firstHalfMean);

        // If most of the change is in a single step, it's sudden
        if (totalChange > 0 && maxStep / totalChange > 0.5)
        {
            return DriftType.Sudden;
        }

        return DriftType.Gradual;
    }
}
