using AiDotNet.FederatedLearning.Infrastructure;
using AiDotNet.Models.Options;
using AiDotNet.Tensors;

namespace AiDotNet.FederatedLearning.DriftDetection;

/// <summary>
/// Drift-adaptive aggregator: wraps any aggregation strategy and adjusts weights based on drift.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> In standard FedAvg, all clients get weight proportional to their
/// data size. But if some clients are experiencing concept drift (their data has changed),
/// their updates may hurt the global model. This aggregator detects drifting clients and
/// reduces their influence while boosting stable clients.</para>
///
/// <para><b>How it works:</b></para>
/// <list type="number">
/// <item><description>After each round, the configured drift detector analyzes client metrics.</description></item>
/// <item><description>Stable clients retain their full aggregation weight.</description></item>
/// <item><description>Warning-level clients get slightly reduced weight.</description></item>
/// <item><description>Drifting clients get significantly reduced weight (down to MinDriftWeight).</description></item>
/// <item><description>Severely drifting clients may be temporarily excluded or asked to retrain.</description></item>
/// </list>
///
/// <para><b>Integration:</b> Use this as a wrapper around your existing aggregation strategy.
/// It modifies the weights before aggregation happens.</para>
/// </remarks>
/// <typeparam name="T">The numeric type used for model parameters.</typeparam>
public class DriftAdaptiveAggregator<T> : FederatedLearningComponentBase<T>
{
    private readonly FederatedDriftOptions _options;
    private readonly IFederatedDriftDetector<T> _detector;
    private DriftReport? _latestReport;

    /// <summary>
    /// Initializes a new instance of <see cref="DriftAdaptiveAggregator{T}"/>.
    /// </summary>
    /// <param name="options">Drift detection configuration.</param>
    /// <param name="detector">The drift detector to use.</param>
    public DriftAdaptiveAggregator(FederatedDriftOptions options, IFederatedDriftDetector<T> detector)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
        _detector = detector ?? throw new ArgumentNullException(nameof(detector));
    }

    /// <summary>
    /// Processes a round of FL by detecting drift and computing adaptive weights.
    /// </summary>
    /// <param name="round">Current FL round number.</param>
    /// <param name="clientModels">Current model updates from each client.</param>
    /// <param name="globalModel">Current global model parameters.</param>
    /// <param name="clientMetrics">Per-client metrics (e.g., loss).</param>
    /// <param name="originalWeights">Original aggregation weights.</param>
    /// <returns>Drift report and adjusted weights.</returns>
    public (DriftReport Report, Dictionary<int, double> AdjustedWeights) ProcessRound(
        int round,
        Dictionary<int, Tensor<T>> clientModels,
        Tensor<T> globalModel,
        Dictionary<int, double> clientMetrics,
        Dictionary<int, double> originalWeights)
    {
        if (!_options.Enabled || round % _options.DetectionFrequency != 0)
        {
            return (CreateEmptyReport(round), new Dictionary<int, double>(originalWeights));
        }

        // Run drift detection
        var report = _detector.DetectDrift(round, clientModels, globalModel, clientMetrics);
        _latestReport = report;

        // Compute adaptive weights
        var adjustedWeights = _detector.GetAdaptiveWeights(originalWeights, report);

        // Remove temporarily excluded clients
        foreach (var result in report.ClientResults)
        {
            if (result.RecommendedAction == DriftAction.TemporaryExclude)
            {
                adjustedWeights.Remove(result.ClientId);
            }
        }

        // Renormalize after exclusions
        double totalWeight = 0;
        foreach (double w in adjustedWeights.Values)
        {
            totalWeight += w;
        }

        if (totalWeight > 1e-12 && adjustedWeights.Count > 0)
        {
            var keys = new List<int>(adjustedWeights.Keys);
            foreach (int key in keys)
            {
                adjustedWeights[key] /= totalWeight;
            }
        }

        return (report, adjustedWeights);
    }

    /// <summary>
    /// Gets the latest drift report.
    /// </summary>
    public DriftReport? LatestReport => _latestReport;

    /// <summary>
    /// Gets the set of currently drifting client IDs.
    /// </summary>
    public HashSet<int> GetDriftingClients()
    {
        var drifting = new HashSet<int>();
        if (_latestReport is null) return drifting;

        foreach (var result in _latestReport.ClientResults)
        {
            if (result.DriftType == DriftType.Sudden || result.DriftType == DriftType.Gradual ||
                result.DriftType == DriftType.Recurring)
            {
                drifting.Add(result.ClientId);
            }
        }

        return drifting;
    }

    /// <summary>
    /// Gets clients that need selective retraining.
    /// </summary>
    public HashSet<int> GetClientsNeedingRetraining()
    {
        var retrain = new HashSet<int>();
        if (_latestReport is null) return retrain;

        foreach (var result in _latestReport.ClientResults)
        {
            if (result.RecommendedAction == DriftAction.SelectiveRetrain)
            {
                retrain.Add(result.ClientId);
            }
        }

        return retrain;
    }

    /// <summary>
    /// Resets all drift detection state.
    /// </summary>
    public void Reset()
    {
        _detector.Reset();
        _latestReport = null;
    }

    private static DriftReport CreateEmptyReport(int round)
    {
        return new DriftReport
        {
            Round = round,
            GlobalDriftDetected = false,
            Summary = $"Round {round}: drift detection skipped (not evaluation round)."
        };
    }
}
