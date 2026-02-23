using AiDotNet.FederatedLearning.Infrastructure;
using AiDotNet.Models.Options;
using AiDotNet.Tensors;

namespace AiDotNet.FederatedLearning.DriftDetection;

/// <summary>
/// Model-based drift detector: uses gradient direction and weight divergence to detect drift.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Instead of monitoring loss values (like StatisticalDriftDetector),
/// this detector looks at HOW the model is changing. If a client's gradients suddenly point in
/// a very different direction than before, their data distribution has likely shifted. Similarly,
/// if their model weights diverge abnormally from the global model, they may be adapting to
/// a changed local distribution.</para>
///
/// <para><b>Methods:</b></para>
/// <list type="bullet">
/// <item><description><b>Gradient Divergence:</b> Track cosine similarity of consecutive gradients.
/// A sudden drop means the optimization landscape changed (drift).</description></item>
/// <item><description><b>Weight Divergence:</b> Monitor L2 distance between client and global model
/// over time. Abnormal increases suggest the client is learning from different data.</description></item>
/// </list>
/// </remarks>
/// <typeparam name="T">The numeric type used for model parameters.</typeparam>
public class ModelDriftDetector<T> : FederatedLearningComponentBase<T>, IFederatedDriftDetector<T>
{
    private readonly FederatedDriftOptions _options;
    private readonly Dictionary<int, List<Tensor<T>>> _clientModelHistory = new();
    private readonly Dictionary<int, List<double>> _clientDivergenceHistory = new();
    private readonly Dictionary<int, int> _clientDriftStartRound = new();

    /// <inheritdoc/>
    public string MethodName => _options.Method.ToString();

    /// <summary>
    /// Initializes a new instance of <see cref="ModelDriftDetector{T}"/>.
    /// </summary>
    /// <param name="options">Drift detection configuration.</param>
    public ModelDriftDetector(FederatedDriftOptions options)
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
        if (clientModels is null) throw new ArgumentNullException(nameof(clientModels));
        if (globalModel is null) throw new ArgumentNullException(nameof(globalModel));

        var report = new DriftReport
        {
            Round = round,
            Method = _options.Method
        };

        int driftingClients = 0;
        double totalDriftScore = 0;

        foreach (var kvp in clientModels)
        {
            int clientId = kvp.Key;
            var clientModel = kvp.Value;

            // Record model history
            if (!_clientModelHistory.ContainsKey(clientId))
            {
                _clientModelHistory[clientId] = new List<Tensor<T>>();
            }

            _clientModelHistory[clientId].Add(clientModel);

            // Trim to lookback window
            while (_clientModelHistory[clientId].Count > _options.LookbackWindowRounds)
            {
                _clientModelHistory[clientId].RemoveAt(0);
            }

            // Compute drift score
            double driftScore;
            if (_options.Method == FederatedDriftMethod.GradientDivergence)
            {
                driftScore = ComputeGradientDivergence(clientId);
            }
            else
            {
                driftScore = ComputeWeightDivergence(clientId, globalModel);
            }

            // Record divergence history for trend analysis
            if (!_clientDivergenceHistory.ContainsKey(clientId))
            {
                _clientDivergenceHistory[clientId] = new List<double>();
            }

            _clientDivergenceHistory[clientId].Add(driftScore);
            while (_clientDivergenceHistory[clientId].Count > _options.LookbackWindowRounds)
            {
                _clientDivergenceHistory[clientId].RemoveAt(0);
            }

            var result = ClassifyDrift(clientId, driftScore, round);
            report.ClientResults.Add(result);

            totalDriftScore += result.DriftScore;
            if (result.DriftType != DriftType.None && result.DriftType != DriftType.Warning)
            {
                driftingClients++;
            }
        }

        int totalClients = clientModels.Count;
        report.DriftingClientFraction = totalClients > 0 ? (double)driftingClients / totalClients : 0;
        report.AverageDriftScore = totalClients > 0 ? totalDriftScore / totalClients : 0;
        report.GlobalDriftDetected = report.DriftingClientFraction >= _options.GlobalDriftThreshold;

        report.Summary = $"Round {round}: {driftingClients}/{totalClients} clients drifting " +
                        $"via {_options.Method}, avg score: {report.AverageDriftScore:F4}. " +
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
            double multiplier = 1.0;

            foreach (var result in driftReport.ClientResults)
            {
                if (result.ClientId == clientId)
                {
                    multiplier = result.SuggestedWeightMultiplier;
                    break;
                }
            }

            adjustedWeights[clientId] = kvp.Value * multiplier;
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
        _clientModelHistory.Clear();
        _clientDivergenceHistory.Clear();
        _clientDriftStartRound.Clear();
    }

    private double ComputeGradientDivergence(int clientId)
    {
        var history = _clientModelHistory[clientId];
        if (history.Count < 2) return 0;

        // Compute cosine similarity between consecutive model updates
        // Low similarity = gradient direction changed = potential drift
        var prev = history[history.Count - 2];
        var curr = history[history.Count - 1];

        double similarity = CosineSimilarity(prev, curr);

        // Map: similarity = 1.0 (no drift) -> score = 0
        //       similarity = 0.0 (orthogonal) -> score = 0.5
        //       similarity = -1.0 (reversed) -> score = 1.0
        return (1.0 - similarity) / 2.0;
    }

    private double ComputeWeightDivergence(int clientId, Tensor<T> globalModel)
    {
        var history = _clientModelHistory[clientId];
        if (history.Count < 2) return 0;

        var currentModel = history[history.Count - 1];

        // Current divergence from global model
        double currentDivergence = L2Distance(currentModel, globalModel);

        // Historical average divergence
        double avgDivergence = 0;
        for (int i = 0; i < history.Count - 1; i++)
        {
            avgDivergence += L2Distance(history[i], globalModel);
        }
        avgDivergence /= (history.Count - 1);

        // Drift score: how much more divergent than usual?
        if (avgDivergence < 1e-12)
        {
            return currentDivergence > 1e-6 ? 1.0 : 0;
        }

        double ratio = currentDivergence / avgDivergence;

        // ratio > 1 means more divergent than usual
        return Math.Min(1.0, Math.Max(0, (ratio - 1.0) / 2.0));
    }

    private double CosineSimilarity(Tensor<T> a, Tensor<T> b)
    {
        int size = Math.Min(a.Shape[0], b.Shape[0]);
        double dot = 0, normA = 0, normB = 0;

        for (int i = 0; i < size; i++)
        {
            double va = NumOps.ToDouble(a[i]);
            double vb = NumOps.ToDouble(b[i]);
            dot += va * vb;
            normA += va * va;
            normB += vb * vb;
        }

        double denom = Math.Sqrt(normA) * Math.Sqrt(normB);
        return denom > 1e-12 ? dot / denom : 0;
    }

    private double L2Distance(Tensor<T> a, Tensor<T> b)
    {
        int size = Math.Min(a.Shape[0], b.Shape[0]);
        double sumSq = 0;

        for (int i = 0; i < size; i++)
        {
            double diff = NumOps.ToDouble(a[i]) - NumOps.ToDouble(b[i]);
            sumSq += diff * diff;
        }

        return Math.Sqrt(sumSq);
    }

    private ClientDriftResult ClassifyDrift(int clientId, double driftScore, int round)
    {
        var result = new ClientDriftResult
        {
            ClientId = clientId,
            DriftScore = driftScore
        };

        if (driftScore >= _options.SensitivityThreshold * 3)
        {
            // Check trend for drift type classification
            result.DriftType = ClassifyDriftType(clientId);
            result.SuggestedWeightMultiplier = Math.Max(_options.MinDriftWeight, 1.0 - driftScore);

            if (!_clientDriftStartRound.ContainsKey(clientId))
            {
                _clientDriftStartRound[clientId] = round;
            }

            result.DriftStartRound = _clientDriftStartRound[clientId];

            result.RecommendedAction = driftScore > 0.8
                ? (_options.TriggerSelectiveRetraining ? DriftAction.SelectiveRetrain : DriftAction.TemporaryExclude)
                : DriftAction.ReduceWeight;
        }
        else if (driftScore >= _options.SensitivityThreshold)
        {
            result.DriftType = DriftType.Warning;
            result.RecommendedAction = DriftAction.Monitor;
            result.SuggestedWeightMultiplier = 1.0 - driftScore * 0.3;
        }
        else
        {
            _clientDriftStartRound.Remove(clientId);
        }

        return result;
    }

    private DriftType ClassifyDriftType(int clientId)
    {
        if (!_clientDivergenceHistory.ContainsKey(clientId))
        {
            return DriftType.Sudden;
        }

        var history = _clientDivergenceHistory[clientId];
        if (history.Count < 3) return DriftType.Sudden;

        // Check if divergence is monotonically increasing (gradual)
        int increasing = 0;
        for (int i = 1; i < history.Count; i++)
        {
            if (history[i] > history[i - 1]) increasing++;
        }

        double fractionIncreasing = (double)increasing / (history.Count - 1);

        if (fractionIncreasing > 0.7)
        {
            return DriftType.Gradual;
        }

        return DriftType.Sudden;
    }
}
