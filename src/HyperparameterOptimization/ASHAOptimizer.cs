using AiDotNet.Helpers;
using AiDotNet.Models;
using AiDotNet.Models.Results;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.HyperparameterOptimization;

/// <summary>
/// Implements ASHA (Asynchronous Successive Halving Algorithm) for hyperparameter optimization.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> ASHA is an improved version of Hyperband that:
/// - Doesn't wait for all configurations at a level before promoting some
/// - Can promote promising configurations as soon as they outperform enough peers
/// - Is naturally suited for parallel/distributed training
/// - Typically converges faster than synchronous Hyperband
///
/// Key concepts:
/// - Rungs: Resource levels at which we evaluate (e.g., epochs 1, 3, 9, 27)
/// - Promotion: Moving a configuration to train with more resources
/// - Early Stopping: Killing configurations that aren't competitive
///
/// ASHA uses the same exponential resource increase as Hyperband but allows
/// asynchronous promotion based on relative performance.
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type for models.</typeparam>
/// <typeparam name="TOutput">The output data type for models.</typeparam>
public class ASHAOptimizer<T, TInput, TOutput> : HyperparameterOptimizerBase<T, TInput, TOutput>
{
    private readonly Random _random;
    private readonly INumericOperations<T> _numOps;
    private readonly int _maxResource;
    private readonly int _reductionFactor;
    private readonly int _minResource;
    private readonly double _promotionThreshold;

    // Track configurations at each rung
    private readonly Dictionary<int, List<RungEntry>> _rungs;

    /// <summary>
    /// Gets the resource levels (rungs) in this ASHA configuration.
    /// </summary>
    public List<int> Rungs { get; }

    /// <summary>
    /// Initializes a new instance of the ASHAOptimizer class.
    /// </summary>
    /// <param name="maximize">Whether to maximize the objective (true) or minimize it (false).</param>
    /// <param name="maxResource">Maximum resource budget per configuration (e.g., max epochs).</param>
    /// <param name="reductionFactor">Factor between successive rungs (typically 2-4).</param>
    /// <param name="minResource">Minimum resource budget (first rung).</param>
    /// <param name="promotionThreshold">Fraction of configurations that must complete a rung before promotion (default 0.5).</param>
    /// <param name="seed">Random seed for reproducibility. If null, uses a random seed.</param>
    public ASHAOptimizer(
        bool maximize = true,
        int maxResource = 81,
        int reductionFactor = 3,
        int minResource = 1,
        double promotionThreshold = 0.5,
        int? seed = null) : base(maximize)
    {
        if (maxResource < 1)
            throw new ArgumentException("maxResource must be at least 1.", nameof(maxResource));
        if (reductionFactor < 2)
            throw new ArgumentException("reductionFactor must be at least 2.", nameof(reductionFactor));
        if (minResource < 1)
            throw new ArgumentException("minResource must be at least 1.", nameof(minResource));
        if (minResource > maxResource)
            throw new ArgumentException("minResource cannot exceed maxResource.", nameof(minResource));
        if (promotionThreshold <= 0 || promotionThreshold > 1)
            throw new ArgumentException("promotionThreshold must be in (0, 1].", nameof(promotionThreshold));

        _random = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.CreateSecureRandom();
        _numOps = MathHelper.GetNumericOperations<T>();
        _maxResource = maxResource;
        _reductionFactor = reductionFactor;
        _minResource = minResource;
        _promotionThreshold = promotionThreshold;

        // Calculate rung levels: r, r*eta, r*eta^2, ..., R
        Rungs = new List<int>();
        int r = _minResource;
        while (r <= _maxResource)
        {
            Rungs.Add(r);
            if (r == _maxResource) break;
            r = Math.Min((int)(r * _reductionFactor), _maxResource);
        }

        _rungs = new Dictionary<int, List<RungEntry>>();
        foreach (var rung in Rungs)
        {
            _rungs[rung] = new List<RungEntry>();
        }
    }

    /// <summary>
    /// Searches for the best hyperparameter configuration using ASHA.
    /// </summary>
    public override HyperparameterOptimizationResult<T> Optimize(
        Func<Dictionary<string, object>, T> objectiveFunction,
        HyperparameterSearchSpace searchSpace,
        int nTrials)
    {
        ValidateOptimizationInputs(objectiveFunction, searchSpace, nTrials);

        SearchSpace = searchSpace;
        Trials.Clear();

        // Reset rungs
        foreach (var rung in Rungs)
        {
            _rungs[rung].Clear();
        }

        var startTime = DateTime.UtcNow;
        var configQueue = new Queue<ConfigurationState>();

        lock (SyncLock)
        {
            int trialId = 0;

            // Main ASHA loop
            while (trialId < nTrials || configQueue.Count > 0)
            {
                // Either promote an existing configuration or start a new one
                ConfigurationState configToRun;

                if (configQueue.Count > 0)
                {
                    // Continue with a configuration waiting for more resources
                    configToRun = configQueue.Dequeue();
                }
                else if (trialId < nTrials)
                {
                    // Start a new configuration at the first rung
                    var newConfig = SampleRandomConfiguration(searchSpace);
                    configToRun = new ConfigurationState(
                        newConfig,
                        Rungs[0],
                        0,
                        trialId++
                    );
                }
                else
                {
                    break;
                }

                // Evaluate at current resource level
                var trial = new HyperparameterTrial<T>(configToRun.TrialId);
                var configWithResource = new Dictionary<string, object>(configToRun.Configuration)
                {
                    ["resource"] = configToRun.CurrentResource
                };

                EvaluateTrialSafely(trial, objectiveFunction, configWithResource);
                Trials.Add(trial);

                if (trial.Status != TrialStatus.Complete || trial.ObjectiveValue == null)
                {
                    continue; // Skip failed configurations
                }

                // Record result at this rung
                var rungEntry = new RungEntry(
                    configToRun.Configuration,
                    trial.ObjectiveValue,
                    configToRun.TrialId
                );
                _rungs[configToRun.CurrentResource].Add(rungEntry);

                // Check if this configuration should be promoted
                if (ShouldPromote(configToRun.CurrentResource, trial.ObjectiveValue))
                {
                    int nextRungIndex = configToRun.RungIndex + 1;
                    if (nextRungIndex < Rungs.Count)
                    {
                        // Promote to next rung
                        var promotedConfig = new ConfigurationState(
                            configToRun.Configuration,
                            Rungs[nextRungIndex],
                            nextRungIndex,
                            configToRun.TrialId
                        );
                        configQueue.Enqueue(promotedConfig);
                    }
                }
            }
        }

        var endTime = DateTime.UtcNow;
        return CreateOptimizationResult(searchSpace, startTime, endTime, Trials.Count);
    }

    /// <summary>
    /// Determines if a configuration should be promoted to the next rung.
    /// </summary>
    private bool ShouldPromote(int currentRung, T objectiveValue)
    {
        var rungEntries = _rungs[currentRung];

        // Need enough configurations at this rung to make a decision
        int requiredCount = Math.Max(1, (int)Math.Ceiling(_reductionFactor * _promotionThreshold));
        if (rungEntries.Count < requiredCount)
        {
            return true; // Not enough data, promote optimistically
        }

        // Check if this configuration is in the top 1/eta
        var sortedValues = rungEntries
            .Select(e => _numOps.ToDouble(e.ObjectiveValue))
            .OrderByDescending(v => Maximize ? v : -v)
            .ToList();

        double currentValue = _numOps.ToDouble(objectiveValue);
        int rank = sortedValues.FindIndex(v => Math.Abs(v - currentValue) < 1e-10);

        if (rank < 0)
        {
            rank = sortedValues.Count; // Not found, assume worst
        }

        int promotionCutoff = Math.Max(1, rungEntries.Count / _reductionFactor);
        return rank < promotionCutoff;
    }

    /// <summary>
    /// Suggests the next hyperparameter configuration to try.
    /// </summary>
    public override Dictionary<string, object> SuggestNext(HyperparameterTrial<T> trial)
    {
        if (SearchSpace == null)
            throw new InvalidOperationException("Search space not initialized. Call Optimize() first.");

        // For ASHA, we start all new configurations at the first rung
        var config = SampleRandomConfiguration(SearchSpace);
        config["resource"] = Rungs[0];
        return config;
    }

    /// <summary>
    /// Samples a random configuration from the search space.
    /// </summary>
    private Dictionary<string, object> SampleRandomConfiguration(HyperparameterSearchSpace searchSpace)
    {
        var config = new Dictionary<string, object>();
        foreach (var param in searchSpace.Parameters)
        {
            config[param.Key] = param.Value.Sample(_random);
        }
        return config;
    }

    /// <summary>
    /// Gets statistics about configurations at each rung.
    /// </summary>
    public Dictionary<int, RungStatistics> GetRungStatistics()
    {
        var stats = new Dictionary<int, RungStatistics>();

        foreach (var rung in Rungs)
        {
            var entries = _rungs[rung];
            if (entries.Count == 0)
            {
                stats[rung] = new RungStatistics(rung, 0, 0, 0, 0);
                continue;
            }

            var values = entries.Select(e => _numOps.ToDouble(e.ObjectiveValue)).ToList();
            stats[rung] = new RungStatistics(
                rung,
                entries.Count,
                values.Average(),
                values.Min(),
                values.Max()
            );
        }

        return stats;
    }

    /// <summary>
    /// Gets the best configuration found at the highest rung.
    /// </summary>
    public (Dictionary<string, object>? Config, T? Score) GetBestConfiguration()
    {
        // Find the highest rung with entries
        for (int i = Rungs.Count - 1; i >= 0; i--)
        {
            var entries = _rungs[Rungs[i]];
            if (entries.Count > 0)
            {
                var best = Maximize
                    ? entries.OrderByDescending(e => _numOps.ToDouble(e.ObjectiveValue)).First()
                    : entries.OrderBy(e => _numOps.ToDouble(e.ObjectiveValue)).First();

                return (best.Configuration, best.ObjectiveValue);
            }
        }

        return (null, default);
    }

    #region Helper Classes

    private class ConfigurationState
    {
        public Dictionary<string, object> Configuration { get; }
        public int CurrentResource { get; }
        public int RungIndex { get; }
        public int TrialId { get; }

        public ConfigurationState(Dictionary<string, object> configuration, int currentResource, int rungIndex, int trialId)
        {
            Configuration = configuration;
            CurrentResource = currentResource;
            RungIndex = rungIndex;
            TrialId = trialId;
        }
    }

    private class RungEntry
    {
        public Dictionary<string, object> Configuration { get; }
        public T ObjectiveValue { get; }
        public int TrialId { get; }

        public RungEntry(Dictionary<string, object> configuration, T objectiveValue, int trialId)
        {
            Configuration = configuration;
            ObjectiveValue = objectiveValue;
            TrialId = trialId;
        }
    }

    #endregion
}

/// <summary>
/// Statistics for a single ASHA rung.
/// </summary>
public class RungStatistics
{
    /// <summary>
    /// The resource level of this rung.
    /// </summary>
    public int Resource { get; }

    /// <summary>
    /// Number of configurations evaluated at this rung.
    /// </summary>
    public int Count { get; }

    /// <summary>
    /// Mean objective value at this rung.
    /// </summary>
    public double MeanObjective { get; }

    /// <summary>
    /// Minimum objective value at this rung.
    /// </summary>
    public double MinObjective { get; }

    /// <summary>
    /// Maximum objective value at this rung.
    /// </summary>
    public double MaxObjective { get; }

    /// <summary>
    /// Initializes a new RungStatistics.
    /// </summary>
    public RungStatistics(int resource, int count, double meanObjective, double minObjective, double maxObjective)
    {
        Resource = resource;
        Count = count;
        MeanObjective = meanObjective;
        MinObjective = minObjective;
        MaxObjective = maxObjective;
    }

    /// <inheritdoc />
    public override string ToString()
    {
        return $"Rung {Resource}: {Count} configs, mean={MeanObjective:F4}, range=[{MinObjective:F4}, {MaxObjective:F4}]";
    }
}
