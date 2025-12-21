using AiDotNet.Helpers;
using AiDotNet.Models;

namespace AiDotNet.HyperparameterOptimization;

/// <summary>
/// Provides trial pruning functionality for hyperparameter optimization.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> Trial pruning allows early termination of unpromising trials:
/// - During training, periodically report intermediate results
/// - Pruner compares against historical data from other trials
/// - If the current trial is clearly worse, it gets pruned (stopped early)
/// - This saves computational resources for more promising trials
///
/// Key concepts:
/// - Intermediate Value: A performance metric reported during training
/// - Step: The training progress when the value was reported
/// - Pruning: Terminating a trial that's unlikely to improve
///
/// Trial pruning is especially useful with Bayesian optimization and Hyperband
/// where many configurations are evaluated.
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
public class TrialPruner<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly PruningStrategy _strategy;
    private readonly double _percentile;
    private readonly int _warmupSteps;
    private readonly int _checkInterval;
    private readonly bool _maximize;

    // Track intermediate values from all trials
    private readonly Dictionary<string, List<(int step, double value)>> _trialHistory;
    private readonly object _lock = new();

    /// <summary>
    /// Initializes a new instance of the TrialPruner class.
    /// </summary>
    /// <param name="maximize">Whether higher values are better (true) or lower (false).</param>
    /// <param name="strategy">Pruning strategy to use.</param>
    /// <param name="percentile">Percentile threshold for pruning (0-100).</param>
    /// <param name="warmupSteps">Minimum steps before pruning can occur.</param>
    /// <param name="checkInterval">Steps between pruning checks.</param>
    public TrialPruner(
        bool maximize = true,
        PruningStrategy strategy = PruningStrategy.MedianPruning,
        double percentile = 50.0,
        int warmupSteps = 3,
        int checkInterval = 1)
    {
        if (percentile <= 0 || percentile > 100)
            throw new ArgumentException("Percentile must be in (0, 100].", nameof(percentile));
        if (warmupSteps < 0)
            throw new ArgumentException("warmupSteps must be non-negative.", nameof(warmupSteps));
        if (checkInterval < 1)
            throw new ArgumentException("checkInterval must be at least 1.", nameof(checkInterval));

        _numOps = MathHelper.GetNumericOperations<T>();
        _maximize = maximize;
        _strategy = strategy;
        _percentile = percentile;
        _warmupSteps = warmupSteps;
        _checkInterval = checkInterval;
        _trialHistory = new Dictionary<string, List<(int, double)>>();
    }

    /// <summary>
    /// Reports an intermediate value and checks if the trial should be pruned.
    /// </summary>
    /// <param name="trial">The trial being evaluated.</param>
    /// <param name="step">The current training step.</param>
    /// <param name="value">The intermediate value to report.</param>
    /// <returns>True if the trial should be pruned, false to continue.</returns>
    public bool ReportAndCheckPrune(HyperparameterTrial<T> trial, int step, T value)
    {
        double doubleValue = _numOps.ToDouble(value);
        return ReportAndCheckPrune(trial.TrialId, step, doubleValue);
    }

    /// <summary>
    /// Reports an intermediate value and checks if the trial should be pruned.
    /// </summary>
    /// <param name="trialId">The trial ID.</param>
    /// <param name="step">The current training step.</param>
    /// <param name="value">The intermediate value to report.</param>
    /// <returns>True if the trial should be pruned, false to continue.</returns>
    public bool ReportAndCheckPrune(string trialId, int step, double value)
    {
        lock (_lock)
        {
            // Record the intermediate value
            if (!_trialHistory.ContainsKey(trialId))
            {
                _trialHistory[trialId] = new List<(int, double)>();
            }
            _trialHistory[trialId].Add((step, value));

            // Check if we should prune
            if (step < _warmupSteps)
                return false;

            if (step % _checkInterval != 0)
                return false;

            return ShouldPrune(trialId, step, value);
        }
    }

    /// <summary>
    /// Determines if a trial should be pruned based on its intermediate value.
    /// </summary>
    private bool ShouldPrune(string trialId, int step, double value)
    {
        return _strategy switch
        {
            PruningStrategy.MedianPruning => MedianPruningCheck(trialId, step, value),
            PruningStrategy.PercentilePruning => PercentilePruningCheck(trialId, step, value),
            PruningStrategy.SuccessiveHalving => SuccessiveHalvingCheck(trialId, step, value),
            PruningStrategy.ThresholdPruning => false, // Requires explicit threshold
            _ => MedianPruningCheck(trialId, step, value)
        };
    }

    /// <summary>
    /// Median pruning: prune if below median of completed trials at this step.
    /// </summary>
    private bool MedianPruningCheck(string trialId, int step, double value)
    {
        var valuesAtStep = GetValuesAtStep(trialId, step);

        if (valuesAtStep.Count < 2)
            return false; // Not enough data

        double median = GetPercentile(valuesAtStep, 50);

        return _maximize ? value < median : value > median;
    }

    /// <summary>
    /// Percentile pruning: prune if in bottom percentile.
    /// </summary>
    private bool PercentilePruningCheck(string trialId, int step, double value)
    {
        var valuesAtStep = GetValuesAtStep(trialId, step);

        if (valuesAtStep.Count < 2)
            return false;

        double threshold = GetPercentile(valuesAtStep, _maximize ? 100 - _percentile : _percentile);

        return _maximize ? value < threshold : value > threshold;
    }

    /// <summary>
    /// Successive halving check: compare with top half of trials.
    /// </summary>
    private bool SuccessiveHalvingCheck(string trialId, int step, double value)
    {
        var valuesAtStep = GetValuesAtStep(trialId, step);

        if (valuesAtStep.Count < 4)
            return false;

        // Get top half threshold
        var sorted = _maximize
            ? valuesAtStep.OrderByDescending(v => v).ToList()
            : valuesAtStep.OrderBy(v => v).ToList();

        int topHalfCount = valuesAtStep.Count / 2;
        double threshold = sorted[topHalfCount];

        return _maximize ? value < threshold : value > threshold;
    }

    /// <summary>
    /// Gets values from other trials at the same step.
    /// </summary>
    private List<double> GetValuesAtStep(string excludeTrialId, int step)
    {
        var values = new List<double>();

        foreach (var kvp in _trialHistory)
        {
            if (kvp.Key == excludeTrialId)
                continue;

            // Find value at or near this step
            var matchingValues = kvp.Value
                .Where(h => h.step <= step)
                .OrderByDescending(h => h.step)
                .ToList();

            // Check if we found any matching values (handles step 0 with value 0.0 correctly)
            if (matchingValues.Count > 0)
            {
                values.Add(matchingValues[0].value);
            }
        }

        return values;
    }

    /// <summary>
    /// Gets the percentile value from a list.
    /// </summary>
    private static double GetPercentile(List<double> values, double percentile)
    {
        if (values.Count == 0)
            return 0;

        var sorted = values.OrderBy(v => v).ToList();
        double index = (percentile / 100.0) * (sorted.Count - 1);

        int lower = (int)Math.Floor(index);
        int upper = (int)Math.Ceiling(index);

        if (lower == upper)
            return sorted[lower];

        double fraction = index - lower;
        return sorted[lower] * (1 - fraction) + sorted[upper] * fraction;
    }

    /// <summary>
    /// Checks if a trial should be pruned based on a threshold.
    /// </summary>
    /// <param name="value">The current value.</param>
    /// <param name="threshold">The threshold to compare against.</param>
    /// <returns>True if the trial should be pruned.</returns>
    public bool CheckThreshold(double value, double threshold)
    {
        return _maximize ? value < threshold : value > threshold;
    }

    /// <summary>
    /// Marks a trial as complete (called when trial finishes without pruning).
    /// </summary>
    public void MarkComplete(string trialId)
    {
        // Currently just stores the history; could be extended
        // to track completion status for more advanced pruning
    }

    /// <summary>
    /// Resets the pruner state.
    /// </summary>
    public void Reset()
    {
        lock (_lock)
        {
            _trialHistory.Clear();
        }
    }

    /// <summary>
    /// Gets statistics about pruning decisions.
    /// </summary>
    public TrialPrunerStatistics GetStatistics()
    {
        lock (_lock)
        {
            int totalTrials = _trialHistory.Count;
            var stepCounts = _trialHistory.Values.Select(v => v.Count).ToList();

            return new TrialPrunerStatistics(
                totalTrials,
                stepCounts.Count > 0 ? stepCounts.Average() : 0,
                stepCounts.Count > 0 ? stepCounts.Max() : 0
            );
        }
    }
}

/// <summary>
/// Strategy for pruning trials.
/// </summary>
public enum PruningStrategy
{
    /// <summary>
    /// Prune trials performing below the median at each step.
    /// </summary>
    MedianPruning,

    /// <summary>
    /// Prune trials in the bottom percentile at each step.
    /// </summary>
    PercentilePruning,

    /// <summary>
    /// Successive halving: keep only the top half at each step.
    /// </summary>
    SuccessiveHalving,

    /// <summary>
    /// Prune based on explicit threshold (manual check required).
    /// </summary>
    ThresholdPruning
}

/// <summary>
/// Statistics about trial pruning.
/// </summary>
public class TrialPrunerStatistics
{
    /// <summary>
    /// Total number of trials tracked.
    /// </summary>
    public int TotalTrials { get; }

    /// <summary>
    /// Average number of steps per trial.
    /// </summary>
    public double AverageSteps { get; }

    /// <summary>
    /// Maximum steps in any trial.
    /// </summary>
    public int MaxSteps { get; }

    /// <summary>
    /// Initializes a new TrialPrunerStatistics.
    /// </summary>
    public TrialPrunerStatistics(int totalTrials, double averageSteps, int maxSteps)
    {
        TotalTrials = totalTrials;
        AverageSteps = averageSteps;
        MaxSteps = maxSteps;
    }

    /// <inheritdoc />
    public override string ToString()
    {
        return $"Trials: {TotalTrials}, Avg steps: {AverageSteps:F1}, Max steps: {MaxSteps}";
    }
}
