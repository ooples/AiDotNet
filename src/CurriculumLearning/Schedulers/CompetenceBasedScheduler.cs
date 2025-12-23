using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.CurriculumLearning.Interfaces;
using AiDotNet.Interfaces;

namespace AiDotNet.CurriculumLearning.Schedulers;

/// <summary>
/// Curriculum scheduler that advances based on model competence/mastery.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This scheduler tracks how well the model is learning
/// and only advances to harder content when the model has "mastered" the current
/// difficulty level. Think of it like a tutor who won't move to harder problems
/// until you've shown you understand the current ones.</para>
///
/// <para><b>How It Works:</b></para>
/// <list type="number">
/// <item><description>Monitor model performance metrics (accuracy, loss improvement)</description></item>
/// <item><description>Compute a competence score based on these metrics</description></item>
/// <item><description>When competence exceeds threshold, advance to next phase</description></item>
/// <item><description>Include more difficult samples in each subsequent phase</description></item>
/// </list>
///
/// <para><b>Competence Metrics:</b></para>
/// <list type="bullet">
/// <item><description><b>Accuracy-Based:</b> Uses validation/training accuracy as competence</description></item>
/// <item><description><b>Loss-Based:</b> Uses loss reduction rate as competence indicator</description></item>
/// <item><description><b>Plateau-Based:</b> Advances when loss stops improving (mastery plateau)</description></item>
/// <item><description><b>Combined:</b> Weighted combination of multiple metrics</description></item>
/// </list>
///
/// <para><b>References:</b></para>
/// <list type="bullet">
/// <item><description>Platanios et al. "Competence-based Curriculum Learning" (NAACL 2019)</description></item>
/// </list>
/// </remarks>
public class CompetenceBasedScheduler<T> : CurriculumSchedulerBase<T>, ICompetenceBasedScheduler<T>
{
    private readonly CompetenceMetricType _metricType;
    private readonly int _patienceEpochs;
    private readonly T _minImprovement;
    private T _currentCompetence;
    private T _competenceThreshold;
    private T _bestLoss;
    private int _epochsWithoutImprovement;
    private readonly List<T> _competenceHistory;
    private readonly T _smoothingFactor;

    /// <summary>
    /// Gets the name of this scheduler.
    /// </summary>
    public override string Name => $"CompetenceBased_{_metricType}";

    /// <summary>
    /// Gets the current competence level of the model.
    /// </summary>
    /// <remarks>
    /// <para>Competence is a value typically between 0 and 1, where higher values
    /// indicate better mastery of the current curriculum content.</para>
    /// </remarks>
    public T CurrentCompetence => _currentCompetence;

    /// <summary>
    /// Gets or sets the competence threshold required to advance phases.
    /// </summary>
    /// <remarks>
    /// <para>When the model's competence exceeds this threshold, it is considered
    /// to have mastered the current content and will advance to the next phase.</para>
    /// </remarks>
    public T CompetenceThreshold
    {
        get => _competenceThreshold;
        set
        {
            if (NumOps.Compare(value, NumOps.Zero) <= 0 ||
                NumOps.Compare(value, NumOps.One) > 0)
            {
                throw new ArgumentOutOfRangeException(nameof(value),
                    "Competence threshold must be in (0, 1].");
            }
            _competenceThreshold = value;
        }
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CompetenceBasedScheduler{T}"/> class.
    /// </summary>
    /// <param name="totalEpochs">Total number of training epochs.</param>
    /// <param name="competenceThreshold">Threshold to advance to next phase (default 0.9).</param>
    /// <param name="metricType">Type of competence metric to use.</param>
    /// <param name="patienceEpochs">Epochs without improvement before advancing (for plateau mode).</param>
    /// <param name="minImprovement">Minimum improvement to reset patience counter.</param>
    /// <param name="smoothingFactor">Exponential smoothing factor for competence updates.</param>
    /// <param name="minFraction">Initial data fraction (default 0.1).</param>
    /// <param name="maxFraction">Final data fraction (default 1.0).</param>
    /// <param name="totalPhases">Number of curriculum phases.</param>
    public CompetenceBasedScheduler(
        int totalEpochs,
        T? competenceThreshold = default,
        CompetenceMetricType metricType = CompetenceMetricType.Combined,
        int patienceEpochs = 5,
        T? minImprovement = default,
        T? smoothingFactor = default,
        T? minFraction = default,
        T? maxFraction = default,
        int? totalPhases = null)
        : base(totalEpochs, minFraction, maxFraction, totalPhases ?? 5)
    {
        if (patienceEpochs <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(patienceEpochs),
                "Patience epochs must be positive.");
        }

        _competenceThreshold = competenceThreshold ?? NumOps.FromDouble(0.9);
        _metricType = metricType;
        _patienceEpochs = patienceEpochs;
        _minImprovement = minImprovement ?? NumOps.FromDouble(0.001);
        _smoothingFactor = smoothingFactor ?? NumOps.FromDouble(0.3);

        _currentCompetence = NumOps.Zero;
        _bestLoss = NumOps.FromDouble(double.MaxValue);
        _epochsWithoutImprovement = 0;
        _competenceHistory = new List<T>();

        ValidateThreshold(_competenceThreshold);
    }

    /// <summary>
    /// Gets the current data fraction based on competence-driven phase.
    /// </summary>
    public override T GetDataFraction()
    {
        // Use phase-based fraction calculation
        var phaseProgress = (double)CurrentPhaseNumber / Math.Max(1, TotalPhases - 1);
        phaseProgress = Math.Min(1.0, phaseProgress);
        return InterpolateFraction(NumOps.FromDouble(phaseProgress));
    }

    /// <summary>
    /// Updates the competence estimate based on model performance.
    /// </summary>
    /// <param name="metrics">Performance metrics from current epoch.</param>
    /// <remarks>
    /// <para>The competence update strategy depends on the configured metric type:</para>
    /// <list type="bullet">
    /// <item><description>Accuracy-based: Uses validation accuracy directly</description></item>
    /// <item><description>Loss-based: Converts loss improvement to competence</description></item>
    /// <item><description>Plateau-based: Tracks epochs without improvement</description></item>
    /// <item><description>Combined: Weighted average of multiple metrics</description></item>
    /// </list>
    /// </remarks>
    public void UpdateCompetence(CurriculumEpochMetrics<T> metrics)
    {
        if (metrics is null) throw new ArgumentNullException(nameof(metrics));

        var newCompetence = _metricType switch
        {
            CompetenceMetricType.Accuracy => ComputeAccuracyCompetence(metrics),
            CompetenceMetricType.LossReduction => ComputeLossReductionCompetence(metrics),
            CompetenceMetricType.Plateau => ComputePlateauCompetence(metrics),
            CompetenceMetricType.Combined => ComputeCombinedCompetence(metrics),
            _ => ComputeCombinedCompetence(metrics)
        };

        // Exponential moving average for smoother competence updates
        _currentCompetence = NumOps.Add(
            NumOps.Multiply(_smoothingFactor, newCompetence),
            NumOps.Multiply(NumOps.Subtract(NumOps.One, _smoothingFactor), _currentCompetence));

        // Clamp to [0, 1]
        if (NumOps.Compare(_currentCompetence, NumOps.Zero) < 0)
        {
            _currentCompetence = NumOps.Zero;
        }
        if (NumOps.Compare(_currentCompetence, NumOps.One) > 0)
        {
            _currentCompetence = NumOps.One;
        }

        _competenceHistory.Add(_currentCompetence);
    }

    /// <summary>
    /// Gets whether the model has mastered the current curriculum content.
    /// </summary>
    /// <returns>True if competence exceeds threshold, false otherwise.</returns>
    public bool HasMasteredCurrentContent()
    {
        return NumOps.Compare(_currentCompetence, _competenceThreshold) >= 0;
    }

    /// <summary>
    /// Updates the scheduler after an epoch, potentially advancing phases.
    /// </summary>
    /// <param name="epochMetrics">Metrics from the completed epoch.</param>
    /// <returns>True if the phase advanced, false otherwise.</returns>
    public override bool StepEpoch(CurriculumEpochMetrics<T> epochMetrics)
    {
        // Update competence based on metrics
        UpdateCompetence(epochMetrics);

        // Increment epoch counter
        CurrentEpoch++;

        // Check if we should advance phase
        if (HasMasteredCurrentContent())
        {
            var advanced = AdvancePhase();
            if (advanced)
            {
                // Reset competence tracking for new phase
                ResetForNewPhase();
            }
            return advanced;
        }

        return false;
    }

    /// <summary>
    /// Resets the scheduler to initial state.
    /// </summary>
    public override void Reset()
    {
        base.Reset();
        _currentCompetence = NumOps.Zero;
        _bestLoss = NumOps.FromDouble(double.MaxValue);
        _epochsWithoutImprovement = 0;
        _competenceHistory.Clear();
    }

    /// <summary>
    /// Gets scheduler-specific statistics.
    /// </summary>
    public override Dictionary<string, object> GetStatistics()
    {
        var stats = base.GetStatistics();
        stats["CurrentCompetence"] = NumOps.ToDouble(_currentCompetence);
        stats["CompetenceThreshold"] = NumOps.ToDouble(_competenceThreshold);
        stats["MetricType"] = _metricType.ToString();
        stats["EpochsWithoutImprovement"] = _epochsWithoutImprovement;
        stats["BestLoss"] = NumOps.ToDouble(_bestLoss);
        stats["CompetenceHistoryLength"] = _competenceHistory.Count;

        if (_competenceHistory.Count > 0)
        {
            stats["AverageCompetence"] = _competenceHistory.Average(c => NumOps.ToDouble(c));
        }

        return stats;
    }

    /// <summary>
    /// Computes competence based on accuracy metrics.
    /// </summary>
    private T ComputeAccuracyCompetence(CurriculumEpochMetrics<T> metrics)
    {
        // Prefer validation accuracy, fall back to training accuracy
        if (metrics.ValidationAccuracy != null)
        {
            return metrics.ValidationAccuracy;
        }

        if (metrics.TrainingAccuracy != null)
        {
            return metrics.TrainingAccuracy;
        }

        // If no accuracy available, use loss-based competence
        return ComputeLossReductionCompetence(metrics);
    }

    /// <summary>
    /// Computes competence based on loss reduction.
    /// </summary>
    private T ComputeLossReductionCompetence(CurriculumEpochMetrics<T> metrics)
    {
        var currentLoss = metrics.ValidationLoss ?? metrics.TrainingLoss;

        // Check if this is an improvement
        if (NumOps.Compare(currentLoss, _bestLoss) < 0)
        {
            var improvement = NumOps.Subtract(_bestLoss, currentLoss);

            if (NumOps.Compare(improvement, _minImprovement) > 0)
            {
                _bestLoss = currentLoss;
                _epochsWithoutImprovement = 0;

                // Competence increases with improvement
                // Convert loss reduction to competence (inverse relationship)
                var improvementRatio = NumOps.Divide(improvement,
                    NumOps.Add(_bestLoss, NumOps.FromDouble(1e-10)));
                return NumOps.Add(_currentCompetence, improvementRatio);
            }
        }

        _epochsWithoutImprovement++;

        // Competence based on how close we are to zero loss
        // Higher competence = lower loss
        var epsilon = NumOps.FromDouble(1e-10);
        var competence = NumOps.Divide(NumOps.One,
            NumOps.Add(NumOps.One, currentLoss));

        return competence;
    }

    /// <summary>
    /// Computes competence based on learning plateau detection.
    /// </summary>
    private T ComputePlateauCompetence(CurriculumEpochMetrics<T> metrics)
    {
        var currentLoss = metrics.ValidationLoss ?? metrics.TrainingLoss;

        // Check for improvement
        var improvement = NumOps.Subtract(_bestLoss, currentLoss);

        if (NumOps.Compare(improvement, _minImprovement) > 0)
        {
            _bestLoss = currentLoss;
            _epochsWithoutImprovement = 0;
        }
        else
        {
            _epochsWithoutImprovement++;
        }

        // Competence increases as we approach plateau (mastery)
        var plateauProgress = (double)_epochsWithoutImprovement / _patienceEpochs;
        plateauProgress = Math.Min(1.0, plateauProgress);

        return NumOps.FromDouble(plateauProgress);
    }

    /// <summary>
    /// Computes competence as a combination of multiple metrics.
    /// </summary>
    private T ComputeCombinedCompetence(CurriculumEpochMetrics<T> metrics)
    {
        var components = new List<T>();

        // Add accuracy component if available
        if (metrics.ValidationAccuracy != null)
        {
            components.Add(metrics.ValidationAccuracy);
        }
        else if (metrics.TrainingAccuracy != null)
        {
            // Discount training accuracy slightly (potential overfitting)
            components.Add(NumOps.Multiply(metrics.TrainingAccuracy, NumOps.FromDouble(0.8)));
        }

        // Add loss-based component
        var lossCompetence = ComputeLossReductionCompetence(metrics);
        components.Add(lossCompetence);

        // Add improvement indicator
        if (metrics.Improved)
        {
            components.Add(NumOps.FromDouble(0.8));
        }
        else
        {
            // Slight competence boost for stability (not regressing)
            components.Add(NumOps.FromDouble(0.3));
        }

        // Average all components
        if (components.Count == 0)
        {
            return NumOps.Zero;
        }

        var sum = NumOps.Zero;
        foreach (var component in components)
        {
            sum = NumOps.Add(sum, component);
        }

        return NumOps.Divide(sum, NumOps.FromDouble(components.Count));
    }

    /// <summary>
    /// Resets tracking metrics for a new phase.
    /// </summary>
    private void ResetForNewPhase()
    {
        // Keep some competence momentum, but reduce threshold expectations
        _currentCompetence = NumOps.Multiply(_currentCompetence, NumOps.FromDouble(0.5));
        _epochsWithoutImprovement = 0;
        // Don't reset best loss - harder samples will naturally have higher loss
    }

    /// <summary>
    /// Validates the competence threshold value.
    /// </summary>
    private void ValidateThreshold(T threshold)
    {
        if (NumOps.Compare(threshold, NumOps.Zero) <= 0 ||
            NumOps.Compare(threshold, NumOps.One) > 0)
        {
            throw new ArgumentOutOfRangeException(nameof(threshold),
                "Competence threshold must be in (0, 1].");
        }
    }
}

/// <summary>
/// Type of competence metric for curriculum advancement.
/// </summary>
public enum CompetenceMetricType
{
    /// <summary>
    /// Uses accuracy (validation preferred) as competence measure.
    /// </summary>
    Accuracy,

    /// <summary>
    /// Uses rate of loss reduction as competence measure.
    /// </summary>
    LossReduction,

    /// <summary>
    /// Advances when learning plateaus (no improvement for N epochs).
    /// </summary>
    Plateau,

    /// <summary>
    /// Combines multiple metrics for robust competence estimation.
    /// </summary>
    Combined
}
