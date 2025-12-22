using AiDotNet.CurriculumLearning.Interfaces;

namespace AiDotNet.CurriculumLearning.Schedulers;

/// <summary>
/// Self-paced curriculum scheduler that adapts sample selection based on model performance.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Unlike fixed schedulers, self-paced learning dynamically
/// selects samples based on the model's current ability. Samples with loss below a
/// threshold are considered "easy enough" and included in training. The threshold
/// increases over time to include progressively harder samples.</para>
///
/// <para><b>How It Works:</b></para>
/// <list type="number">
/// <item><description>Calculate per-sample losses from the model</description></item>
/// <item><description>Include samples where loss &lt; current threshold (λ)</description></item>
/// <item><description>Increase λ each epoch to include more samples</description></item>
/// <item><description>Samples the model struggles with are added later</description></item>
/// </list>
///
/// <para><b>Self-Paced Regularizers:</b></para>
/// <list type="bullet">
/// <item><description><b>Hard:</b> Binary selection (loss &lt; λ)</description></item>
/// <item><description><b>Linear:</b> Weighted selection (max(0, 1 - loss/λ))</description></item>
/// <item><description><b>Mixture:</b> Soft-hard combination for smoother transitions</description></item>
/// </list>
///
/// <para><b>References:</b></para>
/// <list type="bullet">
/// <item><description>Kumar et al. "Self-Paced Learning for Latent Variable Models" (NIPS 2010)</description></item>
/// <item><description>Jiang et al. "Self-Paced Curriculum Learning" (AAAI 2015)</description></item>
/// </list>
/// </remarks>
public class SelfPacedScheduler<T> : CurriculumSchedulerBase<T>, ISelfPacedScheduler<T>
{
    private readonly SelfPaceRegularizer _regularizer;
    private readonly T _initialLambda;
    private readonly T _maxLambda;
    private T _currentLambda;
    private T _lambdaGrowthRate;
    private Vector<T>? _sampleWeights;

    /// <summary>
    /// Gets the name of this scheduler.
    /// </summary>
    public override string Name => $"SelfPaced_{_regularizer}";

    /// <summary>
    /// Gets or sets the current pace threshold (lambda).
    /// </summary>
    /// <remarks>
    /// <para>The pace parameter controls which samples are included in training.
    /// Samples with loss below this threshold are considered learnable.</para>
    /// </remarks>
    public T PaceParameter
    {
        get => _currentLambda;
        set
        {
            if (NumOps.Compare(value, NumOps.Zero) <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(value),
                    "Pace parameter must be positive.");
            }
            _currentLambda = value;
        }
    }

    /// <summary>
    /// Gets or sets the growth rate for the pace parameter.
    /// </summary>
    /// <remarks>
    /// <para>Controls how quickly the pace parameter increases each epoch,
    /// determining the rate at which harder samples are introduced.</para>
    /// </remarks>
    public T GrowthRate
    {
        get => _lambdaGrowthRate;
        set
        {
            if (NumOps.Compare(value, NumOps.Zero) < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(value),
                    "Growth rate cannot be negative.");
            }
            _lambdaGrowthRate = value;
        }
    }

    /// <summary>
    /// Gets the current pace threshold (lambda). Alias for PaceParameter.
    /// </summary>
    public T CurrentThreshold => _currentLambda;

    /// <summary>
    /// Gets sample weights from the last selection.
    /// </summary>
    public Vector<T>? SampleWeights => _sampleWeights;

    /// <summary>
    /// Initializes a new instance of the <see cref="SelfPacedScheduler{T}"/> class.
    /// </summary>
    /// <param name="totalEpochs">Total number of training epochs.</param>
    /// <param name="initialLambda">Initial pace threshold (default 0.1).</param>
    /// <param name="lambdaGrowthRate">How much to increase lambda each epoch (default 0.1).</param>
    /// <param name="maxLambda">Maximum lambda value (default 10.0).</param>
    /// <param name="regularizer">Self-pace regularizer type.</param>
    public SelfPacedScheduler(
        int totalEpochs,
        T? initialLambda = default,
        T? lambdaGrowthRate = default,
        T? maxLambda = default,
        SelfPaceRegularizer regularizer = SelfPaceRegularizer.Hard)
        : base(totalEpochs)
    {
        _initialLambda = initialLambda ?? NumOps.FromDouble(0.1);
        _lambdaGrowthRate = lambdaGrowthRate ?? NumOps.FromDouble(0.1);
        _maxLambda = maxLambda ?? NumOps.FromDouble(10.0);
        _currentLambda = _initialLambda;
        _regularizer = regularizer;

        if (NumOps.Compare(_currentLambda, NumOps.Zero) <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(initialLambda),
                "Initial lambda must be positive.");
        }
    }

    /// <summary>
    /// Gets the current data fraction (estimated from lambda progression).
    /// </summary>
    public override T GetDataFraction()
    {
        // For self-paced, this is an estimate based on lambda progression
        var progress = NumOps.FromDouble(
            Math.Min(1.0, (double)CurrentEpoch / TotalEpochs));
        return InterpolateFraction(progress);
    }

    /// <summary>
    /// Computes sample weights based on current losses and pace threshold.
    /// </summary>
    /// <param name="losses">Per-sample losses from the model.</param>
    /// <returns>Vector of weights for all samples (0 for excluded samples).</returns>
    /// <remarks>
    /// <para>This is the primary method for self-paced sample selection.
    /// Samples with loss below the current pace parameter receive positive weights,
    /// while samples above the threshold receive zero weight.</para>
    /// </remarks>
    public Vector<T> ComputeSampleWeights(Vector<T> losses)
    {
        if (losses is null) throw new ArgumentNullException(nameof(losses));

        var weights = new T[losses.Length];

        for (int i = 0; i < losses.Length; i++)
        {
            var (isSelected, weight) = ComputeSampleWeight(losses[i]);
            weights[i] = isSelected ? weight : NumOps.Zero;
        }

        _sampleWeights = new Vector<T>(weights);
        return _sampleWeights;
    }

    /// <summary>
    /// Selects samples based on current losses and pace threshold.
    /// </summary>
    /// <param name="losses">Per-sample losses from the model.</param>
    /// <returns>Indices and weights for selected samples.</returns>
    public (int[] Indices, Vector<T> Weights) SelectSamplesWithWeights(Vector<T> losses)
    {
        if (losses is null) throw new ArgumentNullException(nameof(losses));

        var selectedIndices = new List<int>();
        var weights = new List<T>();

        for (int i = 0; i < losses.Length; i++)
        {
            var (isSelected, weight) = ComputeSampleWeight(losses[i]);

            if (isSelected)
            {
                selectedIndices.Add(i);
                weights.Add(weight);
            }
        }

        // Ensure at least some samples are selected
        if (selectedIndices.Count == 0)
        {
            // Select the easiest sample if none qualify
            int easiest = 0;
            var minLoss = losses[0];
            for (int i = 1; i < losses.Length; i++)
            {
                if (NumOps.Compare(losses[i], minLoss) < 0)
                {
                    minLoss = losses[i];
                    easiest = i;
                }
            }
            selectedIndices.Add(easiest);
            weights.Add(NumOps.One);
        }

        _sampleWeights = new Vector<T>(weights);
        return (selectedIndices.ToArray(), _sampleWeights);
    }

    /// <summary>
    /// Selects samples based on difficulty scores.
    /// </summary>
    /// <param name="sortedIndices">Indices sorted by difficulty (easy to hard).</param>
    /// <param name="totalSamples">Total number of samples.</param>
    /// <returns>Indices of selected samples.</returns>
    public override int[] GetCurrentIndices(int[] sortedIndices, int totalSamples)
    {
        // For self-paced, we use the base implementation which relies on GetDataFraction
        return base.GetCurrentIndices(sortedIndices, totalSamples);
    }

    /// <summary>
    /// Updates lambda threshold based on epoch metrics.
    /// </summary>
    /// <param name="epochMetrics">Metrics from the completed epoch.</param>
    /// <returns>True if the phase should advance, false otherwise.</returns>
    public override bool StepEpoch(CurriculumEpochMetrics<T> epochMetrics)
    {
        var phaseAdvanced = base.StepEpoch(epochMetrics);

        // Increase lambda (include harder samples over time)
        _currentLambda = NumOps.Add(_currentLambda, _lambdaGrowthRate);

        // Cap at maximum
        if (NumOps.Compare(_currentLambda, _maxLambda) > 0)
        {
            _currentLambda = _maxLambda;
        }

        return phaseAdvanced;
    }

    /// <summary>
    /// Computes the weight for a sample based on its loss and current lambda.
    /// </summary>
    private (bool IsSelected, T Weight) ComputeSampleWeight(T loss)
    {
        return _regularizer switch
        {
            SelfPaceRegularizer.Hard => ComputeHardWeight(loss),
            SelfPaceRegularizer.Linear => ComputeLinearWeight(loss),
            SelfPaceRegularizer.Mixture => ComputeMixtureWeight(loss),
            SelfPaceRegularizer.Logarithmic => ComputeLogarithmicWeight(loss),
            _ => ComputeHardWeight(loss)
        };
    }

    /// <summary>
    /// Hard (binary) self-pace regularizer.
    /// </summary>
    private (bool, T) ComputeHardWeight(T loss)
    {
        // v = 1 if loss < lambda, else 0
        var isSelected = NumOps.Compare(loss, _currentLambda) < 0;
        return (isSelected, isSelected ? NumOps.One : NumOps.Zero);
    }

    /// <summary>
    /// Linear soft-weighting regularizer.
    /// </summary>
    private (bool, T) ComputeLinearWeight(T loss)
    {
        // v = max(0, 1 - loss/lambda)
        var ratio = NumOps.Divide(loss, _currentLambda);
        var weight = NumOps.Subtract(NumOps.One, ratio);

        if (NumOps.Compare(weight, NumOps.Zero) <= 0)
        {
            return (false, NumOps.Zero);
        }

        return (true, weight);
    }

    /// <summary>
    /// Mixture (soft-hard) regularizer.
    /// </summary>
    private (bool, T) ComputeMixtureWeight(T loss)
    {
        // Combination of hard and linear weighting
        var zeta = NumOps.FromDouble(0.5); // Soft-hard mixture parameter

        if (NumOps.Compare(loss, _currentLambda) >= 0)
        {
            return (false, NumOps.Zero);
        }

        // Weight based on distance from threshold
        var ratio = NumOps.Divide(loss, _currentLambda);
        var linearPart = NumOps.Subtract(NumOps.One, ratio);
        var weight = NumOps.Add(
            NumOps.Multiply(zeta, NumOps.One),
            NumOps.Multiply(NumOps.Subtract(NumOps.One, zeta), linearPart));

        return (true, weight);
    }

    /// <summary>
    /// Logarithmic regularizer for smoother transitions.
    /// </summary>
    private (bool, T) ComputeLogarithmicWeight(T loss)
    {
        // v = max(0, log(lambda) - log(loss + epsilon)) / log(lambda)
        var epsilon = NumOps.FromDouble(1e-10);
        var logLambda = NumOps.Log(NumOps.Add(_currentLambda, epsilon));
        var logLoss = NumOps.Log(NumOps.Add(loss, epsilon));

        var diff = NumOps.Subtract(logLambda, logLoss);

        if (NumOps.Compare(diff, NumOps.Zero) <= 0)
        {
            return (false, NumOps.Zero);
        }

        var weight = NumOps.Divide(diff, logLambda);
        return (true, weight);
    }

    /// <summary>
    /// Resets the scheduler to initial state.
    /// </summary>
    public override void Reset()
    {
        base.Reset();
        _currentLambda = _initialLambda;
        _sampleWeights = null;
    }

    /// <summary>
    /// Gets scheduler-specific statistics.
    /// </summary>
    public override Dictionary<string, object> GetStatistics()
    {
        var stats = base.GetStatistics();
        stats["CurrentLambda"] = NumOps.ToDouble(_currentLambda);
        stats["MaxLambda"] = NumOps.ToDouble(_maxLambda);
        stats["LambdaGrowthRate"] = NumOps.ToDouble(_lambdaGrowthRate);
        stats["Regularizer"] = _regularizer.ToString();

        if (_sampleWeights != null)
        {
            stats["SelectedSamples"] = _sampleWeights.Length;
        }

        return stats;
    }
}

/// <summary>
/// Type of self-pace regularizer for sample weighting.
/// </summary>
public enum SelfPaceRegularizer
{
    /// <summary>
    /// Binary selection: include if loss &lt; lambda.
    /// </summary>
    Hard,

    /// <summary>
    /// Linear soft weighting: weight = max(0, 1 - loss/lambda).
    /// </summary>
    Linear,

    /// <summary>
    /// Mixture of hard and linear for balanced selection.
    /// </summary>
    Mixture,

    /// <summary>
    /// Logarithmic weighting for smoother transitions.
    /// </summary>
    Logarithmic
}
