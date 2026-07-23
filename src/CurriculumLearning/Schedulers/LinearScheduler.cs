using AiDotNet.CurriculumLearning.Interfaces;

namespace AiDotNet.CurriculumLearning.Schedulers;

/// <summary>
/// Curriculum scheduler with linear progression from easy to hard samples.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This scheduler increases the data fraction linearly over
/// training epochs. It starts with easy samples and gradually includes more difficult
/// samples at a constant rate.</para>
///
/// <para><b>Progression Pattern:</b></para>
/// <code>
/// fraction(epoch) = min_fraction + (epoch / total_epochs) * (max_fraction - min_fraction)
/// </code>
///
/// <para><b>Example:</b> With minFraction=0.2 and maxFraction=1.0 over 10 epochs:</para>
/// <list type="bullet">
/// <item><description>Epoch 0: 20% easiest samples</description></item>
/// <item><description>Epoch 5: 60% easiest samples</description></item>
/// <item><description>Epoch 10: 100% of all samples</description></item>
/// </list>
///
/// <para><b>Best For:</b></para>
/// <list type="bullet">
/// <item><description>Tasks with evenly distributed difficulty</description></item>
/// <item><description>When constant progression rate is desired</description></item>
/// <item><description>Simple, predictable curriculum schedules</description></item>
/// </list>
/// </remarks>
public class LinearScheduler<T> : CurriculumSchedulerBase<T>
{
    /// <summary>
    /// Gets the name of this scheduler.
    /// </summary>
    public override string Name => "Linear";

    /// <summary>
    /// Initializes a new instance of the <see cref="LinearScheduler{T}"/> class.
    /// </summary>
    /// <param name="totalEpochs">Total number of training epochs.</param>
    /// <param name="minFraction">Initial data fraction (default 0.1).</param>
    /// <param name="maxFraction">Final data fraction (default 1.0).</param>
    public LinearScheduler(
        int totalEpochs,
        T? minFraction = default,
        T? maxFraction = default)
        : base(totalEpochs, minFraction, maxFraction)
    {
    }

    /// <summary>
    /// Gets the current data fraction using linear interpolation.
    /// </summary>
    public override T GetDataFraction()
    {
        // Linear: fraction = progress (clamped to [0, 1])
        var progress = NumOps.FromDouble(
            Math.Min(1.0, (double)CurrentEpoch / Math.Max(1, TotalEpochs - 1)));

        return InterpolateFraction(progress);
    }
}

/// <summary>
/// Curriculum scheduler with exponential (slow start) progression.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This scheduler starts slowly with easy samples and
/// accelerates the addition of harder samples later in training. It follows an
/// exponential curve that reaches the maximum fraction at the end.</para>
///
/// <para><b>Progression Pattern:</b></para>
/// <code>
/// t = epoch / total_epochs
/// fraction = min + (1 - e^(-rate * t)) / (1 - e^(-rate)) * (max - min)
/// </code>
///
/// <para><b>Growth Rate Parameter:</b></para>
/// <list type="bullet">
/// <item><description>Low rate (1-2): Gradual progression</description></item>
/// <item><description>Medium rate (3-5): Balanced curve</description></item>
/// <item><description>High rate (>5): Rapid early growth, slow late growth</description></item>
/// </list>
///
/// <para><b>Best For:</b></para>
/// <list type="bullet">
/// <item><description>Tasks where early easy samples are crucial</description></item>
/// <item><description>When the model needs time on simpler patterns first</description></item>
/// <item><description>Datasets with many hard samples that require solid foundations</description></item>
/// </list>
/// </remarks>
public class ExponentialScheduler<T> : CurriculumSchedulerBase<T>
{
    private readonly double _growthRate;

    /// <summary>
    /// Gets the name of this scheduler.
    /// </summary>
    public override string Name => "Exponential";

    /// <summary>
    /// Initializes a new instance of the <see cref="ExponentialScheduler{T}"/> class.
    /// </summary>
    /// <param name="totalEpochs">Total number of training epochs.</param>
    /// <param name="growthRate">Exponential growth rate (default 3.0).</param>
    /// <param name="minFraction">Initial data fraction (default 0.1).</param>
    /// <param name="maxFraction">Final data fraction (default 1.0).</param>
    public ExponentialScheduler(
        int totalEpochs,
        double growthRate = 3.0,
        T? minFraction = default,
        T? maxFraction = default)
        : base(totalEpochs, minFraction, maxFraction)
    {
        if (growthRate <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(growthRate),
                "Growth rate must be positive.");
        }

        _growthRate = growthRate;
    }

    /// <summary>
    /// Gets the current data fraction using exponential curve.
    /// </summary>
    public override T GetDataFraction()
    {
        var t = (double)CurrentEpoch / Math.Max(1, TotalEpochs - 1);
        t = Math.Min(1.0, t);

        // Exponential curve: (1 - e^(-rate * t)) / (1 - e^(-rate))
        var numerator = 1.0 - Math.Exp(-_growthRate * t);
        var denominator = 1.0 - Math.Exp(-_growthRate);

        var progress = numerator / denominator;
        progress = Math.Min(1.0, Math.Max(0.0, progress));

        return InterpolateFraction(NumOps.FromDouble(progress));
    }

    /// <summary>
    /// Gets scheduler-specific statistics.
    /// </summary>
    public override Dictionary<string, object> GetStatistics()
    {
        var stats = base.GetStatistics();
        stats["GrowthRate"] = _growthRate;
        return stats;
    }
}

/// <summary>
/// Curriculum scheduler with discrete step-based progression.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This scheduler divides training into discrete phases,
/// with the data fraction jumping at specific epochs rather than changing continuously.</para>
///
/// <para><b>Example:</b> With 3 steps over 12 epochs:</para>
/// <list type="bullet">
/// <item><description>Epochs 0-3: 33% easiest samples</description></item>
/// <item><description>Epochs 4-7: 66% easiest samples</description></item>
/// <item><description>Epochs 8-11: 100% of all samples</description></item>
/// </list>
///
/// <para><b>Best For:</b></para>
/// <list type="bullet">
/// <item><description>Clear curriculum phases with distinct difficulty levels</description></item>
/// <item><description>When you want model to fully adapt to each phase before progressing</description></item>
/// <item><description>Educational datasets with natural difficulty tiers</description></item>
/// </list>
/// </remarks>
public class StepScheduler<T> : CurriculumSchedulerBase<T>
{
    private readonly int _numSteps;
    private readonly T[]? _customFractions;

    /// <summary>
    /// Gets the name of this scheduler.
    /// </summary>
    public override string Name => "Step";

    /// <summary>
    /// Gets the total number of phases (steps) in this scheduler.
    /// </summary>
    public override int TotalPhases => _numSteps;

    /// <summary>
    /// Initializes a new instance with uniform steps.
    /// </summary>
    /// <param name="totalEpochs">Total number of training epochs.</param>
    /// <param name="numSteps">Number of curriculum phases.</param>
    /// <param name="minFraction">Initial data fraction (default 0.1).</param>
    /// <param name="maxFraction">Final data fraction (default 1.0).</param>
    public StepScheduler(
        int totalEpochs,
        int numSteps = 5,
        T? minFraction = default,
        T? maxFraction = default)
        : base(totalEpochs, minFraction, maxFraction, numSteps)
    {
        if (numSteps <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(numSteps),
                "Number of steps must be positive.");
        }

        _numSteps = numSteps;
        _customFractions = null;
    }

    /// <summary>
    /// Initializes a new instance with custom step fractions.
    /// </summary>
    /// <param name="totalEpochs">Total number of training epochs.</param>
    /// <param name="stepFractions">Custom data fractions for each step.</param>
    public StepScheduler(int totalEpochs, IEnumerable<T> stepFractions)
        : base(totalEpochs)
    {
        if (stepFractions is null) throw new ArgumentNullException(nameof(stepFractions));

        _customFractions = stepFractions.ToArray();

        if (_customFractions.Length == 0)
        {
            throw new ArgumentException(
                "Step fractions cannot be empty.", nameof(stepFractions));
        }

        _numSteps = _customFractions.Length;

        // Validate fractions are in [0, 1] and non-decreasing
        var prev = NumOps.Zero;
        foreach (var fraction in _customFractions)
        {
            if (NumOps.Compare(fraction, NumOps.Zero) < 0 ||
                NumOps.Compare(fraction, NumOps.One) > 0)
            {
                throw new ArgumentException(
                    "All step fractions must be between 0 and 1.");
            }

            if (NumOps.Compare(fraction, prev) < 0)
            {
                throw new ArgumentException(
                    "Step fractions must be non-decreasing.");
            }

            prev = fraction;
        }
    }

    /// <summary>
    /// Gets the current data fraction based on the current step.
    /// </summary>
    public override T GetDataFraction()
    {
        // Determine which step we're in
        var epochsPerStep = (double)TotalEpochs / _numSteps;
        var currentStep = (int)Math.Floor(CurrentEpoch / epochsPerStep);
        currentStep = Math.Min(currentStep, _numSteps - 1);

        if (_customFractions != null)
        {
            return _customFractions[currentStep];
        }

        // Uniform steps: calculate fraction for this step
        var stepFraction = (double)(currentStep + 1) / _numSteps;
        return InterpolateFraction(NumOps.FromDouble(stepFraction));
    }

    /// <summary>
    /// Gets scheduler-specific statistics.
    /// </summary>
    public override Dictionary<string, object> GetStatistics()
    {
        var stats = base.GetStatistics();
        stats["NumSteps"] = _numSteps;

        var epochsPerStep = (double)TotalEpochs / _numSteps;
        var currentStep = (int)Math.Floor(CurrentEpoch / epochsPerStep);
        stats["CurrentStep"] = Math.Min(currentStep, _numSteps - 1);

        if (_customFractions != null)
        {
            stats["CustomFractions"] = _customFractions.Select(f => NumOps.ToDouble(f)).ToArray();
        }

        return stats;
    }
}

/// <summary>
/// Curriculum scheduler with polynomial progression curve.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This scheduler uses a polynomial curve to control
/// progression speed. The power parameter determines the curve shape.</para>
///
/// <para><b>Power Parameter Effects:</b></para>
/// <list type="bullet">
/// <item><description>Power &lt; 1: Fast start, slow finish (concave curve)</description></item>
/// <item><description>Power = 1: Linear progression</description></item>
/// <item><description>Power &gt; 1: Slow start, fast finish (convex curve)</description></item>
/// </list>
/// </remarks>
public class PolynomialScheduler<T> : CurriculumSchedulerBase<T>
{
    private readonly double _power;

    /// <summary>
    /// Gets the name of this scheduler.
    /// </summary>
    public override string Name => $"Polynomial_{_power:F1}";

    /// <summary>
    /// Initializes a new instance of the <see cref="PolynomialScheduler{T}"/> class.
    /// </summary>
    /// <param name="totalEpochs">Total number of training epochs.</param>
    /// <param name="power">Polynomial power (default 2.0 for quadratic).</param>
    /// <param name="minFraction">Initial data fraction (default 0.1).</param>
    /// <param name="maxFraction">Final data fraction (default 1.0).</param>
    public PolynomialScheduler(
        int totalEpochs,
        double power = 2.0,
        T? minFraction = default,
        T? maxFraction = default)
        : base(totalEpochs, minFraction, maxFraction)
    {
        if (power <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(power),
                "Power must be positive.");
        }

        _power = power;
    }

    /// <summary>
    /// Gets the current data fraction using polynomial curve.
    /// </summary>
    public override T GetDataFraction()
    {
        var t = (double)CurrentEpoch / Math.Max(1, TotalEpochs - 1);
        t = Math.Min(1.0, t);

        // Polynomial: t^power
        var progress = Math.Pow(t, _power);

        return InterpolateFraction(NumOps.FromDouble(progress));
    }

    /// <summary>
    /// Gets scheduler-specific statistics.
    /// </summary>
    public override Dictionary<string, object> GetStatistics()
    {
        var stats = base.GetStatistics();
        stats["Power"] = _power;
        return stats;
    }
}

/// <summary>
/// Curriculum scheduler with cosine annealing curve.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This scheduler follows a cosine curve, providing
/// smooth progression that starts slow, accelerates in the middle, and slows
/// down again near the end. This can help with convergence stability.</para>
///
/// <para><b>Progression Pattern:</b></para>
/// <code>
/// fraction = 0.5 * (1 - cos(π * progress))
/// </code>
/// </remarks>
public class CosineScheduler<T> : CurriculumSchedulerBase<T>
{
    /// <summary>
    /// Gets the name of this scheduler.
    /// </summary>
    public override string Name => "Cosine";

    /// <summary>
    /// Initializes a new instance of the <see cref="CosineScheduler{T}"/> class.
    /// </summary>
    /// <param name="totalEpochs">Total number of training epochs.</param>
    /// <param name="minFraction">Initial data fraction (default 0.1).</param>
    /// <param name="maxFraction">Final data fraction (default 1.0).</param>
    public CosineScheduler(
        int totalEpochs,
        T? minFraction = default,
        T? maxFraction = default)
        : base(totalEpochs, minFraction, maxFraction)
    {
    }

    /// <summary>
    /// Gets the current data fraction using cosine annealing.
    /// </summary>
    public override T GetDataFraction()
    {
        var t = (double)CurrentEpoch / Math.Max(1, TotalEpochs - 1);
        t = Math.Min(1.0, t);

        // Cosine annealing: 0.5 * (1 - cos(π * t))
        var progress = 0.5 * (1.0 - Math.Cos(Math.PI * t));

        return InterpolateFraction(NumOps.FromDouble(progress));
    }
}
