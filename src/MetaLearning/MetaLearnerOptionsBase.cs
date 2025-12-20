using AiDotNet.Interfaces;

namespace AiDotNet.MetaLearning;

/// <summary>
/// Base implementation of IMetaLearnerOptions with industry-standard defaults.
/// </summary>
/// <typeparam name="T">The numeric type (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// This class provides sensible defaults for meta-learning configurations based on
/// established practices from the meta-learning literature (MAML, Reptile, etc.).
/// </para>
/// <para><b>For Beginners:</b> You can use this class directly or extend it to create
/// algorithm-specific options classes. The defaults work well for most scenarios.
/// </para>
/// </remarks>
public class MetaLearnerOptionsBase<T> : IMetaLearnerOptions<T>
{
    /// <inheritdoc/>
    /// <remarks>Default: 0.01 (standard for MAML-like algorithms)</remarks>
    public double InnerLearningRate { get; set; } = 0.01;

    /// <inheritdoc/>
    /// <remarks>Default: 0.001 (typically 10x smaller than inner rate)</remarks>
    public double OuterLearningRate { get; set; } = 0.001;

    /// <inheritdoc/>
    /// <remarks>Default: 5 (balanced between 1-step and 10-step adaptation)</remarks>
    public int AdaptationSteps { get; set; } = 5;

    /// <inheritdoc/>
    /// <remarks>Default: 4 (small batch for faster iteration during development)</remarks>
    public int MetaBatchSize { get; set; } = 4;

    /// <inheritdoc/>
    /// <remarks>Default: 1000 (reasonable for initial training runs)</remarks>
    public int NumMetaIterations { get; set; } = 1000;

    /// <inheritdoc/>
    /// <remarks>Default: false (full second-order gradients for accuracy)</remarks>
    public bool UseFirstOrder { get; set; } = false;

    /// <inheritdoc/>
    /// <remarks>Default: 10.0 (prevents gradient explosion)</remarks>
    public double? GradientClipThreshold { get; set; } = 10.0;

    /// <inheritdoc/>
    /// <remarks>Default: null (non-deterministic for variety in training)</remarks>
    public int? RandomSeed { get; set; } = null;

    /// <inheritdoc/>
    /// <remarks>Default: 100 (quick evaluation during training)</remarks>
    public int EvaluationTasks { get; set; } = 100;

    /// <inheritdoc/>
    /// <remarks>Default: 100 (evaluate every 100 iterations)</remarks>
    public int EvaluationFrequency { get; set; } = 100;

    /// <inheritdoc/>
    /// <remarks>Default: true (save progress during training)</remarks>
    public bool EnableCheckpointing { get; set; } = true;

    /// <inheritdoc/>
    /// <remarks>Default: 500 (save every 500 iterations)</remarks>
    public int CheckpointFrequency { get; set; } = 500;

    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public MetaLearnerOptionsBase()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying values from another options instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    protected MetaLearnerOptionsBase(IMetaLearnerOptions<T> other)
    {
        if (other == null)
        {
            throw new ArgumentNullException(nameof(other));
        }

        InnerLearningRate = other.InnerLearningRate;
        OuterLearningRate = other.OuterLearningRate;
        AdaptationSteps = other.AdaptationSteps;
        MetaBatchSize = other.MetaBatchSize;
        NumMetaIterations = other.NumMetaIterations;
        UseFirstOrder = other.UseFirstOrder;
        GradientClipThreshold = other.GradientClipThreshold;
        RandomSeed = other.RandomSeed;
        EvaluationTasks = other.EvaluationTasks;
        EvaluationFrequency = other.EvaluationFrequency;
        EnableCheckpointing = other.EnableCheckpointing;
        CheckpointFrequency = other.CheckpointFrequency;
    }

    /// <inheritdoc/>
    public virtual bool IsValid()
    {
        if (InnerLearningRate <= 0)
        {
            return false;
        }

        if (OuterLearningRate <= 0)
        {
            return false;
        }

        if (AdaptationSteps < 1)
        {
            return false;
        }

        if (MetaBatchSize < 1)
        {
            return false;
        }

        if (NumMetaIterations < 1)
        {
            return false;
        }

        if (EvaluationTasks < 1)
        {
            return false;
        }

        if (GradientClipThreshold.HasValue && GradientClipThreshold.Value <= 0)
        {
            return false;
        }

        return true;
    }

    /// <inheritdoc/>
    public virtual IMetaLearnerOptions<T> Clone()
    {
        return new MetaLearnerOptionsBase<T>(this);
    }

    /// <summary>
    /// Creates a builder for fluent configuration.
    /// </summary>
    /// <returns>A new builder instance.</returns>
    public static MetaLearnerOptionsBuilder<T> CreateBuilder()
    {
        return new MetaLearnerOptionsBuilder<T>();
    }
}

/// <summary>
/// Fluent builder for MetaLearnerOptionsBase.
/// </summary>
/// <typeparam name="T">The numeric type (e.g., float, double).</typeparam>
public class MetaLearnerOptionsBuilder<T>
{
    private readonly MetaLearnerOptionsBase<T> _options = new();

    /// <summary>
    /// Sets the inner loop learning rate.
    /// </summary>
    public MetaLearnerOptionsBuilder<T> WithInnerLearningRate(double rate)
    {
        _options.InnerLearningRate = rate;
        return this;
    }

    /// <summary>
    /// Sets the outer loop learning rate.
    /// </summary>
    public MetaLearnerOptionsBuilder<T> WithOuterLearningRate(double rate)
    {
        _options.OuterLearningRate = rate;
        return this;
    }

    /// <summary>
    /// Sets the number of adaptation steps.
    /// </summary>
    public MetaLearnerOptionsBuilder<T> WithAdaptationSteps(int steps)
    {
        _options.AdaptationSteps = steps;
        return this;
    }

    /// <summary>
    /// Sets the meta-batch size.
    /// </summary>
    public MetaLearnerOptionsBuilder<T> WithMetaBatchSize(int size)
    {
        _options.MetaBatchSize = size;
        return this;
    }

    /// <summary>
    /// Sets the number of meta-iterations.
    /// </summary>
    public MetaLearnerOptionsBuilder<T> WithNumMetaIterations(int iterations)
    {
        _options.NumMetaIterations = iterations;
        return this;
    }

    /// <summary>
    /// Enables first-order approximation (FOMAML).
    /// </summary>
    public MetaLearnerOptionsBuilder<T> WithFirstOrder(bool useFirstOrder = true)
    {
        _options.UseFirstOrder = useFirstOrder;
        return this;
    }

    /// <summary>
    /// Sets the gradient clipping threshold.
    /// </summary>
    public MetaLearnerOptionsBuilder<T> WithGradientClipping(double? threshold)
    {
        _options.GradientClipThreshold = threshold;
        return this;
    }

    /// <summary>
    /// Sets the random seed for reproducibility.
    /// </summary>
    public MetaLearnerOptionsBuilder<T> WithRandomSeed(int? seed)
    {
        _options.RandomSeed = seed;
        return this;
    }

    /// <summary>
    /// Configures evaluation settings.
    /// </summary>
    public MetaLearnerOptionsBuilder<T> WithEvaluation(int tasks = 100, int frequency = 100)
    {
        _options.EvaluationTasks = tasks;
        _options.EvaluationFrequency = frequency;
        return this;
    }

    /// <summary>
    /// Configures checkpointing settings.
    /// </summary>
    public MetaLearnerOptionsBuilder<T> WithCheckpointing(bool enabled = true, int frequency = 500)
    {
        _options.EnableCheckpointing = enabled;
        _options.CheckpointFrequency = frequency;
        return this;
    }

    /// <summary>
    /// Builds the options instance.
    /// </summary>
    /// <returns>The configured options.</returns>
    /// <exception cref="InvalidOperationException">Thrown if validation fails.</exception>
    public MetaLearnerOptionsBase<T> Build()
    {
        if (!_options.IsValid())
        {
            throw new InvalidOperationException("Invalid meta-learner options configuration.");
        }

        return _options;
    }
}
