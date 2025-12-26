using AiDotNet.Augmentation.Interfaces;
using AiDotNet.Augmentation.Policies;

namespace AiDotNet.Augmentation.Integration;

/// <summary>
/// Configuration for training-time data augmentation.
/// </summary>
/// <remarks>
/// <para>
/// This configuration integrates data augmentation with the PredictionModelBuilder
/// training pipeline. Augmentations are applied on-the-fly during training to
/// increase data diversity and improve model generalization.
/// </para>
/// <para><b>For Beginners:</b> Data augmentation creates variations of your training
/// data (like rotating images, adding noise) to help your model learn better.
/// This configuration tells the training system what augmentations to apply.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <typeparam name="TData">The data type being augmented.</typeparam>
public class TrainingAugmentationConfiguration<T, TData>
{
    /// <summary>
    /// Gets or sets the augmentation pipeline for training.
    /// </summary>
    public IAugmentationPolicy<T, TData>? Pipeline { get; set; }

    /// <summary>
    /// Gets or sets whether augmentation is enabled.
    /// </summary>
    public bool IsEnabled { get; set; } = true;

    /// <summary>
    /// Gets or sets the random seed for reproducibility.
    /// </summary>
    public int? Seed { get; set; }

    /// <summary>
    /// Gets or sets whether to apply augmentation to the validation set.
    /// </summary>
    /// <remarks>
    /// Generally false, but can be true for certain techniques like
    /// knowledge distillation or consistency regularization.
    /// </remarks>
    public bool ApplyToValidation { get; set; }

    /// <summary>
    /// Gets or sets the number of augmented versions to generate per sample.
    /// </summary>
    /// <remarks>
    /// Setting this > 1 creates multiple augmented versions of each sample.
    /// Useful for methods like Mixup/CutMix or consistency regularization.
    /// </remarks>
    public int NumAugmentations { get; set; } = 1;

    /// <summary>
    /// Gets or sets whether to prefetch augmented data for performance.
    /// </summary>
    public bool PrefetchAugmentations { get; set; } = true;

    /// <summary>
    /// Gets or sets the number of worker threads for augmentation.
    /// </summary>
    public int NumWorkers { get; set; } = 1;

    /// <summary>
    /// Creates a default configuration with no augmentation.
    /// </summary>
    public TrainingAugmentationConfiguration()
    {
    }

    /// <summary>
    /// Creates a configuration with the specified pipeline.
    /// </summary>
    /// <param name="pipeline">The augmentation pipeline.</param>
    public TrainingAugmentationConfiguration(IAugmentationPolicy<T, TData> pipeline)
    {
        Pipeline = pipeline;
    }

    /// <summary>
    /// Creates a configuration from a single augmentation.
    /// </summary>
    /// <param name="augmentation">The augmentation to use.</param>
    public TrainingAugmentationConfiguration(IAugmentation<T, TData> augmentation)
    {
        var pipeline = new AugmentationPipeline<T, TData>();
        pipeline.Add(augmentation);
        Pipeline = pipeline;
    }

    /// <summary>
    /// Applies the configured augmentation to input data.
    /// </summary>
    /// <param name="data">The input data.</param>
    /// <param name="context">The augmentation context.</param>
    /// <returns>The augmented data.</returns>
    public TData Apply(TData data, AugmentationContext<T>? context = null)
    {
        if (!IsEnabled || Pipeline is null)
        {
            return data;
        }

        context = context ?? new AugmentationContext<T>(isTraining: true, seed: Seed);
        return Pipeline.Apply(data, context);
    }

    /// <summary>
    /// Gets the configuration as a dictionary for serialization/logging.
    /// </summary>
    /// <returns>A dictionary of configuration values.</returns>
    public IDictionary<string, object> GetConfiguration()
    {
        var config = new Dictionary<string, object>
        {
            ["isEnabled"] = IsEnabled,
            ["applyToValidation"] = ApplyToValidation,
            ["numAugmentations"] = NumAugmentations,
            ["prefetchAugmentations"] = PrefetchAugmentations,
            ["numWorkers"] = NumWorkers
        };

        if (Seed.HasValue)
        {
            config["seed"] = Seed.Value;
        }

        if (Pipeline is not null)
        {
            config["pipeline"] = Pipeline.GetConfiguration();
        }

        return config;
    }
}

/// <summary>
/// Builder for creating training augmentation configurations with a fluent API.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <typeparam name="TData">The data type being augmented.</typeparam>
public class TrainingAugmentationBuilder<T, TData>
{
    private readonly AugmentationPipeline<T, TData> _pipeline = new();
    private int? _seed;
    private bool _applyToValidation;
    private int _numAugmentations = 1;
    private int _numWorkers = 1;
    private bool _prefetch = true;

    /// <summary>
    /// Adds an augmentation to the pipeline.
    /// </summary>
    /// <param name="augmentation">The augmentation to add.</param>
    /// <returns>This builder for method chaining.</returns>
    public TrainingAugmentationBuilder<T, TData> Add(IAugmentation<T, TData> augmentation)
    {
        _pipeline.Add(augmentation);
        return this;
    }

    /// <summary>
    /// Adds a one-of choice of augmentations.
    /// </summary>
    /// <param name="augmentations">The augmentations to choose from.</param>
    /// <returns>This builder for method chaining.</returns>
    public TrainingAugmentationBuilder<T, TData> OneOf(params IAugmentation<T, TData>[] augmentations)
    {
        _pipeline.OneOf(augmentations);
        return this;
    }

    /// <summary>
    /// Adds augmentations that will be applied in random order.
    /// </summary>
    /// <param name="augmentations">The augmentations to shuffle.</param>
    /// <returns>This builder for method chaining.</returns>
    public TrainingAugmentationBuilder<T, TData> Shuffle(params IAugmentation<T, TData>[] augmentations)
    {
        _pipeline.Shuffle(augmentations);
        return this;
    }

    /// <summary>
    /// Sets a random seed for reproducibility.
    /// </summary>
    /// <param name="seed">The random seed.</param>
    /// <returns>This builder for method chaining.</returns>
    public TrainingAugmentationBuilder<T, TData> WithSeed(int seed)
    {
        _seed = seed;
        return this;
    }

    /// <summary>
    /// Configures whether to apply augmentation to validation data.
    /// </summary>
    /// <param name="apply">Whether to apply to validation.</param>
    /// <returns>This builder for method chaining.</returns>
    public TrainingAugmentationBuilder<T, TData> ApplyToValidation(bool apply = true)
    {
        _applyToValidation = apply;
        return this;
    }

    /// <summary>
    /// Sets the number of augmented versions per sample.
    /// </summary>
    /// <param name="count">The number of augmentations.</param>
    /// <returns>This builder for method chaining.</returns>
    public TrainingAugmentationBuilder<T, TData> WithMultipleAugmentations(int count)
    {
        _numAugmentations = count;
        return this;
    }

    /// <summary>
    /// Sets the number of worker threads.
    /// </summary>
    /// <param name="workers">The number of workers.</param>
    /// <returns>This builder for method chaining.</returns>
    public TrainingAugmentationBuilder<T, TData> WithWorkers(int workers)
    {
        _numWorkers = workers;
        return this;
    }

    /// <summary>
    /// Enables or disables prefetching.
    /// </summary>
    /// <param name="prefetch">Whether to prefetch.</param>
    /// <returns>This builder for method chaining.</returns>
    public TrainingAugmentationBuilder<T, TData> Prefetch(bool prefetch = true)
    {
        _prefetch = prefetch;
        return this;
    }

    /// <summary>
    /// Builds the configuration.
    /// </summary>
    /// <returns>The training augmentation configuration.</returns>
    public TrainingAugmentationConfiguration<T, TData> Build()
    {
        return new TrainingAugmentationConfiguration<T, TData>
        {
            Pipeline = _pipeline,
            Seed = _seed,
            ApplyToValidation = _applyToValidation,
            NumAugmentations = _numAugmentations,
            NumWorkers = _numWorkers,
            PrefetchAugmentations = _prefetch
        };
    }
}
