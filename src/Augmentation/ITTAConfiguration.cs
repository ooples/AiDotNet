namespace AiDotNet.Augmentation;

/// <summary>
/// Non-generic interface for Test-Time Augmentation configuration.
/// </summary>
/// <remarks>
/// This interface allows TTA configurations to be stored in non-generic containers
/// (like PredictionModelResultOptions) while maintaining type safety.
/// Use <see cref="ITTAConfiguration{T,TData}"/> for the full generic implementation.
/// </remarks>
public interface ITTAConfiguration
{
    /// <summary>
    /// Gets whether Test-Time Augmentation is enabled.
    /// </summary>
    bool IsEnabled { get; }

    /// <summary>
    /// Gets how many augmented versions to create.
    /// </summary>
    int NumberOfAugmentations { get; }

    /// <summary>
    /// Gets whether to include the original (unaugmented) input in predictions.
    /// </summary>
    bool IncludeOriginal { get; }

    /// <summary>
    /// Gets how predictions from augmented versions are combined.
    /// </summary>
    PredictionAggregationMethod AggregationMethod { get; }

    /// <summary>
    /// Gets the random seed for reproducible augmentations (null for random).
    /// </summary>
    int? Seed { get; }

    /// <summary>
    /// Gets whether to apply inverse transforms to spatial predictions.
    /// </summary>
    bool ApplyInverseTransforms { get; }

    /// <summary>
    /// Gets the minimum confidence threshold for including predictions.
    /// </summary>
    double? ConfidenceThreshold { get; }

    /// <summary>
    /// Gets the configuration as a dictionary for logging/serialization.
    /// </summary>
    IDictionary<string, object> GetConfiguration();
}

/// <summary>
/// Generic interface for Test-Time Augmentation configuration with pipeline access.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <typeparam name="TData">The data type being augmented.</typeparam>
public interface ITTAConfiguration<T, TData> : ITTAConfiguration
{
    /// <summary>
    /// Gets the augmentation pipeline that creates variations of input.
    /// </summary>
    IAugmentationPolicy<T, TData>? Pipeline { get; }
}
