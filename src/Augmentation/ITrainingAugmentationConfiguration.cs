namespace AiDotNet.Augmentation;

/// <summary>
/// Non-generic interface for training augmentation configuration.
/// Allows storing configuration without knowing the specific type parameters.
/// </summary>
/// <remarks>
/// This interface provides a type-safe way to store training augmentation
/// configuration in non-generic contexts like PredictionModelBuilder.
/// </remarks>
public interface ITrainingAugmentationConfiguration
{
    /// <summary>
    /// Gets whether augmentation is enabled.
    /// </summary>
    bool IsEnabled { get; }

    /// <summary>
    /// Gets the random seed for reproducibility.
    /// </summary>
    int? Seed { get; }

    /// <summary>
    /// Gets whether to apply augmentation to the validation set.
    /// </summary>
    bool ApplyToValidation { get; }

    /// <summary>
    /// Gets the number of augmented versions to generate per sample.
    /// </summary>
    int NumAugmentations { get; }

    /// <summary>
    /// Gets whether to prefetch augmented data for performance.
    /// </summary>
    bool PrefetchAugmentations { get; }

    /// <summary>
    /// Gets the number of worker threads for augmentation.
    /// </summary>
    int NumWorkers { get; }

    /// <summary>
    /// Gets the configuration as a dictionary for serialization/logging.
    /// </summary>
    IDictionary<string, object> GetConfiguration();
}

/// <summary>
/// Generic interface for training augmentation configuration with pipeline access.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <typeparam name="TData">The data type being augmented.</typeparam>
public interface ITrainingAugmentationConfiguration<T, TData> : ITrainingAugmentationConfiguration
{
    /// <summary>
    /// Gets the augmentation pipeline for training.
    /// </summary>
    IAugmentationPolicy<T, TData>? Pipeline { get; }

    /// <summary>
    /// Applies the configured augmentation to input data.
    /// </summary>
    /// <param name="data">The input data.</param>
    /// <param name="context">The augmentation context.</param>
    /// <returns>The augmented data.</returns>
    TData Apply(TData data, AugmentationContext<T>? context = null);
}
