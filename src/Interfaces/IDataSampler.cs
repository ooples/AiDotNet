namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the contract for sampling indices from a dataset during batch iteration.
/// </summary>
/// <remarks>
/// <para>
/// Data samplers control how samples are selected for each epoch of training.
/// Different sampling strategies can improve training convergence and handle
/// imbalanced datasets.
/// </para>
/// <para><b>For Beginners:</b> A sampler decides which data points to include in each batch
/// and in what order. The default is random sampling, but you might want:
///
/// - **Stratified sampling**: Ensures each class is represented proportionally in every batch
/// - **Weighted sampling**: Gives more weight to underrepresented or important samples
/// - **Curriculum learning**: Starts with easy examples and gradually increases difficulty
///
/// Example usage:
/// <code>
/// // Use weighted sampling to handle class imbalance
/// var sampler = new WeightedSampler&lt;float&gt;(weights);
/// foreach (var batch in dataLoader.GetBatches(sampler: sampler))
/// {
///     model.TrainOnBatch(batch);
/// }
/// </code>
/// </para>
/// </remarks>
public interface IDataSampler
{
    /// <summary>
    /// Gets the total number of samples this sampler will produce per epoch.
    /// </summary>
    /// <remarks>
    /// This may differ from the dataset size for oversampling or undersampling strategies.
    /// </remarks>
    int Length { get; }

    /// <summary>
    /// Returns an enumerable of indices for one epoch of sampling.
    /// </summary>
    /// <returns>An enumerable of sample indices in the order they should be processed.</returns>
    /// <remarks>
    /// <para>
    /// Each call to this method starts a new epoch. The returned indices determine
    /// which samples are included and in what order.
    /// </para>
    /// <para><b>For Beginners:</b> This method provides the "shopping list" of data points
    /// to include in this round of training. The order matters for learning!
    /// </para>
    /// </remarks>
    IEnumerable<int> GetIndices();

    /// <summary>
    /// Sets the random seed for reproducible sampling.
    /// </summary>
    /// <param name="seed">The random seed value.</param>
    /// <remarks>
    /// Setting a seed ensures the same sampling order is produced each time,
    /// which is important for reproducibility and debugging.
    /// </remarks>
    void SetSeed(int seed);
}

/// <summary>
/// Extended interface for samplers that support batch-level sampling.
/// </summary>
/// <remarks>
/// <para>
/// Some samplers need to operate at the batch level rather than sample level,
/// for example to ensure each batch contains samples from all classes (stratified batching).
/// </para>
/// </remarks>
public interface IBatchSampler : IDataSampler
{
    /// <summary>
    /// Gets or sets the batch size for batch-level sampling.
    /// </summary>
    int BatchSize { get; set; }

    /// <summary>
    /// Gets or sets whether to drop the last incomplete batch.
    /// </summary>
    bool DropLast { get; set; }

    /// <summary>
    /// Returns an enumerable of index arrays, where each array represents one batch.
    /// </summary>
    /// <returns>An enumerable of batch index arrays.</returns>
    IEnumerable<int[]> GetBatchIndices();
}

/// <summary>
/// Interface for samplers that use sample weights.
/// </summary>
/// <typeparam name="T">The numeric type for weights.</typeparam>
public interface IWeightedSampler<T> : IDataSampler
{
    /// <summary>
    /// Gets or sets the weights for each sample.
    /// </summary>
    /// <remarks>
    /// Higher weights increase the probability of a sample being selected.
    /// Weights should be non-negative.
    /// </remarks>
    IReadOnlyList<T> Weights { get; set; }

    /// <summary>
    /// Gets or sets whether to sample with replacement.
    /// </summary>
    /// <remarks>
    /// With replacement: samples can be selected multiple times per epoch.
    /// Without replacement: each sample appears at most once per epoch.
    /// </remarks>
    bool Replacement { get; set; }

    /// <summary>
    /// Gets or sets the number of samples to draw per epoch.
    /// </summary>
    /// <remarks>
    /// If null, defaults to the dataset size.
    /// Can be larger than dataset size when Replacement is true (oversampling).
    /// </remarks>
    int? NumSamples { get; set; }
}

/// <summary>
/// Interface for samplers that use class labels for stratification.
/// </summary>
public interface IStratifiedSampler : IDataSampler
{
    /// <summary>
    /// Gets or sets the class labels for each sample.
    /// </summary>
    IReadOnlyList<int> Labels { get; set; }

    /// <summary>
    /// Gets the number of unique classes.
    /// </summary>
    int NumClasses { get; }
}
