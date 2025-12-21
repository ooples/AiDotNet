namespace AiDotNet.ActiveLearning.Interfaces;

/// <summary>
/// Interface for datasets used in active learning scenarios.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The type of input features.</typeparam>
/// <typeparam name="TOutput">The type of output labels.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> A dataset in machine learning is a collection of samples,
/// where each sample has input features (X) and optionally output labels (Y). This interface
/// provides a unified way to work with datasets in active learning.</para>
///
/// <para><b>Key Concepts:</b></para>
/// <list type="bullet">
/// <item><description><b>Inputs:</b> The feature vectors (X) used for prediction</description></item>
/// <item><description><b>Outputs:</b> The labels or targets (Y) we want to predict</description></item>
/// <item><description><b>Indexing:</b> Access individual samples by their position</description></item>
/// <item><description><b>Subsetting:</b> Create new datasets from selected indices</description></item>
/// </list>
///
/// <para><b>Active Learning Usage:</b></para>
/// <list type="bullet">
/// <item><description>Labeled pool: Samples where outputs are known</description></item>
/// <item><description>Unlabeled pool: Samples where only inputs are known</description></item>
/// <item><description>Subsets are created when selecting samples for labeling</description></item>
/// </list>
/// </remarks>
public interface IDataset<T, TInput, TOutput>
{
    /// <summary>
    /// Gets the number of samples in the dataset.
    /// </summary>
    int Count { get; }

    /// <summary>
    /// Gets the input features for all samples.
    /// </summary>
    IReadOnlyList<TInput> Inputs { get; }

    /// <summary>
    /// Gets the output labels for all samples.
    /// </summary>
    /// <remarks>
    /// For unlabeled datasets, this may contain default values or be empty.
    /// Use <see cref="HasLabels"/> to check if labels are available.
    /// </remarks>
    IReadOnlyList<TOutput> Outputs { get; }

    /// <summary>
    /// Gets whether this dataset has labels for all samples.
    /// </summary>
    bool HasLabels { get; }

    /// <summary>
    /// Gets the input features for a specific sample.
    /// </summary>
    /// <param name="index">The index of the sample.</param>
    /// <returns>The input features at the specified index.</returns>
    TInput GetInput(int index);

    /// <summary>
    /// Gets the output label for a specific sample.
    /// </summary>
    /// <param name="index">The index of the sample.</param>
    /// <returns>The output label at the specified index.</returns>
    TOutput GetOutput(int index);

    /// <summary>
    /// Gets both input and output for a specific sample.
    /// </summary>
    /// <param name="index">The index of the sample.</param>
    /// <returns>A tuple containing the input and output.</returns>
    (TInput Input, TOutput Output) GetSample(int index);

    /// <summary>
    /// Creates a subset of the dataset containing only the specified indices.
    /// </summary>
    /// <param name="indices">The indices to include in the subset.</param>
    /// <returns>A new dataset containing only the specified samples.</returns>
    IDataset<T, TInput, TOutput> Subset(int[] indices);

    /// <summary>
    /// Creates a subset of the dataset excluding the specified indices.
    /// </summary>
    /// <param name="indices">The indices to exclude from the subset.</param>
    /// <returns>A new dataset without the specified samples.</returns>
    IDataset<T, TInput, TOutput> Except(int[] indices);

    /// <summary>
    /// Merges another dataset into this one.
    /// </summary>
    /// <param name="other">The dataset to merge.</param>
    /// <returns>A new dataset containing samples from both datasets.</returns>
    IDataset<T, TInput, TOutput> Merge(IDataset<T, TInput, TOutput> other);

    /// <summary>
    /// Adds samples with labels to the dataset.
    /// </summary>
    /// <param name="inputs">The input features to add.</param>
    /// <param name="outputs">The output labels to add.</param>
    /// <returns>A new dataset with the added samples.</returns>
    IDataset<T, TInput, TOutput> AddSamples(TInput[] inputs, TOutput[] outputs);

    /// <summary>
    /// Removes samples at the specified indices from the dataset.
    /// </summary>
    /// <param name="indices">The indices to remove.</param>
    /// <returns>A new dataset without the specified samples.</returns>
    IDataset<T, TInput, TOutput> RemoveSamples(int[] indices);

    /// <summary>
    /// Updates the labels for specific samples.
    /// </summary>
    /// <param name="indices">The indices to update.</param>
    /// <param name="labels">The new labels.</param>
    /// <returns>A new dataset with updated labels.</returns>
    IDataset<T, TInput, TOutput> UpdateLabels(int[] indices, TOutput[] labels);

    /// <summary>
    /// Shuffles the dataset and returns a new shuffled dataset.
    /// </summary>
    /// <param name="random">Optional random generator for reproducibility.</param>
    /// <returns>A new shuffled dataset.</returns>
    IDataset<T, TInput, TOutput> Shuffle(Random? random = null);

    /// <summary>
    /// Splits the dataset into training and test sets.
    /// </summary>
    /// <param name="trainRatio">The fraction of data for training (0.0 to 1.0).</param>
    /// <param name="random">Optional random generator for reproducibility.</param>
    /// <returns>A tuple containing training and test datasets.</returns>
    (IDataset<T, TInput, TOutput> Train, IDataset<T, TInput, TOutput> Test) Split(
        double trainRatio = 0.8,
        Random? random = null);

    /// <summary>
    /// Gets the indices of all samples in this dataset.
    /// </summary>
    /// <returns>An array of indices from 0 to Count-1.</returns>
    int[] GetIndices();

    /// <summary>
    /// Creates a shallow copy of this dataset.
    /// </summary>
    /// <returns>A new dataset with the same samples.</returns>
    IDataset<T, TInput, TOutput> Clone();
}

/// <summary>
/// Extended dataset interface with additional metadata and features.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The type of input features.</typeparam>
/// <typeparam name="TOutput">The type of output labels.</typeparam>
public interface IExtendedDataset<T, TInput, TOutput> : IDataset<T, TInput, TOutput>
{
    /// <summary>
    /// Gets the feature names if available.
    /// </summary>
    IReadOnlyList<string>? FeatureNames { get; }

    /// <summary>
    /// Gets the number of features per input sample.
    /// </summary>
    int FeatureCount { get; }

    /// <summary>
    /// Gets the number of classes for classification problems.
    /// </summary>
    /// <remarks>
    /// Returns 0 for regression problems.
    /// </remarks>
    int ClassCount { get; }

    /// <summary>
    /// Gets the unique class labels for classification problems.
    /// </summary>
    IReadOnlyList<TOutput>? ClassLabels { get; }

    /// <summary>
    /// Gets the sample weights if available.
    /// </summary>
    IReadOnlyList<T>? SampleWeights { get; }

    /// <summary>
    /// Gets whether this is a classification or regression dataset.
    /// </summary>
    bool IsClassification { get; }

    /// <summary>
    /// Gets metadata for the dataset.
    /// </summary>
    DatasetMetadata? Metadata { get; }
}

/// <summary>
/// Metadata about a dataset.
/// </summary>
public class DatasetMetadata
{
    /// <summary>
    /// Gets or sets the name of the dataset.
    /// </summary>
    public string? Name { get; set; }

    /// <summary>
    /// Gets or sets a description of the dataset.
    /// </summary>
    public string? Description { get; set; }

    /// <summary>
    /// Gets or sets the source of the dataset.
    /// </summary>
    public string? Source { get; set; }

    /// <summary>
    /// Gets or sets when the dataset was created.
    /// </summary>
    public DateTime? CreatedAt { get; set; }

    /// <summary>
    /// Gets or sets custom properties.
    /// </summary>
    public Dictionary<string, object>? Properties { get; set; }
}

/// <summary>
/// Factory for creating datasets.
/// </summary>
public interface IDatasetFactory<T, TInput, TOutput>
{
    /// <summary>
    /// Creates a dataset from inputs and outputs.
    /// </summary>
    /// <param name="inputs">The input features.</param>
    /// <param name="outputs">The output labels.</param>
    /// <returns>A new dataset.</returns>
    IDataset<T, TInput, TOutput> Create(TInput[] inputs, TOutput[] outputs);

    /// <summary>
    /// Creates an unlabeled dataset from inputs only.
    /// </summary>
    /// <param name="inputs">The input features.</param>
    /// <returns>A new unlabeled dataset.</returns>
    IDataset<T, TInput, TOutput> CreateUnlabeled(TInput[] inputs);

    /// <summary>
    /// Creates an empty dataset.
    /// </summary>
    /// <returns>An empty dataset.</returns>
    IDataset<T, TInput, TOutput> CreateEmpty();
}
