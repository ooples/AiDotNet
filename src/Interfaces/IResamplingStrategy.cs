using AiDotNet.LinearAlgebra;

namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the interface for resampling strategies used to handle imbalanced datasets.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Resampling strategies modify the training data to address class imbalance, either by
/// creating synthetic samples for minority classes (oversampling) or removing samples
/// from majority classes (undersampling).
/// </para>
/// <para>
/// <b>For Beginners:</b> In many real-world problems, one class has far fewer examples than another.
/// For example:
/// - Fraud detection: 99% normal transactions, 1% fraudulent
/// - Disease diagnosis: 95% healthy, 5% diseased
/// - Spam filtering: 80% legitimate, 20% spam
///
/// This imbalance causes problems because machine learning models tend to ignore the minority class
/// and just predict the majority class for everything. Resampling strategies fix this by:
///
/// 1. <b>Oversampling:</b> Creating more examples of the minority class (like SMOTE, ADASYN)
/// 2. <b>Undersampling:</b> Removing examples from the majority class (like RandomUnderSampler)
/// 3. <b>Combined:</b> Doing both (like SMOTEENN, SMOTETomek)
///
/// After resampling, the model sees a more balanced dataset and learns to recognize both classes.
/// </para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("ResamplingStrategy")]
public interface IResamplingStrategy<T>
{
    /// <summary>
    /// Gets the name of the resampling strategy.
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Resamples the dataset to address class imbalance.
    /// </summary>
    /// <param name="x">The feature matrix.</param>
    /// <param name="y">The class labels.</param>
    /// <returns>A tuple containing the resampled features and labels.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method takes your imbalanced training data and returns
    /// a new, more balanced dataset. The returned data will have either:
    /// - More samples (if oversampling was used)
    /// - Fewer samples (if undersampling was used)
    /// - Different samples (if combined methods were used)
    ///
    /// You should only use this on training data, never on test data!
    /// </para>
    /// </remarks>
    (Matrix<T> resampledX, Vector<T> resampledY) Resample(Matrix<T> x, Vector<T> y);

    /// <summary>
    /// Gets statistics about the resampling operation.
    /// </summary>
    /// <returns>Information about how many samples were added/removed per class.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> After resampling, you can call this method to see exactly
    /// what changed - how many synthetic samples were created, how many were removed, etc.
    /// </para>
    /// </remarks>
    ResamplingStatistics<T> GetStatistics();
}

/// <summary>
/// Contains statistics about a resampling operation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This class tells you what the resampling strategy did to your data,
/// so you can verify it worked as expected.
/// </para>
/// </remarks>
public class ResamplingStatistics<T>
{
    /// <summary>
    /// The original number of samples per class.
    /// </summary>
    public NumericDictionary<T, int> OriginalClassCounts { get; set; } = new();

    /// <summary>
    /// The new number of samples per class after resampling.
    /// </summary>
    public NumericDictionary<T, int> ResampledClassCounts { get; set; } = new();

    /// <summary>
    /// The number of samples added per class (positive for oversampling).
    /// </summary>
    public NumericDictionary<T, int> SamplesAddedPerClass { get; set; } = new();

    /// <summary>
    /// The number of samples removed per class (positive for undersampling).
    /// </summary>
    public NumericDictionary<T, int> SamplesRemovedPerClass { get; set; } = new();

    /// <summary>
    /// Total samples before resampling.
    /// </summary>
    public int TotalOriginalSamples { get; set; }

    /// <summary>
    /// Total samples after resampling.
    /// </summary>
    public int TotalResampledSamples { get; set; }
}
