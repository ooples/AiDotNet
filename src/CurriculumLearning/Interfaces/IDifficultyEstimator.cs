using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.Interfaces;

namespace AiDotNet.CurriculumLearning.Interfaces;

/// <summary>
/// Interface for estimating the difficulty of training samples.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> A difficulty estimator measures how "hard" each training
/// sample is for the model to learn. This is crucial for curriculum learning, as we
/// want to present easy samples first and gradually introduce harder ones.</para>
///
/// <para><b>Common Difficulty Measures:</b></para>
/// <list type="bullet">
/// <item><description><b>Loss-based:</b> Samples with high loss are harder</description></item>
/// <item><description><b>Confidence-based:</b> Low-confidence predictions indicate difficulty</description></item>
/// <item><description><b>Transfer-based:</b> Performance gap between simple and complex models</description></item>
/// <item><description><b>Expert-defined:</b> Domain knowledge about sample complexity</description></item>
/// </list>
///
/// <para><b>Research Background:</b></para>
/// <list type="bullet">
/// <item><description>Bengio et al. (2009): Originally proposed using prediction loss</description></item>
/// <item><description>Kumar et al. (2010): Self-paced learning using model confidence</description></item>
/// <item><description>Weinshall et al. (2018): Transfer teacher approach</description></item>
/// </list>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("DifficultyEstimator")]
public interface IDifficultyEstimator<T, TInput, TOutput>
{
    /// <summary>
    /// Gets the name of the difficulty estimator.
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Gets whether this estimator requires the model to estimate difficulty.
    /// </summary>
    /// <remarks>
    /// <para>Some estimators (like loss-based) need the model to compute predictions,
    /// while others (like expert-defined) don't need the model at all.</para>
    /// </remarks>
    bool RequiresModel { get; }

    /// <summary>
    /// Estimates the difficulty of a single sample.
    /// </summary>
    /// <param name="input">The input sample.</param>
    /// <param name="expectedOutput">The expected output/label.</param>
    /// <param name="model">The model to use for estimation (optional for some estimators).</param>
    /// <returns>Difficulty score (higher = harder). Typically in range [0, 1] but not required.</returns>
    T EstimateDifficulty(
        TInput input,
        TOutput expectedOutput,
        IFullModel<T, TInput, TOutput>? model = null);

    /// <summary>
    /// Estimates difficulty scores for all samples in a dataset.
    /// </summary>
    /// <param name="dataset">The dataset to estimate difficulties for.</param>
    /// <param name="model">The model to use for estimation (optional for some estimators).</param>
    /// <returns>Vector of difficulty scores (higher = harder).</returns>
    /// <remarks>
    /// <para>This method may be more efficient than calling EstimateDifficulty
    /// individually for each sample, as it can batch operations.</para>
    /// </remarks>
    Vector<T> EstimateDifficulties(
        IDataset<T, TInput, TOutput> dataset,
        IFullModel<T, TInput, TOutput>? model = null);

    /// <summary>
    /// Updates the difficulty estimator based on training progress.
    /// </summary>
    /// <param name="epoch">Current epoch number.</param>
    /// <param name="model">Current model state.</param>
    /// <remarks>
    /// <para>Some estimators adapt over time. For example, a loss-based estimator
    /// might recalculate difficulties as the model improves, since what was "hard"
    /// at epoch 1 might be "easy" at epoch 100.</para>
    /// </remarks>
    void Update(int epoch, IFullModel<T, TInput, TOutput> model);

    /// <summary>
    /// Resets the estimator to its initial state.
    /// </summary>
    void Reset();

    /// <summary>
    /// Gets the indices of samples sorted by difficulty (easy to hard).
    /// </summary>
    /// <param name="difficulties">The difficulty scores.</param>
    /// <returns>Indices sorted from easiest to hardest.</returns>
    int[] GetSortedIndices(Vector<T> difficulties);
}

/// <summary>
/// Interface for difficulty estimators that can provide confidence in their estimates.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
[AiDotNet.Configuration.YamlConfigurable("ConfidentDifficultyEstimator")]
public interface IConfidentDifficultyEstimator<T, TInput, TOutput> : IDifficultyEstimator<T, TInput, TOutput>
{
    /// <summary>
    /// Gets both difficulty estimate and confidence for a sample.
    /// </summary>
    /// <param name="input">The input sample.</param>
    /// <param name="expectedOutput">The expected output.</param>
    /// <param name="model">The model to use.</param>
    /// <returns>Tuple of (difficulty, confidence) where confidence is typically in [0, 1].</returns>
    (T Difficulty, T Confidence) EstimateDifficultyWithConfidence(
        TInput input,
        TOutput expectedOutput,
        IFullModel<T, TInput, TOutput>? model = null);
}

/// <summary>
/// Interface for ensemble difficulty estimators that combine multiple estimators.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
[AiDotNet.Configuration.YamlConfigurable("EnsembleDifficultyEstimator")]
public interface IEnsembleDifficultyEstimator<T, TInput, TOutput> : IDifficultyEstimator<T, TInput, TOutput>
{
    /// <summary>
    /// Gets the individual estimators in this ensemble.
    /// </summary>
    IReadOnlyList<IDifficultyEstimator<T, TInput, TOutput>> Estimators { get; }

    /// <summary>
    /// Gets or sets the weights for each estimator in the ensemble.
    /// </summary>
    Vector<T> Weights { get; set; }

    /// <summary>
    /// Adds an estimator to the ensemble.
    /// </summary>
    /// <param name="estimator">The estimator to add.</param>
    /// <param name="weight">The weight for this estimator.</param>
    void AddEstimator(IDifficultyEstimator<T, TInput, TOutput> estimator, T weight);

    /// <summary>
    /// Removes an estimator from the ensemble.
    /// </summary>
    /// <param name="index">Index of the estimator to remove.</param>
    void RemoveEstimator(int index);
}

/// <summary>
/// Types of difficulty estimation methods.
/// </summary>
public enum DifficultyEstimationType
{
    /// <summary>
    /// Uses training loss as difficulty measure.
    /// Higher loss = harder sample.
    /// </summary>
    LossBased,

    /// <summary>
    /// Uses model confidence/margin as difficulty measure.
    /// Lower confidence = harder sample.
    /// </summary>
    ConfidenceBased,

    /// <summary>
    /// Uses gap between simple and complex model performance.
    /// Larger gap = harder sample.
    /// </summary>
    TransferBased,

    /// <summary>
    /// Uses prediction variance across model ensemble.
    /// Higher variance = harder sample.
    /// </summary>
    EnsembleBased,

    /// <summary>
    /// Uses domain expert-defined difficulty scores.
    /// </summary>
    ExpertDefined,

    /// <summary>
    /// Uses sample complexity metrics (e.g., input magnitude).
    /// </summary>
    ComplexityBased,

    /// <summary>
    /// Combines multiple estimators.
    /// </summary>
    Ensemble
}
