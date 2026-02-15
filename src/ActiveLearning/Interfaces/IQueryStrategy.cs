namespace AiDotNet.ActiveLearning.Interfaces;

/// <summary>
/// Interface for query strategies in active learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> A query strategy determines which unlabeled samples
/// should be selected for labeling by the oracle (human expert). Different strategies
/// use different criteria for selection.</para>
///
/// <para><b>Common Query Strategies:</b></para>
/// <list type="bullet">
/// <item><description><b>Uncertainty Sampling:</b> Select samples where model is most uncertain</description></item>
/// <item><description><b>Query By Committee:</b> Select samples where multiple models disagree</description></item>
/// <item><description><b>Expected Model Change:</b> Select samples that would change the model most</description></item>
/// <item><description><b>Diversity Sampling:</b> Select samples that are diverse in feature space</description></item>
/// </list>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("QueryStrategy")]
public interface IQueryStrategy<T, TInput, TOutput>
{
    /// <summary>
    /// Gets the name of the query strategy.
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Gets a description of how the strategy works.
    /// </summary>
    string Description { get; }

    /// <summary>
    /// Computes informativeness scores for all samples in the unlabeled pool.
    /// </summary>
    /// <param name="model">The current model being trained.</param>
    /// <param name="unlabeledPool">The pool of unlabeled samples.</param>
    /// <returns>Informativeness scores for each sample (higher = more informative).</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method calculates how "informative" each unlabeled
    /// sample would be if we labeled it. Higher scores mean the sample would provide
    /// more useful information for training.</para>
    /// </remarks>
    Vector<T> ComputeScores(
        IFullModel<T, TInput, TOutput> model,
        IDataset<T, TInput, TOutput> unlabeledPool);

    /// <summary>
    /// Selects the most informative samples from the unlabeled pool.
    /// </summary>
    /// <param name="model">The current model being trained.</param>
    /// <param name="unlabeledPool">The pool of unlabeled samples.</param>
    /// <param name="batchSize">Number of samples to select.</param>
    /// <returns>Indices of the selected samples in the unlabeled pool.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the main method for selecting which samples
    /// to label next. It returns the indices of the most informative samples.</para>
    /// </remarks>
    int[] SelectSamples(
        IFullModel<T, TInput, TOutput> model,
        IDataset<T, TInput, TOutput> unlabeledPool,
        int batchSize);

    /// <summary>
    /// Updates the strategy's internal state after new samples are labeled.
    /// </summary>
    /// <param name="newlyLabeledIndices">Indices of samples that were just labeled.</param>
    /// <param name="labels">The labels that were assigned.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Some strategies need to update their internal state
    /// when new samples are labeled. For example, diversity-based strategies might
    /// need to update their representation of the labeled set.</para>
    /// </remarks>
    void UpdateState(int[] newlyLabeledIndices, TOutput[] labels);

    /// <summary>
    /// Resets the strategy to its initial state.
    /// </summary>
    void Reset();
}

/// <summary>
/// Interface for query strategies that support uncertainty-based selection.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
[AiDotNet.Configuration.YamlConfigurable("UncertaintyStrategy")]
public interface IUncertaintyStrategy<T, TInput, TOutput> : IQueryStrategy<T, TInput, TOutput>
{
    /// <summary>
    /// Computes the uncertainty for a single sample.
    /// </summary>
    /// <param name="model">The current model.</param>
    /// <param name="input">The input sample.</param>
    /// <returns>The uncertainty score (higher = more uncertain).</returns>
    T ComputeUncertainty(IFullModel<T, TInput, TOutput> model, TInput input);

    /// <summary>
    /// Gets the predicted probabilities for a sample.
    /// </summary>
    /// <param name="model">The current model.</param>
    /// <param name="input">The input sample.</param>
    /// <returns>Probability distribution over classes.</returns>
    Vector<T> GetPredictionProbabilities(IFullModel<T, TInput, TOutput> model, TInput input);
}

/// <summary>
/// Interface for committee-based query strategies (Query By Committee).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
[AiDotNet.Configuration.YamlConfigurable("CommitteeStrategy")]
public interface ICommitteeStrategy<T, TInput, TOutput> : IQueryStrategy<T, TInput, TOutput>
{
    /// <summary>
    /// Gets the committee of models.
    /// </summary>
    IReadOnlyList<IFullModel<T, TInput, TOutput>> Committee { get; }

    /// <summary>
    /// Computes the disagreement among committee members for a sample.
    /// </summary>
    /// <param name="input">The input sample.</param>
    /// <returns>The disagreement score (higher = more disagreement).</returns>
    T ComputeDisagreement(TInput input);

    /// <summary>
    /// Updates all committee members with new training data.
    /// </summary>
    /// <param name="trainingData">The updated training dataset.</param>
    void UpdateCommittee(IDataset<T, TInput, TOutput> trainingData);
}

/// <summary>
/// Interface for density-weighted query strategies.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
[AiDotNet.Configuration.YamlConfigurable("DensityWeightedStrategy")]
public interface IDensityWeightedStrategy<T, TInput, TOutput> : IQueryStrategy<T, TInput, TOutput>
{
    /// <summary>
    /// Computes the density of samples in the feature space.
    /// </summary>
    /// <param name="unlabeledPool">The pool of unlabeled samples.</param>
    /// <returns>Density estimates for each sample.</returns>
    Vector<T> ComputeDensities(IDataset<T, TInput, TOutput> unlabeledPool);

    /// <summary>
    /// Gets or sets the weight given to density in the final score.
    /// </summary>
    /// <remarks>
    /// <para>Score = Informativeness × Density^Beta</para>
    /// </remarks>
    T DensityWeight { get; set; }
}

/// <summary>
/// Interface for Bayesian query strategies (e.g., BALD).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
[AiDotNet.Configuration.YamlConfigurable("BayesianStrategy")]
public interface IBayesianStrategy<T, TInput, TOutput> : IQueryStrategy<T, TInput, TOutput>
{
    /// <summary>
    /// Gets the number of Monte Carlo samples for uncertainty estimation.
    /// </summary>
    int MonteCarloSamples { get; }

    /// <summary>
    /// Computes the mutual information between predictions and model parameters.
    /// </summary>
    /// <param name="model">The current model.</param>
    /// <param name="input">The input sample.</param>
    /// <returns>The mutual information score.</returns>
    T ComputeMutualInformation(IFullModel<T, TInput, TOutput> model, TInput input);

    /// <summary>
    /// Computes the predictive entropy for a sample.
    /// </summary>
    /// <param name="model">The current model.</param>
    /// <param name="input">The input sample.</param>
    /// <returns>The predictive entropy.</returns>
    T ComputePredictiveEntropy(IFullModel<T, TInput, TOutput> model, TInput input);
}
