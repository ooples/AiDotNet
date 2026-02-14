namespace AiDotNet.Interfaces;

/// <summary>
/// Defines a strategy for active learning that selects the most informative samples
/// for labeling from a pool of unlabeled data.
/// </summary>
/// <typeparam name="T">The numeric type for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Active learning helps when labeling data is expensive or time-consuming.
/// Instead of randomly selecting samples to label, active learning intelligently picks the samples
/// that would be most helpful for training the model. This can dramatically reduce the number of
/// labels needed while achieving similar or better performance.</para>
///
/// <para><b>Common strategies include:</b></para>
/// <list type="bullet">
/// <item><description><b>Uncertainty Sampling:</b> Select samples where the model is most uncertain.
/// The idea is that uncertain samples are near the decision boundary and most informative.</description></item>
/// <item><description><b>Query-by-Committee:</b> Use multiple models and select samples where
/// they disagree the most. Disagreement suggests the sample is in an unclear region.</description></item>
/// <item><description><b>Expected Model Change:</b> Select samples that would cause the largest
/// change to model parameters. These samples have high learning potential.</description></item>
/// <item><description><b>Diversity Sampling:</b> Select samples that are representative of
/// different regions of the input space, ensuring good coverage.</description></item>
/// </list>
///
/// <para><b>Typical Usage Flow:</b></para>
/// <code>
/// // Initial training with small labeled set
/// model.Train(labeledData);
///
/// // Active learning loop
/// while (labelingBudget > 0)
/// {
///     // Select most informative samples
///     var samplesToLabel = strategy.SelectSamples(model, unlabeledPool, batchSize);
///
///     // Get labels (from human annotator or oracle)
///     var newLabels = GetLabels(samplesToLabel);
///
///     // Update training data and retrain
///     labeledData.AddRange(newLabels);
///     model.Train(labeledData);
///
///     labelingBudget -= batchSize;
/// }
/// </code>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("ActiveLearningStrategy")]
public interface IActiveLearningStrategy<T>
{
    /// <summary>
    /// Selects the most informative samples from the unlabeled pool for labeling.
    /// </summary>
    /// <param name="model">The current trained model used to evaluate samples.</param>
    /// <param name="unlabeledPool">Pool of unlabeled samples to select from.</param>
    /// <param name="batchSize">Number of samples to select for labeling.</param>
    /// <returns>Indices of the selected samples in the unlabeled pool.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the main method of active learning. It looks at all
    /// the unlabeled samples and picks the ones that would be most valuable to have labeled.
    /// The returned indices tell you which samples from the unlabeled pool to label next.</para>
    ///
    /// <para>The selection is based on the strategy's informativeness criterion (uncertainty,
    /// diversity, expected change, etc.).</para>
    /// </remarks>
    int[] SelectSamples(IFullModel<T, Tensor<T>, Tensor<T>> model, Tensor<T> unlabeledPool, int batchSize);

    /// <summary>
    /// Computes informativeness scores for all samples in the unlabeled pool.
    /// </summary>
    /// <param name="model">The current trained model used to evaluate samples.</param>
    /// <param name="unlabeledPool">Pool of unlabeled samples to score.</param>
    /// <returns>A vector of scores where higher values indicate more informative samples.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method assigns a score to each unlabeled sample indicating
    /// how "informative" or "valuable" it would be to label. Higher scores mean the model would
    /// benefit more from having that sample labeled.</para>
    ///
    /// <para>Different strategies compute informativeness differently:</para>
    /// <list type="bullet">
    /// <item><description>Uncertainty: How unsure the model is about the prediction</description></item>
    /// <item><description>Diversity: How different the sample is from already labeled data</description></item>
    /// <item><description>Expected change: How much the model would change if trained on this sample</description></item>
    /// </list>
    /// </remarks>
    Vector<T> ComputeInformativenessScores(IFullModel<T, Tensor<T>, Tensor<T>> model, Tensor<T> unlabeledPool);

    /// <summary>
    /// Gets the name of this active learning strategy.
    /// </summary>
    /// <remarks>
    /// <para>Used for logging, debugging, and identifying which strategy is being used.</para>
    /// </remarks>
    string Name { get; }

    /// <summary>
    /// Gets or sets whether to use batch-mode selection that considers diversity among selected samples.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> When selecting multiple samples at once (batch mode), simply
    /// picking the top-scoring samples might lead to redundancy - they might all be similar.
    /// Enabling batch diversity ensures the selected samples are not only informative but also
    /// different from each other, providing better coverage of the uncertain regions.</para>
    /// </remarks>
    bool UseBatchDiversity { get; set; }

    /// <summary>
    /// Gets statistics about the most recent sample selection.
    /// </summary>
    /// <returns>Dictionary containing selection statistics (e.g., score distribution, diversity metrics).</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method returns information about the last selection,
    /// which is useful for understanding how the strategy is performing and debugging.
    /// Statistics might include average score of selected samples, score variance, etc.</para>
    /// </remarks>
    Dictionary<string, T> GetSelectionStatistics();
}
