namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Explainable Boosting Machine (EBM) classifier.
/// </summary>
/// <remarks>
/// <para>
/// EBM is an interpretable machine learning algorithm that learns additive models with
/// optional pairwise interactions. It provides accuracy comparable to black-box models
/// while remaining fully interpretable.
/// </para>
/// <para>
/// <b>For Beginners:</b> EBM learns a separate "effect" for each feature, and the final
/// prediction is just the sum of these effects. This makes it easy to understand
/// exactly how each feature influences the prediction.
///
/// Key options:
/// - MaxBins: How finely to discretize continuous features (more bins = more detailed patterns)
/// - OuterBags: Number of boosting rounds over all features
/// - MaxInteractions: Number of feature pairs to consider for interactions
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class ExplainableBoostingClassifierOptions<T> : ClassifierOptions<T>
{
    /// <summary>
    /// Gets or sets the maximum number of bins for continuous features.
    /// </summary>
    /// <value>Default is 256.</value>
    /// <remarks>
    /// <para>
    /// More bins allow capturing more detailed patterns but increase computation
    /// and risk of overfitting. Typical values range from 128 to 512.
    /// </para>
    /// </remarks>
    public int MaxBins { get; set; } = 256;

    /// <summary>
    /// Gets or sets the number of outer bags (boosting rounds over all features).
    /// </summary>
    /// <value>Default is 50.</value>
    public int OuterBags { get; set; } = 50;

    /// <summary>
    /// Gets or sets the number of inner bags (boosting rounds per feature).
    /// </summary>
    /// <value>Default is 50.</value>
    public int InnerBags { get; set; } = 50;

    /// <summary>
    /// Gets or sets the learning rate for boosting updates.
    /// </summary>
    /// <value>Default is 0.01.</value>
    public double LearningRate { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets the L2 regularization strength.
    /// </summary>
    /// <value>Default is 1.0.</value>
    /// <remarks>
    /// Higher values add more regularization to prevent overfitting.
    /// </remarks>
    public double L2Regularization { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the maximum number of pairwise interactions to detect.
    /// </summary>
    /// <value>Default is 10.</value>
    /// <remarks>
    /// Set to 0 to disable interaction detection (pure additive model).
    /// </remarks>
    public int MaxInteractions { get; set; } = 10;

    /// <summary>
    /// Gets or sets the maximum number of bins for interaction terms.
    /// </summary>
    /// <value>Default is 32.</value>
    /// <remarks>
    /// Interaction terms use fewer bins than main effects to reduce complexity.
    /// </remarks>
    public int MaxInteractionBins { get; set; } = 32;

    /// <summary>
    /// Gets or sets early stopping rounds.
    /// </summary>
    /// <value>Default is null (no early stopping).</value>
    public int? EarlyStoppingRounds { get; set; }

    /// <summary>
    /// Gets or sets whether to print verbose output.
    /// </summary>
    public bool Verbose { get; set; }

    /// <summary>
    /// Gets or sets how often to print progress.
    /// </summary>
    public int VerboseEval { get; set; } = 10;
}
