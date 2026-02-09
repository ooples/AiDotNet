namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for NGBoost (Natural Gradient Boosting) regression models.
/// </summary>
/// <remarks>
/// <para>
/// NGBoost is a probabilistic gradient boosting algorithm that outputs full probability
/// distributions instead of point predictions. It uses natural gradients (gradients
/// preconditioned by the Fisher Information Matrix) for more stable optimization.
/// </para>
/// <para>
/// <b>For Beginners:</b> NGBoost is like regular gradient boosting, but instead of predicting
/// a single number, it predicts a full probability distribution. This tells you not just
/// "what the prediction is" but also "how confident the model is."
///
/// For example, instead of predicting "house price = $300,000", NGBoost might predict
/// "house price is normally distributed with mean $300,000 and standard deviation $50,000."
/// This uncertainty information is valuable for decision-making.
/// </para>
/// </remarks>
public class NGBoostRegressionOptions : DecisionTreeOptions
{
    /// <summary>
    /// Gets or sets the number of boosting iterations (trees).
    /// </summary>
    /// <value>Default is 500.</value>
    /// <remarks>
    /// <para>
    /// More iterations generally improve performance but increase training time
    /// and can lead to overfitting. Use early stopping to find the optimal number.
    /// </para>
    /// <para><b>For Beginners:</b> This is how many "experts" to add to your team.
    /// More experts usually means better predictions, but takes longer to train.
    /// Start with the default and use validation data to determine if you need more.
    /// </para>
    /// </remarks>
    public int NumberOfIterations { get; set; } = 500;

    /// <summary>
    /// Gets or sets the learning rate (shrinkage factor).
    /// </summary>
    /// <value>Default is 0.01.</value>
    /// <remarks>
    /// <para>
    /// Lower learning rates require more iterations but often achieve better
    /// generalization. NGBoost typically works well with lower learning rates
    /// than traditional gradient boosting.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how much each tree contributes.
    /// Smaller values mean each tree has less influence, making learning more gradual
    /// but more stable. The default of 0.01 is recommended for NGBoost.
    /// </para>
    /// </remarks>
    public double LearningRate { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets the minimum number of samples required to split an internal node.
    /// </summary>
    /// <value>Default is 2.</value>
    public new int MinSamplesSplit { get; set; } = 2;

    /// <summary>
    /// Gets or sets the minimum number of samples required to be at a leaf node.
    /// </summary>
    /// <value>Default is 1.</value>
    public int MinSamplesLeaf { get; set; } = 1;

    /// <summary>
    /// Gets or sets the subsampling ratio for each iteration.
    /// </summary>
    /// <value>Default is 0.5.</value>
    /// <remarks>
    /// <para>
    /// Values less than 1.0 enable stochastic gradient boosting where each tree
    /// is trained on a random subset of the data. This can improve generalization.
    /// </para>
    /// <para><b>For Beginners:</b> This determines what fraction of your data
    /// is used to train each tree. Using 0.5 means each tree only sees half the
    /// data, which often leads to better generalization and faster training.
    /// </para>
    /// </remarks>
    public double SubsampleRatio { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the type of distribution to fit.
    /// </summary>
    /// <value>Default is Normal.</value>
    /// <remarks>
    /// <para>
    /// Choose the distribution that best matches your target variable:
    /// - Normal: Continuous data with symmetric errors
    /// - Laplace: Continuous data with heavy-tailed or outlier-prone errors
    /// - LogNormal: Positive-only data with right skew
    /// - Exponential: Positive-only data like survival times
    /// - Poisson: Count data
    /// </para>
    /// </remarks>
    public NGBoostDistributionType DistributionType { get; set; } = NGBoostDistributionType.Normal;

    /// <summary>
    /// Gets or sets the type of scoring rule used for optimization.
    /// </summary>
    /// <value>Default is LogScore.</value>
    /// <remarks>
    /// <para>
    /// The scoring rule defines how to evaluate probabilistic predictions:
    /// - LogScore (NLL): Most common, optimizes negative log likelihood
    /// - CRPS: More robust to outliers, has same units as target variable
    /// </para>
    /// </remarks>
    public NGBoostScoringRuleType ScoringRule { get; set; } = NGBoostScoringRuleType.LogScore;

    /// <summary>
    /// Gets or sets whether to use natural gradients.
    /// </summary>
    /// <value>Default is true.</value>
    /// <remarks>
    /// <para>
    /// Natural gradients precondition the gradient by the Fisher Information Matrix,
    /// which typically leads to faster convergence and more stable optimization.
    /// </para>
    /// <para><b>For Beginners:</b> Natural gradients make the learning process smarter
    /// by accounting for the geometry of the probability distribution. This usually
    /// results in faster and more stable training. Keep this enabled.
    /// </para>
    /// </remarks>
    public bool UseNaturalGradient { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to apply column subsampling.
    /// </summary>
    /// <value>Default is 1.0 (use all columns).</value>
    /// <remarks>
    /// Values less than 1.0 randomly select a subset of features for each tree,
    /// similar to Random Forest. This can prevent overfitting.
    /// </remarks>
    public double ColumnSubsampleRatio { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the number of early stopping rounds.
    /// </summary>
    /// <value>Default is null (no early stopping).</value>
    /// <remarks>
    /// If set, training stops when the validation score doesn't improve for
    /// this many consecutive rounds. Requires validation data to be provided.
    /// </remarks>
    public int? EarlyStoppingRounds { get; set; }

    /// <summary>
    /// Gets or sets whether to verbose output during training.
    /// </summary>
    /// <value>Default is false.</value>
    public bool Verbose { get; set; }

    /// <summary>
    /// Gets or sets how often to print progress (every N iterations).
    /// </summary>
    /// <value>Default is 100.</value>
    public int VerboseEval { get; set; } = 100;
}

/// <summary>
/// Types of distributions supported by NGBoost.
/// </summary>
public enum NGBoostDistributionType
{
    /// <summary>
    /// Normal (Gaussian) distribution for continuous symmetric data.
    /// </summary>
    Normal,

    /// <summary>
    /// Laplace distribution for robust regression with heavy tails.
    /// </summary>
    Laplace,

    /// <summary>
    /// Student-t distribution for heavy-tailed continuous data.
    /// </summary>
    StudentT,

    /// <summary>
    /// Log-Normal distribution for positive, right-skewed data.
    /// </summary>
    LogNormal,

    /// <summary>
    /// Exponential distribution for survival/waiting times.
    /// </summary>
    Exponential,

    /// <summary>
    /// Poisson distribution for count data.
    /// </summary>
    Poisson,

    /// <summary>
    /// Gamma distribution for positive continuous data.
    /// </summary>
    Gamma
}

/// <summary>
/// Types of scoring rules supported by NGBoost.
/// </summary>
public enum NGBoostScoringRuleType
{
    /// <summary>
    /// Logarithmic score (negative log likelihood).
    /// </summary>
    LogScore,

    /// <summary>
    /// Continuous Ranked Probability Score.
    /// </summary>
    CRPS
}
