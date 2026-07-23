using AiDotNet.Enums;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for NGBoost (Natural Gradient Boosting) classifier models.
/// </summary>
/// <remarks>
/// <para>
/// NGBoost is a probabilistic gradient boosting algorithm that outputs well-calibrated
/// class probabilities. It uses natural gradients (gradients preconditioned by the
/// Fisher Information Matrix) for more stable optimization.
/// </para>
/// <para>
/// <b>For Beginners:</b> NGBoost is like regular gradient boosting for classification,
/// but it produces better-calibrated probability estimates. When NGBoost says there's
/// a 70% chance of class A, approximately 70% of similar predictions will actually
/// be class A.
///
/// This calibration is valuable because you can trust the probability estimates
/// for decision-making, not just the predicted class.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class NGBoostClassifierOptions<T> : ClassifierOptions<T>
{
    /// <summary>
    /// Gets or sets the number of boosting iterations (trees per class).
    /// </summary>
    /// <value>Default is 500.</value>
    /// <remarks>
    /// <para>
    /// More iterations generally improve performance but increase training time
    /// and can lead to overfitting. Use early stopping to find the optimal number.
    /// </para>
    /// <para><b>For Beginners:</b> This is how many rounds of boosting to perform.
    /// More rounds usually means better predictions, but takes longer to train.
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
    /// Gets or sets the maximum depth of trees.
    /// </summary>
    /// <value>Default is 3.</value>
    public int MaxDepth { get; set; } = 3;

    /// <summary>
    /// Gets or sets the minimum number of samples required to split an internal node.
    /// </summary>
    /// <value>Default is 2.</value>
    public int MinSamplesSplit { get; set; } = 2;

    /// <summary>
    /// Gets or sets the minimum number of samples required to be at a leaf node.
    /// </summary>
    /// <value>Default is 1.</value>
    public int MinSamplesLeaf { get; set; } = 1;

    /// <summary>
    /// Gets or sets the fraction of features to consider for splits.
    /// </summary>
    /// <value>Default is 1.0 (use all features).</value>
    /// <remarks>
    /// Value should be between 0 and 1. Set to 1.0 to use all features.
    /// </remarks>
    public double MaxFeatures { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the split criterion for the base trees.
    /// </summary>
    /// <value>Default is VarianceReduction.</value>
    public SplitCriterion SplitCriterion { get; set; } = SplitCriterion.VarianceReduction;

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
    /// Gets or sets whether to use natural gradients.
    /// </summary>
    /// <value>Default is true.</value>
    /// <remarks>
    /// <para>
    /// Natural gradients precondition the gradient by the Fisher Information Matrix,
    /// which typically leads to faster convergence and more stable optimization.
    /// For classification, this helps produce better-calibrated probabilities.
    /// </para>
    /// <para><b>For Beginners:</b> Natural gradients make the learning process smarter
    /// by accounting for the geometry of the probability distribution. This usually
    /// results in faster training and better probability estimates. Keep this enabled.
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
