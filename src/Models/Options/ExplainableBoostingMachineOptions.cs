namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Explainable Boosting Machine (EBM) models.
/// </summary>
/// <remarks>
/// <para>
/// EBM is a glass-box machine learning model that maintains high accuracy while being
/// fully interpretable. It's based on Generalized Additive Models (GAMs) with boosting,
/// allowing you to see exactly how each feature contributes to predictions.
/// </para>
/// <para>
/// <b>For Beginners:</b> EBM is one of the few machine learning models that gives you
/// both high accuracy AND full explainability. Unlike "black box" models where you can't
/// see why they made a prediction, EBM shows you exactly how each feature affects the
/// outcome through interpretable graphs called "shape functions."
///
/// For example, if you're predicting house prices, EBM will show you:
/// - How square footage affects price (e.g., "each 100 sqft adds $10,000")
/// - How age affects price (e.g., "older houses are worth less")
/// - How these factors combine
///
/// This transparency is crucial for healthcare, finance, and other domains where
/// you need to explain and justify predictions.
/// </para>
/// </remarks>
public class ExplainableBoostingMachineOptions : DecisionTreeOptions
{
    /// <summary>
    /// Gets or sets the number of outer boosting iterations.
    /// </summary>
    /// <value>Default is 5000.</value>
    /// <remarks>
    /// More iterations generally improve accuracy but increase training time.
    /// EBM typically needs more iterations than traditional gradient boosting
    /// because it trains features one at a time.
    /// </remarks>
    public int NumberOfOuterIterations { get; set; } = 5000;

    /// <summary>
    /// Gets or sets the number of inner boosting iterations per feature.
    /// </summary>
    /// <value>Default is 1.</value>
    /// <remarks>
    /// This controls how much each feature is updated during each outer iteration.
    /// Higher values can speed up training but may reduce interpretability.
    /// </remarks>
    public int NumberOfInnerIterations { get; set; } = 1;

    /// <summary>
    /// Gets or sets the learning rate.
    /// </summary>
    /// <value>Default is 0.01.</value>
    /// <remarks>
    /// Lower learning rates typically produce better generalization but require
    /// more iterations. EBM works best with small learning rates.
    /// </remarks>
    public double LearningRate { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets the maximum number of bins for continuous features.
    /// </summary>
    /// <value>Default is 256.</value>
    /// <remarks>
    /// <para>
    /// Continuous features are binned to speed up training and create interpretable
    /// step functions. More bins allow finer-grained shape functions but increase
    /// memory usage and computation time.
    /// </para>
    /// <para><b>For Beginners:</b> Instead of treating a feature like "age" as having
    /// infinite possible values, EBM groups similar values into "bins" (like 0-10,
    /// 10-20, etc.). More bins means more precise shapes but takes longer to train.
    /// </para>
    /// </remarks>
    public int MaxBins { get; set; } = 256;

    /// <summary>
    /// Gets or sets the maximum number of interaction terms to consider.
    /// </summary>
    /// <value>Default is 10.</value>
    /// <remarks>
    /// <para>
    /// Interaction terms capture how pairs of features work together. For example,
    /// the effect of "age" might depend on "income". More interactions can improve
    /// accuracy but reduce interpretability.
    /// </para>
    /// <para><b>For Beginners:</b> Sometimes two features together have an effect that
    /// neither has alone. For example, "hot weather" and "ice cream store nearby" together
    /// boost ice cream sales more than you'd expect from each alone. Interactions capture this.
    /// </para>
    /// </remarks>
    public int MaxInteractionBins { get; set; } = 10;

    /// <summary>
    /// Gets or sets whether to automatically detect and include pairwise interactions.
    /// </summary>
    /// <value>Default is true.</value>
    /// <remarks>
    /// When enabled, EBM will automatically detect important feature interactions
    /// using FAST algorithm. Disable for a pure additive model.
    /// </remarks>
    public bool DetectInteractions { get; set; } = true;

    /// <summary>
    /// Gets or sets the minimum samples required in each bin.
    /// </summary>
    /// <value>Default is 2.</value>
    /// <remarks>
    /// Bins with fewer samples than this will be merged with adjacent bins.
    /// Higher values provide more robust estimates but coarser shape functions.
    /// </remarks>
    public int MinSamplesPerBin { get; set; } = 2;

    /// <summary>
    /// Gets or sets the subsampling ratio for each iteration.
    /// </summary>
    /// <value>Default is 0.5.</value>
    /// <remarks>
    /// Using less than 100% of data per iteration helps prevent overfitting
    /// and speeds up training.
    /// </remarks>
    public double SubsampleRatio { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets whether to use cyclic training order.
    /// </summary>
    /// <value>Default is true.</value>
    /// <remarks>
    /// When true, features are updated in a fixed cyclic order. When false,
    /// features are randomly selected each iteration (can improve robustness).
    /// </remarks>
    public bool CyclicTraining { get; set; } = true;

    /// <summary>
    /// Gets or sets the number of early stopping rounds.
    /// </summary>
    /// <value>Default is null (no early stopping).</value>
    public int? EarlyStoppingRounds { get; set; }

    /// <summary>
    /// Gets or sets the fraction of data to use for validation during early stopping.
    /// </summary>
    /// <value>Default is 0.15.</value>
    public double ValidationFraction { get; set; } = 0.15;

    /// <summary>
    /// Gets or sets the regularization strength for smoothing shape functions.
    /// </summary>
    /// <value>Default is 1.0.</value>
    /// <remarks>
    /// Higher values produce smoother, more interpretable shape functions but
    /// may reduce accuracy. Set to 0 to disable smoothing.
    /// </remarks>
    public double RegularizationStrength { get; set; } = 1.0;
}
