namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Super Learner (Stacking) ensemble.
/// </summary>
/// <remarks>
/// <para>
/// Super Learner is an ensemble method that optimally combines predictions from multiple
/// base models using a meta-learner. It uses cross-validation to avoid overfitting and
/// is proven to perform at least as well as the best individual base learner.
/// </para>
/// <para>
/// <b>For Beginners:</b> Super Learner is like having a team of experts (base models) vote on
/// a prediction, but instead of equal votes, a "manager" (meta-learner) learns how to weight
/// each expert's opinion based on their track record.
///
/// <b>How it works:</b>
/// 1. Train each base model (e.g., linear regression, random forest, neural network)
/// 2. Get cross-validated predictions from each base model
/// 3. Train a meta-learner to combine these predictions optimally
/// 4. For new data: get predictions from all base models, then combine using meta-learner
///
/// <b>Why it's powerful:</b>
/// - Automatically figures out which models work best
/// - Can combine very different types of models
/// - Mathematically proven to be optimal (or close to it)
/// - Less prone to overfitting than simple averaging
///
/// <b>Example:</b>
/// You have a linear model (good for simple patterns), a tree model (good for interactions),
/// and a neural network (good for complex patterns). Super Learner learns to trust the tree
/// model more for some types of predictions and the neural network for others.
/// </para>
/// <para>
/// Reference: van der Laan, M.J., Polley, E.C., &amp; Hubbard, A.E. (2007). "Super Learner".
/// Statistical Applications in Genetics and Molecular Biology.
/// </para>
/// </remarks>
public class SuperLearnerOptions
{
    /// <summary>
    /// Gets or sets the number of cross-validation folds for generating meta-features.
    /// </summary>
    /// <value>Default is 5.</value>
    /// <remarks>
    /// More folds means more training data for each base model (but slower training).
    /// </remarks>
    public int NumFolds { get; set; } = 5;

    /// <summary>
    /// Gets or sets the meta-learner type.
    /// </summary>
    /// <value>Default is NonNegativeLeastSquares.</value>
    /// <remarks>
    /// The meta-learner combines base model predictions. Non-negative least squares
    /// ensures interpretable, positive weights.
    /// </remarks>
    public SuperLearnerMetaLearner MetaLearnerType { get; set; } = SuperLearnerMetaLearner.NonNegativeLeastSquares;

    /// <summary>
    /// Gets or sets whether to include the original features in the meta-learner.
    /// </summary>
    /// <value>Default is false.</value>
    /// <remarks>
    /// When true, the meta-learner sees both base model predictions AND original features.
    /// This can improve performance but increases complexity.
    /// </remarks>
    public bool IncludeOriginalFeatures { get; set; } = false;

    /// <summary>
    /// Gets or sets whether to use stratified cross-validation (for classification).
    /// </summary>
    /// <value>Default is true.</value>
    public bool UseStratifiedFolds { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to normalize base model predictions before meta-learning.
    /// </summary>
    /// <value>Default is true.</value>
    public bool NormalizeBasePredictions { get; set; } = true;

    /// <summary>
    /// Gets or sets the regularization strength for the meta-learner.
    /// </summary>
    /// <value>Default is 0.01.</value>
    public double MetaLearnerRegularization { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets the random seed for reproducibility.
    /// </summary>
    public int? Seed { get; set; }

    /// <summary>
    /// Gets or sets whether to retrain base models on full data after cross-validation.
    /// </summary>
    /// <value>Default is true.</value>
    /// <remarks>
    /// When true, base models are retrained on the full dataset after determining
    /// optimal weights. This typically improves final performance.
    /// </remarks>
    public bool RetrainOnFullData { get; set; } = true;

    /// <summary>
    /// Gets or sets the maximum number of iterations for the meta-learner.
    /// </summary>
    /// <value>Default is 1000.</value>
    public int MetaLearnerMaxIterations { get; set; } = 1000;

    /// <summary>
    /// Gets or sets the tolerance for meta-learner convergence.
    /// </summary>
    /// <value>Default is 1e-6.</value>
    public double MetaLearnerTolerance { get; set; } = 1e-6;
}

/// <summary>
/// Meta-learner types for Super Learner.
/// </summary>
public enum SuperLearnerMetaLearner
{
    /// <summary>
    /// Non-negative least squares. Ensures interpretable positive weights.
    /// </summary>
    NonNegativeLeastSquares,

    /// <summary>
    /// Ridge regression. Adds L2 regularization for stability.
    /// </summary>
    Ridge,

    /// <summary>
    /// Simple averaging (equal weights). Fast but may not be optimal.
    /// </summary>
    SimpleAverage,

    /// <summary>
    /// Weighted average based on cross-validation performance.
    /// </summary>
    PerformanceWeighted,

    /// <summary>
    /// Linear regression. Standard least squares (may have negative weights).
    /// </summary>
    LinearRegression,

    /// <summary>
    /// Lasso regression. L1 regularization can select a subset of models.
    /// </summary>
    Lasso
}
