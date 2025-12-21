namespace AiDotNet.Enums;

/// <summary>
/// Defines the types of cross-validation strategies available.
/// </summary>
public enum CrossValidationType
{
    /// <summary>
    /// K-Fold cross-validation splits the data into k equal folds.
    /// </summary>
    KFold,

    /// <summary>
    /// Stratified K-Fold maintains class distribution in each fold.
    /// </summary>
    StratifiedKFold,

    /// <summary>
    /// Leave-One-Out uses a single sample for validation.
    /// </summary>
    LeaveOneOut,

    /// <summary>
    /// Time Series cross-validation respects temporal ordering.
    /// </summary>
    TimeSeries,

    /// <summary>
    /// Group K-Fold keeps samples from the same group together.
    /// </summary>
    GroupKFold,

    /// <summary>
    /// Nested cross-validation for hyperparameter tuning and evaluation.
    /// </summary>
    Nested,

    /// <summary>
    /// Monte Carlo uses repeated random sampling.
    /// </summary>
    MonteCarlo,

    /// <summary>
    /// Standard cross-validation with no special considerations.
    /// </summary>
    Standard
}
