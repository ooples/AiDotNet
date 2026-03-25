namespace AiDotNet.Enums;

/// <summary>
/// Describes what kind of test data a loss function expects.
/// The test scaffold generator uses this to create appropriate test vectors.
/// </summary>
public enum LossTestInputFormat
{
    /// <summary>
    /// Standard continuous values in [0, 1] range.
    /// Used by: MSE, MAE, Huber, RMSE, LogCosh, Charbonnier, etc.
    /// Test data: predicted=[0.2, 0.5, 0.8], actual=[0.3, 0.6, 0.7]
    /// </summary>
    Continuous,

    /// <summary>
    /// Signed labels in {-1, +1} with predictions as real-valued scores.
    /// Used by: HingeLoss, SquaredHingeLoss, ModifiedHuberLoss, ExponentialLoss.
    /// Test data: predicted=[0.5, -0.3, 1.2], actual=[1.0, -1.0, 1.0]
    /// </summary>
    SignedLabels,

    /// <summary>
    /// Probability distribution where values are in [0, 1].
    /// Used by: CrossEntropy, CategoricalCrossEntropy, FocalLoss, WeightedCrossEntropy.
    /// Test data: predicted=[0.7, 0.2, 0.1], actual=[1.0, 0.0, 0.0]
    /// </summary>
    ProbabilityDistribution,

    /// <summary>
    /// Binary similarity labels in {0, 1} with distance-based predictions.
    /// Used by: ContrastiveLoss.
    /// Test data: predicted=[0.5, 1.2, 0.8], actual=[1.0, 0.0, 1.0]
    /// </summary>
    SimilarityLabels,

    /// <summary>
    /// Wasserstein critic scores with signed labels.
    /// Used by: WassersteinLoss.
    /// Test data: predicted=[2.5, -1.3, 0.8], actual=[1.0, -1.0, 1.0]
    /// </summary>
    CriticScores,

    /// <summary>
    /// Segmentation masks in [0, 1] range.
    /// Used by: DiceLoss, JaccardLoss.
    /// Test data: predicted=[0.8, 0.1, 0.9], actual=[1.0, 0.0, 1.0]
    /// </summary>
    SegmentationMask,

    /// <summary>
    /// Margin-based with predictions in [0, 1] and binary labels.
    /// Used by: MarginLoss (capsule networks).
    /// Test data: predicted=[0.9, 0.1, 0.8], actual=[1.0, 0.0, 1.0]
    /// </summary>
    MarginBased,

    /// <summary>
    /// Ordinal category indices.
    /// Used by: OrdinalRegressionLoss.
    /// Test data: predicted=[0.5, 0.3, 0.8], actual=[1.0, 2.0, 3.0]
    /// </summary>
    OrdinalCategories
}
