namespace AiDotNet.Models.Options;

/// <summary>
/// Specifies how to handle missing feature blocks when not all parties have data for all entities.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> In vertical FL, different parties may not have data for all entities.
/// For example, Hospital A has records for patients 1-1000 and Hospital B has records for patients
/// 500-1500. For patients 1-499 (only in A) and 1001-1500 (only in B), the other party's features
/// are "missing". This enum controls how those missing features are filled in during training.</para>
/// </remarks>
public enum MissingFeatureStrategy
{
    /// <summary>
    /// Replace missing features with zeros. Simple and fast, but may bias the model.
    /// </summary>
    Zero,

    /// <summary>
    /// Replace missing features with the column-wise mean of available data.
    /// Better than zero imputation for centered data distributions.
    /// </summary>
    Mean,

    /// <summary>
    /// Use a learned imputation model that predicts missing features from available features.
    /// Most accurate but requires additional training.
    /// </summary>
    Learned,

    /// <summary>
    /// Skip entities with missing features entirely. Only train on fully-aligned entities.
    /// Safest but reduces the training set size.
    /// </summary>
    Skip
}
