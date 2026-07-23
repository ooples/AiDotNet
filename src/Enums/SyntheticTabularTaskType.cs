namespace AiDotNet.Enums;

/// <summary>
/// Defines the task type for the synthetic federated tabular benchmark suite.
/// </summary>
public enum SyntheticTabularTaskType
{
    /// <summary>
    /// Binary classification (labels 0/1).
    /// </summary>
    BinaryClassification,

    /// <summary>
    /// Multi-class classification (labels 0..K-1).
    /// </summary>
    MultiClassClassification,

    /// <summary>
    /// Regression (continuous targets).
    /// </summary>
    Regression
}

