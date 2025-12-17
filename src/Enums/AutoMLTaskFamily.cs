namespace AiDotNet.Enums;

/// <summary>
/// Defines the high-level task family for an AutoML run.
/// </summary>
/// <remarks>
/// <para>
/// AutoML uses the task family to choose sensible defaults for metrics, evaluation protocols, and candidate selection.
/// </para>
/// <para>
/// <b>For Beginners:</b> This is the kind of problem you're solving:
/// <list type="bullet">
/// <item><description><see cref="Regression"/> predicts numbers.</description></item>
/// <item><description><see cref="BinaryClassification"/> predicts one of two outcomes.</description></item>
/// <item><description><see cref="MultiClassClassification"/> predicts one of many outcomes.</description></item>
/// <item><description><see cref="TimeSeriesForecasting"/> predicts future values from past time-ordered values.</description></item>
/// <item><description><see cref="ReinforcementLearning"/> learns by interacting with an environment to maximize reward.</description></item>
/// </list>
/// </para>
/// </remarks>
public enum AutoMLTaskFamily
{
    /// <summary>
    /// Supervised regression.
    /// </summary>
    Regression,

    /// <summary>
    /// Binary classification.
    /// </summary>
    BinaryClassification,

    /// <summary>
    /// Multi-class (single-label) classification.
    /// </summary>
    MultiClassClassification,

    /// <summary>
    /// Time-series forecasting.
    /// </summary>
    TimeSeriesForecasting,

    /// <summary>
    /// Reinforcement learning.
    /// </summary>
    ReinforcementLearning
}

