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
    /// Supervised regression (predicting continuous numeric values).
    /// </summary>
    Regression,

    /// <summary>
    /// Binary (two-class) classification.
    /// </summary>
    BinaryClassification,

    /// <summary>
    /// Multi-class (single-label) classification.
    /// </summary>
    MultiClassClassification,

    /// <summary>
    /// Multi-label classification (multiple labels can be true for one sample).
    /// </summary>
    MultiLabelClassification,

    /// <summary>
    /// Time-series forecasting (predicting future values from past time-ordered values).
    /// </summary>
    TimeSeriesForecasting,

    /// <summary>
    /// Time-series anomaly detection (detecting rare/abnormal events in time-ordered data).
    /// </summary>
    TimeSeriesAnomalyDetection,

    /// <summary>
    /// Ranking (ordering items by relevance, e.g., search results).
    /// </summary>
    Ranking,

    /// <summary>
    /// Recommendation (ranking or scoring items for users).
    /// </summary>
    Recommendation,

    /// <summary>
    /// Graph node classification.
    /// </summary>
    GraphNodeClassification,

    /// <summary>
    /// Graph (whole-graph) classification.
    /// </summary>
    GraphClassification,

    /// <summary>
    /// Graph link prediction.
    /// </summary>
    GraphLinkPrediction,

    /// <summary>
    /// Graph generation.
    /// </summary>
    GraphGeneration,

    /// <summary>
    /// Text classification.
    /// </summary>
    TextClassification,

    /// <summary>
    /// Sequence tagging (token-level labels like NER, POS).
    /// </summary>
    SequenceTagging,

    /// <summary>
    /// Machine translation.
    /// </summary>
    Translation,

    /// <summary>
    /// Text generation (language modeling / free-form generation).
    /// </summary>
    TextGeneration,

    /// <summary>
    /// Speech recognition (ASR).
    /// </summary>
    SpeechRecognition,

    /// <summary>
    /// Image classification.
    /// </summary>
    ImageClassification,

    /// <summary>
    /// Object detection.
    /// </summary>
    ObjectDetection,

    /// <summary>
    /// Image segmentation.
    /// </summary>
    ImageSegmentation,

    /// <summary>
    /// Reinforcement learning.
    /// </summary>
    ReinforcementLearning
}
