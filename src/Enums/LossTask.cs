namespace AiDotNet.Enums;

/// <summary>
/// Specific tasks a loss function is designed or commonly used for.
/// </summary>
public enum LossTask
{
    /// <summary>Binary classification (two classes).</summary>
    BinaryClassification,
    /// <summary>Multi-class classification (mutually exclusive classes).</summary>
    MultiClass,
    /// <summary>Multi-label classification (non-exclusive labels).</summary>
    MultiLabel,
    /// <summary>Continuous value regression.</summary>
    Regression,
    /// <summary>Object detection (bounding box + class).</summary>
    ObjectDetection,
    /// <summary>Semantic segmentation (per-pixel classification).</summary>
    SemanticSegmentation,
    /// <summary>Instance segmentation (per-object masks).</summary>
    InstanceSegmentation,
    /// <summary>Image generation/reconstruction.</summary>
    ImageGeneration,
    /// <summary>Text generation/language modeling.</summary>
    TextGeneration,
    /// <summary>Ranking/retrieval tasks.</summary>
    Ranking,
    /// <summary>Embedding/metric learning.</summary>
    Embedding,
    /// <summary>Denoising (noise removal).</summary>
    Denoising,
    /// <summary>Super-resolution (upscaling).</summary>
    SuperResolution,
    /// <summary>Depth estimation.</summary>
    DepthEstimation,
    /// <summary>Anomaly/outlier detection.</summary>
    AnomalyDetection,
    /// <summary>Time series forecasting.</summary>
    TimeSeriesForecasting,
    /// <summary>Survival analysis.</summary>
    SurvivalAnalysis
}
