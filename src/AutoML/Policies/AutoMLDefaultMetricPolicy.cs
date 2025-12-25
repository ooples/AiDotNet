using AiDotNet.Enums;

namespace AiDotNet.AutoML.Policies;

/// <summary>
/// Provides industry-standard default optimization metrics for AutoML task families.
/// </summary>
/// <remarks>
/// <para>
/// This policy is used when the user does not explicitly provide an optimization metric override.
/// It prefers metrics that are commonly used in industry for each task family.
/// </para>
/// <para>
/// <b>For Beginners:</b> AutoML needs a "score" to decide which trial is best.
/// This class chooses that default score (like RMSE for regression or AUC for binary classification).
/// </para>
/// </remarks>
internal static class AutoMLDefaultMetricPolicy
{
    public static (MetricType Metric, bool Maximize) GetDefault(AutoMLTaskFamily taskFamily)
    {
        return taskFamily switch
        {
            AutoMLTaskFamily.Regression => (MetricType.RMSE, Maximize: false),
            AutoMLTaskFamily.BinaryClassification => (MetricType.AUCROC, Maximize: true),
            AutoMLTaskFamily.MultiClassClassification => (MetricType.F1Score, Maximize: true),
            AutoMLTaskFamily.MultiLabelClassification => (MetricType.F1Score, Maximize: true),
            AutoMLTaskFamily.TimeSeriesForecasting => (MetricType.SMAPE, Maximize: false),
            AutoMLTaskFamily.TimeSeriesAnomalyDetection => (MetricType.AUCPR, Maximize: true),
            AutoMLTaskFamily.Ranking => (MetricType.NormalizedDiscountedCumulativeGain, Maximize: true),
            AutoMLTaskFamily.Recommendation => (MetricType.NormalizedDiscountedCumulativeGain, Maximize: true),
            AutoMLTaskFamily.GraphNodeClassification => (MetricType.F1Score, Maximize: true),
            AutoMLTaskFamily.GraphClassification => (MetricType.F1Score, Maximize: true),
            AutoMLTaskFamily.GraphLinkPrediction => (MetricType.AUCROC, Maximize: true),
            AutoMLTaskFamily.GraphGeneration => (MetricType.MeanSquaredError, Maximize: false),
            AutoMLTaskFamily.TextClassification => (MetricType.F1Score, Maximize: true),
            AutoMLTaskFamily.SequenceTagging => (MetricType.F1Score, Maximize: true),
            AutoMLTaskFamily.Translation => (MetricType.Perplexity, Maximize: false),
            AutoMLTaskFamily.TextGeneration => (MetricType.Perplexity, Maximize: false),
            AutoMLTaskFamily.SpeechRecognition => (MetricType.Accuracy, Maximize: true),
            AutoMLTaskFamily.ImageClassification => (MetricType.Accuracy, Maximize: true),
            AutoMLTaskFamily.ObjectDetection => (MetricType.MeanAveragePrecision, Maximize: true),
            AutoMLTaskFamily.ImageSegmentation => (MetricType.F1Score, Maximize: true),
            AutoMLTaskFamily.ReinforcementLearning => (MetricType.AverageEpisodeReward, Maximize: true),
            
            // 3D Point Cloud Tasks
            AutoMLTaskFamily.PointCloudClassification => (MetricType.Accuracy, Maximize: true),
            AutoMLTaskFamily.PointCloudSegmentation => (MetricType.MeanIoU, Maximize: true),
            AutoMLTaskFamily.PointCloudCompletion => (MetricType.ChamferDistance, Maximize: false),
            
            // 3D Volumetric Tasks
            AutoMLTaskFamily.VolumetricClassification => (MetricType.Accuracy, Maximize: true),
            AutoMLTaskFamily.VolumetricSegmentation => (MetricType.MeanIoU, Maximize: true),
            
            // 3D Mesh Tasks
            AutoMLTaskFamily.MeshClassification => (MetricType.Accuracy, Maximize: true),
            AutoMLTaskFamily.MeshSegmentation => (MetricType.MeanIoU, Maximize: true),
            
            // Neural Radiance Fields
            AutoMLTaskFamily.RadianceFieldReconstruction => (MetricType.PSNR, Maximize: true),
            
            // 3D Detection and Depth
            AutoMLTaskFamily.ThreeDObjectDetection => (MetricType.MeanAveragePrecision, Maximize: true),
            AutoMLTaskFamily.DepthEstimation => (MetricType.AbsoluteRelativeError, Maximize: false),
            
            _ => (MetricType.Accuracy, Maximize: true)
        };
    }
}
