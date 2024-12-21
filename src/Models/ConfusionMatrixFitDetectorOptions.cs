namespace AiDotNet.Models;

public class ConfusionMatrixFitDetectorOptions
{
    public double GoodFitThreshold { get; set; } = 0.8;
    public double ModerateFitThreshold { get; set; } = 0.6;
    public double ClassImbalanceThreshold { get; set; } = 0.2;
    public MetricType PrimaryMetric { get; set; } = MetricType.F1Score;
    public double ConfidenceThreshold { get; set; } = 0.5;
}