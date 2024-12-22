namespace AiDotNet.Models;

public class StratifiedKFoldCrossValidationFitDetectorOptions
{
    public double OverfitThreshold { get; set; } = 0.1;
    public double UnderfitThreshold { get; set; } = 0.6;
    public double HighVarianceThreshold { get; set; } = 0.1;
    public double GoodFitThreshold { get; set; } = 0.8;
    public double StabilityThreshold { get; set; } = 0.05;
    public MetricType PrimaryMetric { get; set; } = MetricType.F1Score;
}