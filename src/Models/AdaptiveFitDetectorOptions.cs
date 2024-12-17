namespace AiDotNet.Models;

public class AdaptiveFitDetectorOptions
{
    public ResidualAnalysisFitDetectorOptions ResidualAnalysisOptions { get; set; } = new ResidualAnalysisFitDetectorOptions();
    public LearningCurveFitDetectorOptions LearningCurveOptions { get; set; } = new LearningCurveFitDetectorOptions();
    public HybridFitDetectorOptions HybridOptions { get; set; } = new HybridFitDetectorOptions();
    public double ComplexityThreshold { get; set; } = 1.0;
    public double PerformanceThreshold { get; set; } = 0.8;
}