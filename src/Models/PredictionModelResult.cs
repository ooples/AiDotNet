namespace AiDotNet.Models;

public class PredictionModelResult
{
    public IRegression? Model { get; set; }
    public (Matrix<double>, Vector<double> y) TrainingData { get; set; }
    public (Matrix<double>, Vector<double> y) ValidationData { get; set; }
    public (Matrix<double>, Vector<double> y) TestingData { get; set; }
    public double? TrainingFitness { get; set; }
    public double? ValidationFitness { get; set; }
    public double? TestingFitness { get; set; }
    public FitDetectionResult FitDetectionResult { get; set; } = new();
    public OptimizationResult OptimizationResult { get; set; } = new();
    public NormalizationInfo NormalizationInfo { get; set; } = new();
}