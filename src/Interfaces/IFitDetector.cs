namespace AiDotNet.Interfaces;

public interface IFitDetector
{
    FitDetectorResult DetectFit(
        ErrorStats trainingErrorStats,
        ErrorStats validationErrorStats,
        ErrorStats testErrorStats,
        BasicStats trainingBasicStats,
        BasicStats validationBasicStats,
        BasicStats testBasicStats);
}