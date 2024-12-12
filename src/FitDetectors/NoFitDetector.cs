namespace AiDotNet.FitDetectors;

public class NoFitDetector<T> : IFitDetector<T>
{
    public FitDetectorResult<T> DetectFit(ErrorStats<T> trainingErrorStats, ErrorStats<T> validationErrorStats, ErrorStats<T> testErrorStats, 
        BasicStats<T> trainingBasicStats, BasicStats<T> validationBasicStats, BasicStats<T> testBasicStats)
    {
        throw new NotImplementedException();
    }
}