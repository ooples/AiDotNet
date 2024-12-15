namespace AiDotNet.FitDetectors;

public class NoFitDetector<T> : IFitDetector<T>
{
    public FitDetectorResult<T> DetectFit(ErrorStats<T> trainingErrorStats, ErrorStats<T> validationErrorStats, 
        ErrorStats<T> testErrorStats, BasicStats<T> trainingActualBasicStats, BasicStats<T> trainingPredictedBasicStats, 
        BasicStats<T> validationActualBasicStats, BasicStats<T> validationPredictedBasicStats, BasicStats<T> testActualBasicStats, 
        BasicStats<T> testPredictedBasicStats, PredictionStats<T> trainingPredictionStats, PredictionStats<T> validationPredictionStats, 
        PredictionStats<T> testPredictionStats)
    {
        throw new NotImplementedException();
    }
}