namespace AiDotNet.FitDetectors;

public class NoFitDetector : IFitDetector
{
    public FitDetectorResult DetectFit(ErrorStats trainingErrorStats, ErrorStats validationErrorStats, ErrorStats testErrorStats, 
        BasicStats trainingBasicStats, BasicStats validationBasicStats, BasicStats testBasicStats)
    {
        throw new NotImplementedException();
    }
}