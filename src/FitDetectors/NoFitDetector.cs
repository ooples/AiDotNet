namespace AiDotNet.FitDetectors;

public class NoFitDetector : IFitDetector
{
    public FitDetectionResult DetectFit(Vector<double> trainingMetrics, Vector<double> validationMetrics)
    {
        throw new NotImplementedException();
    }
}