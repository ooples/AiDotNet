namespace AiDotNet.Interfaces;

public interface IFitDetector
{
    FitDetectionResult DetectFit(
        Vector<double> trainingErrors,
        Vector<double> validationErrors,
        Vector<double> testErrors,
        double trainingFitness,
        double validationFitness,
        double testFitness);
}