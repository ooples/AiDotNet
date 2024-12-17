namespace AiDotNet.Interfaces;

public interface IOptimizationAlgorithm<T>
{
    public OptimizationResult<T> Optimize(
        Matrix<T> XTrain,
        Vector<T> yTrain,
        Matrix<T> XVal,
        Vector<T> yVal,
        Matrix<T> XTest,
        Vector<T> yTest,
        IRegression<T> regressionMethod,
        IRegularization<T> regularization,
        INormalizer<T> normalizer,
        NormalizationInfo<T> normInfo,
        IFitnessCalculator<T> fitnessCalculator,
        IFitDetector<T> fitDetector);

    bool ShouldEarlyStop(List<OptimizationIterationInfo<T>> iterationHistory, IFitnessCalculator<T> fitnessCalculator);
}