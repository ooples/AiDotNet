namespace AiDotNet.Interfaces;

public interface IOptimizationAlgorithm
{
    public OptimizationResult Optimize(
        Matrix<double> XTrain,
        Vector<double> yTrain,
        Matrix<double> XVal,
        Vector<double> yVal,
        Matrix<double> XTest,
        Vector<double> yTest,
        PredictionModelOptions modelOptions,
        OptimizationAlgorithmOptions optimizationOptions,
        IRegression regressionMethod,
        IRegularization regularization,
        INormalizer normalizer,
        NormalizationInfo normInfo,
        IFitnessCalculator fitnessCalculator,
        IFitDetector fitDetector);

    bool ShouldEarlyStop(List<OptimizationIteration> iterationHistory, OptimizationAlgorithmOptions options);
}