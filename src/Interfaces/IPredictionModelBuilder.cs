namespace AiDotNet.Interfaces;

public interface IPredictionModelBuilder
{
    IPredictionModelBuilder WithFeatureSelector(IFeatureSelector selector);
    IPredictionModelBuilder WithNormalizer(INormalizer normalizer);
    IPredictionModelBuilder WithRegularization(IRegularization regularization);
    IPredictionModelBuilder WithFitnessCalculator(IFitnessCalculator calculator);
    IPredictionModelBuilder WithFitDetector(IFitDetector detector);
    IPredictionModelBuilder WithRegression(IRegression regression);
    IPredictionModelBuilder WithOptimizer(IOptimizationAlgorithm optimizationAlgorithm, OptimizationAlgorithmOptions optimizationOptions);
    IPredictionModelBuilder WithDataPreprocessor(IDataPreprocessor dataPreprocessor);
    PredictionModelResult Build(Matrix<double> x, Vector<double> y);
    Vector<double> Predict(Matrix<double> newData, PredictionModelResult model);
    void SaveModel(PredictionModelResult model, string filePath);
    PredictionModelResult LoadModel(string filePath);
    string SerializeModel(PredictionModelResult model);
    PredictionModelResult DeserializeModel(string jsonString);
}