namespace AiDotNet.Interfaces;

public interface IPredictionModelBuilder<T>
{
    IPredictionModelBuilder<T> WithFeatureSelector(IFeatureSelector<T> selector);
    IPredictionModelBuilder<T> WithNormalizer(INormalizer<T> normalizer);
    IPredictionModelBuilder<T> WithRegularization(IRegularization<T> regularization);
    IPredictionModelBuilder<T> WithFitnessCalculator(IFitnessCalculator<T> calculator);
    IPredictionModelBuilder<T> WithFitDetector(IFitDetector<T> detector);
    IPredictionModelBuilder<T> WithRegression(IRegression<T> regression);
    IPredictionModelBuilder<T> WithOptimizer(IOptimizationAlgorithm<T> optimizationAlgorithm, OptimizationAlgorithmOptions optimizationOptions);
    IPredictionModelBuilder<T> WithDataPreprocessor(IDataPreprocessor<T> dataPreprocessor);
    PredictionModelResult<T> Build(Matrix<T> x, Vector<T> y);
    Vector<T> Predict(Matrix<T> newData, PredictionModelResult<T> model);
    void SaveModel(PredictionModelResult<T> model, string filePath);
    PredictionModelResult<T> LoadModel(string filePath);
    string SerializeModel(PredictionModelResult<T> model);
    PredictionModelResult<T> DeserializeModel(string jsonString);
}