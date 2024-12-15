namespace AiDotNet.Interfaces;

public interface IPredictionModelBuilder<T>
{
    IPredictionModelBuilder<T> WithFeatureSelector(IFeatureSelector<T> selector);
    IPredictionModelBuilder<T> WithNormalizer(INormalizer<T> normalizer);
    IPredictionModelBuilder<T> WithRegularization(IRegularization<T> regularization, RegularizationOptions? regularizationOptions = null);
    IPredictionModelBuilder<T> WithFitnessCalculator(IFitnessCalculator<T> calculator, FitnessCalculatorOptions? _fitnessCalculatorOptions = null);
    IPredictionModelBuilder<T> WithFitDetector(IFitDetector<T> detector);
    IPredictionModelBuilder<T> WithRegression(IRegression<T> regression, RegressionOptions<T>? regressionOptions = null);
    IPredictionModelBuilder<T> WithOptimizer(IOptimizationAlgorithm<T> optimizationAlgorithm, OptimizationAlgorithmOptions? optimizationOptions = null);
    IPredictionModelBuilder<T> WithDataPreprocessor(IDataPreprocessor<T> dataPreprocessor);
    IPredictionModelBuilder<T> WithOutlierRemoval(IOutlierRemoval<T> outlierRemoval);
    PredictionModelResult<T> Build(Matrix<T> x, Vector<T> y);
    Vector<T> Predict(Matrix<T> newData, PredictionModelResult<T> model);
    void SaveModel(PredictionModelResult<T> model, string filePath);
    PredictionModelResult<T> LoadModel(string filePath);
    string SerializeModel(PredictionModelResult<T> model);
    PredictionModelResult<T> DeserializeModel(string jsonString);
}