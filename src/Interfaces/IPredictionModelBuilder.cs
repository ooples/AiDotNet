namespace AiDotNet.Interfaces;

public interface IPredictionModelBuilder<T>
{
    IPredictionModelBuilder<T> ConfigureFeatureSelector(IFeatureSelector<T> selector);
    IPredictionModelBuilder<T> ConfigureNormalizer(INormalizer<T> normalizer);
    IPredictionModelBuilder<T> ConfigureRegularization(IRegularization<T> regularization);
    IPredictionModelBuilder<T> ConfigureFitnessCalculator(IFitnessCalculator<T> calculator);
    IPredictionModelBuilder<T> ConfigureFitDetector(IFitDetector<T> detector);
    IPredictionModelBuilder<T> ConfigureRegression(IRegression<T> regression);
    IPredictionModelBuilder<T> ConfigureOptimizer(IOptimizationAlgorithm<T> optimizationAlgorithm);
    IPredictionModelBuilder<T> ConfigureDataPreprocessor(IDataPreprocessor<T> dataPreprocessor);
    IPredictionModelBuilder<T> ConfigureOutlierRemoval(IOutlierRemoval<T> outlierRemoval);
    PredictionModelResult<T> Build(Matrix<T> x, Vector<T> y);
    Vector<T> Predict(Matrix<T> newData, PredictionModelResult<T> model);
    void SaveModel(PredictionModelResult<T> model, string filePath);
    PredictionModelResult<T> LoadModel(string filePath);
    string SerializeModel(PredictionModelResult<T> model);
    PredictionModelResult<T> DeserializeModel(string jsonString);
}