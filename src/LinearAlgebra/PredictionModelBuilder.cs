﻿global using AiDotNet.FeatureSelectors;
global using AiDotNet.FitnessCalculators;
global using AiDotNet.Regularization;
global using AiDotNet.Optimizers;
global using AiDotNet.Normalizers;
global using AiDotNet.OutlierRemoval;
global using AiDotNet.DataProcessor;
global using AiDotNet.FitDetectors;

namespace AiDotNet.LinearAlgebra;

public class PredictionModelBuilder<T> : IPredictionModelBuilder<T>
{
    private readonly PredictionStatsOptions _options;
    private IFeatureSelector<T>? _featureSelector;
    private INormalizer<T>? _normalizer;
    private IRegularization<T>? _regularization;
    private IFitnessCalculator<T>? _fitnessCalculator;
    private IFitDetector<T>? _fitDetector;
    private IRegression<T>? _regression;
    private IOptimizer<T>? _optimizer;
    private IDataPreprocessor<T>? _dataPreprocessor;
    private IOutlierRemoval<T>? _outlierRemoval;

    public PredictionModelBuilder(PredictionStatsOptions? options = null)
    {
        _options = options ?? new PredictionStatsOptions();
    }

    public IPredictionModelBuilder<T> ConfigureFeatureSelector(IFeatureSelector<T> selector)
    {
        _featureSelector = selector;
        return this;
    }

    public IPredictionModelBuilder<T> ConfigureNormalizer(INormalizer<T> normalizer)
    {
        _normalizer = normalizer;
        return this;
    }

    public IPredictionModelBuilder<T> ConfigureRegularization(IRegularization<T> regularization)
    {
        _regularization = regularization;
        return this;
    }

    public IPredictionModelBuilder<T> ConfigureFitnessCalculator(IFitnessCalculator<T> calculator)
    {
        _fitnessCalculator = calculator;
        return this;
    }

    public IPredictionModelBuilder<T> ConfigureFitDetector(IFitDetector<T> detector)
    {
        _fitDetector = detector;
        return this;
    }

    public IPredictionModelBuilder<T> ConfigureRegression(IRegression<T> regression)
    {
        _regression = regression;
        return this;
    }

    public IPredictionModelBuilder<T> ConfigureOptimizer(IOptimizer<T> optimizationAlgorithm)
    {
        _optimizer = optimizationAlgorithm;
        return this;
    }

    public IPredictionModelBuilder<T> ConfigureDataPreprocessor(IDataPreprocessor<T> dataPreprocessor)
    {
        _dataPreprocessor = dataPreprocessor;
        return this;
    }

    public IPredictionModelBuilder<T> ConfigureOutlierRemoval(IOutlierRemoval<T> outlierRemoval)
    {
        _outlierRemoval = outlierRemoval;
        return this;
    }

    public IPredictiveModel<T> Build(Matrix<T> x, Vector<T> y)
    {
        // Validate inputs
        if (x == null)
            throw new ArgumentNullException(nameof(x), "Input features matrix can't be null");
        if (y == null)
            throw new ArgumentNullException(nameof(y), "Output vector can't be null");
        if (x.Rows != y.Length)
            throw new ArgumentException("Number of rows in features matrix must match length of actual values vector", nameof(x));
        if (_regression == null)
            throw new InvalidOperationException("Regression method must be specified");

        // Use defaults for these interfaces if they aren't set
        var normalizer = _normalizer ?? new NoNormalizer<T>();
        var optimizer = _optimizer ?? new NormalOptimizer<T>();
        var featureSelector = _featureSelector ?? new NoFeatureSelector<T>();
        var fitDetector = _fitDetector ?? new DefaultFitDetector<T>();
        var fitnessCalculator = _fitnessCalculator ?? new RSquaredFitnessCalculator<T>();
        var regularization = _regularization ?? new NoRegularization<T>();
        var outlierRemoval = _outlierRemoval ?? new NoOutlierRemoval<T>();
        var dataPreprocessor = _dataPreprocessor ?? new DefaultDataPreprocessor<T>(normalizer, featureSelector, outlierRemoval);

        // Preprocess the data
        var (preprocessedX, preprocessedY, normInfo) = dataPreprocessor.PreprocessData(x, y);

        // Split the data
        var (XTrain, yTrain, XVal, yVal, XTest, yTest) = dataPreprocessor.SplitData(preprocessedX, preprocessedY);

        // Optimize the model
        var optimizationResult = optimizer.Optimize(OptimizerHelper<T>.CreateOptimizationInputData(XTrain, yTrain, XVal, yVal, XTest, yTest));

        return new PredictionModelResult<T>(_regression, optimizationResult, normInfo);
    }

    public Vector<T> Predict(Matrix<T> newData, IPredictiveModel<T> modelResult)
    {
        return modelResult.Predict(newData);
    }

    public void SaveModel(IPredictiveModel<T> modelResult, string filePath)
    {
        File.WriteAllBytes(filePath, SerializeModel(modelResult));
    }

    public IPredictiveModel<T> LoadModel(string filePath)
    {
        byte[] modelData = File.ReadAllBytes(filePath);
        return DeserializeModel(modelData);
    }

    public byte[] SerializeModel(IPredictiveModel<T> modelResult)
    {
        return modelResult.Serialize();
    }

    public IPredictiveModel<T> DeserializeModel(byte[] modelData)
    {
        var result = new PredictionModelResult<T>();
        result.Deserialize(modelData);

        return result;
    }
}