global using AiDotNet.FeatureSelectors;
global using AiDotNet.FitnessCalculators;
global using AiDotNet.Optimizers;
global using AiDotNet.Regularization;

namespace AiDotNet.LinearAlgebra;

public class PredictionModelBuilder : IPredictionModelBuilder
{
    private readonly PredictionModelOptions _options;
    private OptimizationAlgorithmOptions? _optimizationOptions;
    private IFeatureSelector? _featureSelector;
    private INormalizer? _normalizer;
    private IRegularization? _regularization;
    private IFitnessCalculator? _fitnessCalculator;
    private IFitDetector? _fitDetector;
    private IRegression? _regression;
    private IOptimizationAlgorithm? _optimizer;
    private IDataPreprocessor? _dataPreprocessor;

    public PredictionModelBuilder(PredictionModelOptions? options = null)
    {
        _options = options ?? new PredictionModelOptions();
    }

    public IPredictionModelBuilder WithFeatureSelector(IFeatureSelector selector)
    {
        _featureSelector = selector;
        return this;
    }

    public IPredictionModelBuilder WithNormalizer(INormalizer normalizer)
    {
        _normalizer = normalizer;
        return this;
    }

    public IPredictionModelBuilder WithRegularization(IRegularization regularization)
    {
        _regularization = regularization;
        return this;
    }

    public IPredictionModelBuilder WithFitnessCalculator(IFitnessCalculator calculator)
    {
        _fitnessCalculator = calculator;
        return this;
    }

    public IPredictionModelBuilder WithFitDetector(IFitDetector detector)
    {
        _fitDetector = detector;
        return this;
    }

    public IPredictionModelBuilder WithRegression(IRegression regression)
    {
        _regression = regression;
        return this;
    }

    public IPredictionModelBuilder WithOptimizer(IOptimizationAlgorithm optimizationAlgorithm, OptimizationAlgorithmOptions? optimizationOptions = null)
    {
        _optimizer = optimizationAlgorithm;
        _optimizationOptions = optimizationOptions;
        return this;
    }

    public IPredictionModelBuilder WithDataPreprocessor(IDataPreprocessor dataPreprocessor)
    {
        _dataPreprocessor = dataPreprocessor;
        return this;
    }

    public PredictionModelResult Build(Matrix<double> x, Vector<double> y)
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
        var normalizer = _normalizer ?? new NoNormalizer();
        var optimizer = _optimizer ?? new NormalOptimizer();
        var optimizerOptions = _optimizationOptions ?? new OptimizationAlgorithmOptions();
        var featureSelector = _featureSelector ?? new NoFeatureSelector();
        var fitDetector = _fitDetector ?? new DefaultFitDetector();
        var fitnessCalculator = _fitnessCalculator ?? new RSquaredFitnessCalculator();
        var regularization = _regularization ?? new NoRegularization();
        var dataPreprocessor = _dataPreprocessor ?? new DataPreprocessor(normalizer, featureSelector, _options);

        // Preprocess the data
        var (preprocessedX, preprocessedY, normInfo) = dataPreprocessor.PreprocessData(x, y);

        // Split the data
        var (XTrain, yTrain, XVal, yVal, XTest, yTest) = dataPreprocessor.SplitData(preprocessedX, preprocessedY);

        // Optimize the model
        var optimizationResult = optimizer.Optimize(XTrain, yTrain, XVal, yVal, XTest, yTest, _options, optimizerOptions, _regression, regularization, normalizer, 
            normInfo, fitnessCalculator, fitDetector);

        return new PredictionModelResult
        {
            Model = _regression,
            OptimizationResult = optimizationResult,
            NormalizationInfo = normInfo
        };
    }

    public Vector<double> Predict(Matrix<double> newData, PredictionModelResult modelResult)
    {
        return modelResult.Predict(newData);
    }

    public void SaveModel(PredictionModelResult modelResult, string filePath)
    {
        modelResult.SaveModel(filePath);
    }

    public PredictionModelResult LoadModel(string filePath)
    {
        return PredictionModelResult.LoadModel(filePath);
    }

    public string SerializeModel(PredictionModelResult modelResult)
    {
        return modelResult.SerializeToJson();
    }

    public PredictionModelResult DeserializeModel(string jsonString)
    {
        return PredictionModelResult.DeserializeFromJson(jsonString);
    }
}