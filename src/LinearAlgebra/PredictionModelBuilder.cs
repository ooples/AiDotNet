global using AiDotNet.FeatureSelectors;
global using AiDotNet.FitnessCalculators;
global using AiDotNet.Regularization;
global using AiDotNet.Optimizers;
global using AiDotNet.Normalizers;
global using AiDotNet.OutlierRemoval;

namespace AiDotNet.LinearAlgebra;

public class PredictionModelBuilder<T> : IPredictionModelBuilder<T>
{
    private readonly PredictionModelOptions _options;
    private OptimizationAlgorithmOptions? _optimizationOptions;
    private RegularizationOptions? _regularizationOptions;
    private RegressionOptions<T>? _regressionOptions;
    private FitnessCalculatorOptions? _fitnessCalculatorOptions;
    private IFeatureSelector<T>? _featureSelector;
    private INormalizer<T>? _normalizer;
    private IRegularization<T>? _regularization;
    private IFitnessCalculator<T>? _fitnessCalculator;
    private IFitDetector<T>? _fitDetector;
    private IRegression<T>? _regression;
    private IOptimizationAlgorithm<T>? _optimizer;
    private IDataPreprocessor<T>? _dataPreprocessor;
    private IOutlierRemoval<T>? _outlierRemoval;

    public PredictionModelBuilder(PredictionModelOptions? options = null)
    {
        _options = options ?? new PredictionModelOptions();
    }

    public IPredictionModelBuilder<T> WithFeatureSelector(IFeatureSelector<T> selector)
    {
        _featureSelector = selector;
        return this;
    }

    public IPredictionModelBuilder<T> WithNormalizer(INormalizer<T> normalizer)
    {
        _normalizer = normalizer;
        return this;
    }

    public IPredictionModelBuilder<T> WithRegularization(IRegularization<T> regularization, RegularizationOptions? regularizationOptions = null)
    {
        _regularization = regularization;
        _regularizationOptions = regularizationOptions;
        return this;
    }

    public IPredictionModelBuilder<T> WithFitnessCalculator(IFitnessCalculator<T> calculator, FitnessCalculatorOptions? fitnessCalculatorOptions = null)
    {
        _fitnessCalculator = calculator;
        _fitnessCalculatorOptions = fitnessCalculatorOptions;
        return this;
    }

    public IPredictionModelBuilder<T> WithFitDetector(IFitDetector<T> detector)
    {
        _fitDetector = detector;
        return this;
    }

    public IPredictionModelBuilder<T> WithRegression(IRegression<T> regression, RegressionOptions<T>? regressionOptions = null)
    {
        _regression = regression;
        _regressionOptions = regressionOptions;
        return this;
    }

    public IPredictionModelBuilder<T> WithOptimizer(IOptimizationAlgorithm<T> optimizationAlgorithm, OptimizationAlgorithmOptions? optimizationOptions = null)
    {
        _optimizer = optimizationAlgorithm;
        _optimizationOptions = optimizationOptions;
        return this;
    }

    public IPredictionModelBuilder<T> WithDataPreprocessor(IDataPreprocessor<T> dataPreprocessor)
    {
        _dataPreprocessor = dataPreprocessor;
        return this;
    }

    public IPredictionModelBuilder<T> WithOutlierRemoval(IOutlierRemoval<T> outlierRemoval)
    {
        _outlierRemoval = outlierRemoval;
        return this;
    }

    public PredictionModelResult<T> Build(Matrix<T> x, Vector<T> y)
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
        var optimizerOptions = _optimizationOptions ?? new OptimizationAlgorithmOptions();
        var featureSelector = _featureSelector ?? new NoFeatureSelector<T>();
        var fitDetector = _fitDetector ?? new DefaultFitDetector<T>();
        var fitnessCalculator = _fitnessCalculator ?? new RSquaredFitnessCalculator<T>();
        var regularization = _regularization ?? new NoRegularization<T>();
        var outlierRemoval = _outlierRemoval ?? new NoOutlierRemoval<T>();
        var dataPreprocessor = _dataPreprocessor ?? new DataPreprocessor<T>(normalizer, featureSelector, outlierRemoval, _options);

        // Preprocess the data
        var (preprocessedX, preprocessedY, normInfo) = dataPreprocessor.PreprocessData(x, y);

        // Split the data
        var (XTrain, yTrain, XVal, yVal, XTest, yTest) = dataPreprocessor.SplitData(preprocessedX, preprocessedY);

        // Optimize the model
        var optimizationResult = optimizer.Optimize(XTrain, yTrain, XVal, yVal, XTest, yTest, _options, optimizerOptions, _regression, regularization, normalizer, 
            normInfo, fitnessCalculator, fitDetector);

        return new PredictionModelResult<T>
        {
            Model = _regression,
            OptimizationResult = optimizationResult,
            NormalizationInfo = normInfo
        };
    }

    public Vector<T> Predict(Matrix<T> newData, PredictionModelResult<T> modelResult)
    {
        return modelResult.Predict(newData);
    }

    public void SaveModel(PredictionModelResult<T> modelResult, string filePath)
    {
        modelResult.SaveModel(filePath);
    }

    public PredictionModelResult<T> LoadModel(string filePath)
    {
        return PredictionModelResult<T>.LoadModel(filePath);
    }

    public string SerializeModel(PredictionModelResult<T> modelResult)
    {
        return modelResult.SerializeToJson();
    }

    public PredictionModelResult<T> DeserializeModel(string jsonString)
    {
        return PredictionModelResult<T>.DeserializeFromJson(jsonString);
    }
}