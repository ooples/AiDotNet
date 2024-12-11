using AiDotNet.FeatureSelectors;
using AiDotNet.FitDetectors;
using AiDotNet.FitnessCalculators;
using AiDotNet.Optimizers;
using AiDotNet.Regularization;

namespace AiDotNet.LinearAlgebra;

public class PredictionModelBuilder
{
    private readonly PredictionModelOptions _options;
    private OptimizationAlgorithmOptions _optimizationOptions { get; set; }
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
        _optimizationOptions = new OptimizationAlgorithmOptions();
    }

    public PredictionModelBuilder WithFeatureSelector(IFeatureSelector selector)
    {
        _featureSelector = selector;
        return this;
    }

    public PredictionModelBuilder WithNormalizer(INormalizer normalizer)
    {
        _normalizer = normalizer;
        return this;
    }

    public PredictionModelBuilder WithRegularization(IRegularization regularization)
    {
        _regularization = regularization;
        return this;
    }

    public PredictionModelBuilder WithFitnessCalculator(IFitnessCalculator calculator)
    {
        _fitnessCalculator = calculator;
        return this;
    }

    public PredictionModelBuilder WithFitDetector(IFitDetector detector)
    {
        _fitDetector = detector;
        return this;
    }

    public PredictionModelBuilder WithRegression(IRegression regression)
    {
        _regression = regression;
        return this;
    }

    public PredictionModelBuilder WithOptimizer(IOptimizationAlgorithm optimizationAlgorithm, OptimizationAlgorithmOptions optimizationOptions)
    {
        _optimizer = optimizationAlgorithm;
        _optimizationOptions = optimizationOptions;
        return this;
    }

    public PredictionModelBuilder WithDataPreprocessor(IDataPreprocessor dataPreprocessor)
    {
        _dataPreprocessor = dataPreprocessor;
        return this;
    }

    public PredictionModelResult Build(Matrix<double> x, Vector<double> y)
    {
        // Validate inputs
        if (x == null || y == null)
            throw new ArgumentNullException("Inputs and outputs can't be null");
        if (x.Rows != y.Length)
            throw new ArgumentException("Number of rows in features matrix must match length of actual values vector");
        if (_regression == null)
            throw new ArgumentException("Regression method must be specified");

        // Use defaults for these interfaces if they aren't set
        var normalizer = _normalizer ?? new NoNormalizer();
        var optimizer = _optimizer ?? new NormalOptimizer();
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
        var optimizationResult = optimizer.Optimize(XTrain, yTrain, XVal, yVal, XTest, yTest, _options, _optimizationOptions, _regression, regularization, normalizer, 
            normInfo, fitnessCalculator, fitDetector);

        return new PredictionModelResult
        {
            Model = _regression,
            TrainingData = (XTrain, yTrain),
            ValidationData = (XVal, yVal),
            TestingData = (XTest, yTest),
            TrainingFitness = optimizationResult.TrainingMetrics["R2"],  // Assuming R2 is used as fitness
            ValidationFitness = optimizationResult.ValidationMetrics["R2"],
            TestingFitness = optimizationResult.TestMetrics["R2"],
            FitDetectionResult = optimizationResult.FitDetectionResult,
            OptimizationResult = optimizationResult,
            NormalizationInfo = normInfo
        };
    }

    public Vector<double> Predict(Matrix<double> newData, PredictionModelResult model)
    {
        return model.Predict(newData);
    }

    public void SaveModel(PredictionModelResult model, string filePath)
    {
        model.SaveModel(filePath);
    }

    public PredictionModelResult LoadModel(string filePath)
    {
        return PredictionModelResult.LoadModel(filePath);
    }

    public string SerializeModel(PredictionModelResult model)
    {
        return model.SerializeToJson();
    }

    public PredictionModelResult DeserializeModel(string jsonString)
    {
        return PredictionModelResult.DeserializeFromJson(jsonString);
    }
}