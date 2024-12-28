namespace AiDotNet.Regression;

public class SymbolicRegression<T> : NonLinearRegressionBase<T>
{
    private readonly SymbolicRegressionOptions _options;
    private readonly IFitnessCalculator<T> _fitnessCalculator;
    private readonly INormalizer<T> _normalizer;
    private readonly IFeatureSelector<T> _featureSelector;
    private readonly IFitDetector<T> _fitDetector;
    private readonly IOutlierRemoval<T> _outlierRemoval;
    private readonly IDataPreprocessor<T> _dataPreprocessor;
    private readonly IOptimizationAlgorithm<T> _optimizer;
    private ISymbolicModel<T> _bestModel;
    private T _bestFitness;

    public SymbolicRegression(
        SymbolicRegressionOptions? options = null, 
        IRegularization<T>? regularization = null,
        IFitnessCalculator<T>? fitnessCalculator = null,
        INormalizer<T>? normalizer = null,
        IFeatureSelector<T>? featureSelector = null,
        IFitDetector<T>? fitDetector = null,
        IOutlierRemoval<T>? outlierRemoval = null,
        IDataPreprocessor<T>? dataPreprocessor = null)
        : base(options, regularization)
    {
        _options = options ?? new SymbolicRegressionOptions();
        _optimizer = new GeneticAlgorithmOptimizer<T>(new GeneticAlgorithmOptions
        {
            PopulationSize = _options.PopulationSize,
            MaxGenerations = _options.MaxGenerations,
            MutationRate = _options.MutationRate,
            CrossoverRate = _options.CrossoverRate
        });
        _fitnessCalculator = fitnessCalculator ?? new RSquaredFitnessCalculator<T>();
        _normalizer = normalizer ?? new NoNormalizer<T>();
        _featureSelector = featureSelector ?? new NoFeatureSelector<T>();
        _fitDetector = fitDetector ?? new DefaultFitDetector<T>();
        _outlierRemoval = outlierRemoval ?? new NoOutlierRemoval<T>();
        _dataPreprocessor = dataPreprocessor ?? new DefaultDataPreprocessor<T>(_normalizer, _featureSelector, _outlierRemoval);
        _bestModel = SymbolicModelFactory<T>.CreateRandomModel(true, 1, NumOps);
        _bestFitness = _fitnessCalculator.IsHigherScoreBetter ? NumOps.MinValue : NumOps.MaxValue;
    }

    protected override void OptimizeModel(Matrix<T> x, Vector<T> y)
    {
        // Preprocess the data
        var (preprocessedX, preprocessedY, normInfo) = _dataPreprocessor.PreprocessData(x, y);

        // Split the data
        var (XTrain, yTrain, XVal, yVal, XTest, yTest) = _dataPreprocessor.SplitData(preprocessedX, preprocessedY);

        // Optimize the model
        var optimizationResult = _optimizer.Optimize(
            XTrain, yTrain, 
            XVal, yVal, 
            XTest, yTest, 
            this, 
            Regularization, 
            _normalizer,
            normInfo, 
            _fitnessCalculator, 
            _fitDetector);

        _bestFitness = optimizationResult.BestFitnessScore;
        _bestModel = optimizationResult.BestSolution ?? throw new InvalidOperationException("Optimization result does not contain a valid symbolic model.");
    }

    public override Vector<T> Predict(Matrix<T> X)
    {
        var predictions = new Vector<T>(X.Rows, NumOps);
        for (int i = 0; i < X.Rows; i++)
        {
            predictions[i] = _bestModel.Evaluate(X.GetRow(i));
        }

        return predictions;
    }

    protected override T PredictSingle(Vector<T> input)
    {
        Vector<T> regularizedInput = Regularization.RegularizeCoefficients(input);
        return _bestModel.Evaluate(regularizedInput);
    }

    private T ModelFromCoefficients(Vector<T> coefficients, Vector<T> input)
    {
        T sum = NumOps.Zero;
        for (int i = 0; i < input.Length; i++)
        {
            sum = NumOps.Add(sum, NumOps.Multiply(coefficients[i], input[i]));
        }

        return sum;
    }

    protected override ModelType GetModelType() => ModelType.SymbolicRegression;
}