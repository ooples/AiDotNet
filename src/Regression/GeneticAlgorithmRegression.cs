namespace AiDotNet.Regression;

public class GeneticAlgorithmRegression<T> : RegressionBase<T>
{
    private readonly GeneticAlgorithmOptimizerOptions _gaOptions;
    private GeneticAlgorithmOptimizer<T> _optimizer;
    private readonly IFitnessCalculator<T> _fitnessCalculator;
    private readonly INormalizer<T> _normalizer;
    private readonly IFeatureSelector<T> _featureSelector;
    private readonly IFitDetector<T> _fitDetector;
    private readonly IOutlierRemoval<T> _outlierRemoval;
    private readonly IDataPreprocessor<T> _dataPreprocessor;
    private VectorModel<T> _bestModel;

    public GeneticAlgorithmRegression(
        RegressionOptions<T>? options = null,
        GeneticAlgorithmOptimizerOptions? gaOptions = null,
        IRegularization<T>? regularization = null,
        IFitnessCalculator<T>? fitnessCalculator = null,
        INormalizer<T>? normalizer = null,
        IFeatureSelector<T>? featureSelector = null,
        IFitDetector<T>? fitDetector = null,
        IOutlierRemoval<T>? outlierRemoval = null,
        IDataPreprocessor<T>? dataPreprocessor = null)
        : base(options, regularization)
    {
        _gaOptions = gaOptions ?? new GeneticAlgorithmOptimizerOptions();
        _optimizer = new GeneticAlgorithmOptimizer<T>(gaOptions);
        _fitnessCalculator = fitnessCalculator ?? new RSquaredFitnessCalculator<T>();
        _normalizer = normalizer ?? new NoNormalizer<T>();
        _featureSelector = featureSelector ?? new NoFeatureSelector<T>();
        _outlierRemoval = outlierRemoval ?? new NoOutlierRemoval<T>();
        _dataPreprocessor = dataPreprocessor ?? new DefaultDataPreprocessor<T>(_normalizer, _featureSelector, _outlierRemoval);
        _fitDetector = fitDetector ?? new DefaultFitDetector<T>();
        
        _bestModel = new VectorModel<T>(Vector<T>.Empty());
    }

    public override void Train(Matrix<T> x, Vector<T> y)
    {
        // Preprocess the data
        var (preprocessedX, preprocessedY, normInfo) = _dataPreprocessor.PreprocessData(x, y);

        // Split the data
        var (xTrain, yTrain, xVal, yVal, xTest, yTest) = _dataPreprocessor.SplitData(preprocessedX, preprocessedY);

        var result = _optimizer.Optimize(OptimizerHelper<T>.CreateOptimizationInputData(xTrain, yTrain, xVal, yVal, xTest, yTest));

        _bestModel = (VectorModel<T>)result.BestSolution;
        UpdateCoefficientsAndIntercept();
    }

    public override Vector<T> Predict(Matrix<T> x)
    {
        return _bestModel.Predict(x);
    }

    protected override ModelType GetModelType() => ModelType.GeneticAlgorithmRegression;

    private void UpdateCoefficientsAndIntercept()
    {
        Coefficients = _bestModel.Coefficients;
        if (HasIntercept)
        {
            Intercept = Coefficients[0];
            Coefficients = Coefficients.Slice(1, Coefficients.Length - 1);
        }
        else
        {
            Intercept = NumOps.Zero;
        }
    }

    public override byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        // Serialize base class data
        byte[] baseData = base.Serialize();
        writer.Write(baseData.Length);
        writer.Write(baseData);

        // Serialize GeneticAlgorithmRegression specific data
        writer.Write(_bestModel.Coefficients.Length);
        for (int i = 0; i < _bestModel.Coefficients.Length; i++)
        {
            writer.Write(Convert.ToDouble(_bestModel.Coefficients[i]));
        }

        // Serialize GeneticAlgorithmOptions
        var gaOptions = _gaOptions;
        writer.Write(gaOptions.MaxGenerations);
        writer.Write(gaOptions.PopulationSize);
        writer.Write(gaOptions.MutationRate);
        writer.Write(gaOptions.CrossoverRate);

        return ms.ToArray();
    }

    public override void Deserialize(byte[] modelData)
    {
        using var ms = new MemoryStream(modelData);
        using var reader = new BinaryReader(ms);

        // Deserialize base class data
        int baseDataLength = reader.ReadInt32();
        byte[] baseData = reader.ReadBytes(baseDataLength);
        base.Deserialize(baseData);

        // Deserialize GeneticAlgorithmRegression specific data
        int coefficientsLength = reader.ReadInt32();
        var coefficients = new T[coefficientsLength];
        for (int i = 0; i < coefficientsLength; i++)
        {
            coefficients[i] = NumOps.FromDouble(reader.ReadDouble());
        }
        _bestModel = new VectorModel<T>(new Vector<T>(coefficients));

        // Deserialize GeneticAlgorithmOptions
        var gaOptions = new GeneticAlgorithmOptimizerOptions
        {
            MaxGenerations = reader.ReadInt32(),
            PopulationSize = reader.ReadInt32(),
            MutationRate = reader.ReadDouble(),
            CrossoverRate = reader.ReadDouble()
        };

        // Recreate the optimizer with the deserialized options
        _optimizer = new GeneticAlgorithmOptimizer<T>(gaOptions);

        // Update coefficients and intercept
        UpdateCoefficientsAndIntercept();
    }
}