namespace AiDotNet.Optimizers;

public class GradientDescentOptimizer<T> : GradientBasedOptimizerBase<T>
{
    private GradientDescentOptimizerOptions _gdOptions;
    private readonly IRegularization<T> _regularization;

    public GradientDescentOptimizer(GradientDescentOptimizerOptions? options = null,
        PredictionStatsOptions? predictionOptions = null,
        ModelStatsOptions? modelOptions = null,
        IModelEvaluator<T>? modelEvaluator = null,
        IFitDetector<T>? fitDetector = null,
        IFitnessCalculator<T>? fitnessCalculator = null,
        IModelCache<T>? modelCache = null,
        IGradientCache<T>? gradientCache = null)
        : base(options, predictionOptions, modelOptions, modelEvaluator, fitDetector, fitnessCalculator, modelCache, gradientCache)
    {
        _gdOptions = options ?? new GradientDescentOptimizerOptions();
        _regularization = CreateRegularization(_gdOptions);
    }

    private static IRegularization<T> CreateRegularization(GradientDescentOptimizerOptions options)
    {
        return RegularizationFactory.CreateRegularization<T>(options.RegularizationOptions);
    }

    public override OptimizationResult<T> Optimize(OptimizationInputData<T> inputData)
    {
        ValidationHelper<T>.ValidateInputData(inputData);

        var currentSolution = InitializeRandomSolution(inputData.XTrain.Columns);
        var bestStepData = EvaluateSolution(currentSolution, inputData);
        var previousStepData = bestStepData;
        InitializeAdaptiveParameters();

        for (int iteration = 0; iteration < _gdOptions.MaxIterations; iteration++)
        {
            var gradient = CalculateGradient(currentSolution, inputData.XTrain, inputData.YTrain);
            gradient = ApplyMomentum(gradient);

            currentSolution = UpdateSolution(currentSolution, gradient);

            var currentStepData = EvaluateSolution(currentSolution, inputData);

            UpdateAdaptiveParameters(currentStepData, previousStepData);

            UpdateBestSolution(currentStepData, ref bestStepData);

            if (UpdateIterationHistoryAndCheckEarlyStopping(iteration, bestStepData))
            {
                break;
            }

            previousStepData = currentStepData;
        }

        return CreateOptimizationResult(bestStepData, inputData);
    }

    private ISymbolicModel<T> UpdateSolution(ISymbolicModel<T> currentSolution, Vector<T> gradient)
    {
        Vector<T> updatedCoefficients = currentSolution.Coefficients.Subtract(gradient.Multiply(CurrentLearningRate));
        return currentSolution.UpdateCoefficients(updatedCoefficients);
    }

    protected new Vector<T> CalculateGradient(ISymbolicModel<T> solution, Matrix<T> X, Vector<T> y)
    {
        string cacheKey = GenerateGradientCacheKey(solution, X, y);
        var cachedGradient = GradientCache.GetCachedGradient(cacheKey);
        if (cachedGradient != null)
        {
            return cachedGradient.Coefficients;
        }

        Vector<T> gradient = new(solution.Coefficients.Length);
        T epsilon = NumOps.FromDouble(1e-8);

        for (int i = 0; i < solution.Coefficients.Length; i++)
        {
            Vector<T> perturbedCoefficientsPlus = solution.Coefficients.Copy();
            perturbedCoefficientsPlus[i] = NumOps.Add(perturbedCoefficientsPlus[i], epsilon);

            Vector<T> perturbedCoefficientsMinus = solution.Coefficients.Copy();
            perturbedCoefficientsMinus[i] = NumOps.Subtract(perturbedCoefficientsMinus[i], epsilon);

            T lossPlus = CalculateLoss(solution.UpdateCoefficients(perturbedCoefficientsPlus), X, y);
            T lossMinus = CalculateLoss(solution.UpdateCoefficients(perturbedCoefficientsMinus), X, y);

            gradient[i] = NumOps.Divide(NumOps.Subtract(lossPlus, lossMinus), NumOps.Multiply(NumOps.FromDouble(2.0), epsilon));
        }

        var gradientModel = gradient.ToSymbolicModel();
        GradientCache.CacheGradient(cacheKey, gradientModel);

        return gradient;
    }

    private T CalculateLoss(ISymbolicModel<T> solution, Matrix<T> X, Vector<T> y)
    {
        Vector<T> predictions = new Vector<T>(X.Rows);
        for (int i = 0; i < X.Rows; i++)
        {
            predictions[i] = solution.Evaluate(X.GetRow(i));
        }

        T mse = StatisticsHelper<T>.CalculateMeanSquaredError(predictions, y);
        Vector<T> regularizedCoefficients = _regularization.RegularizeCoefficients(solution.Coefficients);
        T regularizationTerm = regularizedCoefficients.Subtract(solution.Coefficients).Transform(NumOps.Abs).Sum();

        return NumOps.Add(mse, regularizationTerm);
    }

    protected override void UpdateOptions(OptimizationAlgorithmOptions options)
    {
        if (options is GradientDescentOptimizerOptions gdOptions)
        {
            _gdOptions = gdOptions;
        }
        else
        {
            throw new ArgumentException("Invalid options type. Expected GradientDescentOptions.");
        }
    }

    public override OptimizationAlgorithmOptions GetOptions()
    {
        return _gdOptions;
    }

    public override byte[] Serialize()
    {
        using MemoryStream ms = new MemoryStream();
        using BinaryWriter writer = new BinaryWriter(ms);

        // Serialize base class data
        byte[] baseData = base.Serialize();
        writer.Write(baseData.Length);
        writer.Write(baseData);

        // Serialize GradientDescentOptions
        string optionsJson = JsonConvert.SerializeObject(_gdOptions);
        writer.Write(optionsJson);

        return ms.ToArray();
    }

    public override void Deserialize(byte[] data)
    {
        using MemoryStream ms = new MemoryStream(data);
        using BinaryReader reader = new BinaryReader(ms);

        // Deserialize base class data
        int baseDataLength = reader.ReadInt32();
        byte[] baseData = reader.ReadBytes(baseDataLength);
        base.Deserialize(baseData);

        // Deserialize GradientDescentOptions
        string optionsJson = reader.ReadString();
        _gdOptions = JsonConvert.DeserializeObject<GradientDescentOptimizerOptions>(optionsJson)
            ?? throw new InvalidOperationException("Failed to deserialize optimizer options.");
    }

    protected override string GenerateGradientCacheKey(ISymbolicModel<T> model, Matrix<T> X, Vector<T> y)
    {
        var baseKey = base.GenerateGradientCacheKey(model, X, y);
        return $"{baseKey}_GD_{CurrentLearningRate}_{_gdOptions.MaxIterations}";
    }
}