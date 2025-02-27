namespace AiDotNet.Optimizers;

public class AdaDeltaOptimizer<T> : GradientBasedOptimizerBase<T>
{
    private AdaDeltaOptimizerOptions _options;
    private Vector<T>? _accumulatedSquaredGradients;
    private Vector<T>? _accumulatedSquaredUpdates;

    public AdaDeltaOptimizer(
        AdaDeltaOptimizerOptions? options = null,
        PredictionStatsOptions? predictionOptions = null,
        ModelStatsOptions? modelOptions = null,
        IModelEvaluator<T>? modelEvaluator = null,
        IFitDetector<T>? fitDetector = null,
        IFitnessCalculator<T>? fitnessCalculator = null,
        IModelCache<T>? modelCache = null,
        IGradientCache<T>? gradientCache = null)
        : base(options, predictionOptions, modelOptions, modelEvaluator, fitDetector, fitnessCalculator, modelCache, gradientCache)
    {
        _options = options ?? new AdaDeltaOptimizerOptions();
        InitializeAdaptiveParameters();
    }

    protected override void InitializeAdaptiveParameters()
    {
        base.InitializeAdaptiveParameters();
        CurrentLearningRate = NumOps.FromDouble(_options.InitialLearningRate);
    }

    public override OptimizationResult<T> Optimize(OptimizationInputData<T> inputData)
    {
        ValidationHelper<T>.ValidateInputData(inputData);

        var currentSolution = InitializeRandomSolution(inputData.XTrain.Columns);
        var bestStepData = new OptimizationStepData<T>();
        var previousStepData = new OptimizationStepData<T>();

        _accumulatedSquaredGradients = new Vector<T>(currentSolution.Coefficients.Length);
        _accumulatedSquaredUpdates = new Vector<T>(currentSolution.Coefficients.Length);
        InitializeAdaptiveParameters();

        for (int iteration = 0; iteration < _options.MaxIterations; iteration++)
        {
            var gradient = CalculateGradient(currentSolution, inputData.XTrain, inputData.YTrain);
            var newSolution = UpdateSolution(currentSolution, gradient);

            var currentStepData = EvaluateSolution(newSolution, inputData);
            UpdateBestSolution(currentStepData, ref bestStepData);

            UpdateAdaptiveParameters(currentStepData, previousStepData);

            if (UpdateIterationHistoryAndCheckEarlyStopping(iteration, bestStepData))
            {
                return CreateOptimizationResult(bestStepData, inputData);
            }

            if (NumOps.LessThan(NumOps.Abs(NumOps.Subtract(bestStepData.FitnessScore, currentStepData.FitnessScore)), NumOps.FromDouble(_options.Tolerance)))
            {
                return CreateOptimizationResult(bestStepData, inputData);
            }

            currentSolution = newSolution;
            previousStepData = currentStepData;
        }

        return CreateOptimizationResult(bestStepData, inputData);
    }

    private ISymbolicModel<T> UpdateSolution(ISymbolicModel<T> currentSolution, Vector<T> gradient)
    {
        var newCoefficients = new Vector<T>(currentSolution.Coefficients.Length);
        for (int i = 0; i < currentSolution.Coefficients.Length; i++)
        {
            // Update accumulated squared gradients
            _accumulatedSquaredGradients![i] = NumOps.Add(
                NumOps.Multiply(NumOps.FromDouble(_options.Rho), _accumulatedSquaredGradients[i]),
                NumOps.Multiply(NumOps.FromDouble(1 - _options.Rho), NumOps.Multiply(gradient[i], gradient[i]))
            );

            // Compute update
            var update = NumOps.Multiply(
                NumOps.Sqrt(NumOps.Add(_accumulatedSquaredUpdates![i], NumOps.FromDouble(_options.Epsilon))),
                NumOps.Divide(gradient[i], NumOps.Sqrt(NumOps.Add(_accumulatedSquaredGradients[i], NumOps.FromDouble(_options.Epsilon))))
            );

            // Update accumulated squared updates
            _accumulatedSquaredUpdates[i] = NumOps.Add(
                NumOps.Multiply(NumOps.FromDouble(_options.Rho), _accumulatedSquaredUpdates[i]),
                NumOps.Multiply(NumOps.FromDouble(1 - _options.Rho), NumOps.Multiply(update, update))
            );

            // Update coefficients
            newCoefficients[i] = NumOps.Subtract(currentSolution.Coefficients[i], update);
        }
        return new VectorModel<T>(newCoefficients);
    }

    protected override void UpdateAdaptiveParameters(OptimizationStepData<T> currentStepData, OptimizationStepData<T> previousStepData)
    {
        base.UpdateAdaptiveParameters(currentStepData, previousStepData);

        if (_options.UseAdaptiveRho)
        {
            if (NumOps.GreaterThan(currentStepData.FitnessScore, previousStepData.FitnessScore))
            {
                _options.Rho = Math.Min(_options.Rho * _options.RhoIncreaseFactor, _options.MaxRho);
            }
            else
            {
                _options.Rho = Math.Max(_options.Rho * _options.RhoDecreaseFactor, _options.MinRho);
            }
        }
    }

    protected override void UpdateOptions(OptimizationAlgorithmOptions options)
    {
        if (options is AdaDeltaOptimizerOptions adaDeltaOptions)
        {
            _options = adaDeltaOptions;
        }
        else
        {
            throw new ArgumentException("Invalid options type. Expected AdaDeltaOptimizerOptions.");
        }
    }

    public override OptimizationAlgorithmOptions GetOptions()
    {
        return _options;
    }

    public override byte[] Serialize()
    {
        using (MemoryStream ms = new MemoryStream())
        using (BinaryWriter writer = new BinaryWriter(ms))
        {
            byte[] baseData = base.Serialize();
            writer.Write(baseData.Length);
            writer.Write(baseData);

            string optionsJson = JsonConvert.SerializeObject(_options);
            writer.Write(optionsJson);

            return ms.ToArray();
        }
    }

    public override void Deserialize(byte[] data)
    {
        using (MemoryStream ms = new MemoryStream(data))
        using (BinaryReader reader = new BinaryReader(ms))
        {
            int baseDataLength = reader.ReadInt32();
            byte[] baseData = reader.ReadBytes(baseDataLength);
            base.Deserialize(baseData);

            string optionsJson = reader.ReadString();
            _options = JsonConvert.DeserializeObject<AdaDeltaOptimizerOptions>(optionsJson)
                ?? throw new InvalidOperationException("Failed to deserialize optimizer options.");
        }
    }

    protected override string GenerateGradientCacheKey(ISymbolicModel<T> model, Matrix<T> X, Vector<T> y)
    {
        var baseKey = base.GenerateGradientCacheKey(model, X, y);
        return $"{baseKey}_AdaDelta_{_options.Rho}_{_options.Epsilon}";
    }
}