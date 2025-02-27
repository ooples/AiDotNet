namespace AiDotNet.Optimizers;

public class NesterovAcceleratedGradientOptimizer<T> : GradientBasedOptimizerBase<T>
{
    private NesterovAcceleratedGradientOptimizerOptions _options;
    private Vector<T>? _velocity;

    public NesterovAcceleratedGradientOptimizer(
        NesterovAcceleratedGradientOptimizerOptions? options = null,
        PredictionStatsOptions? predictionOptions = null,
        ModelStatsOptions? modelOptions = null,
        IModelEvaluator<T>? modelEvaluator = null,
        IFitDetector<T>? fitDetector = null,
        IFitnessCalculator<T>? fitnessCalculator = null,
        IModelCache<T>? modelCache = null,
        IGradientCache<T>? gradientCache = null)
        : base(options, predictionOptions, modelOptions, modelEvaluator, fitDetector, fitnessCalculator, modelCache, gradientCache)
    {
        _options = options ?? new NesterovAcceleratedGradientOptimizerOptions();
        InitializeAdaptiveParameters();
    }

    protected override void InitializeAdaptiveParameters()
    {
        base.InitializeAdaptiveParameters();
        CurrentLearningRate = NumOps.FromDouble(_options.InitialLearningRate);
        CurrentMomentum = NumOps.FromDouble(_options.InitialMomentum);
    }

    public override OptimizationResult<T> Optimize(OptimizationInputData<T> inputData)
    {
        ValidationHelper<T>.ValidateInputData(inputData);

        var currentSolution = InitializeRandomSolution(inputData.XTrain.Columns);
        var bestStepData = new OptimizationStepData<T>();
        var previousStepData = new OptimizationStepData<T>();

        _velocity = new Vector<T>(currentSolution.Coefficients.Length);
        InitializeAdaptiveParameters();

        for (int iteration = 0; iteration < _options.MaxIterations; iteration++)
        {
            var lookaheadSolution = GetLookaheadSolution(currentSolution);
            var gradient = CalculateGradient(lookaheadSolution, inputData.XTrain, inputData.YTrain);
            _velocity = UpdateVelocity(gradient);
            var newSolution = UpdateSolution(currentSolution, _velocity);

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

    private ISymbolicModel<T> GetLookaheadSolution(ISymbolicModel<T> currentSolution)
    {
        var lookaheadCoefficients = new Vector<T>(currentSolution.Coefficients.Length);
        for (int i = 0; i < currentSolution.Coefficients.Length; i++)
        {
            lookaheadCoefficients[i] = NumOps.Subtract(currentSolution.Coefficients[i], NumOps.Multiply(CurrentMomentum, _velocity![i]));
        }
        return new VectorModel<T>(lookaheadCoefficients);
    }

    private Vector<T> UpdateVelocity(Vector<T> gradient)
    {
        for (int i = 0; i < _velocity!.Length; i++)
        {
            _velocity[i] = NumOps.Add(
                NumOps.Multiply(CurrentMomentum, _velocity[i]),
                NumOps.Multiply(CurrentLearningRate, gradient[i])
            );
        }
        return _velocity;
    }

    private ISymbolicModel<T> UpdateSolution(ISymbolicModel<T> currentSolution, Vector<T> velocity)
    {
        var newCoefficients = new Vector<T>(currentSolution.Coefficients.Length);
        for (int i = 0; i < currentSolution.Coefficients.Length; i++)
        {
            newCoefficients[i] = NumOps.Subtract(currentSolution.Coefficients[i], velocity[i]);
        }
        return new VectorModel<T>(newCoefficients);
    }

    protected override void UpdateAdaptiveParameters(OptimizationStepData<T> currentStepData, OptimizationStepData<T> previousStepData)
    {
        base.UpdateAdaptiveParameters(currentStepData, previousStepData);

        if (_options.UseAdaptiveLearningRate)
        {
            if (NumOps.GreaterThan(currentStepData.FitnessScore, previousStepData.FitnessScore))
            {
                CurrentLearningRate = NumOps.Multiply(CurrentLearningRate, NumOps.FromDouble(_options.LearningRateIncreaseFactor));
            }
            else
            {
                CurrentLearningRate = NumOps.Multiply(CurrentLearningRate, NumOps.FromDouble(_options.LearningRateDecreaseFactor));
            }

            CurrentLearningRate = MathHelper.Max(NumOps.FromDouble(_options.MinLearningRate),
                MathHelper.Min(NumOps.FromDouble(_options.MaxLearningRate), CurrentLearningRate));
        }

        if (_options.UseAdaptiveMomentum)
        {
            if (NumOps.GreaterThan(currentStepData.FitnessScore, previousStepData.FitnessScore))
            {
                CurrentMomentum = NumOps.Multiply(CurrentMomentum, NumOps.FromDouble(_options.MomentumIncreaseFactor));
            }
            else
            {
                CurrentMomentum = NumOps.Multiply(CurrentMomentum, NumOps.FromDouble(_options.MomentumDecreaseFactor));
            }

            CurrentMomentum = MathHelper.Max(NumOps.FromDouble(_options.MinMomentum),
                MathHelper.Min(NumOps.FromDouble(_options.MaxMomentum), CurrentMomentum));
        }
    }

    protected override void UpdateOptions(OptimizationAlgorithmOptions options)
    {
        if (options is NesterovAcceleratedGradientOptimizerOptions nagOptions)
        {
            _options = nagOptions;
        }
        else
        {
            throw new ArgumentException("Invalid options type. Expected NesterovAcceleratedGradientOptimizerOptions.");
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
            _options = JsonConvert.DeserializeObject<NesterovAcceleratedGradientOptimizerOptions>(optionsJson)
                ?? throw new InvalidOperationException("Failed to deserialize optimizer options.");
        }
    }

    protected override string GenerateGradientCacheKey(ISymbolicModel<T> model, Matrix<T> X, Vector<T> y)
    {
        var baseKey = base.GenerateGradientCacheKey(model, X, y);
        return $"{baseKey}_NAG_{_options.InitialMomentum}_{_options.InitialLearningRate}";
    }
}