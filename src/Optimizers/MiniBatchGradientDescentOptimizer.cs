namespace AiDotNet.Optimizers;

public class MiniBatchGradientDescentOptimizer<T> : GradientBasedOptimizerBase<T>
{
    private MiniBatchGradientDescentOptions _options;
    private Random _random;

    public MiniBatchGradientDescentOptimizer(
        MiniBatchGradientDescentOptions? options = null,
        PredictionStatsOptions? predictionOptions = null,
        ModelStatsOptions? modelOptions = null,
        IModelEvaluator<T>? modelEvaluator = null,
        IFitDetector<T>? fitDetector = null,
        IFitnessCalculator<T>? fitnessCalculator = null,
        IModelCache<T>? modelCache = null,
        IGradientCache<T>? gradientCache = null)
        : base(options, predictionOptions, modelOptions, modelEvaluator, fitDetector, fitnessCalculator, modelCache, gradientCache)
    {
        _options = options ?? new MiniBatchGradientDescentOptions();
        _random = new Random(_options.Seed ?? Environment.TickCount);
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

        InitializeAdaptiveParameters();

        for (int epoch = 0; epoch < _options.MaxEpochs; epoch++)
        {
            var shuffledIndices = Enumerable.Range(0, inputData.XTrain.Rows).OrderBy(x => _random.Next());
            
            for (int i = 0; i < inputData.XTrain.Rows; i += _options.BatchSize)
            {
                var batchIndices = shuffledIndices.Skip(i).Take(_options.BatchSize);
                var xBatch = inputData.XTrain.GetRows(batchIndices);
                var yBatch = new Vector<T>([.. batchIndices.Select(index => inputData.YTrain[index])]);

                var gradient = CalculateGradient(currentSolution, xBatch, yBatch);
                gradient = ApplyMomentum(gradient);
                var newSolution = UpdateSolution(currentSolution, gradient);

                var currentStepData = EvaluateSolution(newSolution, inputData);
                UpdateBestSolution(currentStepData, ref bestStepData);

                UpdateAdaptiveParameters(currentStepData, previousStepData);

                if (UpdateIterationHistoryAndCheckEarlyStopping(epoch * (inputData.XTrain.Rows / _options.BatchSize) + i / _options.BatchSize, bestStepData))
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
        }

        return CreateOptimizationResult(bestStepData, inputData);
    }

    private ISymbolicModel<T> UpdateSolution(ISymbolicModel<T> currentSolution, Vector<T> gradient)
    {
        var newCoefficients = new Vector<T>(currentSolution.Coefficients.Length);
        for (int i = 0; i < currentSolution.Coefficients.Length; i++)
        {
            newCoefficients[i] = NumOps.Subtract(currentSolution.Coefficients[i], NumOps.Multiply(CurrentLearningRate, gradient[i]));
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
    }

    protected override void UpdateOptions(OptimizationAlgorithmOptions options)
    {
        if (options is MiniBatchGradientDescentOptions mbgdOptions)
        {
            _options = mbgdOptions;
        }
        else
        {
            throw new ArgumentException("Invalid options type. Expected MiniBatchGradientDescentOptions.");
        }
    }

    public override OptimizationAlgorithmOptions GetOptions()
    {
        return _options;
    }

    public override byte[] Serialize()
    {
        using MemoryStream ms = new MemoryStream();
        using BinaryWriter writer = new BinaryWriter(ms);

        byte[] baseData = base.Serialize();
        writer.Write(baseData.Length);
        writer.Write(baseData);

        string optionsJson = JsonConvert.SerializeObject(_options);
        writer.Write(optionsJson);

        return ms.ToArray();
    }

    public override void Deserialize(byte[] data)
    {
        using MemoryStream ms = new MemoryStream(data);
        using BinaryReader reader = new BinaryReader(ms);

        int baseDataLength = reader.ReadInt32();
        byte[] baseData = reader.ReadBytes(baseDataLength);
        base.Deserialize(baseData);

        string optionsJson = reader.ReadString();
        _options = JsonConvert.DeserializeObject<MiniBatchGradientDescentOptions>(optionsJson)
            ?? throw new InvalidOperationException("Failed to deserialize optimizer options.");
    }

    protected override string GenerateGradientCacheKey(ISymbolicModel<T> model, Matrix<T> X, Vector<T> y)
    {
        var baseKey = base.GenerateGradientCacheKey(model, X, y);
        return $"{baseKey}_MiniBatchGD_{_options.BatchSize}_{_options.MaxEpochs}";
    }
}