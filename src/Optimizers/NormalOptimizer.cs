namespace AiDotNet.Optimizers;

public class NormalOptimizer<T> : OptimizerBase<T>
{
    private readonly Random _random = new();
    private OptimizationAlgorithmOptions _normalOptions;

    public NormalOptimizer(OptimizationAlgorithmOptions? options = null, PredictionStatsOptions? predictionOptions = null,
        ModelStatsOptions? modelOptions = null,
        IModelEvaluator<T>? modelEvaluator = null,
        IFitDetector<T>? fitDetector = null,
        IFitnessCalculator<T>? fitnessCalculator = null)
        : base(options, predictionOptions, modelOptions, modelEvaluator, fitDetector, fitnessCalculator)
    {
        _normalOptions = options ?? new();
    }

    public override OptimizationResult<T> Optimize(OptimizationInputData<T> inputData)
    {
        ValidationHelper<T>.ValidateInputData(inputData);

        var bestStepData = new OptimizationStepData<T>
        {
            Solution = SymbolicModelFactory<T>.CreateEmptyModel(Options.UseExpressionTrees, inputData.XTrain.Columns, NumOps),
            FitnessScore = _fitnessCalculator.IsHigherScoreBetter ? NumOps.MinValue : NumOps.MaxValue
        };

        for (int iteration = 0; iteration < Options.MaxIterations; iteration++)
        {
            var currentSolution = CreateRandomSolution(inputData.XTrain.Columns);
            var currentStepData = PrepareAndEvaluateSolution(currentSolution, inputData);

            UpdateBestSolution(currentStepData, ref bestStepData);

            if (UpdateIterationHistoryAndCheckEarlyStopping(iteration, bestStepData))
            {
                break;
            }
        }

        return CreateOptimizationResult(bestStepData, inputData);
    }

    private ISymbolicModel<T> CreateRandomSolution(int totalFeatures)
    {
        var selectedFeatures = RandomlySelectFeatures(totalFeatures);
        return SymbolicModelFactory<T>.CreateRandomModel(Options.UseExpressionTrees, selectedFeatures.Count, NumOps);
    }

    private List<int> RandomlySelectFeatures(int totalFeatures)
    {
        var selectedFeatures = new List<int>();
        int numFeatures = _random.Next(Options.MinimumFeatures, Math.Min(Options.MaximumFeatures, totalFeatures) + 1);

        while (selectedFeatures.Count < numFeatures)
        {
            int feature = _random.Next(totalFeatures);
            if (!selectedFeatures.Contains(feature))
            {
                selectedFeatures.Add(feature);
            }
        }

        return selectedFeatures;
    }

    public override OptimizationAlgorithmOptions GetOptions()
    {
        return _normalOptions;
    }

    protected override void UpdateOptions(OptimizationAlgorithmOptions options)
    {
        if (options is OptimizationAlgorithmOptions normalOptions)
        {
            _normalOptions = normalOptions;
        }
        else
        {
            throw new ArgumentException("Invalid options type. Expected NormalOptimizerOptions.");
        }
    }

    public override byte[] Serialize()
    {
        using (MemoryStream ms = new MemoryStream())
        using (BinaryWriter writer = new BinaryWriter(ms))
        {
            // Serialize base class data
            byte[] baseData = base.Serialize();
            writer.Write(baseData.Length);
            writer.Write(baseData);

            // Serialize NormalOptimizerOptions
            string optionsJson = JsonConvert.SerializeObject(_normalOptions);
            writer.Write(optionsJson);

            return ms.ToArray();
        }
    }

    public override void Deserialize(byte[] data)
    {
        using (MemoryStream ms = new MemoryStream(data))
        using (BinaryReader reader = new BinaryReader(ms))
        {
            // Deserialize base class data
            int baseDataLength = reader.ReadInt32();
            byte[] baseData = reader.ReadBytes(baseDataLength);
            base.Deserialize(baseData);

            // Deserialize NormalOptimizerOptions
            string optionsJson = reader.ReadString();
            _normalOptions = JsonConvert.DeserializeObject<OptimizationAlgorithmOptions>(optionsJson)
                ?? throw new InvalidOperationException("Failed to deserialize optimizer options.");
        }
    }
}