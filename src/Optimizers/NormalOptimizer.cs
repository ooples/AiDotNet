namespace AiDotNet.Optimizers;

public class NormalOptimizer<T> : OptimizerBase<T>
{
    private readonly Random _random = new();
    private GeneticAlgorithmOptimizerOptions _normalOptions;

    public NormalOptimizer(GeneticAlgorithmOptimizerOptions? options = null, PredictionStatsOptions? predictionOptions = null,
        ModelStatsOptions? modelOptions = null,
        IModelEvaluator<T>? modelEvaluator = null,
        IFitDetector<T>? fitDetector = null,
        IFitnessCalculator<T>? fitnessCalculator = null,
        IModelCache<T>? modelCache = null)
        : base(options, predictionOptions, modelOptions, modelEvaluator, fitDetector, fitnessCalculator, modelCache)
    {
        _normalOptions = options ?? new();
        InitializeAdaptiveParameters();
    }

    public override OptimizationResult<T> Optimize(OptimizationInputData<T> inputData)
    {
        ValidationHelper<T>.ValidateInputData(inputData);

        var bestStepData = new OptimizationStepData<T>
        {
            Solution = SymbolicModelFactory<T>.CreateEmptyModel(Options.UseExpressionTrees, inputData.XTrain.Columns),
            FitnessScore = _fitnessCalculator.IsHigherScoreBetter ? NumOps.MinValue : NumOps.MaxValue
        };
        var previousStepData = new OptimizationStepData<T>();

        for (int iteration = 0; iteration < Options.MaxIterations; iteration++)
        {
            var currentSolution = CreateRandomSolution(inputData.XTrain.Columns);
            var currentStepData = EvaluateSolution(currentSolution, inputData);

            UpdateBestSolution(currentStepData, ref bestStepData);

            // Update adaptive parameters
            UpdateAdaptiveParameters(currentStepData, previousStepData);

            if (UpdateIterationHistoryAndCheckEarlyStopping(iteration, bestStepData))
            {
                break;
            }

            previousStepData = currentStepData;
        }

        return CreateOptimizationResult(bestStepData, inputData);
    }

    protected override void UpdateAdaptiveParameters(OptimizationStepData<T> currentStepData, OptimizationStepData<T> previousStepData)
    {
        base.UpdateAdaptiveParameters(currentStepData, previousStepData);

        // Adaptive feature selection
        UpdateFeatureSelectionParameters(currentStepData, previousStepData);

        // Adaptive mutation rate
        UpdateMutationRate(currentStepData, previousStepData);

        // Adaptive exploration vs exploitation balance
        UpdateExplorationExploitationBalance(currentStepData, previousStepData);

        // Adaptive population size
        UpdatePopulationSize(currentStepData, previousStepData);

        // Adaptive crossover rate
        UpdateCrossoverRate(currentStepData, previousStepData);
    }

    private void UpdateFeatureSelectionParameters(OptimizationStepData<T> currentStepData, OptimizationStepData<T> previousStepData)
    {
        if (_fitnessCalculator.IsBetterFitness(currentStepData.FitnessScore, previousStepData.FitnessScore))
        {
            Options.MinimumFeatures = Math.Max(1, Options.MinimumFeatures - 1);
            Options.MaximumFeatures = Math.Min(Options.MaximumFeatures + 1, _normalOptions.MaximumFeatures);
        }
        else
        {
            Options.MinimumFeatures = Math.Min(Options.MinimumFeatures + 1, _normalOptions.MaximumFeatures - 1);
            Options.MaximumFeatures = Math.Max(Options.MaximumFeatures - 1, Options.MinimumFeatures + 1);
        }
    }

    private void UpdateExplorationExploitationBalance(OptimizationStepData<T> currentStepData, OptimizationStepData<T> previousStepData)
    {
        if (_fitnessCalculator.IsBetterFitness(currentStepData.FitnessScore, previousStepData.FitnessScore))
        {
            Options.ExplorationRate *= 0.98; // Decrease exploration if improving
        }
        else
        {
            Options.ExplorationRate *= 1.02; // Increase exploration if not improving
        }
        Options.ExplorationRate = MathHelper.Clamp(Options.ExplorationRate, Options.MinExplorationRate, Options.MaxExplorationRate);
    }

    private void UpdateMutationRate(OptimizationStepData<T> currentStepData, OptimizationStepData<T> previousStepData)
    {
        if (_fitnessCalculator.IsBetterFitness(currentStepData.FitnessScore, previousStepData.FitnessScore))
        {
            _normalOptions.MutationRate *= 0.95; // Decrease mutation rate if improving
        }
        else
        {
            _normalOptions.MutationRate *= 1.05; // Increase mutation rate if not improving
        }

        _normalOptions.MutationRate = MathHelper.Clamp(_normalOptions.MutationRate, _normalOptions.MinMutationRate, _normalOptions.MaxMutationRate);
    }

    private void UpdatePopulationSize(OptimizationStepData<T> currentStepData, OptimizationStepData<T> previousStepData)
    {
        if (_fitnessCalculator.IsBetterFitness(currentStepData.FitnessScore, previousStepData.FitnessScore))
        {
            _normalOptions.PopulationSize = Math.Max(_normalOptions.MinPopulationSize, _normalOptions.PopulationSize - 1);
        }
        else
        {
            _normalOptions.PopulationSize = Math.Min(_normalOptions.PopulationSize + 1, _normalOptions.MaxPopulationSize);
        }
    }

    private void UpdateCrossoverRate(OptimizationStepData<T> currentStepData, OptimizationStepData<T> previousStepData)
    {
        if (_fitnessCalculator.IsBetterFitness(currentStepData.FitnessScore, previousStepData.FitnessScore))
        {
            _normalOptions.CrossoverRate *= 1.02; // Increase crossover rate if improving
        }
        else
        {
            _normalOptions.CrossoverRate *= 0.98; // Decrease crossover rate if not improving
        }

        _normalOptions.CrossoverRate = MathHelper.Clamp(_normalOptions.CrossoverRate, _normalOptions.MinCrossoverRate, _normalOptions.MaxCrossoverRate);
    }

    private ISymbolicModel<T> CreateRandomSolution(int totalFeatures)
    {
        var selectedFeatures = RandomlySelectFeatures(totalFeatures);
        return SymbolicModelFactory<T>.CreateRandomModel(Options.UseExpressionTrees, selectedFeatures.Count);
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
        if (options is GeneticAlgorithmOptimizerOptions geneticOptions)
        {
            _normalOptions = geneticOptions;
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
            _normalOptions = JsonConvert.DeserializeObject<GeneticAlgorithmOptimizerOptions>(optionsJson)
                ?? throw new InvalidOperationException("Failed to deserialize optimizer options.");
        }
    }
}