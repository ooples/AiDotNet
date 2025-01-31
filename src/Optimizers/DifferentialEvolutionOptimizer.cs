namespace AiDotNet.Optimizers;

public class DifferentialEvolutionOptimizer<T> : OptimizerBase<T>
{
    private DifferentialEvolutionOptions _deOptions;
    private Random _random;
    private T _currentCrossoverRate;
    private T _currentMutationRate;

    public DifferentialEvolutionOptimizer(
        DifferentialEvolutionOptions? options = null,
        PredictionStatsOptions? predictionOptions = null,
        ModelStatsOptions? modelOptions = null,
        IModelEvaluator<T>? modelEvaluator = null,
        IFitDetector<T>? fitDetector = null,
        IFitnessCalculator<T>? fitnessCalculator = null,
        IModelCache<T>? modelCache = null)
        : base(options, predictionOptions, modelOptions, modelEvaluator, fitDetector, fitnessCalculator, modelCache)
    {
        _deOptions = options ?? new DifferentialEvolutionOptions();
        _random = new Random();
        _currentCrossoverRate = NumOps.Zero;
        _currentMutationRate = NumOps.Zero;
        InitializeAdaptiveParameters();
    }

    protected override void UpdateAdaptiveParameters(OptimizationStepData<T> currentStepData, OptimizationStepData<T> previousStepData)
    {
        base.UpdateAdaptiveParameters(currentStepData, previousStepData);
        AdaptiveParametersHelper<T>.UpdateAdaptiveGeneticParameters(ref _currentCrossoverRate, ref _currentMutationRate, currentStepData, previousStepData, _deOptions);
    }

    private new void InitializeAdaptiveParameters()
    {
        _currentCrossoverRate = NumOps.FromDouble(_deOptions.CrossoverRate);
        _currentMutationRate = NumOps.FromDouble(_deOptions.MutationRate);
    }

    public override OptimizationResult<T> Optimize(OptimizationInputData<T> inputData)
    {
        int dimensions = inputData.XTrain.Columns;
        var population = InitializePopulation(dimensions, _deOptions.PopulationSize);
        var bestStepData = new OptimizationStepData<T>();
        var prevStepData = new OptimizationStepData<T>();
        var currentStepData = new OptimizationStepData<T>();

        for (int generation = 0; generation < Options.MaxIterations; generation++)
        {
            for (int i = 0; i < _deOptions.PopulationSize; i++)
            {
                var trial = GenerateTrialModel(population, i, dimensions);
                currentStepData = EvaluateSolution(trial, inputData);
                UpdateBestSolution(currentStepData, ref bestStepData);
                population[i] = currentStepData.Solution;
            }

            // Update adaptive parameters
            UpdateAdaptiveParameters(currentStepData, prevStepData);

            if (UpdateIterationHistoryAndCheckEarlyStopping(generation, bestStepData))
            {
                break; // Early stopping criteria met, exit the loop
            }

            prevStepData = currentStepData;
        }

        return CreateOptimizationResult(bestStepData, inputData);
    }

    private List<ISymbolicModel<T>> InitializePopulation(int dimensions, int populationSize)
    {
        var population = new List<ISymbolicModel<T>>();
        for (int i = 0; i < populationSize; i++)
        {
            population.Add(SymbolicModelFactory<T>.CreateRandomModel(Options.UseExpressionTrees, dimensions));
        }

        return population;
    }

    private ISymbolicModel<T> GenerateTrialModel(List<ISymbolicModel<T>> population, int currentIndex, int dimensions)
    {
        int a, b, c;
        do
        {
            a = _random.Next(population.Count);
            b = _random.Next(population.Count);
            c = _random.Next(population.Count);
        } while (a == currentIndex || b == currentIndex || c == currentIndex || a == b || a == c || b == c);

        var currentModel = population[currentIndex];
        ISymbolicModel<T> trialModel;
        var currentCrossOverRate = Convert.ToDouble(_currentCrossoverRate);
        var currentMutationRate = Convert.ToDouble(_currentMutationRate);

        if (Options.UseExpressionTrees)
        {
            // For expression trees, we'll use crossover and mutation
            trialModel = SymbolicModelFactory<T>.Crossover(population[a], population[b], currentCrossOverRate).Item1;
            trialModel = SymbolicModelFactory<T>.Mutate(trialModel, currentMutationRate);
        }
        else
        {
            var aVector = ((VectorModel<T>)population[a]).Coefficients;
            var bVector = ((VectorModel<T>)population[b]).Coefficients;
            var cVector = ((VectorModel<T>)population[c]).Coefficients;
            var currentVector = ((VectorModel<T>)currentModel).Coefficients;

            var trialVector = new Vector<T>(dimensions);
            int R = _random.Next(dimensions);

            for (int i = 0; i < dimensions; i++)
            {
                if (_random.NextDouble() < currentCrossOverRate || i == R)
                {
                    trialVector[i] = NumOps.Add(aVector[i],
                        NumOps.Multiply(NumOps.FromDouble(currentMutationRate),
                            NumOps.Subtract(bVector[i], cVector[i])));
                }
                else
                {
                    trialVector[i] = currentVector[i];
                }
            }
            trialModel = new VectorModel<T>(trialVector);
        }

        return trialModel;
    }

    protected override void UpdateOptions(OptimizationAlgorithmOptions options)
    {
        if (options is DifferentialEvolutionOptions deOptions)
        {
            _deOptions = deOptions;
        }
        else
        {
            throw new ArgumentException("Options must be of type DifferentialEvolutionOptions", nameof(options));
        }
    }

    public override OptimizationAlgorithmOptions GetOptions()
    {
        return _deOptions;
    }

    public override byte[] Serialize()
    {
        using MemoryStream ms = new MemoryStream();
        using BinaryWriter writer = new BinaryWriter(ms);

        // Serialize base class data
        byte[] baseData = base.Serialize();
        writer.Write(baseData.Length);
        writer.Write(baseData);

        // Serialize DifferentialEvolutionOptions
        string optionsJson = JsonConvert.SerializeObject(_deOptions);
        writer.Write(optionsJson);

        // Serialize Random state
        writer.Write(_random.Next());

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

        // Deserialize DifferentialEvolutionOptions
        string optionsJson = reader.ReadString();
        _deOptions = JsonConvert.DeserializeObject<DifferentialEvolutionOptions>(optionsJson)
            ?? throw new InvalidOperationException("Failed to deserialize optimizer options.");

        // Deserialize Random state
        int randomSeed = reader.ReadInt32();
        _random = new Random(randomSeed);

        InitializeAdaptiveParameters();
    }
}