namespace AiDotNet.Optimizers;

public class GeneticAlgorithmOptimizer<T> : OptimizerBase<T>
{
    private GeneticAlgorithmOptimizerOptions _geneticOptions;
    private readonly Random _random;
    private T _currentCrossoverRate;
    private T _currentMutationRate;

    public GeneticAlgorithmOptimizer(
        GeneticAlgorithmOptimizerOptions? options = null,
        PredictionStatsOptions? predictionOptions = null,
        ModelStatsOptions? modelOptions = null,
        IModelEvaluator<T>? modelEvaluator = null,
        IFitDetector<T>? fitDetector = null,
        IFitnessCalculator<T>? fitnessCalculator = null,
        IModelCache<T>? modelCache = null)
        : base(options, predictionOptions, modelOptions, modelEvaluator, fitDetector, fitnessCalculator, modelCache)
    {
        _geneticOptions = options ?? new GeneticAlgorithmOptimizerOptions();
        _random = new Random();
        _currentCrossoverRate = NumOps.Zero;
        _currentMutationRate = NumOps.Zero;
        InitializeAdaptiveParameters();
    }

    protected override void UpdateAdaptiveParameters(OptimizationStepData<T> currentStepData, OptimizationStepData<T> previousStepData)
    {
        base.UpdateAdaptiveParameters(currentStepData, previousStepData);
        AdaptiveParametersHelper<T>.UpdateAdaptiveGeneticParameters(ref _currentCrossoverRate, ref _currentMutationRate, currentStepData, previousStepData, _geneticOptions);
    }

    private new void InitializeAdaptiveParameters()
    {
        _currentCrossoverRate = NumOps.FromDouble(_geneticOptions.CrossoverRate);
        _currentMutationRate = NumOps.FromDouble(_geneticOptions.MutationRate);
    }

    public override OptimizationResult<T> Optimize(OptimizationInputData<T> inputData)
    {
        int dimensions = inputData.XTrain.Columns;
        var population = InitializePopulation(dimensions, _geneticOptions.PopulationSize);
        var bestStepData = new OptimizationStepData<T>();
        var prevStepData = new OptimizationStepData<T>();
        var currentStepData = new OptimizationStepData<T>();

        for (int generation = 0; generation < Options.MaxIterations; generation++)
        {
            var populationStepData = new List<OptimizationStepData<T>>();

            // Evaluate all individuals in the population
            foreach (var individual in population)
            {
                currentStepData = EvaluateSolution(individual, inputData);
                populationStepData.Add(currentStepData);

                UpdateBestSolution(currentStepData, ref bestStepData);
            }

            // Update adaptive parameters
            UpdateAdaptiveParameters(currentStepData, prevStepData);

            // Update iteration history and check for early stopping
            if (UpdateIterationHistoryAndCheckEarlyStopping(generation, bestStepData))
            {
                break;
            }

            // Perform genetic operations
            population = PerformSelection(populationStepData);
            population = PerformCrossover(population);
            population = PerformMutation(population);

            prevStepData = currentStepData;
        }

        return CreateOptimizationResult(bestStepData, inputData);
    }

    private List<ISymbolicModel<T>> InitializePopulation(int dimensions, int populationSize)
    {
        var population = new List<ISymbolicModel<T>>();
        for (int i = 0; i < populationSize; i++)
        {
            population.Add(SymbolicModelFactory<T>.CreateRandomModel(Options.UseExpressionTrees, dimensions, NumOps));
        }

        return population;
    }

    private List<ISymbolicModel<T>> PerformSelection(List<OptimizationStepData<T>> populationStepData)
    {
        return TournamentSelection(populationStepData, _geneticOptions.PopulationSize);
    }

    private List<ISymbolicModel<T>> TournamentSelection(List<OptimizationStepData<T>> populationStepData, int selectionSize)
    {
        var selected = new List<ISymbolicModel<T>>();
        for (int i = 0; i < selectionSize; i++)
        {
            int index1 = _random.Next(populationStepData.Count);
            int index2 = _random.Next(populationStepData.Count);
            selected.Add(_fitnessCalculator.IsBetterFitness(populationStepData[index1].FitnessScore, populationStepData[index2].FitnessScore)
                ? populationStepData[index1].Solution
                : populationStepData[index2].Solution);
        }

        return selected;
    }

    private List<ISymbolicModel<T>> PerformCrossover(List<ISymbolicModel<T>> population)
    {
        var newPopulation = new List<ISymbolicModel<T>>();
        for (int i = 0; i < population.Count; i += 2)
        {
            var parent1 = population[i];
            var parent2 = i + 1 < population.Count ? population[i + 1] : population[0];

            var (child1, child2) = Crossover(parent1, parent2);
            newPopulation.Add(child1);
            newPopulation.Add(child2);
        }

        return newPopulation;
    }

    private (ISymbolicModel<T>, ISymbolicModel<T>) Crossover(ISymbolicModel<T> parent1, ISymbolicModel<T> parent2)
    {
        if (NumOps.LessThan(NumOps.FromDouble(_random.NextDouble()), _currentCrossoverRate))
        {
            return SymbolicModelFactory<T>.Crossover(parent1, parent2, Convert.ToDouble(_currentCrossoverRate), NumOps);
        }
        else
        {
            return (parent1, parent2);
        }
    }

    private List<ISymbolicModel<T>> PerformMutation(List<ISymbolicModel<T>> population)
    {
        return [.. population.Select(Mutate)];
    }

    private ISymbolicModel<T> Mutate(ISymbolicModel<T> individual)
    {
        if (NumOps.LessThan(NumOps.FromDouble(_random.NextDouble()), _currentMutationRate))
        {
            return SymbolicModelFactory<T>.Mutate(individual, Convert.ToDouble(_currentMutationRate), NumOps);
        }

        return individual;
    }

    protected override void UpdateOptions(OptimizationAlgorithmOptions options)
    {
        if (options is GeneticAlgorithmOptimizerOptions geneticOptions)
        {
            _geneticOptions = geneticOptions;
        }
        else
        {
            throw new ArgumentException("Invalid options type. Expected GeneticAlgorithmOptimizerOptions.");
        }
    }

    public override OptimizationAlgorithmOptions GetOptions()
    {
        return _geneticOptions;
    }

    public override byte[] Serialize()
    {
        using MemoryStream ms = new MemoryStream();
        using BinaryWriter writer = new BinaryWriter(ms);

        // Serialize base class data
        byte[] baseData = base.Serialize();
        writer.Write(baseData.Length);
        writer.Write(baseData);

        // Serialize GeneticAlgorithmOptimizerOptions
        string optionsJson = JsonConvert.SerializeObject(_geneticOptions);
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

        // Deserialize GeneticAlgorithmOptimizerOptions
        string optionsJson = reader.ReadString();
        _geneticOptions = JsonConvert.DeserializeObject<GeneticAlgorithmOptimizerOptions>(optionsJson)
            ?? throw new InvalidOperationException("Failed to deserialize optimizer options.");

        InitializeAdaptiveParameters();
    }
}