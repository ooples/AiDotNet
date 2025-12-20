namespace AiDotNet.Genetics;

public class IslandModelGeneticAlgorithm<T, TInput, TOutput> :
    StandardGeneticAlgorithm<T, TInput, TOutput>
{
    private readonly int _islandCount;
    private readonly int _migrationInterval;
    private readonly double _migrationRate;
    private List<List<ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>>> _islands = new();

    public IslandModelGeneticAlgorithm(
        Func<IFullModel<T, TInput, TOutput>> modelFactory,
        IFitnessCalculator<T, TInput, TOutput> fitnessCalculator,
        IModelEvaluator<T, TInput, TOutput> modelEvaluator,
        int islandCount = 5,
        int migrationInterval = 10,
        double migrationRate = 0.1)
        : base(modelFactory, fitnessCalculator, modelEvaluator)
    {
        _islandCount = islandCount;
        _migrationInterval = migrationInterval;
        _migrationRate = migrationRate;
    }

    public override EvolutionStats<T, TInput, TOutput> Evolve(
        int generations,
        TInput trainingInput,
        TOutput trainingOutput,
        TInput? validationInput = default,
        TOutput? validationOutput = default,
        Func<EvolutionStats<T, TInput, TOutput>, bool>? stopCriteria = null)
    {
        // Initialize population if it's empty
        if (Population.Count == 0)
        {
            Population = InitializePopulation(GeneticParams.PopulationSize, GeneticParams.InitializationMethod).ToList();
        }

        // Divide population into islands
        InitializeIslands();

        // Reset evolution tracking
        EvolutionStopwatch.Restart();
        CurrentStats = new EvolutionStats<T, TInput, TOutput>(FitnessCalculator);
        CurrentStats.Generation = 0;

        // Initial evaluation of the population
        EvaluatePopulation(trainingInput, trainingOutput, validationInput, validationOutput);

        // Update statistics
        UpdateEvolutionStats();

        // Store initial best individual
        BestIndividual = FindBestIndividual();

        // Main evolution loop
        for (int gen = 0; gen < generations; gen++)
        {
            CurrentStats.Generation = gen + 1;

            // Evolve each island separately
            if (GeneticParams.UseParallelEvaluation)
            {
                Parallel.For(0, _islandCount, i =>
                {
                    EvolveIsland(i, trainingInput, trainingOutput, validationInput, validationOutput);
                });
            }
            else
            {
                for (int i = 0; i < _islandCount; i++)
                {
                    EvolveIsland(i, trainingInput, trainingOutput, validationInput, validationOutput);
                }
            }

            // Perform migration at specified intervals
            if (gen % _migrationInterval == 0 && gen > 0)
            {
                PerformMigration();
            }

            // Merge islands back into main population
            MergeIslands();

            // Update statistics
            UpdateEvolutionStats();

            // Check for improvement
            var currentBest = FindBestIndividual();
            bool improved = IsBetterFitness(currentBest.GetFitness(), BestIndividual.GetFitness());

            if (improved)
            {
                BestIndividual = currentBest;
                CurrentStats.ImprovedInLastGeneration = true;
                CurrentStats.GenerationsSinceImprovement = 0;
            }
            else
            {
                CurrentStats.ImprovedInLastGeneration = false;
                CurrentStats.GenerationsSinceImprovement++;
            }

            // Re-divide population into islands
            SplitIntoIslands();

            // Check stopping criteria
            if (stopCriteria != null && stopCriteria(CurrentStats))
            {
                break;
            }

            // Check for stagnation
            if (CurrentStats.GenerationsSinceImprovement >= GeneticParams.MaxGenerationsWithoutImprovement)
            {
                break;
            }

            // Check for time limit
            if (EvolutionStopwatch.Elapsed >= GeneticParams.MaxTime)
            {
                break;
            }
        }

        // Merge islands for final population
        MergeIslands();

        // Set final stats
        CurrentStats.TimeElapsed = EvolutionStopwatch.Elapsed;
        CurrentStats.BestIndividual = BestIndividual;

        return CurrentStats;
    }

    private void InitializeIslands()
    {
        _islands = new List<List<ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>>>();

        for (int i = 0; i < _islandCount; i++)
        {
            _islands.Add(new List<ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>>());
        }

        // Distribute individuals randomly to islands
        var shuffledPop = Population.OrderBy(_ => Random.Next()).ToList();

        for (int i = 0; i < shuffledPop.Count; i++)
        {
            int islandIndex = i % _islandCount;
            _islands[islandIndex].Add(shuffledPop[i]);
        }
    }

    private void EvolveIsland(
        int islandIndex,
        TInput trainingInput,
        TOutput trainingOutput,
        TInput? validationInput,
        TOutput? validationOutput)
    {
        var island = _islands[islandIndex];
        var tempPopulation = Population;

        // Temporarily set Population to just this island
        Population = island;

        // Use standard CreateNextGeneration for this island
        var newGeneration = base.CreateNextGeneration(
            trainingInput, trainingOutput, validationInput, validationOutput);

        // Update the island
        _islands[islandIndex] = newGeneration.ToList();

        // Restore the full population
        Population = tempPopulation;
    }

    private void PerformMigration()
    {
        // For ring topology migration
        for (int i = 0; i < _islandCount; i++)
        {
            // Determine source and destination islands
            int sourceIsland = i;
            int destIsland = (i + 1) % _islandCount;

            // Calculate number of migrants
            int migrantCount = Math.Max(1, (int)(_islands[sourceIsland].Count * _migrationRate));

            // Select migrants from source (best individuals)
            // Clone and filter out any failed clones
            var migrants = _islands[sourceIsland]
                .OrderByDescending(ind => FitnessCalculator.IsHigherScoreBetter ?
                    ind.GetFitness() : InvertFitness(ind.GetFitness()))
                .Take(migrantCount)
                .Select(ind => ind.Clone() as ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>)
                .Where(ind => ind != null)
                .Cast<ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>>()
                .ToList();

            // Replace worst individuals in destination
            var destination = _islands[destIsland];
            destination = destination
                .OrderByDescending(ind => FitnessCalculator.IsHigherScoreBetter ?
                    ind.GetFitness() : InvertFitness(ind.GetFitness()))
                .Take(destination.Count - migrantCount)
                .ToList();

            // Add migrants to destination
            destination.AddRange(migrants);

            // Update destination island
            _islands[destIsland] = destination;
        }
    }

    private void MergeIslands()
    {
        Population.Clear();

        foreach (var island in _islands)
        {
            Population.AddRange(island);
        }
    }

    private void SplitIntoIslands()
    {
        InitializeIslands();
    }

    public override ModelMetadata<T> GetMetaData()
    {
        var metadata = base.GetMetaData();
        metadata.ModelType = ModelType.GeneticAlgorithmRegression;
        metadata.Description = "Model evolved using an island model genetic algorithm";
        metadata.AdditionalInfo["IslandCount"] = _islandCount.ToString();
        metadata.AdditionalInfo["MigrationInterval"] = _migrationInterval.ToString();
        metadata.AdditionalInfo["MigrationRate"] = _migrationRate.ToString();
        return metadata;
    }
}
