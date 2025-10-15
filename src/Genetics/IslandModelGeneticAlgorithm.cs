namespace AiDotNet.Genetics;

/// <summary>
/// Implements a genetic algorithm using the island model for population subdivision and periodic migration.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type for the model.</typeparam>
/// <typeparam name="TOutput">The output data type for the model.</typeparam>
/// <remarks>
/// <para>
/// The Island Model Genetic Algorithm divides the population into separate subpopulations (islands) that evolve in isolation
/// for several generations before exchanging individuals through migration. This approach helps maintain genetic diversity
/// and can overcome local optima by allowing each island to explore different regions of the search space.
/// </para>
/// <para><b>For Beginners:</b> Think of this algorithm like several isolated villages that develop their own traditions.
/// 
/// Imagine several villages on different islands:
/// - Each village (island) develops its own unique culture and ideas
/// - The villages mostly evolve independently, developing different solutions to similar problems
/// - Occasionally, people travel between islands, bringing new ideas and techniques
/// - This exchange of ideas helps each village discover solutions they might have missed on their own
/// - The isolation preserves diversity while migration spreads good ideas
/// 
/// This approach often finds better solutions than having everyone in one large group,
/// because each isolated group can explore different approaches without being immediately
/// influenced by what others are doing.
/// </para>
/// </remarks>
public class IslandModelGeneticAlgorithm<T, TInput, TOutput> :
    StandardGeneticAlgorithm<T, TInput, TOutput>
{
    /// <summary>
    /// The number of islands (subpopulations) used in the algorithm.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field determines how many subpopulations the main population will be divided into.
    /// More islands can increase exploration but require a larger total population size to maintain
    /// effective evolution on each island.
    /// </para>
    /// <para><b>For Beginners:</b> This is like deciding how many separate villages to establish.
    /// 
    /// If you have too few villages:
    /// - There's less diversity in how problems are approached
    /// - Good ideas spread quickly but may dominate too early
    /// 
    /// If you have too many villages:
    /// - Each village might be too small to develop good ideas effectively
    /// - More computational resources are required
    /// 
    /// Finding the right balance depends on your specific problem and available resources.
    /// </para>
    /// </remarks>
    private readonly int _islandCount;

    /// <summary>
    /// The number of generations between migration events.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field specifies how many generations each island evolves in isolation before individuals
    /// migrate between islands. This parameter balances between exploration (longer intervals)
    /// and exploitation (shorter intervals).
    /// </para>
    /// <para><b>For Beginners:</b> This is like deciding how often the villages exchange travelers.
    /// 
    /// If migrations happen very frequently:
    /// - Good ideas spread quickly throughout all villages
    /// - But villages might not have time to develop their own unique approaches
    /// 
    /// If migrations happen rarely:
    /// - Villages develop more distinct and diverse solutions
    /// - But it takes longer for beneficial discoveries to spread
    /// 
    /// The best interval depends on how quickly you need results and how important diversity is for your problem.
    /// </para>
    /// </remarks>
    private readonly int _migrationInterval;

    /// <summary>
    /// The fraction of individuals that migrate between islands during each migration event.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field determines what percentage of each island's population migrates during
    /// migration events. Higher rates promote faster convergence while lower rates preserve
    /// more diversity.
    /// </para>
    /// <para><b>For Beginners:</b> This is like deciding what percentage of each village travels to another village.
    /// 
    /// If many people travel between villages:
    /// - Ideas spread quickly and influence all villages
    /// - Villages may quickly become more similar to each other
    /// 
    /// If only a few people travel:
    /// - Each village maintains more of its unique character
    /// - But beneficial ideas spread more slowly
    /// 
    /// This balance helps control how much the villages influence each other while still allowing
    /// some exchange of ideas.
    /// </para>
    /// </remarks>
    private readonly double _migrationRate;

    /// <summary>
    /// The collection of islands, each containing a subpopulation of individuals.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field holds the actual islands (subpopulations) that evolve separately.
    /// Each island is a list of individuals that undergoes its own evolutionary process.
    /// </para>
    /// <para><b>For Beginners:</b> This represents the actual villages with their populations.
    /// 
    /// Each element in this collection:
    /// - Contains a group of solution candidates (the villagers)
    /// - Evolves separately from other groups
    /// - Occasionally exchanges members with other groups
    /// - Contributes to the overall diversity of solutions
    /// 
    /// This structure is the core of what makes the Island Model different from
    /// standard genetic algorithms with a single population.
    /// </para>
    /// </remarks>
    private List<List<ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>>> _islands = new();

    /// <summary>
    /// Initializes a new instance of the IslandModelGeneticAlgorithm class.
    /// </summary>
    /// <param name="modelFactory">Factory function that creates new model instances.</param>
    /// <param name="fitnessCalculator">The calculator used to determine model fitness.</param>
    /// <param name="modelEvaluator">The evaluator used to assess model performance.</param>
    /// <param name="islandCount">The number of islands to divide the population into (default: 5).</param>
    /// <param name="migrationInterval">The number of generations between migration events (default: 10).</param>
    /// <param name="migrationRate">The fraction of individuals that migrate between islands (default: 0.1).</param>
    /// <remarks>
    /// <para>
    /// This constructor sets up the island model genetic algorithm with its initial configuration.
    /// The default values are reasonable starting points for many problems, but may need adjustment
    /// based on the specific characteristics of the problem being solved.
    /// </para>
    /// <para><b>For Beginners:</b> This is like planning how to set up your villages.
    /// 
    /// When creating this algorithm:
    /// - You provide the basics needed for any genetic algorithm
    /// - You specify how many separate villages (islands) to create
    /// - You decide how often people should travel between villages
    /// - You determine what percentage of villagers should travel
    /// 
    /// These settings control the balance between having diverse, independent approaches
    /// and sharing good ideas between different groups.
    /// </para>
    /// </remarks>
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

    /// <summary>
    /// Evolves the population over a specified number of generations using the island model.
    /// </summary>
    /// <param name="generations">The number of generations to evolve.</param>
    /// <param name="trainingInput">The input data used for training.</param>
    /// <param name="trainingOutput">The expected output data for training.</param>
    /// <param name="validationInput">Optional validation input data.</param>
    /// <param name="validationOutput">Optional validation output data.</param>
    /// <param name="stopCriteria">Optional function to determine when to stop evolution early.</param>
    /// <returns>Statistics about the evolution process.</returns>
    /// <remarks>
    /// <para>
    /// This method overrides the standard evolution process to implement the island model approach.
    /// It divides the population into islands, evolves each island separately, performs periodic
    /// migrations between islands, and tracks the overall best solution found across all islands.
    /// </para>
    /// <para><b>For Beginners:</b> This method runs the entire village-based evolution process.
    /// 
    /// The process works like this:
    /// 1. People are divided into separate villages (islands)
    /// 2. Each village develops independently for a while
    /// 3. Periodically, some people move between villages, bringing new ideas
    /// 4. Villages continue to develop with these new influences
    /// 5. The process repeats until we've gone through all generations
    /// 6. At the end, we find the best solution from any village
    /// 
    /// This approach often finds better solutions than a standard genetic algorithm
    /// because it maintains more diversity while still allowing good ideas to spread.
    /// </para>
    /// </remarks>
    public override GeneticStats<T, TInput, TOutput> Evolve(
        int generations,
        TInput trainingInput,
        TOutput trainingOutput,
        TInput? validationInput = default,
        TOutput? validationOutput = default,
        Func<GeneticStats<T, TInput, TOutput>, bool>? stopCriteria = null)
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
        CurrentStats = new GeneticStats<T, TInput, TOutput>(FitnessCalculator);
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

    /// <summary>
    /// Initializes the islands by dividing the population into subpopulations.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method creates the island structure and distributes individuals from the main population
    /// randomly across the islands. It ensures that each island receives approximately the same
    /// number of individuals to start with.
    /// </para>
    /// <para><b>For Beginners:</b> This is like randomly assigning people to different villages.
    /// 
    /// The process works like this:
    /// 1. Create empty villages (islands)
    /// 2. Shuffle all the people randomly
    /// 3. Assign each person to a village in a round-robin fashion
    /// 
    /// This random distribution ensures that each village starts with a diverse
    /// mix of potential solutions and isn't biased toward particular approaches.
    /// </para>
    /// </remarks>
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

    /// <summary>
    /// Evolves a single island for one generation.
    /// </summary>
    /// <param name="islandIndex">The index of the island to evolve.</param>
    /// <param name="trainingInput">The input data used for training.</param>
    /// <param name="trainingOutput">The expected output data for training.</param>
    /// <param name="validationInput">Optional validation input data.</param>
    /// <param name="validationOutput">Optional validation output data.</param>
    /// <remarks>
    /// <para>
    /// This method evolves a specific island by temporarily replacing the main population with just
    /// the island's population, applying standard evolution operations, and then updating the
    /// island with the new generation.
    /// </para>
    /// <para><b>For Beginners:</b> This is like letting one village develop for a generation.
    /// 
    /// For a single village, the method:
    /// 1. Focuses only on the people in this specific village
    /// 2. Lets them go through typical evolution (selection, crossover, mutation)
    /// 3. Creates a new generation of villagers
    /// 4. Updates the village with this new generation
    /// 
    /// This allows each village to evolve independently, potentially developing
    /// unique approaches to solving the problem.
    /// </para>
    /// </remarks>
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

    /// <summary>
    /// Performs migration of individuals between islands.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method implements a ring topology migration scheme where the best individuals from each
    /// island migrate to the next island, replacing the worst individuals there. This helps spread
    /// beneficial genetic material throughout the entire population while maintaining isolation.
    /// </para>
    /// <para><b>For Beginners:</b> This is like having travelers move between villages.
    /// 
    /// The migration process works like this:
    /// 1. From each village, select the people with the best ideas (highest fitness)
    /// 2. These selected people travel to the next village in a circle pattern
    /// 3. In each destination village, the travelers replace the people with the worst ideas
    /// 4. This spreads good ideas around while maintaining most of each village's uniqueness
    /// 
    /// This periodic exchange helps prevent villages from getting stuck with suboptimal
    /// solutions by introducing proven good ideas from elsewhere.
    /// </para>
    /// </remarks>
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
            var migrants = _islands[sourceIsland]
                .OrderByDescending(ind => FitnessCalculator.IsHigherScoreBetter ?
                    ind.GetFitness() : InvertFitness(ind.GetFitness()))
                .Take(migrantCount)
                .Select(ind => ind.Clone() as ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>)
                .Where(ind => ind != null)
                .ToList();

            // Replace worst individuals in destination
            var destination = _islands[destIsland];
            destination = destination
                .OrderByDescending(ind => FitnessCalculator.IsHigherScoreBetter ?
                    ind.GetFitness() : InvertFitness(ind.GetFitness()))
                .Take(destination.Count - migrantCount)
                .ToList();

            // Add migrants to destination
            destination.AddRange(migrants!);

            // Update destination island
            _islands[destIsland] = destination;
        }
    }

    /// <summary>
    /// Merges all islands back into the main population.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method combines all island subpopulations back into the main population. It is typically
    /// called at the end of evolution or before evaluating the overall best solution.
    /// </para>
    /// <para><b>For Beginners:</b> This is like bringing all villagers back together.
    /// 
    /// After letting villages develop separately:
    /// 1. Everyone is gathered back together into one main group
    /// 2. This allows us to look at all solutions together
    /// 3. We can then find the overall best solution across all villages
    /// 
    /// This reassembly step is important for evaluating the algorithm's overall performance
    /// and preparing for the next cycle of island-based evolution.
    /// </para>
    /// </remarks>
    private void MergeIslands()
    {
        Population.Clear();

        foreach (var island in _islands)
        {
            Population.AddRange(island);
        }
    }

    /// <summary>
    /// Divides the current population into islands.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method re-divides the main population into islands. It is typically called after
    /// merging islands and before continuing evolution.
    /// </para>
    /// <para><b>For Beginners:</b> This is like redistributing people back to their villages.
    /// 
    /// After temporarily bringing everyone together:
    /// 1. People are reassigned to different villages
    /// 2. This redistribution creates fresh groupings
    /// 3. It helps maintain diversity by reshuffling which solutions are grouped together
    /// 
    /// This process helps prevent premature convergence by periodically
    /// reorganizing which solutions evolve together.
    /// </para>
    /// </remarks>
    private void SplitIntoIslands()
    {
        InitializeIslands();
    }

    /// <summary>
    /// Gets metadata about the algorithm and its current state.
    /// </summary>
    /// <returns>A ModelMetadata object containing information about the algorithm.</returns>
    /// <remarks>
    /// <para>
    /// This method extends the base implementation to include island model-specific information
    /// in the metadata. It adds details about the island count, migration interval, and migration rate.
    /// </para>
    /// <para><b>For Beginners:</b> This is like creating a report about how the villages were organized.
    /// 
    /// The metadata includes:
    /// - Basic information about the genetic algorithm
    /// - How many separate villages were used
    /// - How often people traveled between villages
    /// - What percentage of each village traveled
    /// 
    /// This information helps document how the algorithm was configured,
    /// which is useful for reproducing results or comparing different approaches.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetMetaData()
    {
        var metadata = base.GetMetaData();
        metadata.ModelType = ModelType.GeneticAlgorithmRegression;
        metadata.Description = "Model evolved using an island model genetic algorithm";
        metadata.AdditionalInfo["IslandCount"] = _islandCount;
        metadata.AdditionalInfo["MigrationInterval"] = _migrationInterval;
        metadata.AdditionalInfo["MigrationRate"] = _migrationRate;

        return metadata;
    }
}