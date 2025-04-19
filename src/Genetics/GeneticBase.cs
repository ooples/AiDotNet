namespace AiDotNet.Genetics;

/// <summary>
/// Provides a base implementation of IGeneticModel that handles common genetic algorithm operations.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <typeparam name="TInput">The type of input data for the model.</typeparam>
/// <typeparam name="TOutput">The type of output data produced by the model.</typeparam>
/// <typeparam name="TIndividual">The type representing an individual in the genetic population.</typeparam>
/// <typeparam name="TGene">The type representing a gene in the genetic model.</typeparam>
/// <remarks>
/// <para>
/// This abstract base class implements the IGeneticModel interface, providing standard
/// implementations for common genetic algorithm operations while allowing derived classes
/// to customize behavior specific to their genetic model type.
/// </para>
/// <para><b>For Beginners:</b>
/// This class provides a ready-to-use foundation for genetic algorithm models.
/// It handles:
/// 
/// - Managing a population of candidate solutions
/// - Evolving the population through selection, crossover, and mutation
/// - Tracking statistics about the evolutionary process
/// - Saving and loading populations
/// 
/// When creating your own genetic model, you can inherit from this class and focus
/// on the specific implementation details of your model type rather than reimplementing
/// the entire genetic algorithm framework.
/// </para>
/// </remarks>
public abstract class GeneticBase<T, TInput, TOutput, TIndividual, TGene> :
    IGeneticAlgorithm<T, TInput, TOutput, TIndividual, TGene>
    where TIndividual : class, IEvolvable<TGene, T>
    where TGene : class
{
    /// <summary>
    /// The current population of individuals.
    /// </summary>
    protected List<TIndividual> Population { get; set; }

    /// <summary>
    /// The best individual found so far.
    /// </summary>
    protected TIndividual BestIndividual { get; set; }

    /// <summary>
    /// The parameters for the genetic algorithm.
    /// </summary>
    protected GeneticParameters GeneticParams { get; set; }

    /// <summary>
    /// The fitness calculator used to evaluate individuals.
    /// </summary>
    protected IFitnessCalculator<T, TInput, TOutput> FitnessCalculator { get; set; }

    /// <summary>
    /// The random number generator used for stochastic operations.
    /// </summary>
    protected Random Random { get; set; }

    /// <summary>
    /// A dictionary mapping crossover operator names to implementations.
    /// </summary>
    protected Dictionary<string, Func<TIndividual, TIndividual, double, ICollection<TIndividual>>> CrossoverOperators { get; set; }

    /// <summary>
    /// A dictionary mapping mutation operator names to implementations.
    /// </summary>
    protected Dictionary<string, Func<TIndividual, double, TIndividual>> MutationOperators { get; set; }

    /// <summary>
    /// A stopwatch for tracking evolution time.
    /// </summary>
    protected Stopwatch EvolutionStopwatch { get; set; }

    /// <summary>
    /// The current evolution statistics.
    /// </summary>
    protected EvolutionStats<T, TInput, TOutput> CurrentStats { get; set; }

    /// <summary>
    /// Initializes a new instance of the GeneticModelBase class.
    /// </summary>
    /// <param name="fitnessCalculator">The fitness calculator to use.</param>
    protected GeneticBase(IFitnessCalculator<T, TInput, TOutput> fitnessCalculator)
    {
        FitnessCalculator = fitnessCalculator ?? throw new ArgumentNullException(nameof(fitnessCalculator));
        Population = [];
        GeneticParams = new GeneticParameters();
        Random = new Random();
        CrossoverOperators = [];
        MutationOperators = [];
        EvolutionStopwatch = new Stopwatch();
        CurrentStats = new EvolutionStats<T, TInput, TOutput>(fitnessCalculator);

        // Register default operators
        AddDefaultCrossoverOperators();
        AddDefaultMutationOperators();
    }

    /// <summary>
    /// Adds default crossover operators.
    /// </summary>
    protected virtual void AddDefaultCrossoverOperators()
    {
        // Single-point crossover
        AddCrossoverOperator("SinglePoint", (parent1, parent2, rate) =>
        {
            var offspring = new List<TIndividual>();
            if (Random.NextDouble() > rate)
            {
                // Use pattern matching with null checks for cloning
                if (parent1.Clone() is TIndividual clone1)
                    offspring.Add(clone1);

                if (parent2.Clone() is TIndividual clone2)
                    offspring.Add(clone2);

                return offspring;
            }

            // Clone with null checks
            var child1 = parent1.Clone() as TIndividual;
            var child2 = parent2.Clone() as TIndividual;

            // Check if either clone operation failed
            if (child1 == null || child2 == null)
            {
                // If cloning failed, add the original parents if possible
                if (child1 != null)
                    offspring.Add(child1);
                if (child2 != null)
                    offspring.Add(child2);

                // If we couldn't add anything, add clones of the parents with null checks
                if (offspring.Count == 0)
                {
                    if (parent1.Clone() is TIndividual clone1)
                        offspring.Add(clone1);
                    if (parent2.Clone() is TIndividual clone2)
                        offspring.Add(clone2);
                }

                return offspring;
            }

            var genes1 = child1.GetGenes().ToList();
            var genes2 = child2.GetGenes().ToList();

            if (genes1.Count == 0 || genes2.Count == 0)
            {
                offspring.Add(child1);
                offspring.Add(child2);

                return offspring;
            }

            int crossoverPoint = Random.Next(Math.Min(genes1.Count, genes2.Count));

            var newGenes1 = new List<TGene>();
            var newGenes2 = new List<TGene>();

            for (int i = 0; i < Math.Max(genes1.Count, genes2.Count); i++)
            {
                if (i < crossoverPoint)
                {
                    if (i < genes1.Count) newGenes1.Add(genes1[i]);
                    if (i < genes2.Count) newGenes2.Add(genes2[i]);
                }
                else
                {
                    if (i < genes2.Count) newGenes1.Add(genes2[i]);
                    if (i < genes1.Count) newGenes2.Add(genes1[i]);
                }
            }

            child1.SetGenes(newGenes1);
            child2.SetGenes(newGenes2);

            offspring.Add(child1);
            offspring.Add(child2);

            return offspring;
        });

        // Uniform crossover
        AddCrossoverOperator("Uniform", (parent1, parent2, rate) =>
        {
            var offspring = new List<TIndividual>();
            if (Random.NextDouble() > rate)
            {
                if (parent1.Clone() is TIndividual clone1)
                    offspring.Add(clone1);

                if (parent2.Clone() is TIndividual clone2)
                    offspring.Add(clone2);

                return offspring;
            }

            var child1 = parent1.Clone() as TIndividual;
            var child2 = parent2.Clone() as TIndividual;

            // Check if either clone operation failed
            if (child1 == null || child2 == null)
            {
                // If cloning failed, add the original parents if possible
                if (child1 != null)
                    offspring.Add(child1);
                if (child2 != null)
                    offspring.Add(child2);

                // If we couldn't add anything, add clones of the parents with null checks
                if (offspring.Count == 0)
                {
                    if (parent1.Clone() is TIndividual clone1)
                        offspring.Add(clone1);
                    if (parent2.Clone() is TIndividual clone2)
                        offspring.Add(clone2);
                }

                return offspring;
            }

            var genes1 = child1.GetGenes().ToList();
            var genes2 = child2.GetGenes().ToList();

            if (genes1.Count == 0 || genes2.Count == 0)
            {
                offspring.Add(child1);
                offspring.Add(child2);

                return offspring;
            }

            var newGenes1 = new List<TGene>();
            var newGenes2 = new List<TGene>();

            for (int i = 0; i < Math.Max(genes1.Count, genes2.Count); i++)
            {
                if (Random.NextDouble() < 0.5)
                {
                    if (i < genes1.Count) newGenes1.Add(genes1[i]);
                    if (i < genes2.Count) newGenes2.Add(genes2[i]);
                }
                else
                {
                    if (i < genes2.Count) newGenes1.Add(genes2[i]);
                    if (i < genes1.Count) newGenes2.Add(genes1[i]);
                }
            }

            child1.SetGenes(newGenes1);
            child2.SetGenes(newGenes2);

            offspring.Add(child1);
            offspring.Add(child2);

            return offspring;
        });
    }

    /// <summary>
    /// Adds default mutation operators.
    /// </summary>
    protected virtual void AddDefaultMutationOperators()
    {
        // Uniform mutation - mutates genes randomly
        AddMutationOperator("Uniform", (individual, rate) =>
        {
            // Check if clone is null
            if (individual.Clone() is not TIndividual clone)
            {
                // If cloning failed, return the original individual
                // This ensures we don't return null
                return individual;
            }

            var genes = clone.GetGenes().ToList();

            for (int i = 0; i < genes.Count; i++)
            {
                if (Random.NextDouble() < rate)
                {
                    genes[i] = MutateGene(genes[i]);
                }
            }

            clone.SetGenes(genes);
            return clone;
        });

        // Gaussian mutation - for numeric genes that support it
        AddMutationOperator("Gaussian", (individual, rate) =>
        {
            // Check if clone is null
            if (individual.Clone() is not TIndividual clone)
            {
                // If cloning failed, return the original individual
                // This ensures we don't return null
                return individual;
            }

            var genes = clone.GetGenes().ToList();

            for (int i = 0; i < genes.Count; i++)
            {
                if (Random.NextDouble() < rate)
                {
                    genes[i] = MutateGeneGaussian(genes[i]);
                }
            }

            clone.SetGenes(genes);
            return clone;
        });
    }

    /// <summary>
    /// Creates a mutated version of a gene.
    /// </summary>
    /// <param name="gene">The gene to mutate.</param>
    /// <returns>A mutated copy of the gene.</returns>
    protected abstract TGene MutateGene(TGene gene);

    /// <summary>
    /// Creates a mutated version of a gene using Gaussian noise.
    /// </summary>
    /// <param name="gene">The gene to mutate.</param>
    /// <returns>A mutated copy of the gene.</returns>
    protected abstract TGene MutateGeneGaussian(TGene gene);

    #region IGeneticModel Implementation

    /// <summary>
    /// Gets the fitness calculator used to evaluate individuals.
    /// </summary>
    /// <returns>The fitness calculator instance used by this genetic model.</returns>
    public IFitnessCalculator<T, TInput, TOutput> GetFitnessCalculator()
    {
        return FitnessCalculator;
    }

    /// <summary>
    /// Sets the fitness calculator to be used for evaluating individuals.
    /// </summary>
    /// <param name="fitnessCalculator">The fitness calculator to use.</param>
    public void SetFitnessCalculator(IFitnessCalculator<T, TInput, TOutput> fitnessCalculator)
    {
        FitnessCalculator = fitnessCalculator ?? throw new ArgumentNullException(nameof(fitnessCalculator));
        CurrentStats = new EvolutionStats<T, TInput, TOutput>(fitnessCalculator);
    }

    /// <summary>
    /// Gets the current population of individuals in the genetic model.
    /// </summary>
    /// <returns>A collection of individuals representing the current population.</returns>
    public ICollection<TIndividual> GetPopulation()
    {
        return Population;
    }

    /// <summary>
    /// Gets the best individual from the current population.
    /// </summary>
    /// <returns>The individual with the highest fitness.</returns>
    public TIndividual GetBestIndividual()
    {
        return BestIndividual;
    }

    /// <summary>
    /// Evaluates an individual by converting it to a model and generating evaluation data.
    /// </summary>
    /// <param name="individual">The individual to evaluate.</param>
    /// <param name="trainingInput">The input training data.</param>
    /// <param name="trainingOutput">The expected output for training.</param>
    /// <param name="validationInput">Optional validation input data.</param>
    /// <param name="validationOutput">Optional validation output data.</param>
    /// <returns>The calculated fitness score for the individual.</returns>
    public T EvaluateIndividual(
        TIndividual individual,
        TInput trainingInput,
        TOutput trainingOutput,
        TInput? validationInput = default,
        TOutput? validationOutput = default)
    {
        var model = IndividualToModel(individual);

        // Generate predictions for training data
        TOutput predictedOutput = model.Predict(trainingInput);

        // Calculate fitness score
        T fitnessScore;

        if (validationInput != null && validationOutput != null)
        {
            // Generate predictions for validation data
            TOutput validationPredicted = model.Predict(validationInput);

            // Use both training and validation data for fitness calculation
            fitnessScore = FitnessCalculator.CalculateFitnessScore(
                predictedOutput, trainingOutput,
                validationPredicted, validationOutput);
        }
        else
        {
            // Use only training data for fitness calculation
            fitnessScore = FitnessCalculator.CalculateFitnessScore(
                predictedOutput, trainingOutput);
        }

        // Update the individual's fitness
        individual.SetFitness(fitnessScore);

        return fitnessScore;
    }

    /// <summary>
    /// Evolves the population for a specified number of generations.
    /// </summary>
    /// <param name="generations">The number of generations to evolve.</param>
    /// <param name="trainingInput">The input training data used for fitness evaluation.</param>
    /// <param name="trainingOutput">The expected output for training used for fitness evaluation.</param>
    /// <param name="validationInput">Optional validation input data.</param>
    /// <param name="validationOutput">Optional validation output data.</param>
    /// <param name="stopCriteria">Optional function that determines when to stop evolution.</param>
    /// <returns>Statistics about the evolutionary process.</returns>
    public EvolutionStats<T, TInput, TOutput> Evolve(
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

            // Create next generation
            Population = CreateNextGeneration(trainingInput, trainingOutput, validationInput, validationOutput).ToList();

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

            // Check for fitness threshold
            if (FitnessCalculator.IsHigherScoreBetter)
            {
                if (Convert.ToDouble(BestIndividual.GetFitness()) >= GeneticParams.FitnessThreshold)
                {
                    break;
                }
            }
            else
            {
                if (Convert.ToDouble(BestIndividual.GetFitness()) <= GeneticParams.FitnessThreshold)
                {
                    break;
                }
            }
        }

        // Set final stats
        CurrentStats.TimeElapsed = EvolutionStopwatch.Elapsed;
        CurrentStats.BestIndividual = BestIndividual;

        return CurrentStats;
    }

    /// <summary>
    /// Creates the next generation of individuals through selection, crossover, and mutation.
    /// </summary>
    /// <param name="trainingInput">The input training data.</param>
    /// <param name="trainingOutput">The expected output for training.</param>
    /// <param name="validationInput">Optional validation input data.</param>
    /// <param name="validationOutput">Optional validation output data.</param>
    /// <returns>The new population.</returns>
    protected virtual ICollection<TIndividual> CreateNextGeneration(
        TInput trainingInput,
        TOutput trainingOutput,
        TInput? validationInput = default,
        TOutput? validationOutput = default)
    {
        List<TIndividual> newPopulation = new List<TIndividual>();

        // Elitism - keep the best individuals unchanged
        int eliteCount = (int)(GeneticParams.PopulationSize * GeneticParams.ElitismRate);
        var elites = GetElites(eliteCount);
        newPopulation.AddRange(elites);

        // Fill the rest of the population with offspring from crossover and mutation
        while (newPopulation.Count < GeneticParams.PopulationSize)
        {
            // Select parents
            var parents = Select(2, GeneticParams.SelectionMethod).ToList();

            if (parents.Count < 2)
            {
                continue;
            }

            // Crossover
            var offspring = Crossover(parents[0], parents[1], GeneticParams.CrossoverRate);

            // Mutation
            foreach (var child in offspring)
            {
                var mutated = Mutate(child, GeneticParams.MutationRate);

                // Evaluate the new individual
                EvaluateIndividual(mutated, trainingInput, trainingOutput, validationInput, validationOutput);

                newPopulation.Add(mutated);

                if (newPopulation.Count >= GeneticParams.PopulationSize)
                {
                    break;
                }
            }
        }

        return newPopulation;
    }

    /// <summary>
    /// Gets the elite individuals (best performers) from the population.
    /// </summary>
    /// <param name="count">The number of elite individuals to get.</param>
    /// <returns>The elite individuals.</returns>
    protected virtual ICollection<TIndividual> GetElites(int count)
    {
        return Population
            .OrderByDescending(i => FitnessCalculator.IsHigherScoreBetter ? i.GetFitness() : InvertFitness(i.GetFitness()))
            .Take(count)
            .Select(i => i.Clone() as TIndividual)
            .Where(i => i != null)  // Filter out any null values
            .ToList()!;
    }

    /// <summary>
    /// Evaluates all individuals in the population.
    /// </summary>
    /// <param name="trainingInput">The input training data.</param>
    /// <param name="trainingOutput">The expected output for training.</param>
    /// <param name="validationInput">Optional validation input data.</param>
    /// <param name="validationOutput">Optional validation output data.</param>
    protected virtual void EvaluatePopulation(
        TInput trainingInput,
        TOutput trainingOutput,
        TInput? validationInput = default,
        TOutput? validationOutput = default)
    {
        if (GeneticParams.UseParallelEvaluation)
        {
            Parallel.ForEach(Population, individual =>
            {
                EvaluateIndividual(individual, trainingInput, trainingOutput, validationInput, validationOutput);
            });
        }
        else
        {
            foreach (var individual in Population)
            {
                EvaluateIndividual(individual, trainingInput, trainingOutput, validationInput, validationOutput);
            }
        }
    }

    /// <summary>
    /// Updates the evolution statistics based on the current population.
    /// </summary>
    protected virtual void UpdateEvolutionStats()
    {
        var numOps = MathHelper.GetNumericOperations<T>();

        // Find best, worst, and average fitness
        T sumFitness = numOps.Zero;
        T bestFit = FitnessCalculator.IsHigherScoreBetter ? numOps.MinValue : numOps.MaxValue;
        T worstFit = FitnessCalculator.IsHigherScoreBetter ? numOps.MaxValue : numOps.MinValue;

        foreach (var individual in Population)
        {
            T fitness = individual.GetFitness();
            sumFitness = numOps.Add(sumFitness, fitness);

            if (IsBetterFitness(fitness, bestFit))
            {
                bestFit = fitness;
            }

            if (IsBetterFitness(worstFit, fitness))
            {
                worstFit = fitness;
            }
        }

        T avgFitness = numOps.Divide(sumFitness, numOps.FromDouble(Population.Count));

        // Calculate fitness standard deviation
        T sumSquaredDiff = numOps.Zero;
        foreach (var individual in Population)
        {
            T diff = numOps.Subtract(individual.GetFitness(), avgFitness);
            sumSquaredDiff = numOps.Add(sumSquaredDiff, numOps.Multiply(diff, diff));
        }

        T stdDev = numOps.Sqrt(numOps.Divide(sumSquaredDiff, numOps.FromDouble(Population.Count)));

        // Update stats
        CurrentStats.BestFitness = bestFit;
        CurrentStats.WorstFitness = worstFit;
        CurrentStats.AverageFitness = avgFitness;
        CurrentStats.FitnessStandardDeviation = stdDev;
        CurrentStats.FitnessHistory.Add(bestFit);
        CurrentStats.TimeElapsed = EvolutionStopwatch.Elapsed;

        // Calculate population diversity
        CurrentStats.Diversity = CalculateDiversity();
    }

    /// <summary>
    /// Finds the best individual in the current population.
    /// </summary>
    /// <returns>The individual with the best fitness.</returns>
    protected virtual TIndividual FindBestIndividual()
    {
        return Population
            .OrderByDescending(i => FitnessCalculator.IsHigherScoreBetter ? i.GetFitness() : InvertFitness(i.GetFitness()))
            .First();
    }

    /// <summary>
    /// Calculates the genetic diversity of the population.
    /// </summary>
    /// <returns>A measure of genetic diversity.</returns>
    protected virtual T CalculateDiversity()
    {
        var numOps = MathHelper.GetNumericOperations<T>();

        if (Population.Count <= 1)
        {
            return numOps.Zero;
        }

        T totalDistance = numOps.Zero;
        int comparisons = 0;

        for (int i = 0; i < Population.Count; i++)
        {
            for (int j = i + 1; j < Population.Count; j++)
            {
                totalDistance = numOps.Add(totalDistance, CalculateGeneticDistance(Population[i], Population[j]));
                comparisons++;
            }
        }

        return numOps.Divide(totalDistance, numOps.FromDouble(comparisons));
    }

    /// <summary>
    /// Calculates the genetic distance between two individuals.
    /// </summary>
    /// <param name="individual1">The first individual.</param>
    /// <param name="individual2">The second individual.</param>
    /// <returns>A measure of genetic distance.</returns>
    protected virtual T CalculateGeneticDistance(TIndividual individual1, TIndividual individual2)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var genes1 = individual1.GetGenes().ToList();
        var genes2 = individual2.GetGenes().ToList();

        // Default implementation - Hamming distance
        // Count the number of corresponding genes that are different

        int minLength = Math.Min(genes1.Count, genes2.Count);
        int differences = 0;

        for (int i = 0; i < minLength; i++)
        {
            if (!genes1[i].Equals(genes2[i]))
            {
                differences++;
            }
        }

        // Add differences for length mismatch
        differences += Math.Abs(genes1.Count - genes2.Count);

        return numOps.FromDouble(differences);
    }

    /// <summary>
    /// Inverts a fitness score for use in comparisons.
    /// </summary>
    /// <param name="fitness">The fitness score to invert.</param>
    /// <returns>The inverted fitness score.</returns>
    protected virtual T InvertFitness(T fitness)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        return numOps.Negate(fitness);
    }

    /// <summary>
    /// Determines if one fitness score is better than another.
    /// </summary>
    /// <param name="fitnessA">The first fitness score.</param>
    /// <param name="fitnessB">The second fitness score.</param>
    /// <returns>True if fitnessA is better than fitnessB; otherwise, false.</returns>
    protected virtual bool IsBetterFitness(T fitnessA, T fitnessB)
    {
        var numOps = MathHelper.GetNumericOperations<T>();

        if (FitnessCalculator.IsHigherScoreBetter)
        {
            return numOps.GreaterThan(fitnessA, fitnessB);
        }
        else
        {
            return numOps.LessThan(fitnessA, fitnessB);
        }
    }

    /// <summary>
    /// Performs crossover between two parent individuals to produce offspring.
    /// </summary>
    /// <param name="parent1">The first parent individual.</param>
    /// <param name="parent2">The second parent individual.</param>
    /// <param name="crossoverRate">The probability of crossover occurring.</param>
    /// <returns>One or more offspring produced by crossover.</returns>
    public virtual ICollection<TIndividual> Crossover(TIndividual parent1, TIndividual parent2, double crossoverRate)
    {
        if (!CrossoverOperators.TryGetValue(GeneticParams.CrossoverOperator, out var crossoverFunc))
        {
            throw new InvalidOperationException($"Crossover operator '{GeneticParams.CrossoverOperator}' not found.");
        }

        return crossoverFunc(parent1, parent2, crossoverRate);
    }

    /// <summary>
    /// Applies mutation to an individual.
    /// </summary>
    /// <param name="individual">The individual to mutate.</param>
    /// <param name="mutationRate">The probability of each gene mutating.</param>
    /// <returns>The mutated individual.</returns>
    public virtual TIndividual Mutate(TIndividual individual, double mutationRate)
    {
        if (!MutationOperators.TryGetValue(GeneticParams.MutationOperator, out var mutationFunc))
        {
            throw new InvalidOperationException($"Mutation operator '{GeneticParams.MutationOperator}' not found.");
        }

        return mutationFunc(individual, mutationRate);
    }

    /// <summary>
    /// Selects individuals from the population for reproduction.
    /// </summary>
    /// <param name="selectionSize">The number of individuals to select.</param>
    /// <param name="selectionMethod">The method to use for selection (e.g., tournament, roulette wheel).</param>
    /// <returns>The selected individuals.</returns>
    public virtual ICollection<TIndividual> Select(int selectionSize, SelectionMethod selectionMethod)
    {
        return selectionMethod switch
        {
            SelectionMethod.Tournament => TournamentSelection(selectionSize),
            SelectionMethod.RouletteWheel => RouletteWheelSelection(selectionSize),
            SelectionMethod.Rank => RankSelection(selectionSize),
            SelectionMethod.Truncation => TruncationSelection(selectionSize),
            SelectionMethod.Uniform => UniformSelection(selectionSize),
            SelectionMethod.StochasticUniversalSampling => StochasticUniversalSamplingSelection(selectionSize),
            SelectionMethod.Elitism => GetElites(selectionSize),
            _ => throw new ArgumentException($"Unsupported selection method: {selectionMethod}"),
        };
    }

    /// <summary>
    /// Selects individuals using truncation selection.
    /// </summary>
    /// <param name="selectionSize">The number of individuals to select.</param>
    /// <returns>The selected individuals.</returns>
    protected virtual ICollection<TIndividual> TruncationSelection(int selectionSize)
    {
        // Sort population by fitness and take the top selectionSize individuals
        return Population
            .OrderByDescending(i => FitnessCalculator.IsHigherScoreBetter ? i.GetFitness() : InvertFitness(i.GetFitness()))
            .Take(selectionSize)
            .Select(i => i.Clone() as TIndividual)
            .Where(i => i != null)  // Filter out any null values
            .ToList()!;
    }

    /// <summary>
    /// Selects individuals using uniform selection (all individuals have equal probability).
    /// </summary>
    /// <param name="selectionSize">The number of individuals to select.</param>
    /// <returns>The selected individuals.</returns>
    protected virtual ICollection<TIndividual> UniformSelection(int selectionSize)
    {
        List<TIndividual> selected = [];

        for (int i = 0; i < selectionSize; i++)
        {
            int randomIndex = Random.Next(Population.Count);
            selected.Add(Population[randomIndex].Clone() as TIndividual);
        }

        return selected;
    }

    /// <summary>
    /// Selects individuals using stochastic universal sampling.
    /// </summary>
    /// <param name="selectionSize">The number of individuals to select.</param>
    /// <returns>The selected individuals.</returns>
    protected virtual ICollection<TIndividual> StochasticUniversalSamplingSelection(int selectionSize)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        List<TIndividual> selected = [];

        // Calculate selection probabilities
        List<double> fitnessValues = [];

        // Adjust for minimization problems
        foreach (var individual in Population)
        {
            if (FitnessCalculator.IsHigherScoreBetter)
            {
                fitnessValues.Add(Convert.ToDouble(individual.GetFitness()));
            }
            else
            {
                // For minimization, invert the fitness
                double maxFitness = Population.Max(i => Convert.ToDouble(i.GetFitness())) + 1.0;
                fitnessValues.Add(maxFitness - Convert.ToDouble(individual.GetFitness()));
            }
        }

        double totalFitness = fitnessValues.Sum();

        // Handle case where all fitness values are 0
        if (totalFitness <= 0)
        {
            // Fall back to uniform selection
            return UniformSelection(selectionSize);
        }

        // Calculate distance between pointers
        double distance = totalFitness / selectionSize;

        // Random start point between 0 and distance
        double start = Random.NextDouble() * distance;

        // Create equally spaced pointers
        List<double> pointers = [];
        for (int i = 0; i < selectionSize; i++)
        {
            pointers.Add(start + i * distance);
        }

        // Calculate cumulative fitness
        List<double> cumulativeFitness = [];
        double sum = 0;

        foreach (double fitness in fitnessValues)
        {
            sum += fitness;
            cumulativeFitness.Add(sum);
        }

        // Select individuals using the pointers
        int index = 0;
        foreach (double pointer in pointers)
        {
            // Find the individual whose cumulative fitness is greater than the pointer
            while (index < Population.Count - 1 && cumulativeFitness[index] < pointer)
            {
                index++;
            }

            selected.Add(Population[index].Clone() as TIndividual);
        }

        return selected;
    }

    /// <summary>
    /// Selects individuals using tournament selection.
    /// </summary>
    /// <param name="selectionSize">The number of individuals to select.</param>
    /// <returns>The selected individuals.</returns>
    protected virtual ICollection<TIndividual> TournamentSelection(int selectionSize)
    {
        List<TIndividual> selected = [];

        for (int i = 0; i < selectionSize; i++)
        {
            TIndividual? best = null;

            for (int j = 0; j < GeneticParams.TournamentSize; j++)
            {
                int randomIndex = Random.Next(Population.Count);
                TIndividual contender = Population[randomIndex];

                if (best == null || IsBetterFitness(contender.GetFitness(), best.GetFitness()))
                {
                    best = contender;
                }
            }

            selected.Add(best.Clone() as TIndividual);
        }

        return selected;
    }

    /// <summary>
    /// Selects individuals using roulette wheel selection.
    /// </summary>
    /// <param name="selectionSize">The number of individuals to select.</param>
    /// <returns>The selected individuals.</returns>
    protected virtual ICollection<TIndividual> RouletteWheelSelection(int selectionSize)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        List<TIndividual> selected = new List<TIndividual>();

        // Calculate selection probabilities
        List<double> fitnessValues = new List<double>();

        // Adjust for minimization problems
        foreach (var individual in Population)
        {
            if (FitnessCalculator.IsHigherScoreBetter)
            {
                fitnessValues.Add(Convert.ToDouble(individual.GetFitness()));
            }
            else
            {
                // For minimization, invert the fitness
                double maxFitness = Population.Max(i => Convert.ToDouble(i.GetFitness())) + 1.0;
                fitnessValues.Add(maxFitness - Convert.ToDouble(individual.GetFitness()));
            }
        }

        double totalFitness = fitnessValues.Sum();

        // Handle case where all fitness values are 0
        if (totalFitness <= 0)
        {
            // Fall back to random selection
            for (int i = 0; i < selectionSize; i++)
            {
                int randomIndex = Random.Next(Population.Count);
                selected.Add(Population[randomIndex].Clone() as TIndividual);
            }

            return selected;
        }

        // Calculate cumulative probabilities
        List<double> cumulativeProbabilities = [];
        double cumulativeProbability = 0;

        for (int i = 0; i < Population.Count; i++)
        {
            cumulativeProbability += fitnessValues[i] / totalFitness;
            cumulativeProbabilities.Add(cumulativeProbability);
        }

        // Select individuals
        for (int i = 0; i < selectionSize; i++)
        {
            double randomValue = Random.NextDouble();

            for (int j = 0; j < cumulativeProbabilities.Count; j++)
            {
                if (randomValue <= cumulativeProbabilities[j])
                {
                    selected.Add(Population[j].Clone() as TIndividual);
                    break;
                }
            }
        }

        return selected;
    }

    /// <summary>
    /// Selects individuals using rank selection.
    /// </summary>
    /// <param name="selectionSize">The number of individuals to select.</param>
    /// <returns>The selected individuals.</returns>
    protected virtual ICollection<TIndividual> RankSelection(int selectionSize)
    {
        List<TIndividual> selected = [];

        // Sort population by fitness
        var sortedPopulation = Population
            .OrderByDescending(i => FitnessCalculator.IsHigherScoreBetter ? i.GetFitness() : InvertFitness(i.GetFitness()))
            .ToList();

        // Calculate rank-based probabilities
        double totalRank = Population.Count * (Population.Count + 1) / 2.0;
        List<double> cumulativeProbabilities = [];
        double cumulativeProbability = 0;

        for (int i = 0; i < sortedPopulation.Count; i++)
        {
            int rank = sortedPopulation.Count - i;
            cumulativeProbability += rank / totalRank;
            cumulativeProbabilities.Add(cumulativeProbability);
        }

        // Select individuals
        for (int i = 0; i < selectionSize; i++)
        {
            double randomValue = Random.NextDouble();

            for (int j = 0; j < cumulativeProbabilities.Count; j++)
            {
                if (randomValue <= cumulativeProbabilities[j])
                {
                    selected.Add(sortedPopulation[j].Clone() as TIndividual);
                    break;
                }
            }
        }

        return selected;
    }

    /// <summary>
    /// Initializes a new population with random individuals.
    /// </summary>
    /// <param name="populationSize">The size of the population to create.</param>
    /// <param name="initializationMethod">The method to use for initialization.</param>
    /// <returns>The newly created population.</returns>
    public abstract ICollection<TIndividual> InitializePopulation(int populationSize, InitializationMethod initializationMethod);

    /// <summary>
    /// Creates a new individual with the specified genes.
    /// </summary>
    /// <param name="genes">The genes to include in the individual.</param>
    /// <returns>A new individual with the specified genes.</returns>
    public abstract TIndividual CreateIndividual(ICollection<TGene> genes);

    /// <summary>
    /// Converts an individual to a trained model that can make predictions.
    /// </summary>
    /// <param name="individual">The individual to convert.</param>
    /// <returns>A model capable of making predictions based on the individual's genes.</returns>
    public abstract IModel<TInput, TOutput, ModelMetaData<T>> IndividualToModel(TIndividual individual);

    /// <summary>
    /// Gets statistics about the current evolutionary state.
    /// </summary>
    /// <returns>Statistics about the current evolutionary state.</returns>
    public virtual EvolutionStats<T, TInput, TOutput> GetEvolutionStats(IFitnessCalculator<T, TInput, TOutput> fitnessCalculator)
    {
        UpdateEvolutionStats();
        return CurrentStats;
    }

    /// <summary>
    /// Configures the genetic algorithm parameters.
    /// </summary>
    /// <param name="parameters">The genetic algorithm parameters to use.</param>
    public virtual void ConfigureGeneticParameters(GeneticParameters parameters)
    {
        GeneticParams = parameters ?? throw new ArgumentNullException(nameof(parameters));
    }

    /// <summary>
    /// Gets the current genetic algorithm parameters.
    /// </summary>
    /// <returns>The current genetic algorithm parameters.</returns>
    public virtual GeneticParameters GetGeneticParameters()
    {
        return GeneticParams;
    }

    /// <summary>
    /// Adds a custom crossover operator.
    /// </summary>
    /// <param name="name">The name of the crossover operator.</param>
    /// <param name="crossoverOperator">The crossover function.</param>
    public virtual void AddCrossoverOperator(string name, Func<TIndividual, TIndividual, double, ICollection<TIndividual>> crossoverOperator)
    {
        if (string.IsNullOrEmpty(name))
        {
            throw new ArgumentException("Operator name cannot be null or empty.", nameof(name));
        }

        CrossoverOperators[name] = crossoverOperator ?? throw new ArgumentNullException(nameof(crossoverOperator));
    }

    /// <summary>
    /// Adds a custom mutation operator.
    /// </summary>
    /// <param name="name">The name of the mutation operator.</param>
    /// <param name="mutationOperator">The mutation function.</param>
    public virtual void AddMutationOperator(string name, Func<TIndividual, double, TIndividual> mutationOperator)
    {
        if (string.IsNullOrEmpty(name))
        {
            throw new ArgumentException("Operator name cannot be null or empty.", nameof(name));
        }

        MutationOperators[name] = mutationOperator ?? throw new ArgumentNullException(nameof(mutationOperator));
    }

    /// <summary>
    /// Saves the current population to a file.
    /// </summary>
    /// <param name="filePath">The path where the population should be saved.</param>
    public virtual void SavePopulation(string filePath)
    {
        using (FileStream fs = new FileStream(filePath, FileMode.Create))
        {
            byte[] data = SerializePopulation();
            fs.Write(data, 0, data.Length);
        }
    }

    /// <summary>
    /// Serializes the population to a byte array.
    /// </summary>
    /// <returns>A byte array containing the serialized population.</returns>
    protected abstract byte[] SerializePopulation();

    /// <summary>
    /// Loads a population from a file.
    /// </summary>
    /// <param name="filePath">The path from which to load the population.</param>
    /// <returns>The loaded population.</returns>
    public virtual ICollection<TIndividual> LoadPopulation(string filePath)
    {
        if (!File.Exists(filePath))
        {
            throw new FileNotFoundException("Population file not found.", filePath);
        }

        using (FileStream fs = new FileStream(filePath, FileMode.Open))
        {
            byte[] data = new byte[fs.Length];
            fs.Read(data, 0, data.Length);

            Population = DeserializePopulation(data).ToList();

            if (Population.Count > 0)
            {
                BestIndividual = FindBestIndividual();
            }

            return Population;
        }
    }

    /// <summary>
    /// Deserializes a population from a byte array.
    /// </summary>
    /// <param name="data">The byte array containing the serialized population.</param>
    /// <returns>The deserialized population.</returns>
    protected abstract ICollection<TIndividual> DeserializePopulation(byte[] data);

    #endregion

    #region IFullModel Implementation

    /// <summary>
    /// Makes a prediction using the current best model.
    /// </summary>
    /// <param name="input">The input data.</param>
    /// <returns>The predicted output.</returns>
    public virtual TOutput Predict(TInput input)
    {
        if (BestIndividual == null)
        {
            throw new InvalidOperationException("No best individual found. The model must be trained before making predictions.");
        }

        var model = IndividualToModel(BestIndividual);
        return model.Predict(input);
    }

    /// <summary>
    /// Gets the metadata for the model.
    /// </summary>
    /// <returns>The model metadata.</returns>
    public abstract ModelMetaData<T> GetMetaData();

    /// <summary>
    /// Serializes the model to a byte array.
    /// </summary>
    /// <returns>A byte array containing the serialized model.</returns>
    public virtual byte[] Serialize()
    {
        using (MemoryStream ms = new MemoryStream())
        using (BinaryWriter writer = new BinaryWriter(ms))
        {
            // Write version
            writer.Write(1); // Version 1

            // Write genetic parameters
            writer.Write(JsonSerializer.Serialize(GeneticParams));

            // Write best individual if available
            bool hasBestIndividual = BestIndividual != null;
            writer.Write(hasBestIndividual);

            if (hasBestIndividual)
            {
                byte[] individualData = SerializeIndividual(BestIndividual);
                writer.Write(individualData.Length);
                writer.Write(individualData);
            }

            // Write model-specific data
            byte[] modelData = SerializeModelData();
            writer.Write(modelData.Length);
            writer.Write(modelData);

            return ms.ToArray();
        }
    }

    /// <summary>
    /// Serializes an individual to a byte array.
    /// </summary>
    /// <param name="individual">The individual to serialize.</param>
    /// <returns>A byte array containing the serialized individual.</returns>
    protected abstract byte[] SerializeIndividual(TIndividual individual);

    /// <summary>
    /// Serializes model-specific data.
    /// </summary>
    /// <returns>A byte array containing the serialized model data.</returns>
    protected abstract byte[] SerializeModelData();

    /// <summary>
    /// Deserializes the model from a byte array.
    /// </summary>
    /// <param name="data">The byte array containing the serialized model.</param>
    public virtual void Deserialize(byte[] data)
    {
        using (MemoryStream ms = new MemoryStream(data))
        using (BinaryReader reader = new BinaryReader(ms))
        {
            // Read version
            int version = reader.ReadInt32();

            if (version != 1)
            {
                throw new InvalidOperationException($"Unsupported model version: {version}");
            }

            // Read genetic parameters
            string paramsJson = reader.ReadString();
            GeneticParams = JsonSerializer.Deserialize<GeneticParameters>(paramsJson);

            // Read best individual if available
            bool hasBestIndividual = reader.ReadBoolean();

            if (hasBestIndividual)
            {
                int individualDataLength = reader.ReadInt32();
                byte[] individualData = reader.ReadBytes(individualDataLength);
                BestIndividual = DeserializeIndividual(individualData);
            }

            // Read model-specific data
            int modelDataLength = reader.ReadInt32();
            byte[] modelData = reader.ReadBytes(modelDataLength);
            DeserializeModelData(modelData);
        }
    }

    /// <summary>
    /// Deserializes an individual from a byte array.
    /// </summary>
    /// <param name="data">The byte array containing the serialized individual.</param>
    /// <returns>The deserialized individual.</returns>
    protected abstract TIndividual DeserializeIndividual(byte[] data);

    /// <summary>
    /// Deserializes model-specific data.
    /// </summary>
    /// <param name="data">The byte array containing the serialized model data.</param>
    protected abstract void DeserializeModelData(byte[] data);

    #endregion
}