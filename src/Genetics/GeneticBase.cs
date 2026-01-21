using Newtonsoft.Json;

namespace AiDotNet.Genetics;

/// <summary>
/// Provides a base implementation of IGeneticModel that handles common genetic algorithm operations.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <typeparam name="TInput">The type of input data for the model.</typeparam>
/// <typeparam name="TOutput">The type of output data produced by the model.</typeparam>
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
public abstract class GeneticBase<T, TInput, TOutput> :
    IGeneticAlgorithm<T, TInput, TOutput, ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>, ModelParameterGene<T>>
{
    /// <summary>
    /// The current population of individuals.
    /// </summary>
    protected List<ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>> Population { get; set; }

    /// <summary>
    /// The best individual found so far.
    /// </summary>
    protected ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>? BestIndividual { get; set; }

    /// <summary>
    /// The parameters for the genetic algorithm.
    /// </summary>
    protected GeneticParameters GeneticParams { get; set; }

    /// <summary>
    /// The fitness calculator used to evaluate individuals.
    /// </summary>
    protected IFitnessCalculator<T, TInput, TOutput> FitnessCalculator { get; set; }

    /// <summary>
    /// The fitness calculator used to evaluate individuals.
    /// </summary>
    protected IModelEvaluator<T, TInput, TOutput> ModelEvaluator { get; set; }

    /// <summary>
    /// The random number generator used for stochastic operations.
    /// </summary>
    protected Random Random { get; set; }

    /// <summary>
    /// A dictionary mapping crossover operator names to implementations.
    /// </summary>
    protected Dictionary<string, Func<ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>, ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>, double, ICollection<ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>>>> CrossoverOperators { get; set; }

    /// <summary>
    /// A dictionary mapping mutation operator names to implementations.
    /// </summary>
    protected Dictionary<string, Func<ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>, double, ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>>> MutationOperators { get; set; }

    /// <summary>
    /// A stopwatch for tracking evolution time.
    /// </summary>
    protected Stopwatch EvolutionStopwatch { get; set; }

    /// <summary>
    /// The current evolution statistics.
    /// </summary>
    protected EvolutionStats<T, TInput, TOutput> CurrentStats { get; set; }

    protected INumericOperations<T> NumOps { get; set; } = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// The training input data, stored before population initialization so derived classes
    /// can use it to determine proper parameter dimensions when models have empty parameters.
    /// </summary>
    protected TInput? TrainingInputForInitialization { get; set; }

    /// <summary>
    /// Initializes a new instance of the GeneticModelBase class.
    /// </summary>
    /// <param name="fitnessCalculator">The fitness calculator to use.</param>
    protected GeneticBase(IFitnessCalculator<T, TInput, TOutput> fitnessCalculator, IModelEvaluator<T, TInput, TOutput> modelEvaluator)
    {
        FitnessCalculator = fitnessCalculator ?? throw new ArgumentNullException(nameof(fitnessCalculator));
        Population = new List<ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>>();
        GeneticParams = new GeneticParameters();
        Random = RandomHelper.CreateSecureRandom();
        CrossoverOperators = new Dictionary<string, Func<ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>, ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>, double, ICollection<ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>>>>();
        MutationOperators = new Dictionary<string, Func<ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>, double, ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>>>();
        EvolutionStopwatch = new Stopwatch();
        CurrentStats = new EvolutionStats<T, TInput, TOutput>(fitnessCalculator);
        ModelEvaluator = modelEvaluator ?? throw new ArgumentNullException(nameof(modelEvaluator));

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
            var offspring = new List<ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>>();
            if (Random.NextDouble() > rate)
            {
                // Use pattern matching with null checks for cloning
                if (parent1.Clone() is ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>> clone1)
                    offspring.Add(clone1);

                if (parent2.Clone() is ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>> clone2)
                    offspring.Add(clone2);

                return offspring;
            }

            // Clone with null checks - simplified fallback logic
            var child1 = parent1.Clone() as ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>;
            var child2 = parent2.Clone() as ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>;

            // If cloning failed for either, return the successfully cloned individuals or originals as fallback
            if (child1 == null || child2 == null)
            {
                offspring.Add(child1 ?? parent1);
                offspring.Add(child2 ?? parent2);
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

            var newGenes1 = new List<ModelParameterGene<T>>();
            var newGenes2 = new List<ModelParameterGene<T>>();

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
            var offspring = new List<ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>>();
            if (Random.NextDouble() > rate)
            {
                if (parent1.Clone() is ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>> clone1)
                    offspring.Add(clone1);

                if (parent2.Clone() is ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>> clone2)
                    offspring.Add(clone2);

                return offspring;
            }

            var child1 = parent1.Clone() as ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>;
            var child2 = parent2.Clone() as ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>;

            // If cloning failed for either, return the successfully cloned individuals or originals as fallback
            if (child1 == null || child2 == null)
            {
                offspring.Add(child1 ?? parent1);
                offspring.Add(child2 ?? parent2);
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

            var newGenes1 = new List<ModelParameterGene<T>>();
            var newGenes2 = new List<ModelParameterGene<T>>();

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
            if (individual.Clone() is not ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>> clone)
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
            if (individual.Clone() is not ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>> clone)
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
    protected abstract ModelParameterGene<T> MutateGene(ModelParameterGene<T> gene);

    /// <summary>
    /// Creates a mutated version of a gene using Gaussian noise.
    /// </summary>
    /// <param name="gene">The gene to mutate.</param>
    /// <returns>A mutated copy of the gene.</returns>
    protected abstract ModelParameterGene<T> MutateGeneGaussian(ModelParameterGene<T> gene);

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
    public ICollection<ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>> GetPopulation()
    {
        return Population;
    }

    /// <summary>
    /// Gets the best individual from the current population.
    /// </summary>
    /// <returns>The individual with the highest fitness, or throws an exception if no best individual exists.</returns>
    public ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>> GetBestIndividual()
    {
        if (BestIndividual == null)
        {
            throw new InvalidOperationException("No best individual found. Ensure the population has been evaluated.");
        }

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
        ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>> individual,
        TInput trainingInput,
        TOutput trainingOutput,
        TInput? validationInput = default,
        TOutput? validationOutput = default)
    {
        var model = IndividualToModel(individual);

        var subsetInputData = new OptimizationInputData<T, TInput, TOutput>
        {
            XTrain = trainingInput,
            YTrain = trainingOutput,
        };

        var input = new ModelEvaluationInput<T, TInput, TOutput>
        {
            Model = model,
            InputData = subsetInputData
        };

        // Note: We do NOT call Train here because in a genetic algorithm context,
        // the model parameters are already set from the individual's genes via IndividualToModel.
        // The genetic algorithm evolves parameters through mutation/crossover, not through
        // traditional training methods like the normal equation.

        // Evaluate the model with its gene-based parameters
        if (validationInput != null && validationOutput != null)
        {
            input.InputData.XValidation = validationInput;
            input.InputData.YValidation = validationOutput;
        }

        var evaluationData = ModelEvaluator.EvaluateModel(input);
        var currentFitnessScore = FitnessCalculator.CalculateFitnessScore(evaluationData);

        // Update the individual's fitness
        individual.SetFitness(currentFitnessScore);

        return currentFitnessScore;
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
    public virtual EvolutionStats<T, TInput, TOutput> Evolve(
        int generations,
        TInput trainingInput,
        TOutput trainingOutput,
        TInput? validationInput = default,
        TOutput? validationOutput = default,
        Func<EvolutionStats<T, TInput, TOutput>, bool>? stopCriteria = null)
    {
        // Store training input for population initialization
        // This allows derived classes to determine proper parameter dimensions
        TrainingInputForInitialization = trainingInput;

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
    protected virtual ICollection<ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>> CreateNextGeneration(
        TInput trainingInput,
        TOutput trainingOutput,
        TInput? validationInput = default,
        TOutput? validationOutput = default)
    {
        List<ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>> newPopulation = new List<ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>>();

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
    protected virtual ICollection<ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>> GetElites(int count)
    {
        return Population
            .OrderByDescending(i => FitnessCalculator.IsHigherScoreBetter ? i.GetFitness() : InvertFitness(i.GetFitness()))
            .Take(count)
            .Select(i => i.Clone() as ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>)
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
        // Find best, worst, and average fitness
        T sumFitness = NumOps.Zero;
        T bestFit = FitnessCalculator.IsHigherScoreBetter ? NumOps.MinValue : NumOps.MaxValue;
        T worstFit = FitnessCalculator.IsHigherScoreBetter ? NumOps.MaxValue : NumOps.MinValue;

        foreach (var individual in Population)
        {
            T fitness = individual.GetFitness();
            sumFitness = NumOps.Add(sumFitness, fitness);

            if (IsBetterFitness(fitness, bestFit))
            {
                bestFit = fitness;
            }

            if (IsBetterFitness(worstFit, fitness))
            {
                worstFit = fitness;
            }
        }

        T avgFitness = NumOps.Divide(sumFitness, NumOps.FromDouble(Population.Count));

        // Calculate fitness standard deviation
        T sumSquaredDiff = NumOps.Zero;
        foreach (var individual in Population)
        {
            T diff = NumOps.Subtract(individual.GetFitness(), avgFitness);
            sumSquaredDiff = NumOps.Add(sumSquaredDiff, NumOps.Multiply(diff, diff));
        }

        T stdDev = NumOps.Sqrt(NumOps.Divide(sumSquaredDiff, NumOps.FromDouble(Population.Count)));

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
    protected virtual ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>> FindBestIndividual()
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
        if (Population.Count <= 1)
        {
            return NumOps.Zero;
        }

        T totalDistance = NumOps.Zero;
        int comparisons = 0;

        for (int i = 0; i < Population.Count; i++)
        {
            for (int j = i + 1; j < Population.Count; j++)
            {
                totalDistance = NumOps.Add(totalDistance, CalculateGeneticDistance(Population[i], Population[j]));
                comparisons++;
            }
        }

        return NumOps.Divide(totalDistance, NumOps.FromDouble(comparisons));
    }

    /// <summary>
    /// Calculates the genetic distance between two individuals.
    /// </summary>
    /// <param name="individual1">The first individual.</param>
    /// <param name="individual2">The second individual.</param>
    /// <returns>A measure of genetic distance.</returns>
    protected virtual T CalculateGeneticDistance(
        ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>> individual1,
        ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>> individual2)
    {
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

        return NumOps.FromDouble(differences);
    }

    /// <summary>
    /// Inverts a fitness score for use in comparisons.
    /// </summary>
    /// <param name="fitness">The fitness score to invert.</param>
    /// <returns>The inverted fitness score.</returns>
    protected virtual T InvertFitness(T fitness)
    {
        return NumOps.Negate(fitness);
    }

    /// <summary>
    /// Determines if one fitness score is better than another.
    /// </summary>
    /// <param name="fitnessA">The first fitness score.</param>
    /// <param name="fitnessB">The second fitness score.</param>
    /// <returns>True if fitnessA is better than fitnessB; otherwise, false.</returns>
    protected virtual bool IsBetterFitness(T fitnessA, T fitnessB)
    {

        if (FitnessCalculator.IsHigherScoreBetter)
        {
            return NumOps.GreaterThan(fitnessA, fitnessB);
        }
        else
        {
            return NumOps.LessThan(fitnessA, fitnessB);
        }
    }

    /// <summary>
    /// Performs crossover between two parent individuals to produce offspring.
    /// </summary>
    /// <param name="parent1">The first parent individual.</param>
    /// <param name="parent2">The second parent individual.</param>
    /// <param name="crossoverRate">The probability of crossover occurring.</param>
    /// <returns>One or more offspring produced by crossover.</returns>
    public virtual ICollection<ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>> Crossover(
        ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>> parent1,
        ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>> parent2,
        double crossoverRate)
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
    public virtual ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>> Mutate(
        ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>> individual,
        double mutationRate)
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
    public virtual ICollection<ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>> Select(int selectionSize, SelectionMethod selectionMethod)
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
    protected virtual ICollection<ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>> TruncationSelection(int selectionSize)
    {
        // Sort population by fitness and take the top selectionSize individuals
        return Population
            .OrderByDescending(i => FitnessCalculator.IsHigherScoreBetter ? i.GetFitness() : InvertFitness(i.GetFitness()))
            .Take(selectionSize)
            .Select(i => i.Clone() as ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>)
            .Where(i => i != null)
            .ToList()!;
    }

    /// <summary>
    /// Selects individuals using uniform selection (all individuals have equal probability).
    /// </summary>
    /// <param name="selectionSize">The number of individuals to select.</param>
    /// <returns>The selected individuals.</returns>
    protected virtual ICollection<ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>> UniformSelection(int selectionSize)
    {
        List<ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>> selected = new List<ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>>();

        for (int i = 0; i < selectionSize; i++)
        {
            int randomIndex = Random.Next(Population.Count);

            // Only add the clone if it's not null
            if (Population[randomIndex].Clone() is ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>> clone)
            {
                selected.Add(clone);
            }
            else
            {
                // If cloning failed, try to add the original individual
                // This is a fallback to ensure we don't skip selections
                i--; // Retry this iteration
            }
        }

        return selected;
    }

    /// <summary>
    /// Selects individuals using stochastic universal sampling.
    /// </summary>
    /// <param name="selectionSize">The number of individuals to select.</param>
    /// <returns>The selected individuals.</returns>
    protected virtual ICollection<ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>> StochasticUniversalSamplingSelection(int selectionSize)
    {
        List<ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>> selected = new List<ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>>();

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
            // Fall back to uniform selection
            return UniformSelection(selectionSize);
        }

        // Calculate distance between pointers
        double distance = totalFitness / selectionSize;

        // Random start point between 0 and distance
        double start = Random.NextDouble() * distance;

        // Create equally spaced pointers
        List<double> pointers = new List<double>();
        for (int i = 0; i < selectionSize; i++)
        {
            pointers.Add(start + i * distance);
        }

        // Calculate cumulative fitness
        List<double> cumulativeFitness = new List<double>();
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

            // Clone with null check
            if (Population[index].Clone() is ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>> clone)
            {
                selected.Add(clone);
            }
            else
            {
                // If cloning failed, try again with a different individual
                // This is a fallback to ensure we don't skip selections
                int alternateIndex = (index + 1) % Population.Count;
                if (Population[alternateIndex].Clone() is ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>> alternateClone)
                {
                    selected.Add(alternateClone);
                }
                // If both attempts fail, we'll just continue without adding to maintain the algorithm's flow
            }
        }

        return selected;
    }

    /// <summary>
    /// Selects individuals using tournament selection.
    /// </summary>
    /// <param name="selectionSize">The number of individuals to select.</param>
    /// <returns>The selected individuals.</returns>
    protected virtual ICollection<ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>> TournamentSelection(int selectionSize)
    {
        List<ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>> selected = new List<ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>>();

        for (int i = 0; i < selectionSize; i++)
        {
            ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>? best = null;

            for (int j = 0; j < GeneticParams.TournamentSize; j++)
            {
                int randomIndex = Random.Next(Population.Count);
                ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>> contender = Population[randomIndex];

                if (best == null || IsBetterFitness(contender.GetFitness(), best.GetFitness()))
                {
                    best = contender;
                }
            }

            // At this point, best should never be null since we're selecting from a non-empty population
            // But we'll add a null check to be safe
            if (best != null)
            {
                if (best.Clone() is ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>> clone)
                {
                    selected.Add(clone);
                }
                else
                {
                    // If cloning failed, try to add the original individual
                    // This is a fallback to ensure we don't skip selections
                    i--; // Retry this iteration
                }
            }
            else
            {
                // This should never happen with a non-empty population
                // But if it does, retry the selection
                i--;
            }
        }

        return selected;
    }

    /// <summary>
    /// Selects individuals using roulette wheel selection.
    /// </summary>
    /// <param name="selectionSize">The number of individuals to select.</param>
    /// <returns>The selected individuals.</returns>
    protected virtual ICollection<ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>> RouletteWheelSelection(int selectionSize)
    {
        List<ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>> selected = new List<ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>>();

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
                if (Population[randomIndex].Clone() is ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>> clone)
                {
                    selected.Add(clone);
                }
                else
                {
                    // If cloning failed, try again
                    i--; // Retry this iteration
                }
            }

            return selected;
        }

        // Calculate cumulative probabilities
        List<double> cumulativeProbabilities = new List<double>();
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
            bool selectedIndividual = false;

            for (int j = 0; j < cumulativeProbabilities.Count; j++)
            {
                if (randomValue <= cumulativeProbabilities[j])
                {
                    if (Population[j].Clone() is ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>> clone)
                    {
                        selected.Add(clone);
                        selectedIndividual = true;
                        break;
                    }
                }
            }

            // If we couldn't select an individual (due to cloning failure), try again
            if (!selectedIndividual)
            {
                i--; // Retry this iteration
            }
        }

        return selected;
    }

    /// <summary>
    /// Selects individuals using rank selection.
    /// </summary>
    /// <param name="selectionSize">The number of individuals to select.</param>
    /// <returns>The selected individuals.</returns>
    protected virtual ICollection<ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>> RankSelection(int selectionSize)
    {
        List<ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>> selected = new List<ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>>();

        // Sort population by fitness
        var sortedPopulation = Population
            .OrderByDescending(i => FitnessCalculator.IsHigherScoreBetter ? i.GetFitness() : InvertFitness(i.GetFitness()))
            .ToList();

        // Calculate rank-based probabilities
        double totalRank = Population.Count * (Population.Count + 1) / 2.0;
        List<double> cumulativeProbabilities = new List<double>();
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
            bool selectedIndividual = false;

            for (int j = 0; j < cumulativeProbabilities.Count; j++)
            {
                if (randomValue <= cumulativeProbabilities[j])
                {
                    if (sortedPopulation[j].Clone() is ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>> clone)
                    {
                        selected.Add(clone);
                        selectedIndividual = true;
                        break;
                    }
                }
            }

            // If we couldn't select an individual (due to cloning failure), try again
            if (!selectedIndividual)
            {
                i--; // Retry this iteration
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
    public abstract ICollection<ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>> InitializePopulation(
        int populationSize,
        InitializationMethod initializationMethod);

    /// <summary>
    /// Creates a new individual with the specified genes.
    /// </summary>
    /// <param name="genes">The genes to include in the individual.</param>
    /// <returns>A new individual with the specified genes.</returns>
    public abstract ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>> CreateIndividual(
        ICollection<ModelParameterGene<T>> genes);

    /// <summary>
    /// Converts an individual to a trained model that can make predictions.
    /// </summary>
    /// <param name="individual">The individual to convert.</param>
    /// <returns>A model capable of making predictions based on the individual's genes.</returns>
    public abstract IFullModel<T, TInput, TOutput> IndividualToModel(
        ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>> individual);

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
    public virtual void AddCrossoverOperator(
        string name,
        Func<ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>,
             ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>,
             double,
             ICollection<ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>>> crossoverOperator)
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
    public virtual void AddMutationOperator(
        string name,
        Func<ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>,
             double,
             ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>> mutationOperator)
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
    public virtual ICollection<ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>> LoadPopulation(string filePath)
    {
        if (!File.Exists(filePath))
        {
            throw new FileNotFoundException("Population file not found.", filePath);
        }

        byte[] data = File.ReadAllBytes(filePath);

        Population = DeserializePopulation(data).ToList();

        if (Population.Count > 0)
        {
            BestIndividual = FindBestIndividual();
        }

        return Population;
    }

    /// <summary>
    /// Deserializes a population from a byte array.
    /// </summary>
    /// <param name="data">The byte array containing the serialized population.</param>
    /// <returns>The deserialized population.</returns>
    protected abstract ICollection<ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>> DeserializePopulation(byte[] data);

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
    public abstract ModelMetadata<T> GetMetaData();

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
            writer.Write(JsonConvert.SerializeObject(GeneticParams));

            // Write best individual if available
            bool hasBestIndividual = BestIndividual != null;
            writer.Write(hasBestIndividual);

            if (hasBestIndividual)
            {
                // BestIndividual is guaranteed to be non-null here because of the check above
                byte[] individualData = SerializeIndividual(BestIndividual!);
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
    protected abstract byte[] SerializeIndividual(ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>> individual);

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
            GeneticParams = JsonConvert.DeserializeObject<GeneticParameters>(paramsJson) ?? new GeneticParameters();

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
    protected abstract ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>> DeserializeIndividual(byte[] data);

    /// <summary>
    /// Deserializes model-specific data.
    /// </summary>
    /// <param name="data">The byte array containing the serialized model data.</param>
    protected abstract void DeserializeModelData(byte[] data);

    #endregion
}
