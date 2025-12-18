namespace AiDotNet.Interfaces;

/// <summary>
/// Represents a machine learning model that uses genetic algorithms or evolutionary computation
/// while maintaining the core capabilities of a full model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <typeparam name="TInput">The type of input data for the model.</typeparam>
/// <typeparam name="TOutput">The type of output data produced by the model.</typeparam>
/// <typeparam name="TIndividual">The type representing an individual in the genetic population.</typeparam>
/// <typeparam name="TGene">The type representing a gene in the genetic model.</typeparam>
/// <remarks>
/// <para>
/// This interface extends the IFullModel interface by adding genetic algorithm capabilities
/// that allow for evolutionary optimization, including population management, crossover operations,
/// mutation, selection, and fitness evaluation through integration with IFitnessCalculator.
/// </para>
/// <para><b>For Beginners:</b>
/// This interface provides functionality for AI models that use genetic algorithms - methods
/// inspired by natural evolution.
/// 
/// Think of a genetic model as a population of potential solutions that evolve over time:
/// 
/// 1. Population Management
///    - The model maintains multiple candidate solutions (individuals)
///    - Each individual represents a possible solution to your problem
/// 
/// 2. Evolution Process
///    - Individuals are evaluated based on how well they solve the problem (fitness)
///    - The best individuals are selected to "reproduce" (selection)
///    - New individuals are created by combining parts of successful ones (crossover)
///    - Random changes are introduced to maintain diversity (mutation)
///    - This process repeats over many generations, with solutions improving over time
/// 
/// 3. Advantages
///    - Can solve complex problems where traditional algorithms struggle
///    - Often finds creative solutions humans might not consider
///    - Good for optimization problems and symbolic regression
///    - Can adapt to changing conditions and problems
/// 
/// This is particularly useful for problems like:
/// - Finding optimal neural network architectures
/// - Symbolic regression (discovering mathematical equations from data)
/// - Optimizing complex systems with many parameters
/// - Evolving game-playing strategies or agent behaviors
/// </para>
/// </remarks>
public interface IGeneticAlgorithm<T, TInput, TOutput, TIndividual, TGene>
    where TIndividual : class, IEvolvable<TGene, T>
    where TGene : class
{
    /// <summary>
    /// Gets the fitness calculator used to evaluate individuals.
    /// </summary>
    /// <returns>The fitness calculator instance used by this genetic model.</returns>
    IFitnessCalculator<T, TInput, TOutput> GetFitnessCalculator();

    /// <summary>
    /// Sets the fitness calculator to be used for evaluating individuals.
    /// </summary>
    /// <param name="fitnessCalculator">The fitness calculator to use.</param>
    void SetFitnessCalculator(IFitnessCalculator<T, TInput, TOutput> fitnessCalculator);

    /// <summary>
    /// Gets the current population of individuals in the genetic model.
    /// </summary>
    /// <returns>A collection of individuals representing the current population.</returns>
    ICollection<TIndividual> GetPopulation();

    /// <summary>
    /// Gets the best individual from the current population.
    /// </summary>
    /// <returns>The individual with the highest fitness.</returns>
    TIndividual GetBestIndividual();

    /// <summary>
    /// Evaluates an individual by converting it to a model and generating evaluation data.
    /// </summary>
    /// <param name="individual">The individual to evaluate.</param>
    /// <param name="trainingInput">The input training data.</param>
    /// <param name="trainingOutput">The expected output for training.</param>
    /// <param name="validationInput">Optional validation input data.</param>
    /// <param name="validationOutput">Optional validation output data.</param>
    /// <returns>The calculated fitness score for the individual.</returns>
    /// <remarks>
    /// This method uses the configured IFitnessCalculator to evaluate the individual's fitness.
    /// It first converts the individual to a model, then evaluates the model against the provided
    /// data to generate evaluation metrics, which are then passed to the fitness calculator.
    /// </remarks>
    T EvaluateIndividual(
        TIndividual individual,
        TInput trainingInput,
        TOutput trainingOutput,
        TInput? validationInput = default,
        TOutput? validationOutput = default);

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
    EvolutionStats<T, TInput, TOutput> Evolve(
        int generations,
        TInput trainingInput,
        TOutput trainingOutput,
        TInput? validationInput = default,
        TOutput? validationOutput = default,
        Func<EvolutionStats<T, TInput, TOutput>, bool>? stopCriteria = null);

    /// <summary>
    /// Performs crossover between two parent individuals to produce offspring.
    /// </summary>
    /// <param name="parent1">The first parent individual.</param>
    /// <param name="parent2">The second parent individual.</param>
    /// <param name="crossoverRate">The probability of crossover occurring.</param>
    /// <returns>One or more offspring produced by crossover.</returns>
    ICollection<TIndividual> Crossover(TIndividual parent1, TIndividual parent2, double crossoverRate);

    /// <summary>
    /// Applies mutation to an individual.
    /// </summary>
    /// <param name="individual">The individual to mutate.</param>
    /// <param name="mutationRate">The probability of each gene mutating.</param>
    /// <returns>The mutated individual.</returns>
    TIndividual Mutate(TIndividual individual, double mutationRate);

    /// <summary>
    /// Selects individuals from the population for reproduction.
    /// </summary>
    /// <param name="selectionSize">The number of individuals to select.</param>
    /// <param name="selectionMethod">The method to use for selection (e.g., tournament, roulette wheel).</param>
    /// <returns>The selected individuals.</returns>
    ICollection<TIndividual> Select(int selectionSize, SelectionMethod selectionMethod);

    /// <summary>
    /// Initializes a new population with random individuals.
    /// </summary>
    /// <param name="populationSize">The size of the population to create.</param>
    /// <param name="initializationMethod">The method to use for initialization.</param>
    /// <returns>The newly created population.</returns>
    ICollection<TIndividual> InitializePopulation(int populationSize, InitializationMethod initializationMethod);

    /// <summary>
    /// Creates a new individual with the specified genes.
    /// </summary>
    /// <param name="genes">The genes to include in the individual.</param>
    /// <returns>A new individual with the specified genes.</returns>
    TIndividual CreateIndividual(ICollection<TGene> genes);

    /// <summary>
    /// Converts an individual to a trained model that can make predictions.
    /// </summary>
    /// <param name="individual">The individual to convert.</param>
    /// <returns>A model capable of making predictions based on the individual's genes.</returns>
    IFullModel<T, TInput, TOutput> IndividualToModel(TIndividual individual);

    /// <summary>
    /// Gets statistics about the current evolutionary state, including generation number,
    /// population diversity, and fitness distribution.
    /// </summary>
    /// <returns>Statistics about the current evolutionary state.</returns>
    EvolutionStats<T, TInput, TOutput> GetEvolutionStats(IFitnessCalculator<T, TInput, TOutput> fitnessCalculator);

    /// <summary>
    /// Configures the genetic algorithm parameters.
    /// </summary>
    /// <param name="parameters">The genetic algorithm parameters to use.</param>
    void ConfigureGeneticParameters(GeneticParameters parameters);

    /// <summary>
    /// Gets the current genetic algorithm parameters.
    /// </summary>
    /// <returns>The current genetic algorithm parameters.</returns>
    GeneticParameters GetGeneticParameters();

    /// <summary>
    /// Adds a custom crossover operator.
    /// </summary>
    /// <param name="name">The name of the crossover operator.</param>
    /// <param name="crossoverOperator">The crossover function.</param>
    void AddCrossoverOperator(string name, Func<TIndividual, TIndividual, double, ICollection<TIndividual>> crossoverOperator);

    /// <summary>
    /// Adds a custom mutation operator.
    /// </summary>
    /// <param name="name">The name of the mutation operator.</param>
    /// <param name="mutationOperator">The mutation function.</param>
    void AddMutationOperator(string name, Func<TIndividual, double, TIndividual> mutationOperator);

    /// <summary>
    /// Saves the current population to a file.
    /// </summary>
    /// <param name="filePath">The path where the population should be saved.</param>
    void SavePopulation(string filePath);

    /// <summary>
    /// Loads a population from a file.
    /// </summary>
    /// <param name="filePath">The path from which to load the population.</param>
    /// <returns>The loaded population.</returns>
    ICollection<TIndividual> LoadPopulation(string filePath);
}




/// <summary>
/// Parameters for configuring a genetic algorithm.
/// </summary>
public class GeneticParameters
{
    /// <summary>
    /// Gets or sets the size of the population.
    /// </summary>
    public int PopulationSize { get; set; } = 100;

    /// <summary>
    /// Gets or sets the probability of crossover occurring.
    /// </summary>
    public double CrossoverRate { get; set; } = 0.8;

    /// <summary>
    /// Gets or sets the probability of mutation occurring.
    /// </summary>
    public double MutationRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the selection method to use.
    /// </summary>
    public SelectionMethod SelectionMethod { get; set; } = SelectionMethod.Tournament;

    /// <summary>
    /// Gets or sets the tournament size for tournament selection.
    /// </summary>
    public int TournamentSize { get; set; } = 3;

    /// <summary>
    /// Gets or sets the elitism rate (percentage of top individuals to preserve unchanged).
    /// </summary>
    public double ElitismRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the maximum number of generations to evolve.
    /// </summary>
    public int MaxGenerations { get; set; } = 100;

    /// <summary>
    /// Gets or sets the fitness threshold for termination.
    /// </summary>
    public double FitnessThreshold { get; set; } = double.MaxValue;

    /// <summary>
    /// Gets or sets the maximum time allowed for evolution.
    /// </summary>
    public TimeSpan MaxTime { get; set; } = TimeSpan.FromMinutes(10);

    /// <summary>
    /// Gets or sets the maximum number of generations without improvement before termination.
    /// </summary>
    public int MaxGenerationsWithoutImprovement { get; set; } = 20;

    /// <summary>
    /// Gets or sets the name of the crossover operator to use.
    /// </summary>
    public string CrossoverOperator { get; set; } = "SinglePoint";

    /// <summary>
    /// Gets or sets the name of the mutation operator to use.
    /// </summary>
    public string MutationOperator { get; set; } = "Uniform";

    /// <summary>
    /// Gets or sets whether to use parallel evaluation of fitness.
    /// </summary>
    public bool UseParallelEvaluation { get; set; } = true;

    /// <summary>
    /// Gets or sets the initialization method to use for creating the initial population.
    /// </summary>
    public InitializationMethod InitializationMethod { get; set; } = InitializationMethod.Random;
}
