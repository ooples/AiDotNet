namespace AiDotNet.Optimizers;

/// <summary>
/// Represents a Genetic Algorithm optimizer for machine learning models.
/// </summary>
/// <remarks>
/// <para>
/// The Genetic Algorithm optimizer is an evolutionary optimization technique inspired by the process of natural selection.
/// It evolves a population of potential solutions over multiple generations to find an optimal or near-optimal solution.
/// </para>
/// <para><b>For Beginners:</b> Think of the Genetic Algorithm optimizer like breeding the best solutions:
/// 
/// - Start with a group of random solutions (like a group of different recipes)
/// - Test how good each solution is (like tasting each recipe)
/// - Choose the best solutions (like picking the tastiest recipes)
/// - Create new solutions by mixing the best ones (like combining ingredients from the best recipes)
/// - Sometimes make small random changes (like accidentally adding a new spice)
/// - Repeat this process many times to find the best solution (or the tastiest recipe!)
/// 
/// This approach is good at finding solutions for complex problems where traditional methods might struggle.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class GeneticAlgorithmOptimizer<T, TInput, TOutput> : OptimizerBase<T, TInput, TOutput>
{
    /// <summary>
    /// The options specific to the Genetic Algorithm.
    /// </summary>
    private GeneticAlgorithmOptimizerOptions<T, TInput, TOutput> _geneticOptions;

    /// <summary>
    /// The current crossover rate, which determines how often solutions are combined.
    /// </summary>
    private T _currentCrossoverRate;

    /// <summary>
    /// The current mutation rate, which determines how often random changes are made to solutions.
    /// </summary>
    private T _currentMutationRate;

    /// <summary>
    /// Initializes a new instance of the GeneticAlgorithmOptimizer class.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This sets up the genetic algorithm with its initial settings.
    /// You can customize various aspects of how it works, or use default settings if you're unsure.
    /// </para>
    /// </remarks>
    /// <param name="options">The options for configuring the genetic algorithm.</param>
    public GeneticAlgorithmOptimizer(
        GeneticAlgorithmOptimizerOptions<T, TInput, TOutput>? options = null)
        : base(options ?? new())
    {
        _geneticOptions = options ?? new GeneticAlgorithmOptimizerOptions<T, TInput, TOutput>();
        _currentCrossoverRate = NumOps.Zero;
        _currentMutationRate = NumOps.Zero;

        InitializeAdaptiveParameters();
    }

    /// <summary>
    /// Updates the adaptive parameters used in the genetic algorithm.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method adjusts how the algorithm behaves based on its recent performance.
    /// It's like a chef adjusting their cooking technique based on how the last few dishes turned out.
    /// </para>
    /// </remarks>
    /// <param name="currentStepData">Data from the current optimization step.</param>
    /// <param name="previousStepData">Data from the previous optimization step.</param>
    protected override void UpdateAdaptiveParameters(OptimizationStepData<T, TInput, TOutput> currentStepData, OptimizationStepData<T, TInput, TOutput> previousStepData)
    {
        base.UpdateAdaptiveParameters(currentStepData, previousStepData);
        AdaptiveParametersHelper<T, TInput, TOutput>.UpdateAdaptiveGeneticParameters(ref _currentCrossoverRate, ref _currentMutationRate, 
            currentStepData, previousStepData, _geneticOptions);
    }

    /// <summary>
    /// Initializes the adaptive parameters used in the genetic algorithm.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method sets up the initial rates for crossover (mixing solutions)
    /// and mutation (making small random changes). It's like setting the initial recipe and how often
    /// you'll try new ingredients.
    /// </para>
    /// </remarks>
    private new void InitializeAdaptiveParameters()
    {
        _currentCrossoverRate = NumOps.FromDouble(_geneticOptions.CrossoverRate);
        _currentMutationRate = NumOps.FromDouble(_geneticOptions.MutationRate);
    }

    /// <summary>
    /// Performs the main optimization process using the genetic algorithm.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the heart of the genetic algorithm. It:
    /// 1. Creates an initial group of random solutions
    /// 2. Evaluates how good each solution is
    /// 3. Selects the best solutions
    /// 4. Creates new solutions by mixing the best ones
    /// 5. Sometimes makes small random changes to solutions
    /// 6. Repeats this process for many generations
    /// 
    /// It's like running a cooking competition where each round you keep the best recipes,
    /// combine them to make new recipes, and occasionally add a surprise ingredient.
    /// </para>
    /// </remarks>
    /// <param name="inputData">The input data for the optimization process.</param>
    /// <returns>The result of the optimization process.</returns>
    public override OptimizationResult<T, TInput, TOutput> Optimize(OptimizationInputData<T, TInput, TOutput> inputData)
    {
        int dimensions = InputHelper<T, TInput>.GetInputSize(inputData.XTrain);
        var population = InitializePopulation(dimensions, _geneticOptions.PopulationSize);
        var bestStepData = new OptimizationStepData<T, TInput, TOutput>();
        var prevStepData = new OptimizationStepData<T, TInput, TOutput>();
        var currentStepData = new OptimizationStepData<T, TInput, TOutput>();

        for (int generation = 0; generation < Options.MaxIterations; generation++)
        {
            var populationStepData = new List<OptimizationStepData<T, TInput, TOutput>>();

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

    /// <summary>
    /// Initializes the population for the genetic algorithm.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method creates the initial group of random solutions.
    /// It's like coming up with a bunch of different recipes to start your cooking competition.
    /// </para>
    /// </remarks>
    /// <param name="dimensions">The number of dimensions (features) in the problem.</param>
    /// <param name="populationSize">The number of solutions to create.</param>
    /// <returns>A list of random solutions.</returns>
    private List<IFullModel<T, TInput, TOutput>> InitializePopulation(int dimensions, int populationSize)
    {
        var population = new List<IFullModel<T, TInput, TOutput>>();
        for (int i = 0; i < populationSize; i++)
        {
            population.Add(SymbolicModelFactory<T, TInput, TOutput>.CreateRandomModel(Options.UseExpressionTrees, dimensions));
        }

        return population;
    }

    /// <summary>
    /// Performs selection of the best solutions from the current population.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method chooses the best solutions to be used for creating
    /// the next generation. It's like picking the best recipes from your cooking competition to
    /// use as inspiration for the next round.
    /// </para>
    /// </remarks>
    /// <param name="populationStepData">Data about each solution in the current population.</param>
    /// <returns>A list of selected solutions.</returns>
    private List<IFullModel<T, TInput, TOutput>> PerformSelection(List<OptimizationStepData<T, TInput, TOutput>> populationStepData)
    {
        return TournamentSelection(populationStepData, _geneticOptions.PopulationSize);
    }

    /// <summary>
    /// Performs tournament selection to choose the best solutions.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method randomly picks pairs of solutions and chooses the better one
    /// from each pair. It's like having many small cooking contests and picking the winner from each.
    /// </para>
    /// </remarks>
    /// <param name="populationStepData">Data about each solution in the current population.</param>
    /// <param name="selectionSize">The number of solutions to select.</param>
    /// <returns>A list of selected solutions.</returns>
    private List<IFullModel<T, TInput, TOutput>> TournamentSelection(List<OptimizationStepData<T, TInput, TOutput>> populationStepData, int selectionSize)
    {
        var selected = new List<IFullModel<T, TInput, TOutput>>();
        for (int i = 0; i < selectionSize; i++)
        {
            int index1 = Random.Next(populationStepData.Count);
            int index2 = Random.Next(populationStepData.Count);
            selected.Add(_fitnessCalculator.IsBetterFitness(populationStepData[index1].FitnessScore, populationStepData[index2].FitnessScore)
                ? populationStepData[index1].Solution
                : populationStepData[index2].Solution);
        }

        return selected;
    }

    /// <summary>
    /// Performs crossover to create new solutions from the selected solutions.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method creates new solutions by combining parts of existing solutions.
    /// It's like creating new recipes by mixing ingredients or techniques from your best existing recipes.
    /// </para>
    /// </remarks>
    /// <param name="population">The current population of solutions.</param>
    /// <returns>A new population after crossover.</returns>
    private List<IFullModel<T, TInput, TOutput>> PerformCrossover(List<IFullModel<T, TInput, TOutput>> population)
    {
        var newPopulation = new List<IFullModel<T, TInput, TOutput>>();
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

    /// <summary>
    /// Performs crossover between two parent solutions to create two child solutions.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method combines two parent solutions to create two new solutions.
    /// It's like mixing ingredients from two different recipes to create two new recipes.
    /// The crossover rate determines how often this mixing occurs.
    /// </para>
    /// </remarks>
    /// <param name="parent1">The first parent solution.</param>
    /// <param name="parent2">The second parent solution.</param>
    /// <returns>A tuple containing two new child solutions.</returns>
    private (IFullModel<T, TInput, TOutput>, IFullModel<T, TInput, TOutput>) Crossover(IFullModel<T, TInput, TOutput> parent1, IFullModel<T, TInput, TOutput> parent2)
    {
        if (NumOps.LessThan(NumOps.FromDouble(Random.NextDouble()), _currentCrossoverRate))
        {
            return SymbolicModelFactory<T>.Crossover(parent1, parent2, Convert.ToDouble(_currentCrossoverRate));
        }
        else
        {
            return (parent1, parent2);
        }
    }

    /// <summary>
    /// Performs mutation on the entire population.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method applies small random changes to each solution in the population.
    /// It's like occasionally adding a surprise ingredient to each recipe to see if it improves the taste.
    /// </para>
    /// </remarks>
    /// <param name="population">The current population of solutions.</param>
    /// <returns>A new population after mutation.</returns>
    private List<IFullModel<T, TInput, TOutput>> PerformMutation(List<IFullModel<T, TInput, TOutput>> population)
    {
        return [.. population.Select(Mutate)];
    }

    /// <summary>
    /// Performs mutation on a single individual solution.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method makes a small random change to a single solution.
    /// It's like slightly adjusting the amount of an ingredient in a recipe or trying a new cooking technique.
    /// The mutation rate determines how often these changes occur.
    /// </para>
    /// </remarks>
    /// <param name="individual">The individual solution to potentially mutate.</param>
    /// <returns>The mutated solution if mutation occurred, otherwise the original solution.</returns>
    private IFullModel<T, TInput, TOutput> Mutate(IFullModel<T, TInput, TOutput> individual)
    {
        if (NumOps.LessThan(NumOps.FromDouble(Random.NextDouble()), _currentMutationRate))
        {
            return SymbolicModelFactory<T>.Mutate(individual, Convert.ToDouble(_currentMutationRate));
        }

        return individual;
    }

    /// <summary>
    /// Updates the options for the genetic algorithm optimizer.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method allows you to change the settings of the genetic algorithm
    /// while it's running. It's like adjusting the rules of your cooking competition mid-way through.
    /// </para>
    /// </remarks>
    /// <param name="options">The new options to apply to the optimizer.</param>
    /// <exception cref="ArgumentException">Thrown when the provided options are not of the correct type.</exception>
    protected override void UpdateOptions(OptimizationAlgorithmOptions<T, TInput, TOutput> options)
    {
        if (options is GeneticAlgorithmOptimizerOptions<T, TInput, TOutput> geneticOptions)
        {
            _geneticOptions = geneticOptions;
        }
        else
        {
            throw new ArgumentException("Invalid options type. Expected GeneticAlgorithmOptimizerOptions.");
        }
    }

    /// <summary>
    /// Gets the current options for the genetic algorithm optimizer.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method returns the current settings of the genetic algorithm.
    /// It's like checking the current rules of your cooking competition.
    /// </para>
    /// </remarks>
    /// <returns>The current genetic algorithm optimizer options.</returns>
    public override OptimizationAlgorithmOptions<T, TInput, TOutput> GetOptions()
    {
        return _geneticOptions;
    }

    /// <summary>
    /// Serializes the genetic algorithm optimizer to a byte array.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method saves all the important information about the current state
    /// of the genetic algorithm into a format that can be easily stored or transmitted.
    /// It's like writing down all the details of your cooking competition so you can recreate it later.
    /// </para>
    /// </remarks>
    /// <returns>A byte array containing the serialized data of the optimizer.</returns>
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

    /// <summary>
    /// Deserializes the genetic algorithm optimizer from a byte array.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method recreates the genetic algorithm optimizer from previously saved data.
    /// It's like using your written notes to set up your cooking competition exactly as it was before.
    /// </para>
    /// </remarks>
    /// <param name="data">The byte array containing the serialized data of the optimizer.</param>
    /// <exception cref="InvalidOperationException">Thrown when deserialization of the optimizer options fails.</exception>
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
        _geneticOptions = JsonConvert.DeserializeObject<GeneticAlgorithmOptimizerOptions<T, TInput, TOutput>>(optionsJson)
            ?? throw new InvalidOperationException("Failed to deserialize optimizer options.");

        InitializeAdaptiveParameters();
    }
}