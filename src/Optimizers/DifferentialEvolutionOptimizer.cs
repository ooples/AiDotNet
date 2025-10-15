namespace AiDotNet.Optimizers;

/// <summary>
/// Implements the Differential Evolution optimization algorithm for numerical optimization problems.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Differential Evolution is a population-based optimization algorithm that is particularly well-suited
/// for solving non-linear, non-differentiable continuous space functions. It's known for its simplicity,
/// robustness, and effectiveness in various optimization scenarios.
/// </para>
/// <para><b>For Beginners:</b> This optimizer works by evolving a population of candidate solutions over time.
/// It's inspired by biological evolution and is good at finding global optima in complex problem spaces.
/// </para>
/// </remarks>
public class DifferentialEvolutionOptimizer<T, TInput, TOutput> : OptimizerBase<T, TInput, TOutput>
{
    /// <summary>
    /// Configuration options specific to the Differential Evolution algorithm.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the parameters that control the behavior of the Differential Evolution algorithm,
    /// such as population size, crossover rate, mutation rate, and other algorithm-specific settings.
    /// </para>
    /// <para><b>For Beginners:</b> This is like the rulebook for our evolutionary process.
    /// It defines how many candidate solutions we'll work with (population size), how often we'll
    /// combine solutions (crossover rate), and how much random variation we'll introduce (mutation rate).
    /// These settings control the balance between exploration (trying new areas) and exploitation
    /// (refining good solutions).
    /// </para>
    /// </remarks>
    private DifferentialEvolutionOptions<T, TInput, TOutput> _deOptions = default!;

    /// <summary>
    /// The current crossover rate used in the optimization process.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The crossover rate determines the probability of exchanging components between solutions
    /// during the creation of new trial solutions. This value may adapt during optimization.
    /// </para>
    /// <para><b>For Beginners:</b> This is like the probability of combining features from different
    /// solutions. A higher value means more mixing of solutions, which helps explore new combinations.
    /// This value can change during optimization to balance exploration and refinement.
    /// </para>
    /// </remarks>
    private T _currentCrossoverRate = default!;

    /// <summary>
    /// The current mutation rate used in the optimization process.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The mutation rate controls the magnitude of random changes applied to solutions
    /// during the differential mutation step. This value may adapt during optimization.
    /// </para>
    /// <para><b>For Beginners:</b> This is like the amount of random tweaking we apply to solutions.
    /// A higher value means bigger changes, which helps explore more diverse possibilities.
    /// This value can change during optimization to balance exploration and refinement.
    /// </para>
    /// </remarks>
    private T _currentMutationRate = default!;

    /// <summary>
    /// Initializes a new instance of the DifferentialEvolutionOptimizer class.
    /// </summary>
    /// <param name="model">The model to be optimized.</param>
    /// <param name="options">The options for configuring the Differential Evolution algorithm.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This constructor sets up the Differential Evolution optimizer with its initial configuration.
    /// You provide the model to optimize and can customize various aspects of how the optimizer works,
    /// or use default settings if you don't specify options.
    /// </para>
    /// </remarks>
    public DifferentialEvolutionOptimizer(
        IFullModel<T, TInput, TOutput> model,
        DifferentialEvolutionOptions<T, TInput, TOutput>? options = null)
        : base(model, options ?? new())
    {
        _deOptions = options ?? new DifferentialEvolutionOptions<T, TInput, TOutput>();
        _currentCrossoverRate = NumOps.Zero;
        _currentMutationRate = NumOps.Zero;

        InitializeAdaptiveParameters();
    }

    /// <summary>
    /// Updates the adaptive parameters based on the optimization progress.
    /// </summary>
    /// <param name="currentStepData">Data from the current optimization step.</param>
    /// <param name="previousStepData">Data from the previous optimization step.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method adjusts the crossover and mutation rates during the optimization process.
    /// It helps the algorithm adapt its behavior based on how well it's performing.
    /// </para>
    /// </remarks>
    protected override void UpdateAdaptiveParameters(OptimizationStepData<T, TInput, TOutput> currentStepData, OptimizationStepData<T, TInput, TOutput> previousStepData)
    {
        base.UpdateAdaptiveParameters(currentStepData, previousStepData);

        // Skip if previous step data is null (first iteration)
        if (previousStepData.Solution == null)
            return;

        AdaptiveParametersHelper<T, TInput, TOutput>.UpdateAdaptiveGeneticParameters(ref _currentCrossoverRate, ref _currentMutationRate, currentStepData, previousStepData, _deOptions);
    }

    /// <summary>
    /// Initializes the adaptive parameters used in the Differential Evolution algorithm.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method sets up the initial crossover and mutation rates.
    /// These rates determine how the algorithm combines and changes solutions during optimization.
    /// </para>
    /// </remarks>
    protected override void InitializeAdaptiveParameters()
    {
        base.InitializeAdaptiveParameters();
        _currentCrossoverRate = NumOps.FromDouble(_deOptions.CrossoverRate);
        _currentMutationRate = NumOps.FromDouble(_deOptions.MutationRate);
    }

    /// <summary>
    /// Performs the main optimization process using the Differential Evolution algorithm.
    /// </summary>
    /// <param name="inputData">The input data for the optimization process.</param>
    /// <returns>The result of the optimization process.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the heart of the Differential Evolution algorithm. It creates an initial population
    /// of solutions and then evolves them over multiple generations. In each generation, it creates new trial solutions,
    /// evaluates them, and keeps the best ones.
    /// </para>
    /// </remarks>
    public override OptimizationResult<T, TInput, TOutput> Optimize(OptimizationInputData<T, TInput, TOutput> inputData)
    {
        ValidationHelper<T>.ValidateInputData(inputData);

        int dimensions = InputHelper<T, TInput>.GetInputSize(inputData.XTrain);
        var population = InitializePopulation(inputData.XTrain, _deOptions.PopulationSize);

        var bestStepData = new OptimizationStepData<T, TInput, TOutput>
        {
            Solution = Model.DeepCopy(),
            FitnessScore = FitnessCalculator.IsHigherScoreBetter ? NumOps.MinValue : NumOps.MaxValue
        };
        var prevStepData = new OptimizationStepData<T, TInput, TOutput>();
        var currentStepData = new OptimizationStepData<T, TInput, TOutput>();

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

    /// <summary>
    /// Initializes the population for the Differential Evolution algorithm.
    /// </summary>
    /// <param name="input">The input training data used to determine problem dimensions.</param>
    /// <param name="populationSize">The size of the population to initialize.</param>
    /// <returns>A list of randomly initialized models.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method creates the initial set of candidate solutions.
    /// Each solution is a random guess at what might be a good answer to the optimization problem.
    /// </para>
    /// </remarks>
    private List<IFullModel<T, TInput, TOutput>> InitializePopulation(TInput input, int populationSize)
    {
        var population = new List<IFullModel<T, TInput, TOutput>>();

        // Always include the base model as the first member of the population
        population.Add(Model.DeepCopy());

        // Create the rest of the population with random variations
        for (int i = 1; i < populationSize; i++)
        {
            population.Add(CreateSolution(input));
        }

        return population;
    }

    /// <summary>
    /// Generates a trial model using the Differential Evolution algorithm's mutation and crossover operations.
    /// </summary>
    /// <param name="population">The current population of models.</param>
    /// <param name="currentIndex">The index of the current model in the population.</param>
    /// <param name="dimensions">The number of dimensions in the problem space.</param>
    /// <returns>A new trial model.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method creates a new candidate solution by combining and mutating
    /// existing solutions. It's how the algorithm explores new possibilities and improves over time.
    /// </para>
    /// </remarks>
    private IFullModel<T, TInput, TOutput> GenerateTrialModel(List<IFullModel<T, TInput, TOutput>> population, int currentIndex, int dimensions)
    {
        int a, b, c;
        do
        {
            a = Random.Next(population.Count);
            b = Random.Next(population.Count);
            c = Random.Next(population.Count);
        } while (a == currentIndex || b == currentIndex || c == currentIndex || a == b || a == c || b == c);

        var currentModel = population[currentIndex];

        // Get parameters from each model
        var aParams = population[a].GetParameters();
        var bParams = population[b].GetParameters();
        var cParams = population[c].GetParameters();
        var currentParams = currentModel.GetParameters();

        var trialParams = new Vector<T>(dimensions);
        int R = Random.Next(dimensions);
        var currentCrossOverRate = Convert.ToDouble(_currentCrossoverRate);
        var currentMutationRate = Convert.ToDouble(_currentMutationRate);

        for (int i = 0; i < dimensions; i++)
        {
            if (Random.NextDouble() < currentCrossOverRate || i == R)
            {
                trialParams[i] = NumOps.Add(aParams[i],
                    NumOps.Multiply(NumOps.FromDouble(currentMutationRate),
                        NumOps.Subtract(bParams[i], cParams[i])));
            }
            else
            {
                trialParams[i] = currentParams[i];
            }
        }

        // Create a new model with the modified parameters
        return currentModel.WithParameters(trialParams);
    }

    /// <summary>
    /// Updates the options for the Differential Evolution optimizer.
    /// </summary>
    /// <param name="options">The new options to be set.</param>
    /// <exception cref="ArgumentException">Thrown when the provided options are not of type DifferentialEvolutionOptions.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method allows you to change the settings of the optimizer during runtime.
    /// It ensures that only the correct type of options (specific to Differential Evolution) can be used.
    /// </para>
    /// </remarks>
    protected override void UpdateOptions(OptimizationAlgorithmOptions<T, TInput, TOutput> options)
    {
        if (options is DifferentialEvolutionOptions<T, TInput, TOutput> deOptions)
        {
            _deOptions = deOptions;
        }
        else
        {
            throw new ArgumentException("Options must be of type DifferentialEvolutionOptions", nameof(options));
        }
    }

    /// <summary>
    /// Retrieves the current options of the Differential Evolution optimizer.
    /// </summary>
    /// <returns>The current optimization algorithm options.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method allows you to check the current settings of the optimizer.
    /// It's useful if you need to inspect or copy the current configuration.
    /// </para>
    /// </remarks>
    public override OptimizationAlgorithmOptions<T, TInput, TOutput> GetOptions()
    {
        return _deOptions;
    }

    /// <summary>
    /// Serializes the Differential Evolution optimizer to a byte array.
    /// </summary>
    /// <returns>A byte array representing the serialized state of the optimizer.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method converts the current state of the optimizer into a series of bytes.
    /// This is useful for saving the optimizer's state to a file or sending it over a network. It allows you to
    /// recreate the exact state of the optimizer later.
    /// </para>
    /// <para>The serialization process includes:
    /// <list type="bullet">
    /// <item>Base class data (from the parent OptimizerBase class)</item>
    /// <item>The DifferentialEvolutionOptions</item>
    /// <item>The current state of the random number generator</item>
    /// </list>
    /// </para>
    /// </remarks>
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

        // Serialize current adaptive parameters
        writer.Write(Convert.ToDouble(_currentCrossoverRate));
        writer.Write(Convert.ToDouble(_currentMutationRate));

        return ms.ToArray();
    }

    /// <summary>
    /// Deserializes the Differential Evolution optimizer from a byte array.
    /// </summary>
    /// <param name="data">The byte array containing the serialized optimizer state.</param>
    /// <exception cref="InvalidOperationException">Thrown when deserialization of optimizer options fails.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method reconstructs the optimizer's state from a series of bytes.
    /// It's used to restore a previously saved state of the optimizer, allowing you to continue from where you left off.
    /// </para>
    /// <para>The deserialization process includes:
    /// <list type="bullet">
    /// <item>Restoring base class data (from the parent OptimizerBase class)</item>
    /// <item>Reconstructing the DifferentialEvolutionOptions</item>
    /// <item>Resetting the random number generator to its previous state</item>
    /// <item>Reinitializing adaptive parameters</item>
    /// </list>
    /// </para>
    /// </remarks>
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
        _deOptions = JsonConvert.DeserializeObject<DifferentialEvolutionOptions<T, TInput, TOutput>>(optionsJson)
            ?? throw new InvalidOperationException("Failed to deserialize optimizer options.");

        // Deserialize current adaptive parameters
        _currentCrossoverRate = NumOps.FromDouble(reader.ReadDouble());
        _currentMutationRate = NumOps.FromDouble(reader.ReadDouble());
    }
}