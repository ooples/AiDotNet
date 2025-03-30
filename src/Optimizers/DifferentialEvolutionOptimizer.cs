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
public class DifferentialEvolutionOptimizer<T> : OptimizerBase<T>
{
    private DifferentialEvolutionOptions _deOptions;
    private Random _random;
    private T _currentCrossoverRate;
    private T _currentMutationRate;

    /// <summary>
    /// Initializes a new instance of the DifferentialEvolutionOptimizer class.
    /// </summary>
    /// <param name="options">The options for configuring the Differential Evolution algorithm.</param>
    /// <param name="predictionOptions">Options for prediction statistics.</param>
    /// <param name="modelOptions">Options for model statistics.</param>
    /// <param name="modelEvaluator">The model evaluator to use.</param>
    /// <param name="fitDetector">The fit detector to use.</param>
    /// <param name="fitnessCalculator">The fitness calculator to use.</param>
    /// <param name="modelCache">The model cache to use.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This constructor sets up the Differential Evolution optimizer with its initial configuration.
    /// You can customize various aspects of how it works, or use default settings.
    /// </para>
    /// </remarks>
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
    protected override void UpdateAdaptiveParameters(OptimizationStepData<T> currentStepData, OptimizationStepData<T> previousStepData)
    {
        base.UpdateAdaptiveParameters(currentStepData, previousStepData);
        AdaptiveParametersHelper<T>.UpdateAdaptiveGeneticParameters(ref _currentCrossoverRate, ref _currentMutationRate, currentStepData, previousStepData, _deOptions);
    }

    /// <summary>
    /// Initializes the adaptive parameters used in the Differential Evolution algorithm.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method sets up the initial crossover and mutation rates.
    /// These rates determine how the algorithm combines and changes solutions during optimization.
    /// </para>
    /// </remarks>
    private new void InitializeAdaptiveParameters()
    {
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

    /// <summary>
    /// Initializes the population for the Differential Evolution algorithm.
    /// </summary>
    /// <param name="dimensions">The number of dimensions in the problem space.</param>
    /// <param name="populationSize">The size of the population to initialize.</param>
    /// <returns>A list of randomly initialized symbolic models.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method creates the initial set of candidate solutions.
    /// Each solution is a random guess at what might be a good answer to the optimization problem.
    /// </para>
    /// </remarks>
    private List<ISymbolicModel<T>> InitializePopulation(int dimensions, int populationSize)
    {
        var population = new List<ISymbolicModel<T>>();
        for (int i = 0; i < populationSize; i++)
        {
            population.Add(SymbolicModelFactory<T>.CreateRandomModel(Options.UseExpressionTrees, dimensions));
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

    /// <summary>
    /// Retrieves the current options of the Differential Evolution optimizer.
    /// </summary>
    /// <returns>The current optimization algorithm options.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method allows you to check the current settings of the optimizer.
    /// It's useful if you need to inspect or copy the current configuration.
    /// </para>
    /// </remarks>
    public override OptimizationAlgorithmOptions GetOptions()
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

        // Serialize Random state
        writer.Write(_random.Next());

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
        _deOptions = JsonConvert.DeserializeObject<DifferentialEvolutionOptions>(optionsJson)
            ?? throw new InvalidOperationException("Failed to deserialize optimizer options.");

        // Deserialize Random state
        int randomSeed = reader.ReadInt32();
        _random = new Random(randomSeed);

        InitializeAdaptiveParameters();
    }
}