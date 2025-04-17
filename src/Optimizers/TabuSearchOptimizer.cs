/// <summary>
/// Represents a Tabu Search optimizer for machine learning models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// The TabuSearchOptimizer implements the Tabu Search algorithm, a metaheuristic search method used in optimization.
/// It explores the solution space by iteratively moving from one solution to the best solution in its neighborhood,
/// while keeping a list of recently visited solutions (the tabu list) to avoid cycling and encourage exploration of new areas.
/// </para>
/// <para><b>For Beginners:</b> Think of Tabu Search as a smart explorer:
/// 
/// - The explorer (optimizer) looks for the best solution in a complex landscape
/// - It remembers recently visited places (tabu list) to avoid going in circles
/// - It adapts its search strategy over time to balance between exploring new areas and refining good solutions
/// 
/// This method is particularly effective for problems with many local optima.
/// </para>
/// </remarks>
public class TabuSearchOptimizer<T, TInput, TOutput> : OptimizerBase<T, TInput, TOutput>
{
    /// <summary>
    /// The options specific to the Tabu Search algorithm.
    /// </summary>
    private TabuSearchOptions<T, TInput, TOutput> _tabuOptions;

    /// <summary>
    /// The current mutation rate used in generating neighboring solutions.
    /// </summary>
    private double _currentMutationRate;

    /// <summary>
    /// The current size of the tabu list.
    /// </summary>
    private int _currentTabuListSize;

    /// <summary>
    /// The current size of the neighborhood to explore in each iteration.
    /// </summary>
    private int _currentNeighborhoodSize;

    /// <summary>
    /// Initializes a new instance of the TabuSearchOptimizer class.
    /// </summary>
    /// <param name="options">Options specific to the Tabu Search algorithm.</param>
    /// <param name="predictionOptions">Options for prediction statistics.</param>
    /// <param name="modelOptions">Options for model statistics.</param>
    /// <param name="modelEvaluator">The model evaluator to use.</param>
    /// <param name="fitDetector">The fit detector to use.</param>
    /// <param name="fitnessCalculator">The fitness calculator to use.</param>
    /// <param name="modelCache">The model cache to use.</param>
    public TabuSearchOptimizer(
        TabuSearchOptions<T, TInput, TOutput>? options = null)
        : base(options ?? new())
    {
        _tabuOptions = options ?? new TabuSearchOptions<T, TInput, TOutput>();

        InitializeAdaptiveParameters();
    }

    /// <summary>
    /// Initializes the adaptive parameters used in the Tabu Search algorithm.
    /// </summary>
    private new void InitializeAdaptiveParameters()
    {
        _currentMutationRate = _tabuOptions.InitialMutationRate;
        _currentTabuListSize = _tabuOptions.InitialTabuListSize;
        _currentNeighborhoodSize = _tabuOptions.InitialNeighborhoodSize;
    }

    /// <summary>
    /// Performs the optimization process to find the best solution for the given input data.
    /// </summary>
    /// <param name="inputData">The input data to optimize against.</param>
    /// <returns>An optimization result containing the best solution found and associated metrics.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the main Tabu Search algorithm. It iteratively generates and evaluates
    /// neighboring solutions, updates the best solution found, and adapts its search parameters.
    /// </para>
    /// <para><b>For Beginners:</b> This is the main journey of our explorer:
    /// 
    /// 1. Start at a random point in the landscape (initialize random solution)
    /// 2. For each step (iteration):
    ///    - Look at nearby places (generate neighbors)
    ///    - Choose the best place that hasn't been visited recently (best non-tabu neighbor)
    ///    - Move to that place (update current solution)
    ///    - Remember this place (update tabu list)
    ///    - Adjust the search strategy (update adaptive parameters)
    ///    - Check if this is the best place found so far (update best solution)
    ///    - Decide whether to stop early if no progress is being made
    /// 3. Return the best place found during the entire journey
    /// 
    /// This process helps find a good solution efficiently, even in complex landscapes.
    /// </para>
    /// </remarks>
    public override OptimizationResult<T, TInput, TOutput> Optimize(OptimizationInputData<T, TInput, TOutput> inputData)
    {
        var currentSolution = InitializeRandomSolution(inputData.XTrain);
        var bestStepData = new OptimizationStepData<T, TInput, TOutput>();
        var tabuList = new Queue<IFullModel<T, TInput, TOutput>>(_tabuOptions.TabuListSize);

        for (int iteration = 0; iteration < Options.MaxIterations; iteration++)
        {
            var neighbors = GenerateNeighbors(currentSolution);
            var bestNeighbor = neighbors
                .Where(n => !IsTabu(n, tabuList))
                .OrderByDescending(n => EvaluateSolution(n, inputData).FitnessScore)
                .FirstOrDefault() ?? neighbors.First();

            currentSolution = bestNeighbor;

            var currentStepData = EvaluateSolution(currentSolution, inputData);
            UpdateBestSolution(currentStepData, ref bestStepData);

            UpdateTabuList(tabuList, currentSolution);
            UpdateAdaptiveParameters(iteration);

            if (UpdateIterationHistoryAndCheckEarlyStopping(iteration, bestStepData))
            {
                break; // Early stopping criteria met, exit the loop
            }
        }

        return CreateOptimizationResult(bestStepData, inputData);
    }

    /// <summary>
    /// Updates the adaptive parameters used in the Tabu Search algorithm.
    /// </summary>
    /// <param name="iteration">The current iteration number.</param>
    /// <remarks>
    /// This method adjusts the mutation rate, tabu list size, and neighborhood size based on the
    /// current iteration. It uses decay and increase factors to oscillate these parameters,
    /// promoting a balance between exploration and exploitation.
    /// </remarks>
    private void UpdateAdaptiveParameters(int iteration)
    {
        // Update mutation rate
        _currentMutationRate *= (iteration % 2 == 0) ? _tabuOptions.MutationRateDecay : _tabuOptions.MutationRateIncrease;
        _currentMutationRate = MathHelper.Clamp(_currentMutationRate, _tabuOptions.MinMutationRate, _tabuOptions.MaxMutationRate);

        // Update tabu list size
        _currentTabuListSize = (int)(_currentTabuListSize * ((iteration % 2 == 0) ? _tabuOptions.TabuListSizeDecay : _tabuOptions.TabuListSizeIncrease));
        _currentTabuListSize = MathHelper.Clamp(_currentTabuListSize, _tabuOptions.MinTabuListSize, _tabuOptions.MaxTabuListSize);

        // Update neighborhood size
        _currentNeighborhoodSize = (int)(_currentNeighborhoodSize * ((iteration % 2 == 0) ? _tabuOptions.NeighborhoodSizeDecay : _tabuOptions.NeighborhoodSizeIncrease));
        _currentNeighborhoodSize = MathHelper.Clamp(_currentNeighborhoodSize, _tabuOptions.MinNeighborhoodSize, _tabuOptions.MaxNeighborhoodSize);
    }

    /// <summary>
    /// Generates a list of neighboring solutions from the current solution.
    /// </summary>
    /// <param name="currentSolution">The current solution to generate neighbors from.</param>
    /// <returns>A list of neighboring solutions.</returns>
    private List<IFullModel<T, TInput, TOutput>> GenerateNeighbors(IFullModel<T, TInput, TOutput> currentSolution)
    {
        var neighbors = new List<IFullModel<T, TInput, TOutput>>();
        for (int i = 0; i < _currentNeighborhoodSize; i++)
        {
            neighbors.Add(currentSolution.Mutate(_currentMutationRate));
        }

        return neighbors;
    }

    /// <summary>
    /// Checks if a given solution is in the tabu list.
    /// </summary>
    /// <param name="solution">The solution to check.</param>
    /// <param name="tabuList">The current tabu list.</param>
    /// <returns>True if the solution is in the tabu list, false otherwise.</returns>
    private bool IsTabu(IFullModel<T, TInput, TOutput> solution, Queue<IFullModel<T, TInput, TOutput>> tabuList)
    {
        return tabuList.Any(tabuSolution => tabuSolution.Equals(solution));
    }

    /// <summary>
    /// Updates the tabu list with a new solution.
    /// </summary>
    /// <param name="tabuList">The current tabu list to update.</param>
    /// <param name="solution">The solution to add to the tabu list.</param>
    private void UpdateTabuList(Queue<IFullModel<T, TInput, TOutput>> tabuList, IFullModel<T, TInput, TOutput> solution)
    {
        if (tabuList.Count >= _currentTabuListSize)
        {
            tabuList.Dequeue();
        }

        tabuList.Enqueue(solution);
    }

    /// <summary>
    /// Updates the options for the Tabu Search algorithm.
    /// </summary>
    /// <param name="options">The new options to set.</param>
    /// <exception cref="ArgumentException">Thrown when the provided options are not of type TabuSearchOptions.</exception>
    protected override void UpdateOptions(OptimizationAlgorithmOptions<T, TInput, TOutput> options)
    {
        if (options is TabuSearchOptions<T, TInput, TOutput> tabuOptions)
        {
            _tabuOptions = tabuOptions;
        }
        else
        {
            throw new ArgumentException("Invalid options type. Expected TabuSearchOptions.");
        }
    }

    /// <summary>
    /// Gets the current options for the Tabu Search algorithm.
    /// </summary>
    /// <returns>The current TabuSearchOptions.</returns>
    public override OptimizationAlgorithmOptions<T, TInput, TOutput> GetOptions()
    {
        return _tabuOptions;
    }

    /// <summary>
    /// Serializes the TabuSearchOptimizer to a byte array.
    /// </summary>
    /// <returns>A byte array representing the serialized optimizer.</returns>
    public override byte[] Serialize()
    {
        using MemoryStream ms = new MemoryStream();
        using BinaryWriter writer = new BinaryWriter(ms);

        // Serialize base class data
        byte[] baseData = base.Serialize();
        writer.Write(baseData.Length);
        writer.Write(baseData);

        // Serialize Tabu Search-specific options
        string optionsJson = JsonConvert.SerializeObject(_tabuOptions);
        writer.Write(optionsJson);

        return ms.ToArray();
    }

    /// <summary>
    /// Deserializes the TabuSearchOptimizer from a byte array.
    /// </summary>
    /// <param name="data">The byte array containing the serialized optimizer data.</param>
    /// <exception cref="InvalidOperationException">Thrown when deserialization of optimizer options fails.</exception>
    /// <remarks>
    /// <para>
    /// This method reconstructs the TabuSearchOptimizer from a serialized byte array. It performs the following steps:
    /// 1. Deserializes the base class data.
    /// 2. Deserializes the Tabu Search-specific options.
    /// 3. Reinitializes the adaptive parameters.
    /// </para>
    /// <para><b>For Beginners:</b> Think of this method as "unpacking" the optimizer's saved state:
    /// 
    /// - It's like opening a saved file in a game to continue where you left off.
    /// - The method reads the saved data and sets up the optimizer to match that saved state.
    /// - It ensures that all the special Tabu Search settings are correctly restored.
    /// - After unpacking, it prepares the optimizer for use by setting up its internal values.
    /// 
    /// This allows you to save the optimizer's state and later restore it exactly as it was.
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

        // Deserialize Tabu Search-specific options
        string optionsJson = reader.ReadString();
        _tabuOptions = JsonConvert.DeserializeObject<TabuSearchOptions<T, TInput, TOutput>>(optionsJson)
            ?? throw new InvalidOperationException("Failed to deserialize optimizer options.");

        // Initialize adaptive parameters after deserialization
        InitializeAdaptiveParameters();
    }
}