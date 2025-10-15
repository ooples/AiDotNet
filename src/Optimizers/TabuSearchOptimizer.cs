namespace AiDotNet.Optimizers;

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
    private TabuSearchOptions<T, TInput, TOutput> _tabuOptions = default!;

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
    /// The genetic algorithm used to handle mutations and generate neighboring solutions.
    /// </summary>
    private GeneticBase<T, TInput, TOutput> _geneticAlgorithm = default!;

    /// <summary>
    /// Initializes a new instance of the TabuSearchOptimizer class.
    /// </summary>
    /// <param name="model">The model to be optimized.</param>
    /// <param name="options">Options specific to the Tabu Search algorithm.</param>
    /// <param name="geneticAlgorithm">The genetic algorithm to use for mutations. If null, a StandardGeneticAlgorithm will be used.</param>
    /// <param name="fitnessCalculator">The fitness calculator to use for evaluating solutions.</param>
    /// <param name="modelEvaluator">The model evaluator to use for evaluating solutions.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new Tabu Search optimizer with the specified model, options, and components.
    /// If no options or components are provided, default implementations are used.
    /// </para>
    /// <para><b>For Beginners:</b> This is the starting point for creating a new optimizer.
    /// 
    /// Think of it like preparing for an exploration:
    /// - You provide the model that needs to be optimized (like the terrain to explore)
    /// - You can provide custom settings (options) for how to explore
    /// - You can provide specialized tools (genetic algorithm, evaluators) or use the basic ones
    /// 
    /// This setup ensures the optimizer is ready to start searching for the best solution.
    /// </para>
    /// </remarks>
    public TabuSearchOptimizer(
        IFullModel<T, TInput, TOutput> model,
        TabuSearchOptions<T, TInput, TOutput>? options = null,
        GeneticBase<T, TInput, TOutput>? geneticAlgorithm = null,
        IFitnessCalculator<T, TInput, TOutput>? fitnessCalculator = null,
        IModelEvaluator<T, TInput, TOutput>? modelEvaluator = null)
        : base(model, options ?? new())
    {
        _tabuOptions = options ?? new TabuSearchOptions<T, TInput, TOutput>();

        // If no genetic algorithm is provided, create a default StandardGeneticAlgorithm
        if (geneticAlgorithm == null)
        {
            // Need to provide a model factory - use a simple model as default
            static IFullModel<T, TInput, TOutput> modelFactory()
            {
                return (IFullModel<T, TInput, TOutput>)new SimpleRegression<T>();
            }

            _geneticAlgorithm = new StandardGeneticAlgorithm<T, TInput, TOutput>(
                modelFactory,
                fitnessCalculator ?? new MeanSquaredErrorFitnessCalculator<T, TInput, TOutput>(),
                modelEvaluator ?? new DefaultModelEvaluator<T, TInput, TOutput>());
        }
        else
        {
            _geneticAlgorithm = geneticAlgorithm;
        }

        InitializeAdaptiveParameters();
    }

    /// <summary>
    /// Initializes the adaptive parameters used in the Tabu Search algorithm.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method sets up the initial values for the adaptive parameters used in the Tabu Search algorithm.
    /// These parameters include the mutation rate, tabu list size, and neighborhood size.
    /// </para>
    /// <para><b>For Beginners:</b> This is like setting up the explorer's initial strategy:
    /// 
    /// - It sets how much variation to introduce when looking for nearby places (mutation rate)
    /// - It sets how many places to remember as already visited (tabu list size)
    /// - It sets how many nearby places to check in each round (neighborhood size)
    /// 
    /// These initial settings help the algorithm start with a balanced exploration strategy.
    /// </para>
    /// </remarks>
    protected override void InitializeAdaptiveParameters()
    {
        base.InitializeAdaptiveParameters();
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
    /// 1. Start with the provided model
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
        ValidationHelper<T>.ValidateInputData(inputData);

        var currentSolution = Model.DeepCopy();
        var bestStepData = new OptimizationStepData<T, TInput, TOutput>
        {
            Solution = currentSolution,
            FitnessScore = FitnessCalculator.IsHigherScoreBetter ? NumOps.MinValue : NumOps.MaxValue
        };

        // Evaluate the initial solution
        var initialStepData = EvaluateSolution(currentSolution, inputData);
        UpdateBestSolution(initialStepData, ref bestStepData);

        // Use a HashSet for faster tabu list lookups, with a custom comparer
        var tabuList = new HashSet<string>();

        for (int iteration = 0; iteration < Options.MaxIterations; iteration++)
        {
            var neighbors = GenerateNeighbors(currentSolution, inputData);

            // Find the best non-tabu neighbor
            IFullModel<T, TInput, TOutput>? bestNeighbor = null;
            OptimizationStepData<T, TInput, TOutput> bestNeighborStepData = new OptimizationStepData<T, TInput, TOutput>
            {
                FitnessScore = FitnessCalculator.IsHigherScoreBetter ? NumOps.MinValue : NumOps.MaxValue
            };

            foreach (var neighbor in neighbors)
            {
                // Create a hash representation of this solution
                string solutionHash = GetSolutionHash(neighbor);

                if (!tabuList.Contains(solutionHash))
                {
                    var neighborStepData = EvaluateSolution(neighbor, inputData);

                    if (bestNeighbor == null ||
                        FitnessCalculator.IsBetterFitness(neighborStepData.FitnessScore, bestNeighborStepData.FitnessScore))
                    {
                        bestNeighbor = neighbor;
                        bestNeighborStepData = neighborStepData;
                    }
                }
            }

            // If all neighbors are tabu, pick the best one anyway (aspiration criteria)
            if (bestNeighbor == null && neighbors.Count > 0)
            {
                foreach (var neighbor in neighbors)
                {
                    var neighborStepData = EvaluateSolution(neighbor, inputData);

                    if (bestNeighbor == null ||
                        FitnessCalculator.IsBetterFitness(neighborStepData.FitnessScore, bestNeighborStepData.FitnessScore))
                    {
                        bestNeighbor = neighbor;
                        bestNeighborStepData = neighborStepData;
                    }
                }
            }

            // If we found a neighbor, update current solution
            if (bestNeighbor != null)
            {
                currentSolution = bestNeighbor;

                // Add current solution to tabu list
                string solutionHash = GetSolutionHash(currentSolution);
                UpdateTabuList(tabuList, solutionHash);

                // Update best solution if needed
                UpdateBestSolution(bestNeighborStepData, ref bestStepData);
            }

            UpdateAdaptiveParameters(iteration);

            if (UpdateIterationHistoryAndCheckEarlyStopping(iteration, bestStepData))
            {
                break; // Early stopping criteria met, exit the loop
            }
        }

        return CreateOptimizationResult(bestStepData, inputData);
    }

    /// <summary>
    /// Creates a hash representation of a solution for the tabu list.
    /// </summary>
    /// <param name="solution">The solution to hash.</param>
    /// <returns>A string hash representation of the solution.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a unique identifier for a solution by hashing its parameters.
    /// This hash is used to efficiently check if a solution has been visited before.
    /// </para>
    /// <para><b>For Beginners:</b> This is like creating a unique fingerprint for each location:
    /// 
    /// - It takes the solution's parameters and creates a compact representation
    /// - This allows the explorer to quickly check if a place has been visited before
    /// - It's more efficient than comparing all the details of two solutions
    /// 
    /// This fingerprinting mechanism is essential for maintaining the tabu list efficiently.
    /// </para>
    /// </remarks>
    private string GetSolutionHash(IFullModel<T, TInput, TOutput> solution)
    {
        var parameters = solution.GetParameters();

        // Create a MD5 hash of the parameters
        using System.Security.Cryptography.MD5 md5 = System.Security.Cryptography.MD5.Create();

        byte[] data = new byte[parameters.Length * sizeof(double)];

        for (int i = 0; i < parameters.Length; i++)
        {
            double value = Convert.ToDouble(parameters[i]);
            byte[] bytes = BitConverter.GetBytes(value);
            Array.Copy(bytes, 0, data, i * sizeof(double), sizeof(double));
        }

        byte[] hashBytes = md5.ComputeHash(data);
        return Convert.ToBase64String(hashBytes);
    }

    /// <summary>
    /// Updates the adaptive parameters used in the Tabu Search algorithm.
    /// </summary>
    /// <param name="iteration">The current iteration number.</param>
    /// <remarks>
    /// <para>
    /// This method adjusts the mutation rate, tabu list size, and neighborhood size based on the
    /// current iteration. It uses decay and increase factors to oscillate these parameters,
    /// promoting a balance between exploration and exploitation.
    /// </para>
    /// <para><b>For Beginners:</b> This is like the explorer adjusting their strategy:
    /// 
    /// - It alternates between focusing on nearby areas and exploring more widely
    /// - It adjusts how many past locations to remember (tabu list size)
    /// - It changes how many nearby places to check in each round (neighborhood size)
    /// - It ensures all parameters stay within reasonable limits
    /// 
    /// This adaptive behavior helps balance exploring new areas and refining searches in promising regions.
    /// </para>
    /// </remarks>
    private void UpdateAdaptiveParameters(int iteration)
    {
        // Update mutation rate
        _currentMutationRate *= iteration % 2 == 0 ? _tabuOptions.MutationRateDecay : _tabuOptions.MutationRateIncrease;
        _currentMutationRate = MathHelper.Clamp(_currentMutationRate, _tabuOptions.MinMutationRate, _tabuOptions.MaxMutationRate);

        // Update tabu list size
        _currentTabuListSize = (int)(_currentTabuListSize * (iteration % 2 == 0 ? _tabuOptions.TabuListSizeDecay : _tabuOptions.TabuListSizeIncrease));
        _currentTabuListSize = MathHelper.Clamp(_currentTabuListSize, _tabuOptions.MinTabuListSize, _tabuOptions.MaxTabuListSize);

        // Update neighborhood size
        _currentNeighborhoodSize = (int)(_currentNeighborhoodSize * (iteration % 2 == 0 ? _tabuOptions.NeighborhoodSizeDecay : _tabuOptions.NeighborhoodSizeIncrease));
        _currentNeighborhoodSize = MathHelper.Clamp(_currentNeighborhoodSize, _tabuOptions.MinNeighborhoodSize, _tabuOptions.MaxNeighborhoodSize);
    }

    /// <summary>
    /// Generates a list of neighboring solutions from the current solution.
    /// </summary>
    /// <param name="currentSolution">The current solution to generate neighbors from.</param>
    /// <param name="inputData">The input data used for evaluating solutions.</param>
    /// <returns>A list of neighboring solutions.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a set of solutions that are similar but slightly different from the current solution.
    /// It uses a genetic algorithm's mutation operator to introduce controlled randomness.
    /// </para>
    /// <para><b>For Beginners:</b> This is like the explorer looking at nearby areas:
    /// 
    /// - It takes the current location and finds places that are slightly different
    /// - It uses the genetic algorithm to introduce random variations
    /// - The amount of variation is controlled by the mutation rate
    /// - It generates multiple nearby places to consider
    /// 
    /// This process helps the algorithm explore the solution space efficiently by focusing on
    /// areas close to the current solution.
    /// </para>
    /// </remarks>
    private List<IFullModel<T, TInput, TOutput>> GenerateNeighbors(
        IFullModel<T, TInput, TOutput> currentSolution,
        OptimizationInputData<T, TInput, TOutput> inputData)
    {
        var neighbors = new List<IFullModel<T, TInput, TOutput>>();

        // Convert the current solution to a ModelIndividual
        var parameters = currentSolution.GetParameters();
        var genes = new List<ModelParameterGene<T>>();

        for (int i = 0; i < parameters.Length; i++)
        {
            genes.Add(new ModelParameterGene<T>(i, parameters[i]));
        }

        // Create an individual from the current solution
        Func<ICollection<ModelParameterGene<T>>, IFullModel<T, TInput, TOutput>> modelFactory =
            g => {
                var model = currentSolution.DeepCopy();

                var newParams = new Vector<T>(g.Count);
                foreach (var gene in g.OrderBy(gene => gene.Index))
                {
                    newParams[gene.Index] = gene.Value;
                }

                return model.WithParameters(newParams);
            };

        var individual = new ModelIndividual<T, TInput, TOutput, ModelParameterGene<T>>(
            genes, modelFactory);

        // Configure genetic algorithm parameters for generating neighbors
        var geneticParams = _geneticAlgorithm.GetGeneticParameters();
        geneticParams.MutationRate = _currentMutationRate;
        _geneticAlgorithm.ConfigureGeneticParameters(geneticParams);

        // Generate neighbors using the genetic algorithm's mutation operator
        for (int i = 0; i < _currentNeighborhoodSize; i++)
        {
            var mutated = _geneticAlgorithm.Mutate(individual, _currentMutationRate);
            var model = _geneticAlgorithm.IndividualToModel(mutated);
            neighbors.Add(model);
        }

        return neighbors;
    }

    /// <summary>
    /// Updates the tabu list with a new solution hash.
    /// </summary>
    /// <param name="tabuList">The current tabu list to update.</param>
    /// <param name="solutionHash">The solution hash to add to the tabu list.</param>
    /// <remarks>
    /// <para>
    /// This method adds a new solution hash to the tabu list and removes the oldest entry if the list is at capacity.
    /// </para>
    /// <para><b>For Beginners:</b> This is like the explorer updating their memory of visited places:
    /// 
    /// - It adds the current location to the list of places to avoid revisiting
    /// - If the memory is getting too full, it removes the oldest memory
    /// - This ensures the explorer doesn't waste time revisiting recent places
    /// 
    /// This "tabu" mechanism is what gives the Tabu Search algorithm its name and is crucial for
    /// preventing the search from cycling between the same solutions.
    /// </para>
    /// </remarks>
    private void UpdateTabuList(HashSet<string> tabuList, string solutionHash)
    {
        // If the tabu list is at capacity, remove a random item
        if (tabuList.Count >= _currentTabuListSize)
        {
            tabuList.Remove(tabuList.First());
        }

        tabuList.Add(solutionHash);
    }

    /// <summary>
    /// Updates the options for the Tabu Search algorithm.
    /// </summary>
    /// <param name="options">The new options to set.</param>
    /// <exception cref="ArgumentException">Thrown when the provided options are not of type TabuSearchOptions.</exception>
    /// <remarks>
    /// <para>
    /// This method validates and applies new optimization options to the Tabu Search algorithm.
    /// </para>
    /// <para><b>For Beginners:</b> This is like giving the explorer new instructions:
    /// 
    /// - It checks that the instructions are the right type for a Tabu Search expedition
    /// - If they are, it updates the explorer's guidelines
    /// - If not, it reports an error
    /// 
    /// This ensures that only appropriate settings are used with this specific optimizer.
    /// </para>
    /// </remarks>
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
    /// <remarks>
    /// <para>
    /// This method returns the current configuration options of the Tabu Search optimizer.
    /// </para>
    /// <para><b>For Beginners:</b> This is like asking to see the explorer's current instructions:
    /// 
    /// - It returns the complete set of settings that control how the Tabu Search works
    /// - This includes parameters for mutation rate, tabu list size, and neighborhood size
    /// 
    /// This is useful for understanding or checking the current configuration of the optimizer.
    /// </para>
    /// </remarks>
    public override OptimizationAlgorithmOptions<T, TInput, TOutput> GetOptions()
    {
        return _tabuOptions;
    }

    /// <summary>
    /// Serializes the TabuSearchOptimizer to a byte array.
    /// </summary>
    /// <returns>A byte array representing the serialized optimizer.</returns>
    /// <remarks>
    /// <para>
    /// This method saves the current state of the optimizer, including its base class data,
    /// Tabu Search-specific options, and genetic algorithm, into a byte array.
    /// </para>
    /// <para><b>For Beginners:</b> This is like taking a snapshot of the explorer's expedition:
    /// 
    /// - It saves all the current settings and progress
    /// - This saved data can be used later to continue from where you left off
    /// - It includes both general optimization info and Tabu Search-specific details
    /// 
    /// This is useful for saving progress or sharing the optimizer's current state.
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

        // Serialize Tabu Search-specific options
        string optionsJson = JsonConvert.SerializeObject(_tabuOptions);
        writer.Write(optionsJson);

        // Serialize the genetic algorithm
        byte[] geneticAlgorithmData = _geneticAlgorithm.Serialize();
        writer.Write(geneticAlgorithmData.Length);
        writer.Write(geneticAlgorithmData);

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
    /// 3. Deserializes the genetic algorithm if available.
    /// 4. Reinitializes the adaptive parameters.
    /// </para>
    /// <para><b>For Beginners:</b> Think of this method as "unpacking" the optimizer's saved state:
    /// 
    /// - It's like opening a saved file in a game to continue where you left off
    /// - The method reads the saved data and sets up the optimizer to match that saved state
    /// - It ensures that all the special Tabu Search settings are correctly restored
    /// - After unpacking, it prepares the optimizer for use by setting up its internal values
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

        // Deserialize the genetic algorithm if available
        if (reader.BaseStream.Position < reader.BaseStream.Length)
        {
            int geneticAlgorithmDataLength = reader.ReadInt32();
            byte[] geneticAlgorithmData = reader.ReadBytes(geneticAlgorithmDataLength);
            _geneticAlgorithm.Deserialize(geneticAlgorithmData);
        }

        // Initialize adaptive parameters after deserialization
        InitializeAdaptiveParameters();
    }
}