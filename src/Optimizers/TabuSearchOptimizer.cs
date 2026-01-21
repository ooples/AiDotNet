using Newtonsoft.Json;

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
    /// The genetic algorithm used to handle mutations and generate neighboring solutions.
    /// </summary>
    private GeneticBase<T, TInput, TOutput> _geneticAlgorithm;

    /// <summary>
    /// Initializes a new instance of the TabuSearchOptimizer class.
    /// </summary>
    /// <param name="model">The model to be optimized.</param>
    /// <param name="options">Options specific to the Tabu Search algorithm.</param>
    /// <param name="geneticAlgorithm">The genetic algorithm to use for mutations. If null, a StandardGeneticAlgorithm will be used.</param>
    /// <param name="engine">The computation engine (CPU or GPU) for vectorized operations.</param>
    public TabuSearchOptimizer(
        IFullModel<T, TInput, TOutput> model,
        TabuSearchOptions<T, TInput, TOutput>? options = null,
        GeneticBase<T, TInput, TOutput>? geneticAlgorithm = null,
        IFitnessCalculator<T, TInput, TOutput>? fitnessCalculator = null,
        IModelEvaluator<T, TInput, TOutput>? modelEvaluator = null,
        IEngine? engine = null)
        : base(model, options ?? new())
    {
        _tabuOptions = options ?? new TabuSearchOptions<T, TInput, TOutput>();

        // If no genetic algorithm is provided, create a default StandardGeneticAlgorithm
        if (geneticAlgorithm == null)
        {
            // Create a model factory that clones the provided model
            // This ensures we use the same model type as was passed to the optimizer
            IFullModel<T, TInput, TOutput> ModelFactory() => model.DeepCopy();

            _geneticAlgorithm = new StandardGeneticAlgorithm<T, TInput, TOutput>(
                ModelFactory,
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

        // Use a HashSet for faster tabu list lookups, with a custom comparer
        var tabuList = new HashSet<string>();

        for (int iteration = 0; iteration < Options.MaxIterations; iteration++)
        {
            var neighbors = GenerateNeighbors(currentSolution, inputData);

            // Find the best non-tabu neighbor
            IFullModel<T, TInput, TOutput>? bestNeighbor = null;
            OptimizationStepData<T, TInput, TOutput> bestNeighborStepData = new OptimizationStepData<T, TInput, TOutput>();

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
    /// This method adjusts the mutation rate, tabu list size, and neighborhood size based on the
    /// current iteration. It uses decay and increase factors to oscillate these parameters,
    /// promoting a balance between exploration and exploitation.
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
            g =>
            {
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
