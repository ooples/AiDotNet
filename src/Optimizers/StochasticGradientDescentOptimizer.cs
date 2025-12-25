using Newtonsoft.Json;

namespace AiDotNet.Optimizers;

/// <summary>
/// Represents a Stochastic Gradient Descent (SGD) optimizer for machine learning models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// The StochasticGradientDescentOptimizer is a gradient-based optimization algorithm that iteratively
/// adjusts model parameters to minimize the loss function. It uses a stochastic approach, updating
/// parameters based on a subset of the training data in each iteration.
/// </para>
/// <para><b>For Beginners:</b> Think of this optimizer as a hiker trying to find the lowest point in a hilly landscape:
/// 
/// - The hiker (optimizer) takes steps downhill to find the lowest point (best model parameters)
/// - Instead of looking at the entire landscape at once, the hiker looks at small patches (subsets of data)
/// - The hiker adjusts their step size (learning rate) as they go
/// - This approach helps the hiker find a good low point quickly, even in a complex landscape
/// 
/// This method is efficient for large datasets and can often find good solutions quickly.
/// </para>
/// </remarks>
public class StochasticGradientDescentOptimizer<T, TInput, TOutput> : GradientBasedOptimizerBase<T, TInput, TOutput>
{
    private StochasticGradientDescentOptimizerOptions<T, TInput, TOutput> _options;

    /// <summary>
    /// Initializes a new instance of the StochasticGradientDescentOptimizer class.
    /// </summary>
    /// <param name="options">Options specific to the SGD optimizer.</param>
    /// <param name="predictionOptions">Options for prediction statistics.</param>
    /// <param name="modelOptions">Options for model statistics.</param>
    /// <param name="modelEvaluator">The model evaluator to use.</param>
    /// <param name="fitDetector">The fit detector to use.</param>
    /// <param name="fitnessCalculator">The fitness calculator to use.</param>
    /// <param name="modelCache">The model cache to use.</param>
    /// <remarks>
    /// <para>
    /// This constructor sets up the SGD optimizer with the specified options and components.
    /// If no options are provided, default options are used.
    /// </para>
    /// <para><b>For Beginners:</b> This is like setting up your hiker with their gear before the hike:
    /// 
    /// - You can give the hiker special instructions (options) for how to search
    /// - You can provide tools to measure progress (evaluator, fit detector, etc.)
    /// - If you don't provide instructions, the hiker will use a standard set
    /// 
    /// This setup ensures the optimizer is ready to start finding the best solution.
    /// </para>
    /// </remarks>
    public StochasticGradientDescentOptimizer(
        IFullModel<T, TInput, TOutput> model,
        StochasticGradientDescentOptimizerOptions<T, TInput, TOutput>? options = null,
        IEngine? engine = null)
        : base(model, options ?? new())
    {
        _options = options ?? new();
    }

    /// <summary>
    /// Performs the optimization process to find the best solution for the given input data.
    /// </summary>
    /// <param name="inputData">The input data to optimize against.</param>
    /// <returns>An optimization result containing the best solution found and associated metrics.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the main SGD algorithm. It iteratively updates the model parameters
    /// based on the calculated gradient, applying momentum and adaptive learning rates if configured.
    /// The process continues until either the maximum number of iterations is reached or early stopping
    /// criteria are met.
    /// </para>
    /// <para><b>For Beginners:</b> This is the main journey of our hiker:
    ///
    /// 1. Start at a random point on the hill (initialize random solution)
    /// 2. For each epoch (pass through the data):
    ///    - Process data in batches (default BatchSize=1 for true stochastic)
    ///    - For each batch:
    ///      - Look around to decide which way is downhill (calculate gradient)
    ///      - Apply momentum if configured
    ///      - Take a step in that direction (update solution)
    ///    - Check if this is the lowest point found so far (evaluate and update best solution)
    ///    - Adjust step size if needed (update adaptive parameters)
    ///    - Decide whether to stop early if no progress is being made
    /// 3. Return the lowest point found during the entire journey
    ///
    /// This process helps find a good solution efficiently, even in complex landscapes.
    /// </para>
    /// <para><b>DataLoader Integration:</b>
    /// This optimizer now uses the DataLoader batching infrastructure which supports:
    /// - Custom samplers (weighted, stratified, curriculum, importance, active learning)
    /// - Reproducible shuffling via RandomSeed
    /// - Option to drop incomplete final batches
    /// - True stochastic behavior with BatchSize=1 (default)
    /// Set these options via GradientBasedOptimizerOptions.DataSampler, ShuffleData, DropLastBatch, and RandomSeed.
    /// </para>
    /// </remarks>
    public override OptimizationResult<T, TInput, TOutput> Optimize(OptimizationInputData<T, TInput, TOutput> inputData)
    {
        // Initialize with random solution
        var currentSolution = InitializeRandomSolution(inputData.XTrain);
        var bestStepData = new OptimizationStepData<T, TInput, TOutput>();
        var previousStepData = PrepareAndEvaluateSolution(currentSolution, inputData);

        // Initialize parameters
        InitializeAdaptiveParameters();

        for (int epoch = 0; epoch < _options.MaxIterations; epoch++)
        {
            // Notify sampler of new epoch (for curriculum/self-paced learning)
            NotifyEpochStart(epoch);

            // Create batcher for the current epoch using DataLoader infrastructure
            // Default BatchSize=1 gives true stochastic gradient descent
            var batcher = CreateBatcher(inputData, _options.BatchSize);

            foreach (var (xBatch, yBatch, batchIndices) in batcher.GetBatches())
            {
                // Calculate gradient on the batch (single sample for true SGD)
                var gradient = CalculateGradient(currentSolution, xBatch, yBatch);

                // Apply momentum if configured
                gradient = ApplyMomentum(gradient);

                // Update solution
                var newSolution = UpdateSolution(currentSolution, gradient);

                currentSolution = newSolution;
            }

            // Evaluate after processing all batches in the epoch
            var currentStepData = EvaluateSolution(currentSolution, inputData);
            UpdateBestSolution(currentStepData, ref bestStepData);
            UpdateAdaptiveParameters(currentStepData, previousStepData);

            // Check early stopping criteria
            if (UpdateIterationHistoryAndCheckEarlyStopping(epoch, bestStepData))
            {
                return CreateOptimizationResult(bestStepData, inputData);
            }

            // Check convergence
            if (NumOps.LessThan(
                NumOps.Abs(NumOps.Subtract(bestStepData.FitnessScore, currentStepData.FitnessScore)),
                NumOps.FromDouble(_options.Tolerance)))
            {
                return CreateOptimizationResult(bestStepData, inputData);
            }

            previousStepData = currentStepData;
        }

        return CreateOptimizationResult(bestStepData, inputData);
    }

    /// <summary>
    /// Updates the current solution based on the calculated gradient.
    /// </summary>
    /// <param name="currentSolution">The current solution to update.</param>
    /// <param name="gradient">The calculated gradient.</param>
    /// <returns>A new ISymbolicModel representing the updated solution.</returns>
    /// <remarks>
    /// <para>
    /// This method applies the gradient descent update rule, subtracting the gradient multiplied by
    /// the learning rate from the current solution's coefficients.
    /// </para>
    /// <para><b>For Beginners:</b> This is like the hiker taking a step:
    /// 
    /// - The direction to step is given by the gradient
    /// - The size of the step is controlled by the learning rate
    /// - The hiker moves from their current position in this direction and distance
    /// 
    /// This small step helps the hiker gradually move towards the lowest point.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, TInput, TOutput> UpdateSolution(IFullModel<T, TInput, TOutput> currentSolution, Vector<T> gradient)
    {
        // === Vectorized SGD Update using IEngine ===
        // Phase B: US-GPU-015 - GPU-accelerated gradient updates
        // params = params - learningRate * gradient

        var parameters = currentSolution.GetParameters();
        var scaledGradient = (Vector<T>)Engine.Multiply(gradient, CurrentLearningRate);
        var updatedCoefficients = (Vector<T>)Engine.Subtract(parameters, scaledGradient);

        return currentSolution.WithParameters(updatedCoefficients);
    }

    /// <summary>
    /// Updates the optimizer's options with the provided options.
    /// </summary>
    /// <param name="options">The options to apply to this optimizer.</param>
    /// <exception cref="ArgumentException">Thrown when the options are not of the expected type.</exception>
    /// <remarks>
    /// <para>
    /// This method ensures that only StochasticGradientDescentOptimizerOptions can be applied to this optimizer.
    /// </para>
    /// <para><b>For Beginners:</b> This is like giving the hiker new instructions mid-journey:
    /// 
    /// - You can only give instructions specific to this type of hike (SGD)
    /// - If you try to give the wrong type of instructions, it will cause an error
    /// 
    /// This ensures that the optimizer always has the correct type of settings.
    /// </para>
    /// </remarks>
    protected override void UpdateOptions(OptimizationAlgorithmOptions<T, TInput, TOutput> options)
    {
        if (options is StochasticGradientDescentOptimizerOptions<T, TInput, TOutput> sgdOptions)
        {
            _options = sgdOptions;
        }
        else
        {
            throw new ArgumentException("Invalid options type. Expected StochasticGradientDescentOptimizerOptions.");
        }
    }

    /// <summary>
    /// Gets the current options for this optimizer.
    /// </summary>
    /// <returns>The current StochasticGradientDescentOptimizerOptions.</returns>
    /// <remarks>
    /// <para>
    /// This method returns the current configuration options of the SGD optimizer.
    /// </para>
    /// <para><b>For Beginners:</b> This is like asking the hiker what their current instructions are:
    /// 
    /// - You can see how the hiker is currently set up to search
    /// - This includes things like how big their steps are, how many steps they're allowed to take, etc.
    /// 
    /// This is useful for understanding or checking the current setup of the optimizer.
    /// </para>
    /// </remarks>
    public override OptimizationAlgorithmOptions<T, TInput, TOutput> GetOptions()
    {
        return _options;
    }

    /// <summary>
    /// Serializes the current state of the StochasticGradientDescentOptimizer to a byte array.
    /// </summary>
    /// <returns>A byte array representing the serialized state of the optimizer.</returns>
    /// <remarks>
    /// <para>
    /// This method saves the current state of the optimizer, including its base class data and
    /// SGD-specific options, into a byte array.
    /// </para>
    /// <para><b>For Beginners:</b> This is like taking a snapshot of the hiker's journey:
    /// 
    /// - It saves all the current settings and progress
    /// - This saved data can be used later to continue from where you left off
    /// - It includes both general hiking info and SGD-specific details
    /// 
    /// This is useful for saving progress or sharing the optimizer's current state.
    /// </para>
    /// </remarks>
    public override byte[] Serialize()
    {
        using (MemoryStream ms = new MemoryStream())
        using (BinaryWriter writer = new BinaryWriter(ms))
        {
            // Serialize base class data
            byte[] baseData = base.Serialize();
            writer.Write(baseData.Length);
            writer.Write(baseData);

            // Serialize SGD-specific options
            string optionsJson = JsonConvert.SerializeObject(_options);
            writer.Write(optionsJson);

            return ms.ToArray();
        }
    }

    /// <summary>
    /// Deserializes a byte array to restore the state of the StochasticGradientDescentOptimizer.
    /// </summary>
    /// <param name="data">The byte array containing the serialized optimizer state.</param>
    /// <exception cref="InvalidOperationException">Thrown when deserialization of optimizer options fails.</exception>
    /// <remarks>
    /// <para>
    /// This method restores the state of the optimizer from a byte array, including its base class data
    /// and SGD-specific options. It uses a BinaryReader to read the serialized data and reconstruct
    /// the optimizer's state.
    /// </para>
    /// <para><b>For Beginners:</b> This is like unpacking the hiker's backpack after a journey:
    /// 
    /// - It reads the saved snapshot of the hiker's journey
    /// - It restores both general hiking info and SGD-specific details
    /// - If there's a problem reading the SGD-specific details, it reports an error
    /// 
    /// This allows you to continue from a previously saved state of the optimizer.
    /// </para>
    /// </remarks>
    public override void Deserialize(byte[] data)
    {
        using (MemoryStream ms = new MemoryStream(data))
        using (BinaryReader reader = new BinaryReader(ms))
        {
            // Deserialize base class data
            int baseDataLength = reader.ReadInt32();
            byte[] baseData = reader.ReadBytes(baseDataLength);
            base.Deserialize(baseData);

            // Deserialize SGD-specific options
            string optionsJson = reader.ReadString();
            _options = JsonConvert.DeserializeObject<StochasticGradientDescentOptimizerOptions<T, TInput, TOutput>>(optionsJson)
                ?? throw new InvalidOperationException("Failed to deserialize optimizer options.");
        }
    }

    /// <summary>
    /// Generates a unique cache key for gradient calculations.
    /// </summary>
    /// <param name="model">The symbolic model for which the gradient is being calculated.</param>
    /// <param name="X">The input data matrix.</param>
    /// <param name="y">The target vector.</param>
    /// <returns>A string representing the unique cache key.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a unique identifier for caching gradient calculations. It combines the base
    /// cache key with SGD-specific parameters to ensure that cached gradients are only reused when all
    /// relevant parameters are identical.
    /// </para>
    /// <para><b>For Beginners:</b> This is like creating a unique label for each calculation the hiker does:
    /// 
    /// - It starts with a basic label (baseKey) that describes the general calculation
    /// - It adds SGD-specific information like the current step size (learning rate) and how many steps
    ///   the hiker is allowed to take (max iterations)
    /// - This unique label helps the hiker remember and quickly recall previous calculations
    ///   instead of redoing them unnecessarily
    /// 
    /// This improves efficiency by avoiding redundant calculations.
    /// </para>
    /// </remarks>
    protected override string GenerateGradientCacheKey(IFullModel<T, TInput, TOutput> model, TInput X, TOutput y)
    {
        var baseKey = base.GenerateGradientCacheKey(model, X, y);
        return $"{baseKey}_SGD_{CurrentLearningRate}_{_options.MaxIterations}";
    }
}
