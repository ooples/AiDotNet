using Newtonsoft.Json;
using AiDotNet.Tensors.Engines.DirectGpu;

namespace AiDotNet.Optimizers;

/// <summary>
/// Implements the Mini-Batch Gradient Descent optimization algorithm.
/// </summary>
/// <remarks>
/// <para>
/// Mini-Batch Gradient Descent is a variation of gradient descent that splits the training data into small batches
/// to calculate model error and update model coefficients. This approach strikes a balance between the efficiency
/// of stochastic gradient descent and the stability of batch gradient descent.
/// </para>
/// <para><b>For Beginners:</b>
/// Imagine you're trying to find the bottom of a valley while blindfolded. Mini-Batch Gradient Descent is like taking 
/// a few steps, checking your position, adjusting your direction, and repeating. It's faster than checking after every 
/// single step (Stochastic Gradient Descent) but more precise than taking a lot of steps before checking (Batch Gradient Descent).
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class MiniBatchGradientDescentOptimizer<T, TInput, TOutput> : GradientBasedOptimizerBase<T, TInput, TOutput>
{
    /// <summary>
    /// The options specific to the Mini-Batch Gradient Descent algorithm.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b>
    /// This is like your hiking plan, containing details such as how many steps to take before checking your position,
    /// how many times to repeat the process, and how to adjust your step size.
    /// </para>
    /// </remarks>
    private MiniBatchGradientDescentOptions<T, TInput, TOutput> _options;

    /// <summary>
    /// Initializes a new instance of the MiniBatchGradientDescentOptimizer class.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This constructor sets up the optimizer with the provided options and dependencies. If no options are provided,
    /// it uses default settings. It also initializes a random number generator for shuffling data.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like setting up your hiking gear before starting the journey to find the valley's bottom. You're 
    /// deciding on your strategy (options) and packing your tools (dependencies) that you'll use along the way.
    /// </para>
    /// </remarks>
    /// <param name="model">The model to optimize.</param>
    public MiniBatchGradientDescentOptimizer(
        IFullModel<T, TInput, TOutput> model,
        MiniBatchGradientDescentOptions<T, TInput, TOutput>? options = null,
        IEngine? engine = null)
        : base(model, options ?? new())
    {
        _options = options ?? new MiniBatchGradientDescentOptions<T, TInput, TOutput>();

        InitializeAdaptiveParameters();
    }

    /// <summary>
    /// Initializes adaptive parameters for the optimization process.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method sets up the initial learning rate for the optimization process based on the options provided.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like deciding how big your steps will be when you start your journey. The learning rate determines 
    /// how much you adjust your position based on each batch of information you process.
    /// </para>
    /// </remarks>
    protected override void InitializeAdaptiveParameters()
    {
        base.InitializeAdaptiveParameters();
        CurrentLearningRate = NumOps.FromDouble(_options.InitialLearningRate);
    }

    /// <summary>
    /// Performs the optimization process using Mini-Batch Gradient Descent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method implements the main optimization loop. It iterates through the data in mini-batches,
    /// calculating gradients and updating the model parameters for each batch. The process continues for
    /// a specified number of epochs or until a stopping criterion is met.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is the actual journey to find the valley's bottom. You're taking steps (processing batches of data),
    /// checking your position (evaluating the model), and adjusting your direction (updating the model parameters).
    /// You do this repeatedly (for each epoch) until you're satisfied with your position or you've taken the
    /// maximum number of steps you allowed yourself.
    /// </para>
    /// <para><b>DataLoader Integration:</b>
    /// This optimizer now uses the DataLoader batching infrastructure which supports:
    /// - Custom samplers (weighted, stratified, curriculum, importance, active learning)
    /// - Reproducible shuffling via RandomSeed
    /// - Option to drop incomplete final batches
    /// Set these options via GradientBasedOptimizerOptions.DataSampler, ShuffleData, DropLastBatch, and RandomSeed.
    /// </para>
    /// </remarks>
    /// <param name="inputData">The input data for the optimization process.</param>
    /// <returns>The result of the optimization process.</returns>
    public override OptimizationResult<T, TInput, TOutput> Optimize(OptimizationInputData<T, TInput, TOutput> inputData)
    {
        // Initialize with random solution
        var currentSolution = InitializeRandomSolution(inputData.XTrain);
        var bestStepData = new OptimizationStepData<T, TInput, TOutput>();
        var previousStepData = PrepareAndEvaluateSolution(currentSolution, inputData);

        // Initialize parameters
        InitializeAdaptiveParameters();

        for (int epoch = 0; epoch < Options.MaxIterations; epoch++)
        {
            // Notify sampler of new epoch (for curriculum/self-paced learning)
            NotifyEpochStart(epoch);

            // Create batcher for the current epoch using DataLoader infrastructure
            // This handles shuffling, sampling strategies, and batch creation
            var batcher = CreateBatcher(inputData, _options.BatchSize);

            foreach (var (xBatch, yBatch, batchIndices) in batcher.GetBatches())
            {
                // Process batch and calculate gradient using the batch data directly
                var gradient = CalculateGradient(currentSolution, xBatch, yBatch);

                // Update solution
                var newSolution = UpdateSolution(currentSolution, gradient);

                // Evaluate the solution
                var currentStepData = EvaluateSolution(newSolution, inputData);
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

                currentSolution = newSolution;
                previousStepData = currentStepData;
            }
        }

        return CreateOptimizationResult(bestStepData, inputData);
    }

    /// <summary>
    /// Updates the current solution based on the calculated gradient.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method applies the gradient to the current solution, adjusting each coefficient by the gradient 
    /// scaled by the learning rate.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like taking a step in the direction you think will lead you closer to the valley's bottom. 
    /// The size of your step is determined by the learning rate, and the direction is given by the gradient.
    /// </para>
    /// </remarks>
    /// <param name="currentSolution">The current model solution.</param>
    /// <param name="gradient">The calculated gradient.</param>
    /// <returns>An updated symbolic model with improved coefficients.</returns>
    protected override IFullModel<T, TInput, TOutput> UpdateSolution(IFullModel<T, TInput, TOutput> currentSolution, Vector<T> gradient)
    {
        // === Vectorized Mini-Batch GD Update using IEngine (Phase B: US-GPU-015) ===
        // params = params - learningRate * gradient

        var parameters = currentSolution.GetParameters();
        var scaledGradient = (Vector<T>)Engine.Multiply(gradient, CurrentLearningRate);
        var newCoefficients = (Vector<T>)Engine.Subtract(parameters, scaledGradient);

        return currentSolution.WithParameters(newCoefficients);
    }

    /// <summary>
    /// Reverses a Mini-Batch Gradient Descent update to recover original parameters.
    /// </summary>
    /// <param name="updatedParameters">Parameters after Mini-Batch GD update</param>
    /// <param name="appliedGradients">The gradients that were applied</param>
    /// <returns>Original parameters before the update</returns>
    /// <remarks>
    /// <para>
    /// Mini-Batch Gradient Descent uses vanilla SGD update rule: params_new = params_old - lr * gradient.
    /// The reverse is straightforward: params_old = params_new + lr * gradient.
    /// </para>
    /// <para><b>For Beginners:</b> This calculates where parameters were before a Mini-Batch GD update.
    /// Since Mini-Batch GD uses simple steps (parameter minus learning_rate times gradient), reversing
    /// just means adding back that step.
    /// </para>
    /// </remarks>
    public override Vector<T> ReverseUpdate(Vector<T> updatedParameters, Vector<T> appliedGradients)
    {
        if (updatedParameters == null)
            throw new ArgumentNullException(nameof(updatedParameters));
        if (appliedGradients == null)
            throw new ArgumentNullException(nameof(appliedGradients));

        if (updatedParameters.Length != appliedGradients.Length)
        {
            throw new ArgumentException(
                $"Updated parameters size ({updatedParameters.Length}) must match applied gradients size ({appliedGradients.Length})",
                nameof(appliedGradients));
        }

        // === Vectorized Reverse Mini-Batch GD Update (Phase B: US-GPU-015) ===
        // Reverse: original = updated + lr * gradient
        var currentLrVec = Vector<T>.CreateDefault(appliedGradients.Length, CurrentLearningRate);
        var gradientStep = (Vector<T>)Engine.Multiply(currentLrVec, appliedGradients);
        return (Vector<T>)Engine.Add(updatedParameters, gradientStep);
    }

    /// <summary>
    /// Updates the adaptive parameters of the optimizer based on the current and previous optimization steps.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method adjusts the learning rate based on the performance of the current step compared to the previous step.
    /// If improvement is seen, the learning rate may be increased, otherwise it may be decreased.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like adjusting your step size based on how well you're doing. If you're making good progress, 
    /// you might take slightly bigger steps. If you're not improving, you might take smaller, more careful steps.
    /// </para>
    /// </remarks>
    /// <param name="currentStepData">Data from the current optimization step.</param>
    /// <param name="previousStepData">Data from the previous optimization step.</param>
    protected override void UpdateAdaptiveParameters(OptimizationStepData<T, TInput, TOutput> currentStepData, OptimizationStepData<T, TInput, TOutput> previousStepData)
    {
        base.UpdateAdaptiveParameters(currentStepData, previousStepData);

        if (_options.UseAdaptiveLearningRate)
        {
            if (NumOps.GreaterThan(currentStepData.FitnessScore, previousStepData.FitnessScore))
            {
                CurrentLearningRate = NumOps.Multiply(CurrentLearningRate, NumOps.FromDouble(_options.LearningRateIncreaseFactor));
            }
            else
            {
                CurrentLearningRate = NumOps.Multiply(CurrentLearningRate, NumOps.FromDouble(_options.LearningRateDecreaseFactor));
            }

            CurrentLearningRate = MathHelper.Max(NumOps.FromDouble(_options.MinLearningRate),
                MathHelper.Min(NumOps.FromDouble(_options.MaxLearningRate), CurrentLearningRate));
        }
    }

    /// <summary>
    /// Updates the optimizer's options with new settings.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method allows updating the optimizer's settings during runtime. It ensures that only compatible
    /// option types are used with this optimizer.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like changing your hiking strategy mid-journey. It makes sure you're only using strategies 
    /// that work for this specific type of journey (Mini-Batch Gradient Descent).
    /// </para>
    /// </remarks>
    /// <param name="options">The new options to be applied to the optimizer.</param>
    /// <exception cref="ArgumentException">Thrown when the provided options are not of the correct type.</exception>
    protected override void UpdateOptions(OptimizationAlgorithmOptions<T, TInput, TOutput> options)
    {
        if (options is MiniBatchGradientDescentOptions<T, TInput, TOutput> mbgdOptions)
        {
            _options = mbgdOptions;
        }
        else
        {
            throw new ArgumentException("Invalid options type. Expected MiniBatchGradientDescentOptions.");
        }
    }

    /// <summary>
    /// Gets the current optimization algorithm options.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method returns the current options used by the Mini-Batch Gradient Descent optimizer.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like checking your current hiking plan. It lets you see all the settings and strategies 
    /// you're currently using in your journey to find the valley's bottom.
    /// </para>
    /// </remarks>
    /// <returns>The current MiniBatchGradientDescentOptions object.</returns>
    public override OptimizationAlgorithmOptions<T, TInput, TOutput> GetOptions()
    {
        return _options;
    }

    /// <summary>
    /// Serializes the optimizer's state into a byte array.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method converts the current state of the optimizer, including its base class state and options, 
    /// into a byte array. This is useful for saving the optimizer's state or transferring it between systems.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// Think of this as taking a snapshot of your entire journey so far. It captures all the details of your 
    /// current position, your hiking plan, and how you got there. This snapshot can be used to continue your 
    /// journey later or share your exact situation with others.
    /// </para>
    /// </remarks>
    /// <returns>A byte array representing the serialized state of the optimizer.</returns>
    public override byte[] Serialize()
    {
        using MemoryStream ms = new MemoryStream();
        using BinaryWriter writer = new BinaryWriter(ms);

        byte[] baseData = base.Serialize();
        writer.Write(baseData.Length);
        writer.Write(baseData);

        string optionsJson = JsonConvert.SerializeObject(_options);
        writer.Write(optionsJson);

        return ms.ToArray();
    }

    /// <summary>
    /// Deserializes a byte array to restore the optimizer's state.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method takes a byte array (previously created by Serialize) and uses it to restore the optimizer's state, 
    /// including its base class state and options.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like using a detailed map and instructions to recreate your exact position and plan from a previous 
    /// point in your journey. It allows you to pick up right where you left off, with all your strategies and progress intact.
    /// </para>
    /// </remarks>
    /// <param name="data">The byte array containing the serialized optimizer state.</param>
    /// <exception cref="InvalidOperationException">Thrown when the optimizer options cannot be deserialized.</exception>
    public override void Deserialize(byte[] data)
    {
        using MemoryStream ms = new MemoryStream(data);
        using BinaryReader reader = new BinaryReader(ms);

        int baseDataLength = reader.ReadInt32();
        byte[] baseData = reader.ReadBytes(baseDataLength);
        base.Deserialize(baseData);

        string optionsJson = reader.ReadString();
        _options = JsonConvert.DeserializeObject<MiniBatchGradientDescentOptions<T, TInput, TOutput>>(optionsJson)
            ?? throw new InvalidOperationException("Failed to deserialize optimizer options.");
    }

    /// <summary>
    /// Generates a unique key for caching gradients based on the model and input data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method creates a unique identifier for caching gradients. It combines the base gradient cache key 
    /// with specific parameters of the Mini-Batch Gradient Descent algorithm.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// Imagine you're leaving markers along your hiking path. This method creates a unique label for each marker, 
    /// combining information about where you are (the model and data) with specifics about how you're hiking 
    /// (batch size and number of rounds). This helps you quickly recognize and use information from similar 
    /// situations you've encountered before.
    /// </para>
    /// </remarks>
    /// <param name="model">The symbolic model being optimized.</param>
    /// <param name="X">The input data matrix.</param>
    /// <param name="y">The target output vector.</param>
    /// <returns>A string representing the unique gradient cache key.</returns>
    protected override string GenerateGradientCacheKey(IFullModel<T, TInput, TOutput> model, TInput X, TOutput y)
    {
        var baseKey = base.GenerateGradientCacheKey(model, X, y);
        return $"{baseKey}_MiniBatchGD_{_options.BatchSize}_{_options.MaxEpochs}";
    }

    /// <summary>
    /// Updates parameters on the GPU using vanilla SGD (same as SGD for parameter updates).
    /// </summary>
    public override void UpdateParametersGpu(IGpuBuffer parameters, IGpuBuffer gradients, int parameterCount, IDirectGpuBackend backend)
    {
        backend.SgdUpdate(
            parameters,
            gradients,
            (float)NumOps.ToDouble(CurrentLearningRate),
            0.0f, // No weight decay
            parameterCount);
    }
}
