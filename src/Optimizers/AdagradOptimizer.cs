using Newtonsoft.Json;

namespace AiDotNet.Optimizers;

/// <summary>
/// Represents an Adagrad (Adaptive Gradient) optimizer for gradient-based optimization.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// The Adagrad optimizer adapts the learning rate for each parameter based on the historical gradients.
/// It performs larger updates for infrequent parameters and smaller updates for frequent ones.
/// </para>
/// <para><b>For Beginners:</b> Adagrad is like a smart learning assistant that adjusts how much it learns
/// for each piece of information based on how often it has seen similar information before.
/// 
/// - It learns more from new or rare information
/// - It learns less from common or frequently seen information
/// - This helps it focus on the most important parts of what it's learning
/// 
/// This can be especially useful when some parts of your data are more important or occur less frequently.
/// </para>
/// </remarks>
public class AdagradOptimizer<T, TInput, TOutput> : GradientBasedOptimizerBase<T, TInput, TOutput>
{
    /// <summary>
    /// The configuration options specific to the Adagrad optimizer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the configuration parameters that control the behavior of the Adagrad algorithm,
    /// such as learning rate, epsilon value, and convergence criteria.
    /// </para>
    /// <para><b>For Beginners:</b> This is like the instruction manual for our learning assistant.
    /// It contains all the settings that determine how the optimizer behaves, such as how fast it learns
    /// and when it should consider its job complete.
    /// 
    /// These settings can be customized to make the optimizer work better for different types of problems.
    /// </para>
    /// </remarks>
    private AdagradOptimizerOptions<T, TInput, TOutput> _options;

    /// <summary>
    /// Stores the sum of squared gradients for each parameter during optimization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This vector accumulates the squared gradients for each parameter throughout the optimization process.
    /// It's used to adapt the learning rate individually for each parameter - parameters with larger
    /// accumulated gradients will have smaller learning rates and vice versa.
    /// </para>
    /// <para><b>For Beginners:</b> This is like the optimizer's memory of how much each part of the solution
    /// has changed over time.
    /// 
    /// Imagine you're learning different subjects:
    /// - For subjects you've practiced a lot (large accumulated gradients), you take smaller learning steps
    /// - For subjects you've rarely practiced (small accumulated gradients), you take larger learning steps
    /// 
    /// This helps the optimizer focus more on the parts of the solution that haven't received much attention,
    /// which is especially useful when some parameters are more important than others or appear less frequently.
    /// </para>
    /// </remarks>
    private Vector<T>? _accumulatedSquaredGradients;

    /// <summary>
    /// Initializes a new instance of the AdagradOptimizer class.
    /// </summary>
    /// <param name="model">The model to optimize.</param>
    /// <param name="options">The options for configuring the Adagrad optimizer.</param>
    /// <remarks>
    /// <para>
    /// This constructor sets up the Adagrad optimizer with the specified options and components.
    /// If no options are provided, it uses default AdagradOptimizerOptions.
    /// </para>
    /// <para><b>For Beginners:</b> This is like setting up your learning assistant with specific instructions.
    ///
    /// You can customize:
    /// - How the assistant learns (options)
    /// - How it measures its progress (predictionOptions, modelOptions)
    /// - How it evaluates its performance (modelEvaluator, fitDetector, fitnessCalculator)
    /// - How it remembers what it has learned (modelCache, gradientCache)
    ///
    /// If you don't specify these, it will use default settings.
    /// </para>
    /// </remarks>
    public AdagradOptimizer(
        IFullModel<T, TInput, TOutput> model,
        AdagradOptimizerOptions<T, TInput, TOutput>? options = null)
        : base(model, options ?? new())
    {
        _options = options ?? new AdagradOptimizerOptions<T, TInput, TOutput>();

        InitializeAdaptiveParameters();
    }

    /// <summary>
    /// Initializes the adaptive parameters for the Adagrad optimizer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method sets up the initial learning rate for the optimizer based on the options.
    /// </para>
    /// <para><b>For Beginners:</b> This is like setting the initial speed at which your assistant learns.
    /// 
    /// The learning rate determines how big the steps are when the optimizer is trying to find the best solution.
    /// A good initial learning rate helps the optimizer start its learning process effectively.
    /// </para>
    /// </remarks>
    protected override void InitializeAdaptiveParameters()
    {
        base.InitializeAdaptiveParameters();
        CurrentLearningRate = NumOps.FromDouble(_options.InitialLearningRate);
    }

    /// <summary>
    /// Performs the optimization process using the Adagrad algorithm.
    /// </summary>
    /// <param name="inputData">The input data for optimization.</param>
    /// <returns>The result of the optimization process.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the main optimization loop of the Adagrad algorithm. It iteratively
    /// updates the solution based on calculated gradients and accumulated squared gradients.
    /// </para>
    /// <para><b>For Beginners:</b> This is the main learning process of the Adagrad optimizer.
    /// 
    /// Here's what happens in each iteration:
    /// 1. Calculate how to improve the current solution (gradient)
    /// 2. Update the memory of past improvements (accumulated squared gradients)
    /// 3. Create a new, hopefully better solution
    /// 4. Check if this new solution is the best so far
    /// 5. Adjust how the optimizer learns (adaptive parameters)
    /// 6. Check if we should stop early (if the solution is good enough)
    /// 
    /// This process repeats until we reach the maximum number of iterations or find a good enough solution.
    /// </para>
    /// <para><b>DataLoader Integration:</b> This method uses the DataLoader API for efficient batch processing.
    /// It creates a batcher using <see cref="GradientBasedOptimizerBase{T,TInput,TOutput}.CreateBatcher"/>
    /// and notifies the sampler of epoch starts using
    /// <see cref="GradientBasedOptimizerBase{T,TInput,TOutput}.NotifyEpochStart"/>.
    /// </para>
    /// </remarks>
    public override OptimizationResult<T, TInput, TOutput> Optimize(OptimizationInputData<T, TInput, TOutput> inputData)
    {
        ValidationHelper<T>.ValidateInputData(inputData);

        var currentSolution = InitializeRandomSolution(inputData.XTrain);
        var bestStepData = new OptimizationStepData<T, TInput, TOutput>();
        var previousStepData = new OptimizationStepData<T, TInput, TOutput>();

        _accumulatedSquaredGradients = new Vector<T>(currentSolution.GetParameters().Length);
        InitializeAdaptiveParameters();

        for (int epoch = 0; epoch < _options.MaxIterations; epoch++)
        {
            NotifyEpochStart(epoch);
            var batcher = CreateBatcher(inputData, _options.BatchSize);

            foreach (var (xBatch, yBatch, batchIndices) in batcher.GetBatches())
            {
                var gradient = CalculateGradient(currentSolution, xBatch, yBatch);
                UpdateAccumulatedSquaredGradients(gradient);
                var newSolution = UpdateSolution(currentSolution, gradient);
                currentSolution = newSolution;
            }

            var currentStepData = EvaluateSolution(currentSolution, inputData);
            UpdateBestSolution(currentStepData, ref bestStepData);

            UpdateAdaptiveParameters(currentStepData, previousStepData);

            if (UpdateIterationHistoryAndCheckEarlyStopping(epoch, bestStepData))
            {
                return CreateOptimizationResult(bestStepData, inputData);
            }

            if (NumOps.LessThan(NumOps.Abs(NumOps.Subtract(bestStepData.FitnessScore, currentStepData.FitnessScore)), NumOps.FromDouble(_options.Tolerance)))
            {
                return CreateOptimizationResult(bestStepData, inputData);
            }

            previousStepData = currentStepData;
        }

        return CreateOptimizationResult(bestStepData, inputData);
    }

    /// <summary>
    /// Updates the accumulated squared gradients used in the Adagrad algorithm.
    /// </summary>
    /// <param name="gradient">The current gradient vector.</param>
    /// <remarks>
    /// <para>
    /// This method updates the accumulated squared gradients by adding the square of each gradient component.
    /// These accumulated values are used to adapt the learning rate for each parameter.
    /// </para>
    /// <para><b>For Beginners:</b> This is like updating the optimizer's memory of past improvements.
    /// 
    /// For each piece of the solution:
    /// 1. Square the current improvement (gradient)
    /// 2. Add this square to the memory of past improvements
    /// 
    /// This memory helps the optimizer decide how much to change each part of the solution in future steps.
    /// Parts with a history of larger improvements will get smaller changes, and vice versa.
    /// </para>
    /// </remarks>
    private void UpdateAccumulatedSquaredGradients(Vector<T> gradient)
    {
        // === Vectorized using IEngine (Phase B: US-GPU-015) ===
        var gradSquared = (Vector<T>)Engine.Multiply(gradient, gradient);
        _accumulatedSquaredGradients = (Vector<T>)Engine.Add(_accumulatedSquaredGradients!, gradSquared);
    }

    /// <summary>
    /// Updates the current solution using the Adagrad update rule.
    /// </summary>
    /// <param name="currentSolution">The current solution model.</param>
    /// <param name="gradient">The calculated gradient.</param>
    /// <returns>A new solution model after applying the Adagrad update.</returns>
    /// <remarks>
    /// <para>
    /// This method applies the Adagrad update rule to each coefficient of the current solution.
    /// It uses the accumulated squared gradients to adapt the learning rate for each parameter.
    /// </para>
    /// <para><b>For Beginners:</b> This is like taking a step towards a better solution.
    /// 
    /// For each part of the solution:
    /// 1. Calculate a custom learning rate based on past improvements
    /// 2. Use this rate to decide how big a step to take
    /// 3. Take the step by updating that part of the solution
    /// 
    /// This adaptive approach allows the optimizer to take larger steps for less frequently updated parts
    /// and smaller steps for more frequently updated parts.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, TInput, TOutput> UpdateSolution(IFullModel<T, TInput, TOutput> currentSolution, Vector<T> gradient)
    {
        var parameters = currentSolution.GetParameters();

        // === Vectorized Adagrad Update using IEngine (Phase B: US-GPU-015) ===
        T epsilon = NumOps.FromDouble(_options.Epsilon);

        // Calculate adaptive learning rates: lr / (sqrt(accSqGrad) + eps)
        var sqrtAccSqGrad = (Vector<T>)Engine.Sqrt(_accumulatedSquaredGradients!);
        var epsilonVec = Vector<T>.CreateDefault(sqrtAccSqGrad.Length, epsilon);
        var denominator = (Vector<T>)Engine.Add(sqrtAccSqGrad, epsilonVec);
        var currentLrVec = Vector<T>.CreateDefault(sqrtAccSqGrad.Length, CurrentLearningRate);
        var adaptiveLearningRates = (Vector<T>)Engine.Divide(currentLrVec, denominator);

        // Calculate updates: adaptiveLr * gradient
        var updates = (Vector<T>)Engine.Multiply(adaptiveLearningRates, gradient);

        // Update parameters: params - updates
        var newCoefficients = (Vector<T>)Engine.Subtract(parameters, updates);

        return currentSolution.WithParameters(newCoefficients);
    }

    /// <summary>
    /// Updates a vector of parameters using the Adagrad optimization algorithm.
    /// </summary>
    /// <param name="parameters">The current parameter vector to be updated.</param>
    /// <param name="gradient">The gradient vector corresponding to the parameters.</param>
    /// <returns>The updated parameter vector.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the Adagrad update rule by accumulating squared gradients for each parameter
    /// and using them to adapt the learning rate individually. Parameters with larger accumulated gradients
    /// receive smaller learning rates, and vice versa.
    /// </para>
    /// <para><b>For Beginners:</b> Adagrad adjusts the learning rate for each parameter based on how much
    /// it has changed in the past. Parameters that have received many large updates get smaller future updates,
    /// while rarely-updated parameters get larger updates. This helps focus learning on less frequent features.
    /// </para>
    /// </remarks>
    public override Vector<T> UpdateParameters(Vector<T> parameters, Vector<T> gradient)
    {
        if (_accumulatedSquaredGradients == null || _accumulatedSquaredGradients.Length != parameters.Length)
        {
            _accumulatedSquaredGradients = new Vector<T>(parameters.Length);
        }

        // === Vectorized Adagrad Update using IEngine (Phase B: US-GPU-015) ===
        T epsilon = NumOps.FromDouble(_options.Epsilon);

        // Accumulate squared gradients: accSqGrad = accSqGrad + gradient^2
        var gradSquared = (Vector<T>)Engine.Multiply(gradient, gradient);
        _accumulatedSquaredGradients = (Vector<T>)Engine.Add(_accumulatedSquaredGradients, gradSquared);

        // Calculate adaptive learning rates: lr / (sqrt(accSqGrad) + eps)
        var sqrtAccSqGrad = (Vector<T>)Engine.Sqrt(_accumulatedSquaredGradients);
        var epsilonVec = Vector<T>.CreateDefault(sqrtAccSqGrad.Length, epsilon);
        var denominator = (Vector<T>)Engine.Add(sqrtAccSqGrad, epsilonVec);
        var currentLrVec = Vector<T>.CreateDefault(sqrtAccSqGrad.Length, CurrentLearningRate);
        var adaptiveLearningRates = (Vector<T>)Engine.Divide(currentLrVec, denominator);

        // Calculate updates: adaptiveLr * gradient
        var updates = (Vector<T>)Engine.Multiply(adaptiveLearningRates, gradient);

        // Update parameters: params - updates
        var updatedParams = (Vector<T>)Engine.Subtract(parameters, updates);

        return updatedParams;
    }


    /// <summary>
    /// Updates the adaptive parameters of the Adagrad optimizer.
    /// </summary>
    /// <param name="currentStepData">The optimization step data for the current iteration.</param>
    /// <param name="previousStepData">The optimization step data for the previous iteration.</param>
    /// <remarks>
    /// <para>
    /// This method updates the learning rate if adaptive learning rate is enabled in the options.
    /// It increases or decreases the learning rate based on whether the current solution is better than the previous one.
    /// </para>
    /// <para><b>For Beginners:</b> This is like adjusting how fast the optimizer learns based on its recent progress.
    /// 
    /// If adaptive learning rate is turned on:
    /// - If the current solution is better, slightly increase the learning rate
    /// - If the current solution is worse, slightly decrease the learning rate
    /// - Keep the learning rate within specified limits
    /// 
    /// This helps the optimizer adapt its learning speed based on how well it's doing,
    /// potentially making the learning process more efficient.
    /// </para>
    /// </remarks>
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
    /// Updates the options for the Adagrad optimizer.
    /// </summary>
    /// <param name="options">The new options to be set.</param>
    /// <exception cref="ArgumentException">Thrown when the provided options are not of type AdagradOptimizerOptions.</exception>
    /// <remarks>
    /// <para>
    /// This method updates the optimizer's configuration with new options. It ensures that only
    /// AdagradOptimizerOptions are used to configure this optimizer.
    /// </para>
    /// <para><b>For Beginners:</b> This is like updating the instructions for your learning assistant.
    /// 
    /// - It checks if the new instructions are the right type for this specific assistant (Adagrad)
    /// - If they are, it updates the assistant's settings
    /// - If they're not, it reports an error
    /// 
    /// This helps prevent accidentally using the wrong type of settings, which could cause problems.
    /// </para>
    /// </remarks>
    protected override void UpdateOptions(OptimizationAlgorithmOptions<T, TInput, TOutput> options)
    {
        if (options is AdagradOptimizerOptions<T, TInput, TOutput> adagradOptions)
        {
            _options = adagradOptions;
        }
        else
        {
            throw new ArgumentException("Invalid options type. Expected AdagradOptimizerOptions.");
        }
    }

    /// <summary>
    /// Retrieves the current options of the Adagrad optimizer.
    /// </summary>
    /// <returns>The current AdagradOptimizerOptions.</returns>
    /// <remarks>
    /// <para>
    /// This method returns the current configuration options of the Adagrad optimizer.
    /// </para>
    /// <para><b>For Beginners:</b> This is like asking your learning assistant for its current instructions.
    /// 
    /// It allows you to check:
    /// - What learning rate the optimizer is using
    /// - How many iterations it will run
    /// - Other specific settings for the Adagrad method
    /// 
    /// This can be useful for understanding how the optimizer is currently set up or for saving its configuration.
    /// </para>
    /// </remarks>
    public override OptimizationAlgorithmOptions<T, TInput, TOutput> GetOptions()
    {
        return _options;
    }

    /// <summary>
    /// Serializes the Adagrad optimizer to a byte array.
    /// </summary>
    /// <returns>A byte array representing the serialized state of the optimizer.</returns>
    /// <remarks>
    /// <para>
    /// This method saves the current state of the Adagrad optimizer, including its base class state and specific options,
    /// into a byte array. This allows the optimizer's state to be stored or transmitted.
    /// </para>
    /// <para><b>For Beginners:</b> This is like taking a snapshot of your learning assistant's current state.
    /// 
    /// The process:
    /// 1. Saves the basic information (from the parent class)
    /// 2. Saves the specific Adagrad settings
    /// 3. Combines all this information into a single package (byte array)
    /// 
    /// This snapshot can be used later to recreate the exact same state of the optimizer,
    /// which is useful for saving progress or sharing the optimizer's configuration.
    /// </para>
    /// </remarks>
    public override byte[] Serialize()
    {
        using (MemoryStream ms = new MemoryStream())
        using (BinaryWriter writer = new BinaryWriter(ms))
        {
            byte[] baseData = base.Serialize();
            writer.Write(baseData.Length);
            writer.Write(baseData);

            string optionsJson = JsonConvert.SerializeObject(_options);
            writer.Write(optionsJson);

            return ms.ToArray();
        }
    }

    /// <summary>
    /// Deserializes the Adagrad optimizer from a byte array.
    /// </summary>
    /// <param name="data">The byte array containing the serialized optimizer state.</param>
    /// <exception cref="InvalidOperationException">Thrown when deserialization of optimizer options fails.</exception>
    /// <remarks>
    /// <para>
    /// This method reconstructs the state of the Adagrad optimizer from a byte array, including its base class state
    /// and specific options. It's used to restore a previously serialized optimizer state.
    /// </para>
    /// <para><b>For Beginners:</b> This is like recreating your learning assistant from a saved snapshot.
    /// 
    /// The process:
    /// 1. Reads the basic information (for the parent class)
    /// 2. Recreates the parent class state
    /// 3. Reads and recreates the specific Adagrad settings
    /// 
    /// This allows you to continue using the optimizer from exactly where you left off,
    /// with all its learned information and settings intact.
    /// </para>
    /// </remarks>
    public override void Deserialize(byte[] data)
    {
        using (MemoryStream ms = new MemoryStream(data))
        using (BinaryReader reader = new BinaryReader(ms))
        {
            int baseDataLength = reader.ReadInt32();
            byte[] baseData = reader.ReadBytes(baseDataLength);
            base.Deserialize(baseData);

            string optionsJson = reader.ReadString();
            _options = JsonConvert.DeserializeObject<AdagradOptimizerOptions<T, TInput, TOutput>>(optionsJson)
                ?? throw new InvalidOperationException("Failed to deserialize optimizer options.");
        }
    }

    /// <summary>
    /// Generates a unique key for caching gradients based on the model, input data, and Adagrad-specific parameters.
    /// </summary>
    /// <param name="model">The symbolic model.</param>
    /// <param name="X">The input feature matrix.</param>
    /// <param name="y">The target vector.</param>
    /// <returns>A string representing the unique gradient cache key.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a unique identifier for caching gradients. It combines the base cache key with
    /// Adagrad-specific parameters to ensure that cached gradients are only reused when all relevant factors are identical.
    /// </para>
    /// <para><b>For Beginners:</b> This is like creating a unique label for each set of calculations.
    /// 
    /// The label includes:
    /// - Information about the model and data (from the base class)
    /// - Specific settings of the Adagrad optimizer (initial learning rate and epsilon)
    /// 
    /// This helps the optimizer quickly find and reuse previous calculations when the same situation occurs again,
    /// which can save time and computational resources.
    /// </para>
    /// </remarks>
    protected override string GenerateGradientCacheKey(IFullModel<T, TInput, TOutput> model, TInput X, TOutput y)
    {
        var baseKey = base.GenerateGradientCacheKey(model, X, y);
        return $"{baseKey}_Adagrad_{_options.InitialLearningRate}_{_options.Epsilon}";
    }

    /// <summary>
    /// Reverses an Adagrad gradient update to recover original parameters.
    /// </summary>
    /// <remarks>
    /// <para>
    /// For Adagrad, the forward update is:
    /// 1. _accumulatedSquaredGradients[i] += gradient[i]^2
    /// 2. adaptiveLearningRate = learning_rate / (sqrt(_accumulatedSquaredGradients[i]) + epsilon)
    /// 3. params_new = params_old - adaptiveLearningRate * gradient
    ///
    /// To reverse: params_old = params_new + adaptiveLearningRate * gradient
    ///
    /// This requires access to the accumulated squared gradients to recalculate the adaptive learning rate.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like undoing a learning step. Given where the optimizer ended up (updated parameters)
    /// and its memory of past improvements (accumulated squared gradients), we can calculate
    /// the exact step that was taken and figure out where it started from.
    /// </para>
    /// </remarks>
    /// <param name="updatedParameters">Parameters after gradient application</param>
    /// <param name="appliedGradients">The gradients that were applied</param>
    /// <returns>Original parameters before the gradient update</returns>
    /// <exception cref="ArgumentNullException">If parameters or gradients are null</exception>
    /// <exception cref="ArgumentException">If parameter and gradient sizes do not match</exception>
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

        // If accumulated gradients are not initialized, fall back to vanilla SGD reversal
        if (_accumulatedSquaredGradients == null || _accumulatedSquaredGradients.Length != updatedParameters.Length)
        {
            return base.ReverseUpdate(updatedParameters, appliedGradients);
        }

        // === Vectorized Reverse Adagrad Update using IEngine (Phase B: US-GPU-015) ===
        // Recalculate the adaptive learning rates that were used
        var sqrtAccSqGrad = (Vector<T>)Engine.Sqrt(_accumulatedSquaredGradients);
        var epsilonVec = Vector<T>.CreateDefault(sqrtAccSqGrad.Length, NumOps.FromDouble(_options.Epsilon));
        var denominator = (Vector<T>)Engine.Add(sqrtAccSqGrad, epsilonVec);
        var currentLrVec = Vector<T>.CreateDefault(sqrtAccSqGrad.Length, CurrentLearningRate);
        var adaptiveLearningRates = (Vector<T>)Engine.Divide(currentLrVec, denominator);

        // Calculate the updates that were applied: adaptiveLr * gradient
        var updates = (Vector<T>)Engine.Multiply(adaptiveLearningRates, appliedGradients);

        // Reverse: params_old = params_new + updates
        return (Vector<T>)Engine.Add(updatedParameters, updates);
    }
}
