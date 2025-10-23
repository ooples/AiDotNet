using Newtonsoft.Json;

/// <summary>
/// Implements the Root Mean Square Propagation (RMSProp) optimization algorithm, an adaptive learning rate method.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// RMSProp is an adaptive learning rate optimization algorithm designed to handle non-stationary
/// objectives and accelerate convergence. It maintains a moving average of the squared gradients
/// for each parameter and divides the learning rate by the square root of this average. This
/// approach allows the algorithm to use a larger learning rate for parameters with small gradients
/// and a smaller learning rate for parameters with large gradients, leading to more efficient optimization.
/// </para>
/// <para><b>For Beginners:</b> RMSProp is like a hiker who adjusts their step size differently for each direction.
/// 
/// Imagine a hiker exploring mountains with different terrains:
/// - On steep slopes (large gradients), the hiker takes small, careful steps
/// - On gentle slopes (small gradients), the hiker takes larger, confident steps
/// - The hiker remembers how steep each direction has been recently (using a moving average)
/// - This memory helps the hiker adjust their steps even as the terrain changes
/// 
/// This adaptive approach helps the algorithm find good solutions more quickly by:
/// - Preventing wild overshooting on steep slopes
/// - Making faster progress on gentle terrain
/// - Adjusting automatically to different parts of the solution space
/// </para>
/// </remarks>
public class RootMeanSquarePropagationOptimizer<T, TInput, TOutput> : GradientBasedOptimizerBase<T, TInput, TOutput>
{
    /// <summary>
    /// Moving average of squared gradients for each parameter.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores a running average of the squared gradients for each parameter in the model.
    /// It is used to adapt the learning rate individually for each parameter, with larger accumulated
    /// squared gradients resulting in smaller effective learning rates.
    /// </para>
    /// <para><b>For Beginners:</b> This is like the hiker's memory of how steep each direction has been.
    /// 
    /// This moving average:
    /// - Keeps track of the squared gradient (steepness) for each direction
    /// - Gives more weight to recent observations and gradually forgets older ones
    /// - Helps determine how cautious to be when stepping in each direction
    /// - A consistently steep direction will have a large value, signaling the need for smaller steps
    /// 
    /// This adaptive memory allows the algorithm to respond differently to different parameters based on their history.
    /// </para>
    /// </remarks>
    private Vector<T> _squaredGradient;

    /// <summary>
    /// The current iteration count of the optimization process.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field keeps track of the number of completed iterations in the optimization process.
    /// It is used for gradient caching and can be useful for monitoring the progress of the optimization.
    /// </para>
    /// <para><b>For Beginners:</b> This is like counting how many steps the hiker has taken.
    /// 
    /// The iteration counter:
    /// - Keeps track of how many rounds of optimization have been completed
    /// - Helps with creating unique cache keys for gradient calculations
    /// - Can be used to monitor how the algorithm is progressing
    /// 
    /// This simple counter plays an important role in the optimization process.
    /// </para>
    /// </remarks>
    private int _t;

    /// <summary>
    /// Configuration options specific to the RMSProp algorithm.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the configuration parameters for the RMSProp algorithm, such as
    /// the decay rate for the moving average, epsilon for numerical stability, and the
    /// maximum number of iterations. These parameters control the behavior of the optimizer
    /// and affect its performance and convergence properties.
    /// </para>
    /// <para><b>For Beginners:</b> This is like the instruction manual for the optimizer.
    /// 
    /// The options control:
    /// - How quickly the algorithm forgets old gradient information (decay rate)
    /// - How to prevent division by very small numbers (epsilon)
    /// - When to stop the optimization process (maximum iterations, tolerance)
    /// 
    /// Adjusting these settings can help the algorithm work better for different types of problems.
    /// </para>
    /// </remarks>
    private RootMeanSquarePropagationOptimizerOptions<T, TInput, TOutput> _options;

    /// <summary>
    /// Initializes a new instance of the <see cref="RootMeanSquarePropagationOptimizer{T}"/> class with the specified options and components.
    /// </summary>
    /// <param name="options">The RMSProp optimization options, or null to use default options.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new RMSProp optimizer with the specified options and components.
    /// If any parameter is null, a default implementation is used. The constructor initializes
    /// the iteration counter, squared gradient vector, and options.
    /// </para>
    /// <para><b>For Beginners:</b> This is the starting point for creating a new optimizer.
    /// 
    /// Think of it like preparing for a hiking expedition:
    /// - You can provide custom settings (options) or use the default ones
    /// - You can provide specialized tools (evaluators, calculators) or use the basic ones
    /// - It initializes everything the optimizer needs to start working
    /// - The squared gradient starts empty because there's no history yet
    /// - The step counter starts at zero because no steps have been taken
    /// 
    /// This constructor gets everything ready so you can start the optimization process.
    /// </para>
    /// </remarks>
    public RootMeanSquarePropagationOptimizer(
        RootMeanSquarePropagationOptimizerOptions<T, TInput, TOutput>? options = null)
        : base(options ?? new())
    {
        _t = 0;
        _squaredGradient = Vector<T>.Empty();
        _options = options ?? new();
    }

    /// <summary>
    /// Performs the RMSProp optimization to find the best solution for the given input data.
    /// </summary>
    /// <param name="inputData">The input data to optimize against.</param>
    /// <returns>An optimization result containing the best solution found and associated metrics.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the main RMSProp algorithm. It starts from a random solution and
    /// iteratively improves it by calculating the gradient, applying momentum, updating the solution
    /// based on the adaptive learning rates, and evaluating the new solution. The process continues
    /// until either the maximum number of iterations is reached, early stopping criteria are met,
    /// or the improvement falls below the specified tolerance.
    /// </para>
    /// <para><b>For Beginners:</b> This is the main search process where the algorithm looks for the best solution.
    /// 
    /// The process works like this:
    /// 1. Start at a random position on the "landscape"
    /// 2. Initialize the squared gradient history and step counter
    /// 3. For each iteration:
    ///    - Figure out which direction is most uphill (calculate gradient)
    ///    - Apply momentum to smooth the movement
    ///    - Take a step using adaptive step sizes for each direction
    ///    - Check if the new position is better than the best found so far
    ///    - Update the adaptive parameters based on progress
    /// 4. Stop when enough iterations are done, when no more improvement is happening, or when the
    ///    improvement is very small
    /// 
    /// This approach efficiently finds good solutions by adapting its behavior based on the shape
    /// of the optimization landscape.
    /// </para>
    /// </remarks>
    public override OptimizationResult<T, TInput, TOutput> Optimize(OptimizationInputData<T, TInput, TOutput> inputData)
    {
        ValidationHelper<T>.ValidateInputData(inputData);

        var currentSolution = InitializeRandomSolution(inputData.XTrain);
        var bestStepData = new OptimizationStepData<T, TInput, TOutput>();
        var previousStepData = new OptimizationStepData<T, TInput, TOutput>();

        _squaredGradient = new Vector<T>(currentSolution.GetParameters().Length);
        _t = 0;
        InitializeAdaptiveParameters();

        for (int iteration = 0; iteration < _options.MaxIterations; iteration++)
        {
            _t++;
            var gradient = CalculateGradient(currentSolution, inputData.XTrain, inputData.YTrain);
            gradient = ApplyMomentum(gradient);
            var newSolution = UpdateSolution(currentSolution, gradient);

            var currentStepData = EvaluateSolution(newSolution, inputData);
            UpdateBestSolution(currentStepData, ref bestStepData);

            UpdateAdaptiveParameters(currentStepData, previousStepData);

            if (UpdateIterationHistoryAndCheckEarlyStopping(iteration, bestStepData))
            {
                break;
            }

            if (NumOps.LessThan(NumOps.Abs(NumOps.Subtract(bestStepData.FitnessScore, currentStepData.FitnessScore)), NumOps.FromDouble(_options.Tolerance)))
            {
                break;
            }

            currentSolution = newSolution;
            previousStepData = currentStepData;
        }

        return CreateOptimizationResult(bestStepData, inputData);
    }

    /// <summary>
    /// Updates a vector of parameters using the RMSProp algorithm.
    /// </summary>
    /// <param name="parameters">The parameters to update.</param>
    /// <param name="gradient">The gradient vector for the parameters.</param>
    /// <returns>The updated parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the core RMSProp update rule. For each parameter, it:
    /// 1. Updates the running average of squared gradients
    /// 2. Calculates an adaptive learning rate by dividing the base learning rate by the square root
    ///    of the running average (plus epsilon for numerical stability)
    /// 3. Updates the parameter by subtracting the product of the adaptive learning rate and the gradient
    /// </para>
    /// <para><b>For Beginners:</b> This method adjusts each parameter based on its gradient history.
    /// 
    /// For each parameter:
    /// - It updates the memory of how steep this direction has been (squared gradient)
    /// - It calculates a custom step size based on the steepness history
    /// - Parameters with consistently large gradients get smaller steps
    /// - Parameters with consistently small gradients get larger steps
    /// - It then updates the parameter value using this custom step size
    /// 
    /// This adaptive approach helps the algorithm converge faster by giving each parameter
    /// exactly the step size it needs.
    /// </para>
    /// </remarks>
    public override Vector<T> UpdateParameters(Vector<T> parameters, Vector<T> gradient)
    {
        for (int i = 0; i < parameters.Length; i++)
        {
            var squaredGrad = NumOps.Multiply(gradient[i], gradient[i]);
            _squaredGradient[i] = NumOps.Add(NumOps.Multiply(NumOps.FromDouble(_options.Decay), _squaredGradient[i]), NumOps.Multiply(NumOps.FromDouble(1 - _options.Decay), squaredGrad));
            
            var adaptiveLearningRate = CurrentLearningRate;
            var denominator = NumOps.Add(NumOps.Sqrt(_squaredGradient[i]), NumOps.FromDouble(_options.Epsilon));
            var update = NumOps.Divide(NumOps.Multiply(adaptiveLearningRate, gradient[i]), denominator);
            
            parameters[i] = NumOps.Subtract(parameters[i], update);
        }

        return parameters;
    }

    /// <summary>
    /// Updates a solution model using the RMSProp algorithm.
    /// </summary>
    /// <param name="currentSolution">The current solution model to update.</param>
    /// <param name="gradient">The gradient vector for the solution.</param>
    /// <returns>The updated solution model.</returns>
    /// <remarks>
    /// <para>
    /// This method applies the RMSProp update rule to the coefficients of a solution model.
    /// It follows the same steps as UpdateVector, but operates directly on the solution model's
    /// coefficients. For each coefficient, it:
    /// 1. Updates the running average of squared gradients
    /// 2. Calculates an adaptive learning rate by dividing the base learning rate by the square root
    ///    of the running average (plus epsilon for numerical stability)
    /// 3. Updates the coefficient by subtracting the product of the adaptive learning rate and the gradient
    /// </para>
    /// <para><b>For Beginners:</b> This method moves the solution in the direction of improvement.
    /// 
    /// Think of it as the hiker taking one step:
    /// - For each direction, it updates the memory of how steep that direction has been
    /// - It calculates custom step sizes for each direction based on their history
    /// - Steeper directions get smaller, more careful steps
    /// - Gentler directions get larger, more confident steps
    /// - The solution then moves according to these personalized step sizes
    /// 
    /// This adaptive movement helps the algorithm navigate efficiently toward better solutions.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, TInput, TOutput> UpdateSolution(IFullModel<T, TInput, TOutput> currentSolution, Vector<T> gradient)
    {
        var parameters = currentSolution.GetParameters();
        for (int i = 0; i < gradient.Length; i++)
        {
            var gradientSquared = NumOps.Multiply(gradient[i], gradient[i]);
            _squaredGradient[i] = NumOps.Add(
                NumOps.Multiply(NumOps.FromDouble(_options.Decay), _squaredGradient[i]),
                NumOps.Multiply(NumOps.FromDouble(1 - _options.Decay), gradientSquared)
            );

            var update = NumOps.Divide(
                NumOps.Multiply(CurrentLearningRate, gradient[i]),
                NumOps.Add(NumOps.Sqrt(_squaredGradient[i]), NumOps.FromDouble(_options.Epsilon))
            );

            parameters[i] = NumOps.Subtract(parameters[i], update);
        }

        return currentSolution;
    }

    /// <summary>
    /// Generates a unique key for caching gradients based on the model, input data, and optimizer state.
    /// </summary>
    /// <param name="model">The model for which the gradient is calculated.</param>
    /// <param name="X">The input features matrix.</param>
    /// <param name="y">The target values vector.</param>
    /// <returns>A string key that uniquely identifies this gradient calculation.</returns>
    /// <remarks>
    /// <para>
    /// This method overrides the base implementation to include RMSProp-specific information in the cache key.
    /// It extends the base key with information about the current learning rate, decay rate, epsilon value,
    /// and iteration count. This ensures that gradients are properly cached and retrieved even as the
    /// optimizer's state changes.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates a unique identification tag for each gradient calculation.
    /// 
    /// Think of it like a file naming system:
    /// - It includes information about the model and data being used
    /// - It adds details specific to the RMSProp optimizer's current state
    /// - This unique tag helps the optimizer avoid redundant calculations
    /// - If the same gradient is needed again, it can be retrieved from cache instead of recalculated
    /// 
    /// This caching mechanism improves efficiency by avoiding duplicate work.
    /// </para>
    /// </remarks>
    protected override string GenerateGradientCacheKey(IFullModel<T, TInput, TOutput> model, TInput X, TOutput y)
    {
        var baseKey = base.GenerateGradientCacheKey(model, X, y);
        return $"{baseKey}_RMSprop_{CurrentLearningRate}_{_options.Decay}_{_options.Epsilon}_{_t}";
    }

    /// <summary>
    /// Resets the optimizer to its initial state.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method overrides the base implementation to reset RMSProp-specific state variables
    /// in addition to the base state. It resets the iteration counter and clears the squared
    /// gradient history, preparing the optimizer for a fresh start.
    /// </para>
    /// <para><b>For Beginners:</b> This method prepares the optimizer to start fresh.
    /// 
    /// It's like a hiker:
    /// - Returning to the starting point
    /// - Resetting their step counter to zero
    /// - Clearing their memory of previous terrain steepness
    /// 
    /// This allows the optimizer to begin a new optimization process without being influenced
    /// by previous runs.
    /// </para>
    /// </remarks>
    public override void Reset()
    {
        base.Reset();
        _t = 0;
        _squaredGradient = Vector<T>.Empty();
    }

    /// <summary>
    /// Gets the current options for this optimizer.
    /// </summary>
    /// <returns>The current RMSProp optimization options.</returns>
    /// <remarks>
    /// <para>
    /// This method overrides the base implementation to return the RMSProp-specific options.
    /// </para>
    /// <para><b>For Beginners:</b> This method returns the current settings of the optimizer.
    /// 
    /// It's like checking what settings are currently active:
    /// - You can see the current decay rate
    /// - You can see the current epsilon value
    /// - You can see all the other parameters that control the optimizer
    /// 
    /// This is useful for understanding how the optimizer is currently configured
    /// or for making a copy of the settings to modify and apply later.
    /// </para>
    /// </remarks>
    public override OptimizationAlgorithmOptions<T, TInput, TOutput> GetOptions()
    {
        return _options;
    }

    /// <summary>
    /// Serializes the RMSProp optimizer to a byte array for storage or transmission.
    /// </summary>
    /// <returns>A byte array containing the serialized optimizer.</returns>
    /// <remarks>
    /// <para>
    /// This method overrides the base implementation to include RMSProp-specific information in the serialization.
    /// It first serializes the base class data, then adds the iteration count, squared gradient vector, and options.
    /// </para>
    /// <para><b>For Beginners:</b> This method saves the current state of the optimizer so it can be restored later.
    /// 
    /// It's like taking a snapshot of the optimizer:
    /// - First, it saves all the general optimizer information
    /// - Then, it saves the RMSProp-specific state and settings
    /// - It packages everything into a format that can be saved to a file or sent over a network
    /// 
    /// This allows you to:
    /// - Save a trained optimizer to use later
    /// - Share an optimizer with others
    /// - Create a backup before making changes
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

        // Serialize RootMeanSquarePropagationOptimizer specific data
        writer.Write(_t);
        writer.Write(JsonConvert.SerializeObject(_squaredGradient));
        writer.Write(JsonConvert.SerializeObject(_options));

        return ms.ToArray();
    }

    /// <summary>
    /// Reconstructs the RMSProp optimizer from a serialized byte array.
    /// </summary>
    /// <param name="data">The byte array containing the serialized optimizer.</param>
    /// <exception cref="InvalidOperationException">Thrown when the data cannot be deserialized.</exception>
    /// <remarks>
    /// <para>
    /// This method overrides the base implementation to handle RMSProp-specific information during deserialization.
    /// It first deserializes the base class data, then reconstructs the iteration count, squared gradient vector,
    /// and options.
    /// </para>
    /// <para><b>For Beginners:</b> This method restores the optimizer from a previously saved state.
    /// 
    /// It's like restoring from a snapshot:
    /// - First, it loads all the general optimizer information
    /// - Then, it loads the RMSProp-specific state and settings
    /// - It reconstructs the optimizer to the exact state it was in when saved
    /// 
    /// This allows you to:
    /// - Continue working with an optimizer you previously saved
    /// - Use an optimizer that someone else created and shared
    /// - Revert to a backup if needed
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

        // Deserialize RootMeanSquarePropagationOptimizer specific data
        _t = reader.ReadInt32();
        _squaredGradient = JsonConvert.DeserializeObject<Vector<T>>(reader.ReadString())
            ?? throw new InvalidOperationException("Failed to deserialize _squaredGradient.");
        _options = JsonConvert.DeserializeObject<RootMeanSquarePropagationOptimizerOptions<T, TInput, TOutput>>(reader.ReadString())
            ?? throw new InvalidOperationException("Failed to deserialize _options.");
    }
}