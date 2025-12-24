using Newtonsoft.Json;

namespace AiDotNet.Optimizers;

/// <summary>
/// Implements the AdaDelta optimization algorithm for training neural networks and other machine learning models.
/// </summary>
/// <remarks>
/// <para>
/// AdaDelta is an adaptive learning rate method that dynamically adjusts the learning rate for each parameter
/// based on a moving window of gradient updates. This optimizer addresses some of the drawbacks of AdaGrad,
/// particularly its aggressive, monotonically decreasing learning rate.
/// </para>
/// <para><b>For Beginners:</b> AdaDelta is like a smart assistant that helps your model learn more efficiently.
/// 
/// Imagine you're learning a new skill:
/// - Sometimes you need to practice more on difficult parts (bigger learning steps)
/// - Other times you need to be more careful with easier parts (smaller learning steps)
/// 
/// AdaDelta does this automatically for each part of your model, helping it learn better and faster.
/// It remembers recent changes and uses this information to decide how big the next learning step should be.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class AdaDeltaOptimizer<T, TInput, TOutput> : GradientBasedOptimizerBase<T, TInput, TOutput>
{
    /// <summary>
    /// The configuration options specific to the AdaDelta optimizer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the configuration parameters that control the behavior of the AdaDelta algorithm,
    /// such as rho (decay rate), epsilon (small constant for numerical stability), and adaptive settings.
    /// </para>
    /// <para><b>For Beginners:</b> This is like the instruction manual for our learning assistant.
    /// It contains all the settings that determine how the optimizer behaves, such as how much it
    /// remembers from previous steps and how it adapts over time.
    /// 
    /// These settings can be customized to make the optimizer work better for different types of problems.
    /// </para>
    /// </remarks>
    private AdaDeltaOptimizerOptions<T, TInput, TOutput> _options;

    /// <summary>
    /// Stores the exponential moving average of squared gradients for each parameter.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This vector accumulates the squared gradients for each parameter with an exponential decay,
    /// controlled by the rho parameter. It's used to normalize the parameter updates in the AdaDelta algorithm.
    /// </para>
    /// <para><b>For Beginners:</b> This is like the optimizer's memory of how much each part of the solution
    /// has been changing recently.
    /// 
    /// Imagine you're learning different subjects:
    /// - For each subject, this keeps track of how much your knowledge has been changing
    /// - It gives more weight to recent changes and gradually forgets older ones
    /// - This information helps determine how big your next learning step should be
    /// </para>
    /// </remarks>
    private Vector<T>? _accumulatedSquaredGradients;

    /// <summary>
    /// Stores the exponential moving average of squared parameter updates for each parameter.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This vector accumulates the squared parameter updates with an exponential decay,
    /// controlled by the rho parameter. It's used to adapt the effective learning rate in the AdaDelta algorithm,
    /// allowing the optimizer to proceed without requiring an explicit learning rate setting.
    /// </para>
    /// <para><b>For Beginners:</b> This is like keeping track of how big your recent learning steps have been
    /// for each part of what you're learning.
    /// 
    /// The optimizer uses this information to:
    /// - Adjust future step sizes based on how big previous steps were
    /// - Make learning more consistent across different parts of the model
    /// - Allow the model to learn effectively without needing to set a specific learning rate
    /// 
    /// This is one of the key features that makes AdaDelta special - it can adapt its learning
    /// process automatically based on past experience.
    /// </para>
    /// </remarks>
    private Vector<T>? _accumulatedSquaredUpdates;

    /// <summary>
    /// Stores the pre-update snapshot of accumulated squared gradients for accurate reverse updates.
    /// </summary>
    private Vector<T>? _previousAccumulatedSquaredGradients;

    /// <summary>
    /// Stores the pre-update snapshot of accumulated squared updates for accurate reverse updates.
    /// </summary>
    private Vector<T>? _previousAccumulatedSquaredUpdates;

    /// <summary>
    /// Initializes a new instance of the <see cref="AdaDeltaOptimizer{T}"/> class.
    /// </summary>
    /// <param name="model">The model to optimize.</param>
    /// <param name="options">The options for configuring the AdaDelta optimizer.</param>
    /// <remarks>
    /// <para>
    /// This constructor sets up the AdaDelta optimizer with the specified options and components.
    /// If no options are provided, default AdaDelta options are used.
    /// </para>
    /// <para><b>For Beginners:</b> This is like setting up your learning assistant (the optimizer) with specific instructions.
    ///
    /// You can customize how it works by providing different options and tools:
    /// - options: Special settings for AdaDelta (like how much it remembers from past steps)
    /// - predictionOptions and modelOptions: Rules for measuring how well the model is doing
    /// - modelEvaluator, fitDetector, fitnessCalculator: Different ways to check the model's performance
    /// - modelCache and gradientCache: Places to store information to speed up learning
    ///
    /// If you don't provide these, the optimizer will use default settings.
    /// </para>
    /// </remarks>
    public AdaDeltaOptimizer(
        IFullModel<T, TInput, TOutput> model,
        AdaDeltaOptimizerOptions<T, TInput, TOutput>? options = null,
        IEngine? engine = null)
        : base(model, options ?? new())
    {
        _options = options ?? new AdaDeltaOptimizerOptions<T, TInput, TOutput>();

        InitializeAdaptiveParameters();
    }

    /// <summary>
    /// Initializes the adaptive parameters for the AdaDelta optimizer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method sets up the initial learning rate based on the options provided.
    /// It's called during the optimizer's initialization.
    /// </para>
    /// <para><b>For Beginners:</b> This is like setting the starting point for how big the learning steps will be.
    /// 
    /// The initial learning rate is like deciding how big your first step will be when starting to learn something new.
    /// This method sets that initial step size based on the options you provided when creating the optimizer.
    /// </para>
    /// </remarks>
    protected override void InitializeAdaptiveParameters()
    {
        base.InitializeAdaptiveParameters();
        CurrentLearningRate = NumOps.FromDouble(_options.InitialLearningRate);
    }

    /// <summary>
    /// Performs the optimization process using the AdaDelta algorithm.
    /// </summary>
    /// <param name="inputData">The input data for optimization, including training data.</param>
    /// <returns>The result of the optimization process.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the main optimization loop of AdaDelta. It iteratively updates the model parameters
    /// using the AdaDelta update rule, evaluates the new solution, and checks for convergence or early stopping conditions.
    /// </para>
    /// <para><b>For Beginners:</b> This is the main learning process of the optimizer.
    /// 
    /// Here's what happens:
    /// 1. It starts with a random guess for the best solution
    /// 2. In each step (iteration):
    ///    - It calculates how to improve the current solution
    ///    - It updates the solution using the AdaDelta method
    ///    - It checks if the new solution is better than the previous best
    ///    - It decides whether to stop early if the solution is good enough
    /// 3. It repeats this process until it reaches the maximum number of steps or finds a good enough solution
    /// 
    /// This is like practicing a skill over and over, getting a little better each time, until you're satisfied with your performance.
    /// </para>
    /// </remarks>
    /// <remarks>
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
        var parameters = currentSolution.GetParameters();

        _accumulatedSquaredGradients = new Vector<T>(parameters.Length);
        _accumulatedSquaredUpdates = new Vector<T>(parameters.Length);
        InitializeAdaptiveParameters();

        for (int epoch = 0; epoch < _options.MaxIterations; epoch++)
        {
            NotifyEpochStart(epoch);
            var batcher = CreateBatcher(inputData, _options.BatchSize);

            foreach (var (xBatch, yBatch, batchIndices) in batcher.GetBatches())
            {
                var gradient = CalculateGradient(currentSolution, xBatch, yBatch);
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
    /// Updates the current solution using the AdaDelta update rule.
    /// </summary>
    /// <param name="currentSolution">The current solution (model parameters).</param>
    /// <param name="gradient">The computed gradient for the current solution.</param>
    /// <returns>A new solution with updated parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method applies the AdaDelta update rule to each parameter of the current solution.
    /// It uses accumulated squared gradients and updates to compute adaptive learning rates for each parameter.
    /// </para>
    /// <para><b>For Beginners:</b> This is where the actual learning happens for each part of the model.
    /// 
    /// For each parameter in the model:
    /// 1. It remembers how much this parameter has changed recently (accumulated squared gradients)
    /// 2. It calculates how much to change the parameter this time (update)
    /// 3. It remembers how big these changes have been (accumulated squared updates)
    /// 4. It applies the change to the parameter
    /// 
    /// This process helps the model learn more efficiently by adjusting bigger for parameters that need more change
    /// and smaller for those that need less change.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, TInput, TOutput> UpdateSolution(IFullModel<T, TInput, TOutput> currentSolution, Vector<T> gradient)
    {
        var parameters = currentSolution.GetParameters();

        // Initialize state vectors if needed
        if (_accumulatedSquaredGradients == null || _accumulatedSquaredUpdates == null || _accumulatedSquaredGradients.Length != parameters.Length)
        {
            _accumulatedSquaredGradients = new Vector<T>(parameters.Length);
            _accumulatedSquaredUpdates = new Vector<T>(parameters.Length);
        }

        // Use shared UpdateParameters method to eliminate duplication
        var newCoefficients = UpdateParameters(parameters, gradient);

        return currentSolution.WithParameters(newCoefficients);
    }

    /// <summary>
    /// Updates a vector of parameters using the AdaDelta optimization algorithm.
    /// </summary>
    /// <param name="parameters">The current parameter vector to be updated.</param>
    /// <param name="gradient">The gradient vector corresponding to the parameters.</param>
    /// <returns>The updated parameter vector.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the AdaDelta update rule by maintaining exponential moving averages of both
    /// squared gradients and squared updates. This allows AdaDelta to adapt the learning rate without
    /// requiring an explicit learning rate parameter.
    /// </para>
    /// <para><b>For Beginners:</b> AdaDelta automatically adjusts learning rates by remembering both
    /// how gradients have changed (squared gradients) and how parameters have been updated (squared updates).
    /// This makes it largely learning-rate-free, adapting automatically to the scale of the problem.
    /// </para>
    /// </remarks>
    public override Vector<T> UpdateParameters(Vector<T> parameters, Vector<T> gradient)
    {
        if (_accumulatedSquaredGradients == null || _accumulatedSquaredUpdates == null || _accumulatedSquaredGradients.Length != parameters.Length)
        {
            _accumulatedSquaredGradients = new Vector<T>(parameters.Length);
            _accumulatedSquaredUpdates = new Vector<T>(parameters.Length);
            _previousAccumulatedSquaredGradients = new Vector<T>(parameters.Length);
            _previousAccumulatedSquaredUpdates = new Vector<T>(parameters.Length);
        }

        // Save pre-update state for accurate reverse updates (element-wise copy required for reverse)
        if (_previousAccumulatedSquaredGradients == null || _previousAccumulatedSquaredUpdates == null)
        {
            _previousAccumulatedSquaredGradients = new Vector<T>(parameters.Length);
            _previousAccumulatedSquaredUpdates = new Vector<T>(parameters.Length);
        }

        // Vectorized copy of accumulated state
        _previousAccumulatedSquaredGradients = new Vector<T>(_accumulatedSquaredGradients);
        _previousAccumulatedSquaredUpdates = new Vector<T>(_accumulatedSquaredUpdates);

        // === Vectorized AdaDelta Update using IEngine (Phase B: US-GPU-015) ===
        T rho = NumOps.FromDouble(_options.Rho);
        T oneMinusRho = NumOps.FromDouble(1 - _options.Rho);
        T epsilon = NumOps.FromDouble(_options.Epsilon);

        // Update accumulated squared gradients: accSqGrad = rho * accSqGrad + (1 - rho) * gradient^2
        var gradSquared = (Vector<T>)Engine.Multiply(gradient, gradient);
        var rhoTimesAccSqGrad = (Vector<T>)Engine.Multiply(_accumulatedSquaredGradients, rho);
        var oneMinusRhoTimesGradSq = (Vector<T>)Engine.Multiply(gradSquared, oneMinusRho);
        _accumulatedSquaredGradients = (Vector<T>)Engine.Add(rhoTimesAccSqGrad, oneMinusRhoTimesGradSq);

        // Compute RMS of accumulated squared updates and gradients
        var epsilonVec = Vector<T>.CreateDefault(_accumulatedSquaredUpdates.Length, epsilon);
        var accSqUpdPlusEps = (Vector<T>)Engine.Add(_accumulatedSquaredUpdates, epsilonVec);
        var rmsUpdate = (Vector<T>)Engine.Sqrt(accSqUpdPlusEps);

        var accSqGradPlusEps = (Vector<T>)Engine.Add(_accumulatedSquaredGradients, epsilonVec);
        var rmsGrad = (Vector<T>)Engine.Sqrt(accSqGradPlusEps);

        // Compute update: update = (RMS[Δ] / RMS[g]) * gradient
        var ratio = (Vector<T>)Engine.Divide(rmsUpdate, rmsGrad);
        var update = (Vector<T>)Engine.Multiply(ratio, gradient);

        // Update accumulated squared updates: accSqUpd = rho * accSqUpd + (1 - rho) * update^2
        var updateSquared = (Vector<T>)Engine.Multiply(update, update);
        var rhoTimesAccSqUpd = (Vector<T>)Engine.Multiply(_accumulatedSquaredUpdates, rho);
        var oneMinusRhoTimesUpdSq = (Vector<T>)Engine.Multiply(updateSquared, oneMinusRho);
        _accumulatedSquaredUpdates = (Vector<T>)Engine.Add(rhoTimesAccSqUpd, oneMinusRhoTimesUpdSq);

        // Update parameters: params = params - update
        var updatedParams = (Vector<T>)Engine.Subtract(parameters, update);

        return updatedParams;
    }

    /// <summary>
    /// Reverses an AdaDelta gradient update to recover original parameters.
    /// </summary>
    /// <param name="updatedParameters">Parameters after AdaDelta update</param>
    /// <param name="appliedGradients">The gradients that were applied</param>
    /// <returns>Original parameters before the update</returns>
    /// <remarks>
    /// <para>
    /// AdaDelta's reverse update requires both accumulated squared gradients and accumulated squared updates
    /// from the forward pass. This method must be called immediately after UpdateParameters while both states are fresh.
    /// It recalculates the adaptive update that was applied based on the accumulated statistics.
    /// </para>
    /// <para><b>For Beginners:</b> This calculates where parameters were before an AdaDelta update.
    /// AdaDelta uses two pieces of memory: one for gradient history and one for update history.
    /// To reverse an update, we need both memories to reconstruct what step was taken. It's like
    /// rewinding a dance where each move depends on previous moves and the music (gradients).
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

        if (_previousAccumulatedSquaredGradients == null || _previousAccumulatedSquaredUpdates == null ||
            _previousAccumulatedSquaredGradients.Length != updatedParameters.Length)
        {
            throw new InvalidOperationException(
                "AdaDelta optimizer state is not initialized. ReverseUpdate must be called after UpdateParameters.");
        }

        if (_accumulatedSquaredGradients == null || _accumulatedSquaredUpdates == null)
        {
            throw new InvalidOperationException(
                "AdaDelta optimizer current state is not initialized.");
        }

        // === Vectorized Reverse AdaDelta Update using IEngine (Phase B: US-GPU-015) ===
        T rho = NumOps.FromDouble(_options.Rho);
        T oneMinusRho = NumOps.FromDouble(1 - _options.Rho);
        T epsilon = NumOps.FromDouble(_options.Epsilon);

        // Recalculate accumulated squared gradients at update time
        var rhoVec = Vector<T>.CreateDefault(appliedGradients.Length, rho);
        var oneMinusRhoVec = Vector<T>.CreateDefault(appliedGradients.Length, oneMinusRho);

        var gradSquared = (Vector<T>)Engine.Multiply(appliedGradients, appliedGradients);
        var rhoTimesPrevAccSqGrad = (Vector<T>)Engine.Multiply(_previousAccumulatedSquaredGradients, rhoVec);
        var oneMinusRhoTimesGradSq = (Vector<T>)Engine.Multiply(gradSquared, oneMinusRhoVec);
        var accGradAtUpdateTime = (Vector<T>)Engine.Add(rhoTimesPrevAccSqGrad, oneMinusRhoTimesGradSq);

        // Recalculate the update that was applied
        var epsilonVec = Vector<T>.CreateDefault(appliedGradients.Length, epsilon);
        var prevAccSqUpdPlusEps = (Vector<T>)Engine.Add(_previousAccumulatedSquaredUpdates, epsilonVec);
        var rmsUpdate = (Vector<T>)Engine.Sqrt(prevAccSqUpdPlusEps);

        var accGradPlusEps = (Vector<T>)Engine.Add(accGradAtUpdateTime, epsilonVec);
        var rmsGrad = (Vector<T>)Engine.Sqrt(accGradPlusEps);

        var ratio = (Vector<T>)Engine.Divide(rmsUpdate, rmsGrad);
        var update = (Vector<T>)Engine.Multiply(ratio, appliedGradients);

        // Reverse: original = updated + update
        var original = (Vector<T>)Engine.Add(updatedParameters, update);

        // Restore state so the rollback fully reverts the step
        _accumulatedSquaredGradients = new Vector<T>(_previousAccumulatedSquaredGradients);
        _accumulatedSquaredUpdates = new Vector<T>(_previousAccumulatedSquaredUpdates);

        return original;
    }

    /// <summary>
    /// Updates the adaptive parameters of the AdaDelta optimizer.
    /// </summary>
    /// <param name="currentStepData">The optimization step data for the current iteration.</param>
    /// <param name="previousStepData">The optimization step data for the previous iteration.</param>
    /// <remarks>
    /// <para>
    /// This method updates the adaptive parameters of the AdaDelta optimizer, specifically the rho value
    /// if adaptive rho is enabled in the options.
    /// </para>
    /// <para><b>For Beginners:</b> This method adjusts how the optimizer learns over time.
    /// 
    /// If adaptive rho is turned on:
    /// - If the current solution is better than the previous one, it slightly increases rho
    /// - If the current solution is worse, it slightly decreases rho
    /// 
    /// Rho controls how much the optimizer remembers from past steps. Adjusting it helps the optimizer
    /// adapt to the current state of learning, potentially making it more efficient.
    /// </para>
    /// </remarks>
    protected override void UpdateAdaptiveParameters(OptimizationStepData<T, TInput, TOutput> currentStepData, OptimizationStepData<T, TInput, TOutput> previousStepData)
    {
        base.UpdateAdaptiveParameters(currentStepData, previousStepData);

        if (_options.UseAdaptiveRho)
        {
            if (NumOps.GreaterThan(currentStepData.FitnessScore, previousStepData.FitnessScore))
            {
                _options.Rho = Math.Min(_options.Rho * _options.RhoIncreaseFactor, _options.MaxRho);
            }
            else
            {
                _options.Rho = Math.Max(_options.Rho * _options.RhoDecreaseFactor, _options.MinRho);
            }
        }
    }

    /// <summary>
    /// Updates the optimizer options.
    /// </summary>
    /// <param name="options">The new options to set.</param>
    /// <exception cref="ArgumentException">Thrown when the provided options are not of type AdaDeltaOptimizerOptions.</exception>
    /// <remarks>
    /// <para>
    /// This method updates the optimizer's options with new settings. It ensures that only
    /// AdaDeltaOptimizerOptions are used with this optimizer.
    /// </para>
    /// <para><b>For Beginners:</b> This is like changing the settings on your learning assistant.
    /// 
    /// You can use this to adjust how the optimizer works, but you need to make sure you're
    /// using the right type of settings (AdaDeltaOptimizerOptions). If you try to use the wrong
    /// type of settings, it will give you an error message.
    /// </para>
    /// </remarks>
    protected override void UpdateOptions(OptimizationAlgorithmOptions<T, TInput, TOutput> options)
    {
        if (options is AdaDeltaOptimizerOptions<T, TInput, TOutput> adaDeltaOptions)
        {
            _options = adaDeltaOptions;
        }
        else
        {
            throw new ArgumentException("Invalid options type. Expected AdaDeltaOptimizerOptions.");
        }
    }

    /// <summary>
    /// Gets the current optimizer options.
    /// </summary>
    /// <returns>The current AdaDeltaOptimizerOptions.</returns>
    /// <remarks>
    /// <para>
    /// This method returns the current options used by the AdaDelta optimizer.
    /// </para>
    /// <para><b>For Beginners:</b> This is like checking the current settings of your learning assistant.
    /// 
    /// You can use this to see how the optimizer is currently configured, which can be helpful
    /// if you want to understand its behavior or make changes.
    /// </para>
    /// </remarks>
    public override OptimizationAlgorithmOptions<T, TInput, TOutput> GetOptions()
    {
        return _options;
    }

    /// <summary>
    /// Serializes the AdaDelta optimizer to a byte array.
    /// </summary>
    /// <returns>A byte array representing the serialized optimizer.</returns>
    /// <remarks>
    /// <para>
    /// This method converts the optimizer's state, including its base class state and options,
    /// into a byte array that can be stored or transmitted.
    /// </para>
    /// <para><b>For Beginners:</b> This is like packing up the optimizer into a compact form.
    /// 
    /// Imagine you're packing a suitcase:
    /// 1. You pack the basic stuff (base class data)
    /// 2. You write down how much basic stuff you packed
    /// 3. You pack your special AdaDelta stuff (options)
    /// 
    /// This packed form can be saved or sent somewhere else, and later unpacked to recreate
    /// the optimizer exactly as it was.
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
    /// Deserializes the AdaDelta optimizer from a byte array.
    /// </summary>
    /// <param name="data">The byte array containing the serialized optimizer data.</param>
    /// <exception cref="InvalidOperationException">Thrown when deserialization of optimizer options fails.</exception>
    /// <remarks>
    /// <para>
    /// This method reconstructs the optimizer's state from a byte array, including its base class state and options.
    /// </para>
    /// <para><b>For Beginners:</b> This is like unpacking the optimizer from its compact form.
    /// 
    /// Continuing the suitcase analogy:
    /// 1. You check how much basic stuff was packed
    /// 2. You unpack the basic stuff (base class data)
    /// 3. You unpack and set up your special AdaDelta stuff (options)
    /// 
    /// If there's a problem unpacking the special stuff, it will let you know with an error message.
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
            _options = JsonConvert.DeserializeObject<AdaDeltaOptimizerOptions<T, TInput, TOutput>>(optionsJson)
                ?? throw new InvalidOperationException("Failed to deserialize optimizer options.");
        }
    }

    /// <summary>
    /// Generates a unique key for caching gradients.
    /// </summary>
    /// <param name="model">The symbolic model.</param>
    /// <param name="X">The input data matrix.</param>
    /// <param name="y">The target values vector.</param>
    /// <returns>A string representing the unique gradient cache key.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a unique identifier for caching gradients based on the model, input data,
    /// and specific AdaDelta parameters.
    /// </para>
    /// <para><b>For Beginners:</b> This is like creating a special label for each set of calculations.
    /// 
    /// Imagine you're organizing your homework:
    /// - You start with a basic label (from the base class)
    /// - Then you add specific information about this AdaDelta optimizer (rho and epsilon values)
    /// 
    /// This helps the optimizer quickly find and reuse calculations it has done before,
    /// which can make the learning process faster.
    /// </para>
    /// </remarks>
    protected override string GenerateGradientCacheKey(IFullModel<T, TInput, TOutput> model, TInput X, TOutput y)
    {
        var baseKey = base.GenerateGradientCacheKey(model, X, y);
        return $"{baseKey}_AdaDelta_{_options.Rho}_{_options.Epsilon}";
    }
}
