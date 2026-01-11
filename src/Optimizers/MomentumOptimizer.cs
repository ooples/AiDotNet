using AiDotNet.Tensors.Engines.DirectGpu;
using Newtonsoft.Json;

namespace AiDotNet.Optimizers;

/// <summary>
/// Implements the Momentum optimization algorithm for gradient-based optimization.
/// </summary>
/// <remarks>
/// <para>
/// The Momentum optimizer is an extension of gradient descent that helps accelerate the optimization process
/// in relevant directions and dampens oscillations. It does this by adding a fraction of the update vector
/// of the past time step to the current update vector.
/// </para>
/// <para><b>For Beginners:</b>
/// Imagine you're rolling a ball down a hill to find the lowest point. The Momentum optimizer is like giving
/// that ball some "memory" of its previous movements. This helps it move faster in consistent directions and
/// resist getting stuck in small bumps or divots along the way.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class MomentumOptimizer<T, TInput, TOutput> : GradientBasedOptimizerBase<T, TInput, TOutput>
{
    /// <summary>
    /// The configuration options specific to the Momentum optimizer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the configuration parameters that control the behavior of the Momentum algorithm,
    /// such as initial momentum value, learning rate, and adaptive parameter settings.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like the instruction manual for your ball-rolling experiment. It contains all the settings
    /// that determine how the ball behaves - how fast it starts rolling (learning rate), how much it remembers
    /// its previous direction (momentum), and whether these properties change automatically during the experiment.
    /// </para>
    /// </remarks>
    private MomentumOptimizerOptions<T, TInput, TOutput> _options;

    /// <summary>
    /// Stores the current velocity vector for each parameter in the model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The velocity vector represents the direction and magnitude of parameter updates in the previous iteration.
    /// It's used to add momentum to the optimization process, helping to accelerate convergence and overcome
    /// local minima.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like keeping track of how fast and in which direction your ball is currently rolling.
    /// For each part of your model (each parameter), it remembers both the speed and direction of movement.
    /// This "memory" helps the ball maintain its course through small bumps and roll faster in consistent directions.
    /// </para>
    /// </remarks>
    private Vector<T>? _velocity;

    /// <summary>
    /// Initializes a new instance of the MomentumOptimizer class.
    /// </summary>
    /// <param name="model">The model to optimize.</param>
    /// <param name="options">The options for configuring the Momentum optimizer.</param>
    /// <remarks>
    /// <para>
    /// This constructor sets up the optimizer with the provided options and dependencies. If no options are provided,
    /// it uses default settings.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like setting up your ball-rolling experiment. You're deciding on the properties of the ball
    /// (like its size and bounciness) and the hill (like its steepness and texture).
    /// </para>
    /// </remarks>
    public MomentumOptimizer(
        IFullModel<T, TInput, TOutput> model,
        MomentumOptimizerOptions<T, TInput, TOutput>? options = null)
        : base(model, options ?? new())
    {
        _options = options ?? new MomentumOptimizerOptions<T, TInput, TOutput>();
        InitializeAdaptiveParameters();
    }

    /// <summary>
    /// Initializes adaptive parameters for the optimization process.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method sets up the initial learning rate and momentum for the optimization process based on the options provided.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like setting the initial speed of your ball (learning rate) and how much it remembers its previous
    /// movements (momentum) before you start rolling it down the hill.
    /// </para>
    /// </remarks>
    protected override void InitializeAdaptiveParameters()
    {
        base.InitializeAdaptiveParameters();
        CurrentLearningRate = NumOps.FromDouble(_options.InitialLearningRate);
        CurrentMomentum = NumOps.FromDouble(_options.InitialMomentum);
    }

    /// <summary>
    /// Performs the optimization process using the Momentum algorithm.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method implements the main optimization loop. It iterates through the data, calculating gradients,
    /// updating the velocity (momentum), and adjusting the model parameters accordingly.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is the actual process of rolling the ball down the hill. In each step, you're calculating which way
    /// the ball should roll (gradient), how fast it's moving (velocity), and where it ends up (new solution).
    /// You keep doing this until the ball finds the lowest point or you've rolled it enough times.
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
        ValidationHelper<T>.ValidateInputData(inputData);

        // Initialize with random solution
        var currentSolution = InitializeRandomSolution(inputData.XTrain);
        var bestStepData = new OptimizationStepData<T, TInput, TOutput>();
        var previousStepData = PrepareAndEvaluateSolution(currentSolution, inputData);

        _velocity = new Vector<T>(currentSolution.GetParameters().Length);
        InitializeAdaptiveParameters();

        for (int epoch = 0; epoch < _options.MaxIterations; epoch++)
        {
            // Notify sampler of new epoch (for curriculum/self-paced learning)
            NotifyEpochStart(epoch);

            // Create batcher for the current epoch using DataLoader infrastructure
            var batcher = CreateBatcher(inputData, _options.BatchSize);

            foreach (var (xBatch, yBatch, batchIndices) in batcher.GetBatches())
            {
                // Calculate gradient on the batch
                var gradient = CalculateGradient(currentSolution, xBatch, yBatch);

                // Update velocity with momentum
                _velocity = UpdateVelocity(gradient);

                // Update solution
                var newSolution = UpdateSolution(currentSolution, _velocity);

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
    /// Updates the velocity vector based on the current gradient and momentum.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method calculates the new velocity by combining the previous velocity (scaled by the momentum)
    /// with the current gradient (scaled by the learning rate).
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like updating how fast and in what direction your ball is rolling. It takes into account
    /// both where the ball was heading before (momentum) and the current slope it's on (gradient).
    /// </para>
    /// </remarks>
    /// <param name="gradient">The current gradient vector.</param>
    /// <returns>The updated velocity vector.</returns>
    private Vector<T> UpdateVelocity(Vector<T> gradient)
    {
        // === Vectorized Momentum Update using IEngine ===
        // Phase B: US-GPU-015 - GPU-accelerated gradient updates
        // velocity = momentum * velocity + learningRate * gradient

        var momentumScaled = (Vector<T>)Engine.Multiply(_velocity!, CurrentMomentum);
        var gradientScaled = (Vector<T>)Engine.Multiply(gradient, CurrentLearningRate);
        _velocity = (Vector<T>)Engine.Add(momentumScaled, gradientScaled);

        return _velocity;
    }

    /// <summary>
    /// Updates the current solution based on the calculated velocity.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method applies the velocity to the current solution, adjusting each coefficient accordingly.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like determining the ball's new position after it has rolled. You're using the ball's
    /// speed and direction (velocity) to figure out where it ends up.
    /// </para>
    /// </remarks>
    /// <param name="currentSolution">The current model solution.</param>
    /// <param name="velocity">The current velocity vector.</param>
    /// <returns>An updated symbolic model with improved coefficients.</returns>
    protected override IFullModel<T, TInput, TOutput> UpdateSolution(IFullModel<T, TInput, TOutput> currentSolution, Vector<T> velocity)
    {
        var parameters = currentSolution.GetParameters();

        // === Vectorized Update using IEngine ===
        // Phase B: US-GPU-015 - GPU-accelerated parameter updates
        // params = params - velocity
        var newCoefficients = (Vector<T>)Engine.Subtract(parameters, velocity);

        return currentSolution.WithParameters(newCoefficients);
    }

    /// <summary>
    /// Updates a vector of parameters using the Momentum optimization algorithm.
    /// </summary>
    /// <param name="parameters">The current parameter vector to be updated.</param>
    /// <param name="gradient">The gradient vector corresponding to the parameters.</param>
    /// <returns>The updated parameter vector.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the Momentum update rule by maintaining a velocity vector that accumulates
    /// a weighted average of past gradients. The velocity combines the previous velocity (scaled by momentum)
    /// with the current gradient (scaled by learning rate).
    /// </para>
    /// <para><b>For Beginners:</b> This method applies Momentum to adjust parameters. Like a ball rolling
    /// down a hill, it remembers its previous direction and speed (velocity) and combines it with the
    /// current slope (gradient) to determine where to go next. This helps the optimizer move faster
    /// in consistent directions and resist getting stuck in small bumps.
    /// </para>
    /// </remarks>
    public override Vector<T> UpdateParameters(Vector<T> parameters, Vector<T> gradient)
    {
        if (_velocity == null || _velocity.Length != parameters.Length)
        {
            _velocity = new Vector<T>(parameters.Length);
        }

        // === Vectorized Momentum Update using IEngine ===
        // Phase B: US-GPU-015 - GPU-accelerated gradient updates

        // Update velocity: velocity = momentum * velocity + learningRate * gradient
        var momentumScaled = (Vector<T>)Engine.Multiply(_velocity, CurrentMomentum);
        var gradientScaled = (Vector<T>)Engine.Multiply(gradient, CurrentLearningRate);
        _velocity = (Vector<T>)Engine.Add(momentumScaled, gradientScaled);

        // Update parameters: params = params - velocity
        var updatedParams = (Vector<T>)Engine.Subtract(parameters, _velocity);

        return updatedParams;
    }


    /// <summary>
    /// Updates the adaptive parameters of the optimizer based on the current and previous optimization steps.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method adjusts the learning rate and momentum based on the performance of the current step compared to the previous step.
    /// If improvement is seen, the learning rate and momentum may be increased, otherwise they may be decreased.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like adjusting how you roll the ball based on how well you're doing. If you're getting closer to the bottom of the hill,
    /// you might roll the ball a bit faster or give it more momentum. If you're not improving, you might slow down or reduce the momentum
    /// to be more careful.
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

        if (_options.UseAdaptiveMomentum)
        {
            if (NumOps.GreaterThan(currentStepData.FitnessScore, previousStepData.FitnessScore))
            {
                CurrentMomentum = NumOps.Multiply(CurrentMomentum, NumOps.FromDouble(_options.MomentumIncreaseFactor));
            }
            else
            {
                CurrentMomentum = NumOps.Multiply(CurrentMomentum, NumOps.FromDouble(_options.MomentumDecreaseFactor));
            }

            CurrentMomentum = MathHelper.Max(NumOps.FromDouble(_options.MinMomentum),
                MathHelper.Min(NumOps.FromDouble(_options.MaxMomentum), CurrentMomentum));
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
    /// This is like changing the rules of how you're rolling the ball mid-experiment. It makes sure you're only
    /// using rules that work for this specific type of ball-rolling (Momentum optimization).
    /// </para>
    /// </remarks>
    /// <param name="options">The new options to be applied to the optimizer.</param>
    /// <exception cref="ArgumentException">Thrown when the provided options are not of the correct type.</exception>
    protected override void UpdateOptions(OptimizationAlgorithmOptions<T, TInput, TOutput> options)
    {
        if (options is MomentumOptimizerOptions<T, TInput, TOutput> momentumOptions)
        {
            _options = momentumOptions;
        }
        else
        {
            throw new ArgumentException("Invalid options type. Expected MomentumOptimizerOptions.");
        }
    }

    /// <summary>
    /// Gets the current optimization algorithm options.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method returns the current options used by the Momentum optimizer.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like checking your current ball-rolling rules. It lets you see all the settings and strategies 
    /// you're currently using in your experiment.
    /// </para>
    /// </remarks>
    /// <returns>The current MomentumOptimizerOptions object.</returns>
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
    /// Think of this as taking a snapshot of your entire ball-rolling experiment. It captures all the details of your 
    /// current setup, including the ball's position, speed, and all your rules. This snapshot can be used to recreate 
    /// the exact same experiment later or share it with others.
    /// </para>
    /// </remarks>
    /// <returns>A byte array representing the serialized state of the optimizer.</returns>
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
    /// Deserializes a byte array to restore the optimizer's state.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method takes a byte array (previously created by Serialize) and uses it to restore the optimizer's state, 
    /// including its base class state and options.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like using a detailed blueprint to recreate your ball-rolling experiment exactly as it was at a certain point. 
    /// It allows you to set up the experiment to match a previous state, with all the same rules and conditions.
    /// </para>
    /// </remarks>
    /// <param name="data">The byte array containing the serialized optimizer state.</param>
    /// <exception cref="InvalidOperationException">Thrown when the optimizer options cannot be deserialized.</exception>
    public override void Deserialize(byte[] data)
    {
        using (MemoryStream ms = new MemoryStream(data))
        using (BinaryReader reader = new BinaryReader(ms))
        {
            int baseDataLength = reader.ReadInt32();
            byte[] baseData = reader.ReadBytes(baseDataLength);
            base.Deserialize(baseData);

            string optionsJson = reader.ReadString();
            _options = JsonConvert.DeserializeObject<MomentumOptimizerOptions<T, TInput, TOutput>>(optionsJson)
                ?? throw new InvalidOperationException("Failed to deserialize optimizer options.");
        }
    }

    /// <summary>
    /// Generates a unique key for caching gradients based on the model and input data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method creates a unique identifier for caching gradients. It combines the base gradient cache key
    /// with specific parameters of the Momentum algorithm.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// Imagine you're leaving markers along your ball-rolling path. This method creates a unique label for each marker,
    /// combining information about the hill (the model and data) with specifics about how you're rolling the ball
    /// (initial momentum and learning rate). This helps you quickly recognize and use information from similar
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
        return $"{baseKey}_Momentum_{_options.InitialMomentum}_{_options.InitialLearningRate}";
    }

    /// <summary>
    /// Reverses a momentum-based gradient update to recover original parameters.
    /// </summary>
    /// <remarks>
    /// <para>
    /// For momentum optimizer, the forward update is:
    /// 1. velocity_new = momentum * velocity_old + learning_rate * gradient
    /// 2. params_new = params_old - velocity_new
    ///
    /// To reverse: params_old = params_new + velocity_new
    ///
    /// This requires access to the current velocity state, which is maintained by the optimizer.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like rewinding your ball-rolling experiment. Given where the ball ended up (updated parameters)
    /// and how fast it was moving (velocity), we can figure out where it started from.
    /// </para>
    /// </remarks>
    /// <param name="updatedParameters">Parameters after gradient application</param>
    /// <param name="appliedGradients">The gradients that were applied (not used directly for momentum reversal)</param>
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

        // If velocity is not initialized, fall back to vanilla SGD reversal
        if (_velocity == null || _velocity.Length != updatedParameters.Length)
        {
            return base.ReverseUpdate(updatedParameters, appliedGradients);
        }

        // === Vectorized Reverse Momentum Update (Phase B: US-GPU-015) ===
        // Reverse momentum update: params_old = params_new + velocity_new
        // The velocity was applied as: params_new = params_old - velocity
        // So: params_old = params_new + velocity
        return (Vector<T>)Engine.Add(updatedParameters, _velocity);
    }

    #region GPU Optimizer Support

    /// <summary>
    /// GPU buffer for velocity state.
    /// </summary>
    private IGpuBuffer? _gpuVelocity;

    /// <summary>
    /// Gets whether this optimizer supports GPU-accelerated parameter updates.
    /// </summary>
    public override bool SupportsGpuUpdate => true;

    /// <summary>
    /// Initializes Momentum optimizer state on the GPU.
    /// </summary>
    public override void InitializeGpuState(int parameterCount, IDirectGpuBackend backend)
    {
        if (_gpuStateInitialized && _gpuVelocity != null)
            return;

        var zeros = new float[parameterCount];
        _gpuVelocity = backend.AllocateBuffer(zeros);

        _gpuStateInitialized = true;
    }

    /// <summary>
    /// Updates parameters on the GPU using the SGD with momentum kernel.
    /// </summary>
    public override void UpdateParametersGpu(IGpuBuffer parameters, IGpuBuffer gradients, int parameterCount, IDirectGpuBackend backend)
    {
        if (!_gpuStateInitialized || _gpuVelocity == null)
        {
            InitializeGpuState(parameterCount, backend);
        }

        backend.SgdMomentumUpdate(
            parameters,
            gradients,
            _gpuVelocity!,
            (float)NumOps.ToDouble(CurrentLearningRate),
            (float)NumOps.ToDouble(CurrentMomentum),
            0.0f, // Basic momentum doesn't have weight decay
            parameterCount
        );
    }

    /// <summary>
    /// Disposes GPU-allocated optimizer state.
    /// </summary>
    public override void DisposeGpuState()
    {
        _gpuVelocity?.Dispose();
        _gpuVelocity = null;
        _gpuStateInitialized = false;
    }

    #endregion
}
