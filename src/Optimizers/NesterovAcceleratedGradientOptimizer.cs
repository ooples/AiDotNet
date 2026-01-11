using AiDotNet.Tensors.Engines.DirectGpu;
using Newtonsoft.Json;

namespace AiDotNet.Optimizers;

/// <summary>
/// Implements the Nesterov Accelerated Gradient optimization algorithm.
/// </summary>
/// <remarks>
/// <para>
/// The Nesterov Accelerated Gradient (NAG) is an optimization algorithm that improves upon standard gradient descent.
/// It introduces a smart prediction of the next position of the parameters, which helps to dampen oscillations and
/// improve convergence, especially in scenarios with high curvature or small but consistent gradients.
/// </para>
/// <para><b>For Beginners:</b>
/// Imagine you're skiing down a hill. Regular gradient descent is like looking at your current position to decide where to go next.
/// NAG is like looking ahead to where you'll be after your next move, and then deciding how to adjust your path.
/// This "look-ahead" helps you navigate the slope more efficiently, especially around tricky turns.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class NesterovAcceleratedGradientOptimizer<T, TInput, TOutput> : GradientBasedOptimizerBase<T, TInput, TOutput>
{
    /// <summary>
    /// The options specific to the Nesterov Accelerated Gradient optimizer.
    /// </summary>
    private NesterovAcceleratedGradientOptimizerOptions<T, TInput, TOutput> _options;

    /// <summary>
    /// The velocity vector used in the NAG algorithm.
    /// </summary>
    private Vector<T>? _velocity;

    /// <summary>
    /// Initializes a new instance of the NesterovAcceleratedGradientOptimizer class.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This constructor sets up the NAG optimizer with the provided options and dependencies.
    /// If no options are provided, it uses default settings.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like preparing your skis and gear before you start your descent. You're setting up all the tools and rules you'll use during your optimization journey.
    /// </para>
    /// </remarks>
    /// <param name="model">The model to optimize.</param>
    /// <param name="options">The NAG-specific optimization options.</param>
    /// <param name="predictionOptions">Options for prediction statistics.</param>
    /// <param name="modelOptions">Options for model statistics.</param>
    /// <param name="modelEvaluator">The model evaluator to use.</param>
    /// <param name="fitDetector">The fit detector to use.</param>
    /// <param name="fitnessCalculator">The fitness calculator to use.</param>
    /// <param name="modelCache">The model cache to use.</param>
    /// <param name="gradientCache">The gradient cache to use.</param>
    public NesterovAcceleratedGradientOptimizer(
        IFullModel<T, TInput, TOutput> model,
        NesterovAcceleratedGradientOptimizerOptions<T, TInput, TOutput>? options = null,
        IEngine? engine = null)
        : base(model, options ?? new())
    {
        _options = options ?? new NesterovAcceleratedGradientOptimizerOptions<T, TInput, TOutput>();

        InitializeAdaptiveParameters();
    }

    /// <summary>
    /// Initializes the adaptive parameters for the NAG optimizer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method sets up the initial values for the learning rate and momentum.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like setting your initial speed and direction before you start skiing. You're deciding how fast to move and how much to consider your previous direction.
    /// </para>
    /// </remarks>
    protected override void InitializeAdaptiveParameters()
    {
        base.InitializeAdaptiveParameters();
        CurrentLearningRate = NumOps.FromDouble(_options.InitialLearningRate);
        CurrentMomentum = NumOps.FromDouble(_options.InitialMomentum);
    }

    /// <summary>
    /// Performs the optimization process using the Nesterov Accelerated Gradient algorithm.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method implements the main optimization loop. It uses the NAG algorithm to update the solution iteratively,
    /// aiming to find the optimal set of parameters that minimize the loss function.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is your actual ski run. You start at the top of the hill (your initial solution) and then repeatedly:
    /// 1. Look ahead to where you might be after your next move.
    /// 2. Check the steepness (gradient) at that future position.
    /// 3. Adjust your speed and direction based on what you see.
    /// 4. Make your move.
    /// You keep doing this until you reach the bottom of the hill or decide you're close enough to the best spot.
    /// </para>
    /// <para><b>DataLoader Integration:</b> This method uses the DataLoader API for efficient batch processing.
    /// It creates a batcher using <see cref="GradientBasedOptimizerBase{T,TInput,TOutput}.CreateBatcher"/>
    /// and notifies the sampler of epoch starts using
    /// <see cref="GradientBasedOptimizerBase{T,TInput,TOutput}.NotifyEpochStart"/>.
    /// </para>
    /// </remarks>
    /// <param name="inputData">The input data for the optimization process.</param>
    /// <returns>The result of the optimization process.</returns>
    public override OptimizationResult<T, TInput, TOutput> Optimize(OptimizationInputData<T, TInput, TOutput> inputData)
    {
        ValidationHelper<T>.ValidateInputData(inputData);

        var currentSolution = InitializeRandomSolution(inputData.XTrain);
        var bestStepData = new OptimizationStepData<T, TInput, TOutput>();
        var previousStepData = new OptimizationStepData<T, TInput, TOutput>();

        _velocity = new Vector<T>(currentSolution.GetParameters().Length);
        InitializeAdaptiveParameters();

        for (int epoch = 0; epoch < _options.MaxIterations; epoch++)
        {
            NotifyEpochStart(epoch);
            var batcher = CreateBatcher(inputData, _options.BatchSize);

            foreach (var (xBatch, yBatch, batchIndices) in batcher.GetBatches())
            {
                var lookaheadSolution = GetLookaheadSolution(currentSolution);
                var gradient = CalculateGradient(lookaheadSolution, xBatch, yBatch);
                _velocity = UpdateVelocity(gradient);
                var newSolution = UpdateSolution(currentSolution, _velocity);
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
    /// Calculates the lookahead solution based on the current solution and velocity.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method computes a predicted future position of the solution, which is a key aspect of the NAG algorithm.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like looking ahead to where you think you'll be after your next move, based on your current position and how fast you're moving (velocity).
    /// </para>
    /// </remarks>
    /// <param name="currentSolution">The current solution.</param>
    /// <returns>A predicted future solution.</returns>
    private IFullModel<T, TInput, TOutput> GetLookaheadSolution(IFullModel<T, TInput, TOutput> currentSolution)
    {
        // === Vectorized NAG Lookahead using IEngine (Phase B: US-GPU-015) ===
        // lookahead = params - momentum * velocity

        var parameters = currentSolution.GetParameters();
        var momentumVelocity = (Vector<T>)Engine.Multiply(_velocity!, CurrentMomentum);
        var lookaheadCoefficients = (Vector<T>)Engine.Subtract(parameters, momentumVelocity);

        return currentSolution.WithParameters(lookaheadCoefficients);
    }

    /// <summary>
    /// Updates the velocity vector based on the current gradient.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method adjusts the velocity using the momentum and learning rate, incorporating the new gradient information.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like adjusting your speed and direction. You consider how fast you were going before (momentum) and the new information about the slope (gradient),
    /// to decide how to change your movement.
    /// </para>
    /// </remarks>
    /// <param name="gradient">The current gradient.</param>
    /// <returns>The updated velocity vector.</returns>
    private Vector<T> UpdateVelocity(Vector<T> gradient)
    {
        // === Vectorized NAG Velocity Update using IEngine (Phase B: US-GPU-015) ===
        // velocity = momentum * velocity + learningRate * gradient

        var momentumVelocity = (Vector<T>)Engine.Multiply(_velocity!, CurrentMomentum);
        var scaledGradient = (Vector<T>)Engine.Multiply(gradient, CurrentLearningRate);
        _velocity = (Vector<T>)Engine.Add(momentumVelocity, scaledGradient);

        return _velocity;
    }

    /// <summary>
    /// Updates the current solution using the velocity vector.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method computes the new solution by applying the velocity to the current solution.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like actually making your move down the slope. You take your current position and adjust it based on your speed and direction (velocity).
    /// </para>
    /// </remarks>
    /// <param name="currentSolution">The current solution.</param>
    /// <param name="velocity">The current velocity vector.</param>
    /// <returns>The updated solution.</returns>
    protected override IFullModel<T, TInput, TOutput> UpdateSolution(IFullModel<T, TInput, TOutput> currentSolution, Vector<T> velocity)
    {
        // === Vectorized NAG Update using IEngine (Phase B: US-GPU-015) ===
        // params = params - velocity

        var parameters = currentSolution.GetParameters();
        var newCoefficients = (Vector<T>)Engine.Subtract(parameters, velocity);

        return currentSolution.WithParameters(newCoefficients);
    }

    /// <summary>
    /// Updates a vector of parameters using the Nesterov Accelerated Gradient algorithm.
    /// </summary>
    /// <param name="parameters">The current parameter vector to be updated.</param>
    /// <param name="gradient">The gradient vector corresponding to the parameters.</param>
    /// <returns>The updated parameter vector.</returns>
    /// <remarks>
    /// <para>
    /// NAG uses a lookahead mechanism where it evaluates the gradient at a predicted future position,
    /// then uses that gradient to update velocity. This lookahead gives NAG better convergence properties
    /// than standard momentum.
    /// </para>
    /// <para><b>For Beginners:</b> NAG is like looking ahead while skiing - you peek at the slope
    /// ahead before making your move, which helps you make smarter adjustments to your speed and direction.
    /// </para>
    /// </remarks>
    public override Vector<T> UpdateParameters(Vector<T> parameters, Vector<T> gradient)
    {
        if (_velocity == null || _velocity.Length != parameters.Length)
        {
            _velocity = new Vector<T>(parameters.Length);
        }

        // === Vectorized NAG Update using IEngine (Phase B: US-GPU-015) ===
        // Note: In NAG, the gradient is evaluated at the lookahead position

        // Update velocity: velocity = momentum * velocity + lr * gradient
        var momentumVelocity = (Vector<T>)Engine.Multiply(_velocity, CurrentMomentum);
        var scaledGradient = (Vector<T>)Engine.Multiply(gradient, CurrentLearningRate);
        _velocity = (Vector<T>)Engine.Add(momentumVelocity, scaledGradient);

        // Update parameters: params = params - velocity
        var updatedParams = (Vector<T>)Engine.Subtract(parameters, _velocity);

        return updatedParams;
    }

    /// <summary>
    /// Reverses a Nesterov Accelerated Gradient update to recover original parameters.
    /// </summary>
    /// <param name="updatedParameters">Parameters after NAG update</param>
    /// <param name="appliedGradients">The gradients that were applied</param>
    /// <returns>Original parameters before the update</returns>
    /// <remarks>
    /// <para>
    /// NAG's reverse update requires the optimizer's internal velocity state from the forward pass.
    /// This method must be called immediately after UpdateParameters while the velocity is fresh.
    /// NAG evaluates gradients at a lookahead position, but the reversal only needs the final velocity.
    /// </para>
    /// <para><b>For Beginners:</b> This calculates where parameters were before a NAG update.
    /// NAG uses velocity (built from lookahead gradients) to update parameters. To reverse,
    /// we just need to know what velocity was used to take the step.
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

        if (_velocity == null || _velocity.Length != updatedParameters.Length)
        {
            throw new InvalidOperationException(
                "NAG optimizer velocity is not initialized. ReverseUpdate must be called after UpdateParameters.");
        }

        // === Vectorized Reverse NAG Update (Phase B: US-GPU-015) ===
        // Reverse the update: original = updated + velocity
        return (Vector<T>)Engine.Add(updatedParameters, _velocity);
    }

    /// <summary>
    /// Updates the adaptive parameters of the NAG optimizer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method adjusts the learning rate and momentum based on the improvement in fitness.
    /// It's used to fine-tune the algorithm's behavior as the optimization progresses.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like adjusting your skiing technique as you go down the hill. If you're making good progress, you might decide to go a bit faster or trust your momentum more.
    /// If you're not improving, you might slow down or be more cautious about following your previous direction.
    /// </para>
    /// </remarks>
    /// <param name="currentStepData">The current optimization step data.</param>
    /// <param name="previousStepData">The previous optimization step data.</param>
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
    /// This method ensures that only compatible option types are used with this optimizer.
    /// It updates the internal options if the provided options are of the correct type.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like changing the rules for how the skier should navigate the slope. It makes sure you're only using rules that work for this specific type of skiing technique (Nesterov Accelerated Gradient method).
    /// </para>
    /// </remarks>
    /// <param name="options">The new options to be applied to the optimizer.</param>
    /// <exception cref="ArgumentException">Thrown when the provided options are not of the correct type.</exception>
    protected override void UpdateOptions(OptimizationAlgorithmOptions<T, TInput, TOutput> options)
    {
        if (options is NesterovAcceleratedGradientOptimizerOptions<T, TInput, TOutput> nagOptions)
        {
            _options = nagOptions;
        }
        else
        {
            throw new ArgumentException("Invalid options type. Expected NesterovAcceleratedGradientOptimizerOptions.");
        }
    }

    /// <summary>
    /// Gets the current options of the Nesterov Accelerated Gradient optimizer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method returns the current configuration options of the optimizer.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like asking to see the current set of rules the skier is following on their descent.
    /// </para>
    /// </remarks>
    /// <returns>The current optimization algorithm options.</returns>
    public override OptimizationAlgorithmOptions<T, TInput, TOutput> GetOptions()
    {
        return _options;
    }

    /// <summary>
    /// Serializes the Nesterov Accelerated Gradient optimizer to a byte array.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method converts the current state of the optimizer, including its options and parameters, into a byte array.
    /// This allows the optimizer's state to be saved or transmitted.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like taking a snapshot of the entire skiing process, including where the skier is on the slope and what techniques they're using, so you can save it or send it to someone else.
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
    /// Deserializes the Nesterov Accelerated Gradient optimizer from a byte array.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method reconstructs the optimizer's state from a byte array, including its options and parameters.
    /// It's used to restore a previously saved or transmitted optimizer state.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like using a saved snapshot to set up the skiing process exactly as it was before, placing the skier back where they were on the slope and restoring the techniques they were using.
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
            _options = JsonConvert.DeserializeObject<NesterovAcceleratedGradientOptimizerOptions<T, TInput, TOutput>>(optionsJson)
                ?? throw new InvalidOperationException("Failed to deserialize optimizer options.");
        }
    }

    /// <summary>
    /// Generates a unique key for caching gradients in the Nesterov Accelerated Gradient optimizer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method creates a unique identifier for caching gradients, incorporating the base key and NAG-specific parameters.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like creating a special label for each unique skiing situation, considering not just the slope (model and data) but also the specific NAG skiing technique being used (initial momentum and learning rate).
    /// </para>
    /// </remarks>
    /// <param name="model">The symbolic model for which the gradient is being calculated.</param>
    /// <param name="X">The input data matrix.</param>
    /// <param name="y">The target vector.</param>
    /// <returns>A string key uniquely identifying the gradient calculation scenario for caching purposes.</returns>
    protected override string GenerateGradientCacheKey(IFullModel<T, TInput, TOutput> model, TInput X, TOutput y)
    {
        var baseKey = base.GenerateGradientCacheKey(model, X, y);
        return $"{baseKey}_NAG_{_options.InitialMomentum}_{_options.InitialLearningRate}";
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
    /// Initializes NAG optimizer state on the GPU.
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
    /// Updates parameters on the GPU using the NAG kernel.
    /// </summary>
    public override void UpdateParametersGpu(IGpuBuffer parameters, IGpuBuffer gradients, int parameterCount, IDirectGpuBackend backend)
    {
        if (!_gpuStateInitialized || _gpuVelocity == null)
        {
            InitializeGpuState(parameterCount, backend);
        }

        backend.NagUpdate(
            parameters,
            gradients,
            _gpuVelocity!,
            (float)NumOps.ToDouble(CurrentLearningRate),
            (float)NumOps.ToDouble(CurrentMomentum),
            0.0f, // NAG doesn't have weight decay in these options
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
