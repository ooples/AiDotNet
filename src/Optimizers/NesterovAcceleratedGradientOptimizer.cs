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
        NesterovAcceleratedGradientOptimizerOptions<T, TInput, TOutput>? options = null)
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

        for (int iteration = 0; iteration < _options.MaxIterations; iteration++)
        {
            var lookaheadSolution = GetLookaheadSolution(currentSolution);
            var gradient = CalculateGradient(lookaheadSolution, inputData.XTrain, inputData.YTrain);
            _velocity = UpdateVelocity(gradient);
            var newSolution = UpdateSolution(currentSolution, _velocity);

            var currentStepData = EvaluateSolution(newSolution, inputData);
            UpdateBestSolution(currentStepData, ref bestStepData);

            UpdateAdaptiveParameters(currentStepData, previousStepData);

            if (UpdateIterationHistoryAndCheckEarlyStopping(iteration, bestStepData))
            {
                return CreateOptimizationResult(bestStepData, inputData);
            }

            if (NumOps.LessThan(NumOps.Abs(NumOps.Subtract(bestStepData.FitnessScore, currentStepData.FitnessScore)), NumOps.FromDouble(_options.Tolerance)))
            {
                return CreateOptimizationResult(bestStepData, inputData);
            }

            currentSolution = newSolution;
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
        var parameters = currentSolution.GetParameters();
        var lookaheadCoefficients = new Vector<T>(parameters.Length);
        for (int i = 0; i < parameters.Length; i++)
        {
            lookaheadCoefficients[i] = NumOps.Subtract(parameters[i], NumOps.Multiply(CurrentMomentum, _velocity![i]));
        }

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
        for (int i = 0; i < _velocity!.Length; i++)
        {
            _velocity[i] = NumOps.Add(
                NumOps.Multiply(CurrentMomentum, _velocity[i]),
                NumOps.Multiply(CurrentLearningRate, gradient[i])
            );
        }
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
        var parameters = currentSolution.GetParameters();
        var newCoefficients = new Vector<T>(parameters.Length);
        for (int i = 0; i < parameters.Length; i++)
        {
            newCoefficients[i] = NumOps.Subtract(parameters[i], velocity[i]);
        }

        return currentSolution.WithParameters(newCoefficients);
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
    /// <param name="model">The model to optimize.</param>
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
}