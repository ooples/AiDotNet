namespace AiDotNet.Optimizers;

/// <summary>
/// Implements a Nesterov Accelerated Gradient optimization algorithm for machine learning models.
/// </summary>
/// <remarks>
/// <para>
/// The Nesterov Accelerated Gradient (NAG) method is an extension of standard gradient descent that
/// incorporates momentum with a "look-ahead" correction. It helps improve convergence rates and can navigate
/// through ravines in the loss landscape more effectively than standard gradient descent.
/// </para>
/// <para><b>For Beginners:</b>
/// Think of this optimizer like skiing down a mountain with momentum and foresight. Unlike regular 
/// gradient descent (which is like carefully walking downhill step by step), NAG is like skiing:
/// 
/// 1. You build up speed (momentum) as you ski downhill
/// 2. You look ahead to where your momentum will take you
/// 3. You adjust your direction based on what you see ahead rather than just where you are now
/// 
/// This approach helps you reach the bottom (optimal solution) faster and avoid getting stuck in small dips
/// along the way.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <typeparam name="TInput">The type of input data for the model.</typeparam>
/// <typeparam name="TOutput">The type of output data for the model.</typeparam>
public class NesterovAcceleratedGradientOptimizer<T, TInput, TOutput> : GradientBasedOptimizerBase<T, TInput, TOutput>
{
    /// <summary>
    /// The options specific to the Nesterov Accelerated Gradient optimizer.
    /// </summary>
    private NesterovAcceleratedGradientOptimizerOptions<T, TInput, TOutput> _nagOptions = default!;

    /// <summary>
    /// The velocity vector used in the NAG algorithm.
    /// </summary>
    private Vector<T>? _velocity;

    /// <summary>
    /// Initializes a new instance of the NesterovAcceleratedGradientOptimizer class.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This constructor sets up the NAG optimizer with the provided model and options.
    /// If no options are provided, it uses default settings.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like preparing your skis and gear before you start your descent. You're setting up all the tools and rules you'll use during your optimization journey.
    /// </para>
    /// </remarks>
    /// <param name="model">The machine learning model to optimize.</param>
    /// <param name="options">The NAG-specific optimization options.</param>
    public NesterovAcceleratedGradientOptimizer(
        IFullModel<T, TInput, TOutput> model,
        NesterovAcceleratedGradientOptimizerOptions<T, TInput, TOutput>? options = null)
        : base(model, options ?? new())
    {
        _nagOptions = options ?? new NesterovAcceleratedGradientOptimizerOptions<T, TInput, TOutput>();

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
        CurrentLearningRate = NumOps.FromDouble(_nagOptions.InitialLearningRate);
        CurrentMomentum = NumOps.FromDouble(_nagOptions.InitialMomentum);
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

        var bestStepData = new OptimizationStepData<T, TInput, TOutput>
        {
            Solution = Model.DeepCopy(),
            FitnessScore = FitnessCalculator.IsHigherScoreBetter ? NumOps.MinValue : NumOps.MaxValue
        };
        var previousStepData = new OptimizationStepData<T, TInput, TOutput>();
        var currentSolution = Model.DeepCopy();

        // Initialize velocity vector with zeros
        _velocity = new Vector<T>(currentSolution.GetParameters().Length);
        for (int i = 0; i < _velocity.Length; i++)
        {
            _velocity[i] = NumOps.Zero;
        }

        for (int iteration = 0; iteration < Options.MaxIterations; iteration++)
        {
            // Apply different optimization strategies based on optimization mode
            var modifiedSolution = CreateSolution(inputData.XTrain);

            if (Options.OptimizationMode == OptimizationMode.FeatureSelectionOnly ||
                Options.OptimizationMode == OptimizationMode.Both)
            {
                // Feature selection logic
                int numFeatures = InputHelper<T, TInput>.GetInputSize(inputData.XTrain);
                if (Random.NextDouble() < _nagOptions.FeatureSelectionProbability)
                {
                    ApplyFeatureSelection(modifiedSolution, numFeatures);
                }
            }

            if (Options.OptimizationMode == OptimizationMode.ParametersOnly ||
                Options.OptimizationMode == OptimizationMode.Both)
            {
                // NAG parameter optimization
                if (Random.NextDouble() < _nagOptions.ParameterAdjustmentProbability)
                {
                    var lookaheadSolution = GetLookaheadSolution(modifiedSolution);
                    var gradient = CalculateGradient(lookaheadSolution, inputData.XTrain, inputData.YTrain);
                    _velocity = UpdateVelocity(gradient);
                    modifiedSolution = UpdateSolution(modifiedSolution, _velocity);
                }
            }

            var currentStepData = EvaluateSolution(modifiedSolution, inputData);
            UpdateBestSolution(currentStepData, ref bestStepData);

            // Update adaptive parameters
            UpdateAdaptiveParameters(currentStepData, previousStepData);

            if (UpdateIterationHistoryAndCheckEarlyStopping(iteration, bestStepData))
            {
                break;
            }

            // Check for convergence
            if (previousStepData.FitnessScore != null &&
                NumOps.LessThan(
                    NumOps.Abs(NumOps.Subtract(bestStepData.FitnessScore, previousStepData.FitnessScore)),
                    NumOps.FromDouble(_nagOptions.Tolerance)))
            {
                break;
            }

            currentSolution = modifiedSolution;
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
        // Call the base implementation to update common parameters
        base.UpdateAdaptiveParameters(currentStepData, previousStepData);

        // Skip if previous step data is null (first iteration)
        if (previousStepData.Solution == null)
            return;

        bool isImproving = FitnessCalculator.IsBetterFitness(currentStepData.FitnessScore, previousStepData.FitnessScore);

        // Adaptive feature selection parameters
        if ((Options.OptimizationMode == OptimizationMode.FeatureSelectionOnly ||
             Options.OptimizationMode == OptimizationMode.Both))
        {
            UpdateFeatureSelectionParameters(isImproving);
        }

        // Adaptive parameter adjustment settings
        if ((Options.OptimizationMode == OptimizationMode.ParametersOnly ||
             Options.OptimizationMode == OptimizationMode.Both))
        {
            UpdateParameterAdjustmentSettings(isImproving);
        }
    }

    /// <summary>
    /// Updates the feature selection parameters based on whether the solution is improving.
    /// </summary>
    /// <param name="isImproving">Indicates whether the solution is improving.</param>
    private void UpdateFeatureSelectionParameters(bool isImproving)
    {
        if (isImproving)
        {
            // If improving, gradually expand the range of features to consider
            _nagOptions.MinimumFeatures = Math.Max(1, _nagOptions.MinimumFeatures - 1);
            _nagOptions.MaximumFeatures = Math.Min(_nagOptions.MaximumFeatures + 1, _nagOptions.AbsoluteMaximumFeatures);

            // Slightly increase the probability of feature selection for future iterations
            _nagOptions.FeatureSelectionProbability *= 1.02;
        }
        else
        {
            // If not improving, narrow the range to focus the search
            _nagOptions.MinimumFeatures = Math.Min(_nagOptions.MinimumFeatures + 1, _nagOptions.AbsoluteMaximumFeatures - 1);
            _nagOptions.MaximumFeatures = Math.Max(_nagOptions.MaximumFeatures - 1, _nagOptions.MinimumFeatures + 1);

            // Slightly decrease the probability of feature selection for future iterations
            _nagOptions.FeatureSelectionProbability *= 0.98;
        }

        // Ensure probabilities stay within bounds
        _nagOptions.FeatureSelectionProbability = MathHelper.Clamp(
            _nagOptions.FeatureSelectionProbability,
            _nagOptions.MinFeatureSelectionProbability,
            _nagOptions.MaxFeatureSelectionProbability);
    }

    /// <summary>
    /// Updates the parameter adjustment settings based on whether the solution is improving.
    /// </summary>
    /// <param name="isImproving">Indicates whether the solution is improving.</param>
    private void UpdateParameterAdjustmentSettings(bool isImproving)
    {
        if (isImproving)
        {
            // If improving, make smaller adjustments to fine-tune
            _nagOptions.ParameterAdjustmentScale *= 0.95;

            // Increase the momentum for smoother progress
            CurrentMomentum = NumOps.Multiply(CurrentMomentum, NumOps.FromDouble(_nagOptions.MomentumIncreaseFactor));

            // Increase the probability of parameter adjustments
            _nagOptions.ParameterAdjustmentProbability *= 1.02;
        }
        else
        {
            // If not improving, make larger adjustments to explore more
            _nagOptions.ParameterAdjustmentScale *= 1.05;

            // Decrease the momentum to try different directions
            CurrentMomentum = NumOps.Multiply(CurrentMomentum, NumOps.FromDouble(_nagOptions.MomentumDecreaseFactor));

            // Slightly decrease the probability of parameter adjustments
            _nagOptions.ParameterAdjustmentProbability *= 0.98;
        }

        // Ensure values stay within bounds
        _nagOptions.ParameterAdjustmentScale = MathHelper.Clamp(
            _nagOptions.ParameterAdjustmentScale,
            _nagOptions.MinParameterAdjustmentScale,
            _nagOptions.MaxParameterAdjustmentScale);

        CurrentMomentum = MathHelper.Clamp(
            CurrentMomentum,
            NumOps.FromDouble(_nagOptions.MinMomentum),
            NumOps.FromDouble(_nagOptions.MaxMomentum));

        _nagOptions.ParameterAdjustmentProbability = MathHelper.Clamp(
            _nagOptions.ParameterAdjustmentProbability,
            _nagOptions.MinParameterAdjustmentProbability,
            _nagOptions.MaxParameterAdjustmentProbability);
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
            _nagOptions = nagOptions;
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
        return _nagOptions;
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
            // Serialize base class data
            byte[] baseData = base.Serialize();
            writer.Write(baseData.Length);
            writer.Write(baseData);

            // Serialize optimization mode
            writer.Write((int)Options.OptimizationMode);

            // Serialize NesterovAcceleratedGradientOptimizerOptions
            string optionsJson = JsonConvert.SerializeObject(_nagOptions);
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
            // Deserialize base class data
            int baseDataLength = reader.ReadInt32();
            byte[] baseData = reader.ReadBytes(baseDataLength);
            base.Deserialize(baseData);

            // Deserialize optimization mode
            Options.OptimizationMode = (OptimizationMode)reader.ReadInt32();

            // Deserialize NesterovAcceleratedGradientOptimizerOptions
            string optionsJson = reader.ReadString();
            _nagOptions = JsonConvert.DeserializeObject<NesterovAcceleratedGradientOptimizerOptions<T, TInput, TOutput>>(optionsJson)
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
        return $"{baseKey}_NAG_{_nagOptions.InitialMomentum}_{_nagOptions.InitialLearningRate}";
    }
}