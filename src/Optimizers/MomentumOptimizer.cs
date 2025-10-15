namespace AiDotNet.Optimizers;

/// <summary>
/// Implements an optimization algorithm that uses momentum to accelerate convergence and overcome local minima.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <typeparam name="TInput">The input data structure type (e.g., Matrix<T>, Tensor<T>).</typeparam>
/// <typeparam name="TOutput">The output data structure type (e.g., Vector<T>, Tensor<T>).</typeparam>
/// <remarks>
/// <para>
/// The Momentum optimizer is a variation of gradient descent that adds a fraction of the previous update
/// to the current one. This helps accelerate convergence and overcome shallow local minima.
/// </para>
/// <para><b>For Beginners:</b>
/// Imagine rolling a ball down a hill to find the lowest point. Unlike standard gradient descent (which is like
/// dropping the ball and letting it roll straight down), momentum is like giving the ball some weight so it
/// builds up speed. This helps it:
/// - Move faster in consistent directions
/// - Roll through small bumps and valleys rather than getting stuck
/// - Find the bottom of the hill more efficiently
/// 
/// The "momentum" parameter controls how much the ball remembers its previous direction and speed.
/// </para>
/// </remarks>
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
    private MomentumOptimizerOptions<T, TInput, TOutput> _options = default!;

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
    /// <remarks>
    /// <para>
    /// This constructor sets up the optimizer with the provided model and options. If no options are provided,
    /// it uses default settings.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like setting up your ball-rolling experiment. You're deciding on the properties of the ball
    /// (like its size and bounciness) and the hill (like its steepness and texture).
    /// </para>
    /// </remarks>
    /// <param name="model">The model to be optimized.</param>
    /// <param name="options">Custom options for the Momentum algorithm.</param>
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
    /// </remarks>
    /// <param name="inputData">The input data for the optimization process.</param>
    /// <returns>The result of the optimization process.</returns>
    public override OptimizationResult<T, TInput, TOutput> Optimize(OptimizationInputData<T, TInput, TOutput> inputData)
    {
        ValidationHelper<T>.ValidateInputData(inputData);

        var currentSolution = Model.DeepCopy();
        var bestStepData = new OptimizationStepData<T, TInput, TOutput>
        {
            Solution = currentSolution,
            FitnessScore = FitnessCalculator.IsHigherScoreBetter ? NumOps.MinValue : NumOps.MaxValue
        };
        var previousStepData = new OptimizationStepData<T, TInput, TOutput>();

        _velocity = new Vector<T>(currentSolution.GetParameters().Length);
        InitializeAdaptiveParameters();

        for (int iteration = 0; iteration < Options.MaxIterations; iteration++)
        {
            // Create solution using the base class method (handles feature selection and parameter adjustments)
            var optimizedSolution = CreateSolution(inputData.XTrain);

            var gradient = CalculateGradient(optimizedSolution, inputData.XTrain, inputData.YTrain);
            _velocity = UpdateVelocity(gradient);
            var newSolution = UpdateSolution(optimizedSolution, _velocity);

            var currentStepData = EvaluateSolution(newSolution, inputData);
            UpdateBestSolution(currentStepData, ref bestStepData);

            UpdateAdaptiveParameters(currentStepData, previousStepData);

            if (UpdateIterationHistoryAndCheckEarlyStopping(iteration, bestStepData))
            {
                return CreateOptimizationResult(bestStepData, inputData);
            }

            if (NumOps.LessThan(NumOps.Abs(NumOps.Subtract(bestStepData.FitnessScore, currentStepData.FitnessScore)),
                                NumOps.FromDouble(_options.Tolerance)))
            {
                return CreateOptimizationResult(bestStepData, inputData);
            }

            currentSolution = newSolution;
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
    /// <returns>An updated model with improved parameters.</returns>
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
        // Call the base implementation to update common parameters
        base.UpdateAdaptiveParameters(currentStepData, previousStepData);

        // Skip if previous step data is null (first iteration)
        if (previousStepData.Solution == null)
            return;

        if (_options.UseAdaptiveLearningRate)
        {
            bool isImproving = FitnessCalculator.IsBetterFitness(currentStepData.FitnessScore, previousStepData.FitnessScore);

            if (isImproving)
            {
                CurrentLearningRate = NumOps.Multiply(CurrentLearningRate, NumOps.FromDouble(_options.LearningRateIncreaseFactor));
            }
            else
            {
                CurrentLearningRate = NumOps.Multiply(CurrentLearningRate, NumOps.FromDouble(_options.LearningRateDecreaseFactor));
            }

            CurrentLearningRate = MathHelper.Clamp(CurrentLearningRate,
                NumOps.FromDouble(_options.MinLearningRate),
                NumOps.FromDouble(_options.MaxLearningRate));
        }

        if (_options.UseAdaptiveMomentum)
        {
            bool isImproving = FitnessCalculator.IsBetterFitness(currentStepData.FitnessScore, previousStepData.FitnessScore);

            if (isImproving)
            {
                CurrentMomentum = NumOps.Multiply(CurrentMomentum, NumOps.FromDouble(_options.MomentumIncreaseFactor));
            }
            else
            {
                CurrentMomentum = NumOps.Multiply(CurrentMomentum, NumOps.FromDouble(_options.MomentumDecreaseFactor));
            }

            CurrentMomentum = MathHelper.Clamp(CurrentMomentum,
                NumOps.FromDouble(_options.MinMomentum),
                NumOps.FromDouble(_options.MaxMomentum));
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
    /// <param name="model">The model being optimized.</param>
    /// <param name="X">The input data matrix.</param>
    /// <param name="y">The target output vector.</param>
    /// <returns>A string representing the unique gradient cache key.</returns>
    protected override string GenerateGradientCacheKey(IFullModel<T, TInput, TOutput> model, TInput X, TOutput y)
    {
        var baseKey = base.GenerateGradientCacheKey(model, X, y);
        return $"{baseKey}_Momentum_{_options.InitialMomentum}_{_options.InitialLearningRate}";
    }
}