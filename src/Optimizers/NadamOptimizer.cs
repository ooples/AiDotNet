namespace AiDotNet.Optimizers;

/// <summary>
/// Implements the Nadam (Nesterov-accelerated Adaptive Moment Estimation) optimization algorithm.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <typeparam name="TInput">The input data structure type (e.g., Matrix<T>, Tensor<T>).</typeparam>
/// <typeparam name="TOutput">The output data structure type (e.g., Vector<T>, Tensor<T>).</typeparam>
/// <remarks>
/// <para>
/// Nadam combines the benefits of Adam (adaptive learning rates) with Nesterov momentum.
/// It's especially effective for deep learning and other complex optimization problems.
/// </para>
/// <para><b>For Beginners:</b>
/// Think of Nadam as a "smart ball" rolling down a hill to find the lowest point:
/// - It remembers its past movement (momentum) to help roll through small bumps
/// - It adjusts its speed differently for each direction (adaptive learning rates)
/// - It can "look ahead" to anticipate where it's going (Nesterov acceleration)
/// 
/// These features help it find the bottom of the hill more efficiently than simpler methods,
/// particularly on complex, bumpy landscapes with many hills and valleys.
/// </para>
/// </remarks>
public class NadamOptimizer<T, TInput, TOutput> : GradientBasedOptimizerBase<T, TInput, TOutput>
{
    /// <summary>
    /// The options specific to the Nadam optimizer.
    /// </summary>
    private NadamOptimizerOptions<T, TInput, TOutput> _options = default!;

    /// <summary>
    /// The first moment vector (momentum).
    /// </summary>
    private Vector<T>? _m;

    /// <summary>
    /// The second moment vector (adaptive learning rates).
    /// </summary>
    private Vector<T>? _v;

    /// <summary>
    /// The current time step.
    /// </summary>
    private int _t;

    /// <summary>
    /// Initializes a new instance of the NadamOptimizer class.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This constructor sets up the Nadam optimizer with the provided model and options.
    /// If no options are provided, it uses default settings.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like preparing your smart ball for the hill-rolling experiment. You're setting up its initial properties
    /// and deciding how it will adapt during its journey.
    /// </para>
    /// </remarks>
    /// <param name="model">The model to be optimized.</param>
    /// <param name="options">The Nadam-specific optimization options.</param>
    public NadamOptimizer(
        IFullModel<T, TInput, TOutput> model,
        NadamOptimizerOptions<T, TInput, TOutput>? options = null)
        : base(model, options ?? new())
    {
        _options = options ?? new NadamOptimizerOptions<T, TInput, TOutput>();

        InitializeAdaptiveParameters();
    }

    /// <summary>
    /// Initializes the adaptive parameters for the Nadam optimizer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method sets up the initial learning rate and resets the time step counter.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like setting the initial speed of your smart ball and resetting its internal clock before it starts rolling.
    /// </para>
    /// </remarks>
    protected override void InitializeAdaptiveParameters()
    {
        base.InitializeAdaptiveParameters();

        CurrentLearningRate = NumOps.FromDouble(_options.LearningRate);
        _t = 0;
    }

    /// <summary>
    /// Performs the optimization process using the Nadam algorithm.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method implements the main optimization loop. It iterates through the data, calculating gradients,
    /// updating the momentum and adaptive learning rates, and adjusting the model parameters accordingly.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is the actual process of rolling your smart ball down the hill. In each step, you're calculating which way
    /// the ball should roll (gradient), how fast it's moving (momentum), and how it should adapt its speed (adaptive learning rates).
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

        var parameters = currentSolution.GetParameters();
        _m = new Vector<T>(parameters.Length);
        _v = new Vector<T>(parameters.Length);

        InitializeAdaptiveParameters();

        for (int iteration = 0; iteration < Options.MaxIterations; iteration++)
        {
            _t++;

            // Create solution using the base class method (handles feature selection and parameter adjustments)
            var optimizedSolution = CreateSolution(inputData.XTrain);

            var gradient = CalculateGradient(optimizedSolution, inputData.XTrain, inputData.YTrain);
            var newSolution = UpdateSolution(optimizedSolution, gradient);

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
    /// Updates the current solution based on the calculated gradient using the Nadam algorithm.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method applies the Nadam update rule to adjust the model parameters. It uses both momentum
    /// and adaptive learning rates, incorporating Nesterov's accelerated gradient.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like adjusting the ball's position based on its current speed, the slope it's on, and its ability
    /// to look ahead. It's a complex calculation that helps the ball move more efficiently towards the lowest point.
    /// </para>
    /// </remarks>
    /// <param name="currentSolution">The current model solution.</param>
    /// <param name="gradient">The calculated gradient.</param>
    /// <returns>An updated model with improved parameters.</returns>
    protected override IFullModel<T, TInput, TOutput> UpdateSolution(IFullModel<T, TInput, TOutput> currentSolution, Vector<T> gradient)
    {
        var parameters = currentSolution.GetParameters();
        var newCoefficients = new Vector<T>(parameters.Length);
        var beta1 = NumOps.FromDouble(_options.Beta1);
        var beta2 = NumOps.FromDouble(_options.Beta2);
        var oneMinusBeta1 = NumOps.FromDouble(1 - _options.Beta1);
        var oneMinusBeta2 = NumOps.FromDouble(1 - _options.Beta2);

        for (int i = 0; i < parameters.Length; i++)
        {
            // Update biased first moment estimate
            _m![i] = NumOps.Add(NumOps.Multiply(beta1, _m[i]), NumOps.Multiply(oneMinusBeta1, gradient[i]));

            // Update biased second raw moment estimate
            _v![i] = NumOps.Add(NumOps.Multiply(beta2, _v[i]), NumOps.Multiply(oneMinusBeta2, NumOps.Multiply(gradient[i], gradient[i])));

            // Compute bias-corrected first moment estimate
            var mHat = NumOps.Divide(_m[i], NumOps.FromDouble(1 - Math.Pow(_options.Beta1, _t)));

            // Compute bias-corrected second raw moment estimate
            var vHat = NumOps.Divide(_v[i], NumOps.FromDouble(1 - Math.Pow(_options.Beta2, _t)));

            // Compute the Nesterov momentum term
            var mHatNesterov = NumOps.Add(NumOps.Multiply(beta1, mHat), NumOps.Multiply(NumOps.Divide(oneMinusBeta1, NumOps.FromDouble(1 - Math.Pow(_options.Beta1, _t))), gradient[i]));

            // Update parameters
            var update = NumOps.Divide(NumOps.Multiply(CurrentLearningRate, mHatNesterov), NumOps.Add(NumOps.Sqrt(vHat), NumOps.FromDouble(_options.Epsilon)));
            newCoefficients[i] = NumOps.Subtract(parameters[i], update);
        }

        return currentSolution.WithParameters(newCoefficients);
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
    /// This is like adjusting how fast your ball rolls based on whether it's getting closer to the bottom of the hill.
    /// If it's improving, you might let it roll a bit faster. If not, you might slow it down to be more careful.
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
    /// This is like changing the rules of how your smart ball rolls mid-experiment. It makes sure you're only
    /// using rules that work for this specific type of smart ball (Nadam optimization).
    /// </para>
    /// </remarks>
    /// <param name="options">The new options to be applied to the optimizer.</param>
    /// <exception cref="ArgumentException">Thrown when the provided options are not of the correct type.</exception>
    protected override void UpdateOptions(OptimizationAlgorithmOptions<T, TInput, TOutput> options)
    {
        if (options is NadamOptimizerOptions<T, TInput, TOutput> nadamOptions)
        {
            _options = nadamOptions;
        }
        else
        {
            throw new ArgumentException("Invalid options type. Expected NadamOptimizerOptions.");
        }
    }

    /// <summary>
    /// Gets the current optimization algorithm options.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method returns the current options used by the Nadam optimizer.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like checking your current smart ball rolling rules. It lets you see all the settings and strategies 
    /// you're currently using in your experiment.
    /// </para>
    /// </remarks>
    /// <returns>The current NadamOptimizerOptions object.</returns>
    public override OptimizationAlgorithmOptions<T, TInput, TOutput> GetOptions()
    {
        return _options;
    }

    /// <summary>
    /// Serializes the optimizer's state into a byte array.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method converts the current state of the optimizer, including its base class state, options, and time step,
    /// into a byte array. This is useful for saving the optimizer's state or transferring it between systems.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// Think of this as taking a snapshot of your entire smart ball rolling experiment. It captures all the details of your 
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

            writer.Write(_t);

            return ms.ToArray();
        }
    }

    /// <summary>
    /// Deserializes a byte array to restore the optimizer's state.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method takes a byte array (previously created by Serialize) and uses it to restore the optimizer's state, 
    /// including its base class state, options, and time step.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like using a detailed blueprint to recreate your smart ball rolling experiment exactly as it was at a certain point. 
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
            _options = JsonConvert.DeserializeObject<NadamOptimizerOptions<T, TInput, TOutput>>(optionsJson)
                ?? throw new InvalidOperationException("Failed to deserialize optimizer options.");

            _t = reader.ReadInt32();
        }
    }

    /// <summary>
    /// Generates a unique key for caching gradients.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method creates a unique identifier for caching gradients based on the current model, input data,
    /// and Nadam-specific parameters. This helps in efficiently reusing previously calculated gradients when possible.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like creating a special label for each unique situation your smart ball encounters. It helps the ball
    /// remember and quickly recall how it should move in similar situations, making the whole process more efficient.
    /// </para>
    /// </remarks>
    /// <param name="model">The current model being optimized.</param>
    /// <param name="X">The input feature matrix.</param>
    /// <param name="y">The target vector.</param>
    /// <returns>A string that uniquely identifies the current gradient calculation scenario.</returns>
    protected override string GenerateGradientCacheKey(IFullModel<T, TInput, TOutput> model, TInput X, TOutput y)
    {
        var baseKey = base.GenerateGradientCacheKey(model, X, y);
        return $"{baseKey}_Nadam_{_options.Beta1}_{_options.Beta2}_{_t}";
    }
}