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
public class MiniBatchGradientDescentOptimizer<T> : GradientBasedOptimizerBase<T>
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
    private MiniBatchGradientDescentOptions _options;

    /// <summary>
    /// A random number generator used for shuffling the data.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b>
    /// This is like a dice you roll to decide which part of the map to look at next. It helps ensure you don't always
    /// take the same path, which could lead to getting stuck in a local valley instead of finding the deepest one.
    /// </para>
    /// </remarks>
    private Random _random;

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
    public MiniBatchGradientDescentOptimizer(
        MiniBatchGradientDescentOptions? options = null,
        PredictionStatsOptions? predictionOptions = null,
        ModelStatsOptions? modelOptions = null,
        IModelEvaluator<T>? modelEvaluator = null,
        IFitDetector<T>? fitDetector = null,
        IFitnessCalculator<T>? fitnessCalculator = null,
        IModelCache<T>? modelCache = null,
        IGradientCache<T>? gradientCache = null)
        : base(options, predictionOptions, modelOptions, modelEvaluator, fitDetector, fitnessCalculator, modelCache, gradientCache)
    {
        _options = options ?? new MiniBatchGradientDescentOptions();
        _random = new Random(_options.Seed ?? Environment.TickCount);
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
    /// </remarks>
    /// <param name="inputData">The input data for the optimization process.</param>
    /// <returns>The result of the optimization process.</returns>
    public override OptimizationResult<T> Optimize(OptimizationInputData<T> inputData)
    {
        ValidationHelper<T>.ValidateInputData(inputData);

        var currentSolution = InitializeRandomSolution(inputData.XTrain.Columns);
        var bestStepData = new OptimizationStepData<T>();
        var previousStepData = new OptimizationStepData<T>();

        InitializeAdaptiveParameters();

        for (int epoch = 0; epoch < _options.MaxEpochs; epoch++)
        {
            var shuffledIndices = Enumerable.Range(0, inputData.XTrain.Rows).OrderBy(x => _random.Next());
            
            for (int i = 0; i < inputData.XTrain.Rows; i += _options.BatchSize)
            {
                var batchIndices = shuffledIndices.Skip(i).Take(_options.BatchSize);
                var xBatch = inputData.XTrain.GetRows(batchIndices);
                var yBatch = new Vector<T>([.. batchIndices.Select(index => inputData.YTrain[index])]);

                var gradient = CalculateGradient(currentSolution, xBatch, yBatch);
                gradient = ApplyMomentum(gradient);
                var newSolution = UpdateSolution(currentSolution, gradient);

                var currentStepData = EvaluateSolution(newSolution, inputData);
                UpdateBestSolution(currentStepData, ref bestStepData);

                UpdateAdaptiveParameters(currentStepData, previousStepData);

                if (UpdateIterationHistoryAndCheckEarlyStopping(epoch * (inputData.XTrain.Rows / _options.BatchSize) + i / _options.BatchSize, bestStepData))
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
    private ISymbolicModel<T> UpdateSolution(ISymbolicModel<T> currentSolution, Vector<T> gradient)
    {
        var newCoefficients = new Vector<T>(currentSolution.Coefficients.Length);
        for (int i = 0; i < currentSolution.Coefficients.Length; i++)
        {
            newCoefficients[i] = NumOps.Subtract(currentSolution.Coefficients[i], NumOps.Multiply(CurrentLearningRate, gradient[i]));
        }

        return new VectorModel<T>(newCoefficients);
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
    protected override void UpdateAdaptiveParameters(OptimizationStepData<T> currentStepData, OptimizationStepData<T> previousStepData)
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
    protected override void UpdateOptions(OptimizationAlgorithmOptions options)
    {
        if (options is MiniBatchGradientDescentOptions mbgdOptions)
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
    public override OptimizationAlgorithmOptions GetOptions()
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
        _options = JsonConvert.DeserializeObject<MiniBatchGradientDescentOptions>(optionsJson)
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
    protected override string GenerateGradientCacheKey(ISymbolicModel<T> model, Matrix<T> X, Vector<T> y)
    {
        var baseKey = base.GenerateGradientCacheKey(model, X, y);
        return $"{baseKey}_MiniBatchGD_{_options.BatchSize}_{_options.MaxEpochs}";
    }
}