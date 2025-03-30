namespace AiDotNet.Optimizers;

/// <summary>
/// Represents a Gradient Descent optimizer for machine learning models.
/// </summary>
/// <remarks>
/// <para>
/// Gradient Descent is a first-order iterative optimization algorithm for finding a local minimum of a differentiable function.
/// It takes steps proportional to the negative of the gradient of the function at the current point.
/// </para>
/// <para><b>For Beginners:</b> Imagine you're trying to find the lowest point in a valley:
/// 
/// - You start at a random point (initial model parameters)
/// - You look around to see which way is steepest downhill (calculate the gradient)
/// - You take a step in that direction (update the parameters)
/// - You repeat this process until you reach the bottom of the valley (optimize the model)
/// 
/// This optimizer helps the model learn by gradually adjusting its parameters to minimize errors.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class GradientDescentOptimizer<T> : GradientBasedOptimizerBase<T>
{
    /// <summary>
    /// Options specific to the Gradient Descent optimizer.
    /// </summary>
    private GradientDescentOptimizerOptions _gdOptions;

    /// <summary>
    /// The regularization technique used to prevent overfitting.
    /// </summary>
    private readonly IRegularization<T> _regularization;

    /// <summary>
    /// Initializes a new instance of the GradientDescentOptimizer class.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This sets up the Gradient Descent optimizer with its initial settings.
    /// It's like preparing for your hike by choosing your starting point, deciding how big your steps
    /// will be, and how you'll adjust your path to avoid getting stuck in small dips.
    /// </para>
    /// </remarks>
    /// <param name="options">Options for the Gradient Descent optimizer.</param>
    /// <param name="predictionOptions">Options for prediction statistics.</param>
    /// <param name="modelOptions">Options for model statistics.</param>
    /// <param name="modelEvaluator">The model evaluator to use.</param>
    /// <param name="fitDetector">The fit detector to use.</param>
    /// <param name="fitnessCalculator">The fitness calculator to use.</param>
    /// <param name="modelCache">The model cache to use.</param>
    /// <param name="gradientCache">The gradient cache to use.</param>
    public GradientDescentOptimizer(GradientDescentOptimizerOptions? options = null,
        PredictionStatsOptions? predictionOptions = null,
        ModelStatsOptions? modelOptions = null,
        IModelEvaluator<T>? modelEvaluator = null,
        IFitDetector<T>? fitDetector = null,
        IFitnessCalculator<T>? fitnessCalculator = null,
        IModelCache<T>? modelCache = null,
        IGradientCache<T>? gradientCache = null)
        : base(options, predictionOptions, modelOptions, modelEvaluator, fitDetector, fitnessCalculator, modelCache, gradientCache)
    {
        _gdOptions = options ?? new GradientDescentOptimizerOptions();
        _regularization = CreateRegularization(_gdOptions);
    }

    /// <summary>
    /// Creates a regularization technique based on the provided options.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method sets up a way to prevent the model from becoming too complex.
    /// It's like adding rules to your hiking strategy to avoid taking unnecessarily complicated paths.
    /// </para>
    /// </remarks>
    /// <param name="options">The options specifying the regularization technique to use.</param>
    /// <returns>An instance of the specified regularization technique.</returns>
    private static IRegularization<T> CreateRegularization(GradientDescentOptimizerOptions options)
    {
        return RegularizationFactory.CreateRegularization<T>(options.RegularizationOptions);
    }

    /// <summary>
    /// Performs the main optimization process using the Gradient Descent algorithm.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the heart of the Gradient Descent algorithm. It:
    /// 1. Starts with a random solution
    /// 2. Calculates how to improve the solution (the gradient)
    /// 3. Updates the solution by taking a step in the direction of improvement
    /// 4. Repeats this process many times
    /// 
    /// It's like repeatedly adjusting your path as you hike, always trying to move towards lower ground.
    /// </para>
    /// </remarks>
    /// <param name="inputData">The input data for the optimization process.</param>
    /// <returns>The result of the optimization process.</returns>
    public override OptimizationResult<T> Optimize(OptimizationInputData<T> inputData)
    {
        ValidationHelper<T>.ValidateInputData(inputData);

        var currentSolution = InitializeRandomSolution(inputData.XTrain.Columns);
        var bestStepData = EvaluateSolution(currentSolution, inputData);
        var previousStepData = bestStepData;
        InitializeAdaptiveParameters();

        for (int iteration = 0; iteration < _gdOptions.MaxIterations; iteration++)
        {
            var gradient = CalculateGradient(currentSolution, inputData.XTrain, inputData.YTrain);
            gradient = ApplyMomentum(gradient);

            currentSolution = UpdateSolution(currentSolution, gradient);

            var currentStepData = EvaluateSolution(currentSolution, inputData);

            UpdateAdaptiveParameters(currentStepData, previousStepData);

            UpdateBestSolution(currentStepData, ref bestStepData);

            if (UpdateIterationHistoryAndCheckEarlyStopping(iteration, bestStepData))
            {
                break;
            }

            previousStepData = currentStepData;
        }

        return CreateOptimizationResult(bestStepData, inputData);
    }

    /// <summary>
    /// Updates the current solution based on the calculated gradient.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method adjusts the current solution to make it better.
    /// It's like taking a step in the direction you've determined will lead you downhill.
    /// </para>
    /// </remarks>
    /// <param name="currentSolution">The current solution.</param>
    /// <param name="gradient">The calculated gradient.</param>
    /// <returns>The updated solution.</returns>
    private ISymbolicModel<T> UpdateSolution(ISymbolicModel<T> currentSolution, Vector<T> gradient)
    {
        Vector<T> updatedCoefficients = currentSolution.Coefficients.Subtract(gradient.Multiply(CurrentLearningRate));
        return currentSolution.UpdateCoefficients(updatedCoefficients);
    }

    /// <summary>
    /// Calculates the gradient for the given solution and input data.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method calculates how steep the hill is and in which direction.
    /// It helps determine which way the optimizer should step to improve the model.
    /// This implementation uses numerical differentiation for flexibility with different model types.
    /// </para>
    /// </remarks>
    /// <param name="solution">The current solution.</param>
    /// <param name="X">The input features.</param>
    /// <param name="y">The target values.</param>
    /// <returns>The calculated gradient.</returns>
    protected new Vector<T> CalculateGradient(ISymbolicModel<T> solution, Matrix<T> X, Vector<T> y)
    {
        string cacheKey = GenerateGradientCacheKey(solution, X, y);
        var cachedGradient = GradientCache.GetCachedGradient(cacheKey);
        if (cachedGradient != null)
        {
            return cachedGradient.Coefficients;
        }

        Vector<T> gradient = new(solution.Coefficients.Length);
        T epsilon = NumOps.FromDouble(1e-8);

        for (int i = 0; i < solution.Coefficients.Length; i++)
        {
            Vector<T> perturbedCoefficientsPlus = solution.Coefficients.Copy();
            perturbedCoefficientsPlus[i] = NumOps.Add(perturbedCoefficientsPlus[i], epsilon);

            Vector<T> perturbedCoefficientsMinus = solution.Coefficients.Copy();
            perturbedCoefficientsMinus[i] = NumOps.Subtract(perturbedCoefficientsMinus[i], epsilon);

            T lossPlus = CalculateLoss(solution.UpdateCoefficients(perturbedCoefficientsPlus), X, y);
            T lossMinus = CalculateLoss(solution.UpdateCoefficients(perturbedCoefficientsMinus), X, y);

            gradient[i] = NumOps.Divide(NumOps.Subtract(lossPlus, lossMinus), NumOps.Multiply(NumOps.FromDouble(2.0), epsilon));
        }

        var gradientModel = gradient.ToSymbolicModel();
        GradientCache.CacheGradient(cacheKey, gradientModel);

        return gradient;
    }

    /// <summary>
    /// Calculates the loss for a given solution and input data.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method measures how well the current solution performs.
    /// It's like checking your altitude to see how close you are to the bottom of the valley.
    /// The method also includes a regularization term to prevent overfitting.
    /// </para>
    /// </remarks>
    /// <param name="solution">The current solution to evaluate.</param>
    /// <param name="X">The input features.</param>
    /// <param name="y">The target values.</param>
    /// <returns>The calculated loss value.</returns>
    private T CalculateLoss(ISymbolicModel<T> solution, Matrix<T> X, Vector<T> y)
    {
        Vector<T> predictions = new Vector<T>(X.Rows);
        for (int i = 0; i < X.Rows; i++)
        {
            predictions[i] = solution.Evaluate(X.GetRow(i));
        }

        T mse = StatisticsHelper<T>.CalculateMeanSquaredError(predictions, y);
        Vector<T> regularizedCoefficients = _regularization.RegularizeCoefficients(solution.Coefficients);
        T regularizationTerm = regularizedCoefficients.Subtract(solution.Coefficients).Transform(NumOps.Abs).Sum();

        return NumOps.Add(mse, regularizationTerm);
    }

    /// <summary>
    /// Updates the options for the Gradient Descent optimizer.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method allows you to change the settings of the optimizer
    /// while it's running. It's like adjusting your hiking strategy mid-journey based on the terrain you encounter.
    /// </para>
    /// </remarks>
    /// <param name="options">The new options to apply to the optimizer.</param>
    /// <exception cref="ArgumentException">Thrown when the provided options are not of the correct type.</exception>
    protected override void UpdateOptions(OptimizationAlgorithmOptions options)
    {
        if (options is GradientDescentOptimizerOptions gdOptions)
        {
            _gdOptions = gdOptions;
        }
        else
        {
            throw new ArgumentException("Invalid options type. Expected GradientDescentOptions.");
        }
    }

    /// <summary>
    /// Retrieves the current options for the Gradient Descent optimizer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method returns the current configuration options for the Gradient Descent optimizer.
    /// These options control various aspects of the optimization process, such as learning rate,
    /// maximum iterations, and regularization settings.
    /// </para>
    /// <para><b>For Beginners:</b> Think of this method as checking your current hiking plan:
    /// 
    /// - It tells you things like how big your steps are (learning rate)
    /// - How long you plan to hike (maximum iterations)
    /// - What rules you're following to avoid getting lost (regularization settings)
    /// 
    /// This information is useful if you want to understand or adjust how the optimizer is currently set up.
    /// </para>
    /// </remarks>
    /// <returns>The current Gradient Descent optimizer options.</returns>
    public override OptimizationAlgorithmOptions GetOptions()
    {
        return _gdOptions;
    }

    /// <summary>
    /// Converts the current state of the Gradient Descent optimizer into a byte array for storage or transmission.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method serializes both the base class data and the Gradient Descent-specific options.
    /// It uses a combination of binary serialization for efficiency and JSON serialization for flexibility.
    /// </para>
    /// <para><b>For Beginners:</b> This is like packing up your hiking gear and writing down your plan:
    /// 
    /// - It saves all the important information about the optimizer's current state
    /// - This saved information can be used later to recreate the optimizer exactly as it is now
    /// - It's useful for saving your progress or sharing your optimizer setup with others
    /// 
    /// Think of it as creating a detailed snapshot of your hiking journey that you can use to continue 
    /// from the same point later or allow someone else to follow your exact path.
    /// </para>
    /// </remarks>
    /// <returns>A byte array representing the serialized state of the optimizer.</returns>
    public override byte[] Serialize()
    {
        using MemoryStream ms = new MemoryStream();
        using BinaryWriter writer = new BinaryWriter(ms);

        // Serialize base class data
        byte[] baseData = base.Serialize();
        writer.Write(baseData.Length);
        writer.Write(baseData);

        // Serialize GradientDescentOptions
        string optionsJson = JsonConvert.SerializeObject(_gdOptions);
        writer.Write(optionsJson);

        return ms.ToArray();
    }

    /// <summary>
    /// Restores the state of the Gradient Descent optimizer from a byte array.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method deserializes both the base class data and the Gradient Descent-specific options
    /// from a byte array, typically created by the Serialize method. It reconstructs the optimizer's
    /// state, including all settings and progress information.
    /// </para>
    /// <para><b>For Beginners:</b> This is like unpacking your hiking gear and reading your saved plan:
    /// 
    /// - It takes the saved information (byte array) and uses it to set up the optimizer
    /// - This allows you to continue optimizing from where you left off, or use someone else's setup
    /// - It's the reverse process of Serialize, turning the saved data back into a working optimizer
    /// 
    /// Imagine you're starting a hike using a very detailed guide someone else wrote. This method
    /// helps you set everything up exactly as described in that guide.
    /// </para>
    /// </remarks>
    /// <param name="data">The byte array containing the serialized optimizer state.</param>
    /// <exception cref="InvalidOperationException">Thrown when deserialization of optimizer options fails.</exception>
    public override void Deserialize(byte[] data)
    {
        using MemoryStream ms = new MemoryStream(data);
        using BinaryReader reader = new BinaryReader(ms);

        // Deserialize base class data
        int baseDataLength = reader.ReadInt32();
        byte[] baseData = reader.ReadBytes(baseDataLength);
        base.Deserialize(baseData);

        // Deserialize GradientDescentOptions
        string optionsJson = reader.ReadString();
        _gdOptions = JsonConvert.DeserializeObject<GradientDescentOptimizerOptions>(optionsJson)
            ?? throw new InvalidOperationException("Failed to deserialize optimizer options.");
    }

    /// <summary>
    /// Generates a unique key for caching gradients specific to the Gradient Descent optimizer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method extends the base class's gradient cache key generation by adding Gradient Descent-specific
    /// parameters. The resulting key is unique to the current state of the optimizer and the input data,
    /// allowing for efficient caching and retrieval of previously calculated gradients.
    /// </para>
    /// <para><b>For Beginners:</b> Think of this method as creating a unique label for each gradient calculation:
    /// 
    /// - It starts with a basic label (from the base class) that describes the model and data
    /// - Then it adds specific details about the Gradient Descent optimizer, like how big steps it's taking (learning rate)
    ///   and how many times it plans to adjust the model (max iterations)
    /// - This unique label helps the optimizer remember and quickly find previous calculations,
    ///   making the whole process faster and more efficient
    /// 
    /// It's like keeping a well-organized hiking journal where you can quickly look up information
    /// about specific points in your journey.
    /// </para>
    /// </remarks>
    /// <param name="model">The current model being optimized.</param>
    /// <param name="X">The input features used for gradient calculation.</param>
    /// <param name="y">The target values used for gradient calculation.</param>
    /// <returns>A string that uniquely identifies the current gradient calculation scenario.</returns>
    protected override string GenerateGradientCacheKey(ISymbolicModel<T> model, Matrix<T> X, Vector<T> y)
    {
        var baseKey = base.GenerateGradientCacheKey(model, X, y);
        return $"{baseKey}_GD_{CurrentLearningRate}_{_gdOptions.MaxIterations}";
    }
}