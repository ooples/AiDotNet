namespace AiDotNet.Optimizers;

/// <summary>
/// Represents an Adagrad (Adaptive Gradient) optimizer for gradient-based optimization.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// The Adagrad optimizer adapts the learning rate for each parameter based on the historical gradients.
/// It performs larger updates for infrequent parameters and smaller updates for frequent ones.
/// </para>
/// <para><b>For Beginners:</b> Adagrad is like a smart learning assistant that adjusts how much it learns
/// for each piece of information based on how often it has seen similar information before.
/// 
/// - It learns more from new or rare information
/// - It learns less from common or frequently seen information
/// - This helps it focus on the most important parts of what it's learning
/// 
/// This can be especially useful when some parts of your data are more important or occur less frequently.
/// </para>
/// </remarks>
public class AdagradOptimizer<T> : GradientBasedOptimizerBase<T>
{
    private AdagradOptimizerOptions _options;
    private Vector<T>? _accumulatedSquaredGradients;

    /// <summary>
    /// Initializes a new instance of the AdagradOptimizer class.
    /// </summary>
    /// <param name="options">The options for configuring the Adagrad optimizer.</param>
    /// <param name="predictionOptions">Options for prediction statistics.</param>
    /// <param name="modelOptions">Options for model statistics.</param>
    /// <param name="modelEvaluator">The model evaluator to use.</param>
    /// <param name="fitDetector">The fit detector to use.</param>
    /// <param name="fitnessCalculator">The fitness calculator to use.</param>
    /// <param name="modelCache">The model cache to use.</param>
    /// <param name="gradientCache">The gradient cache to use.</param>
    /// <remarks>
    /// <para>
    /// This constructor sets up the Adagrad optimizer with the specified options and components.
    /// If no options are provided, it uses default AdagradOptimizerOptions.
    /// </para>
    /// <para><b>For Beginners:</b> This is like setting up your learning assistant with specific instructions.
    /// 
    /// You can customize:
    /// - How the assistant learns (options)
    /// - How it measures its progress (predictionOptions, modelOptions)
    /// - How it evaluates its performance (modelEvaluator, fitDetector, fitnessCalculator)
    /// - How it remembers what it has learned (modelCache, gradientCache)
    /// 
    /// If you don't specify these, it will use default settings.
    /// </para>
    /// </remarks>
    public AdagradOptimizer(
        AdagradOptimizerOptions? options = null,
        PredictionStatsOptions? predictionOptions = null,
        ModelStatsOptions? modelOptions = null,
        IModelEvaluator<T>? modelEvaluator = null,
        IFitDetector<T>? fitDetector = null,
        IFitnessCalculator<T>? fitnessCalculator = null,
        IModelCache<T>? modelCache = null,
        IGradientCache<T>? gradientCache = null)
        : base(options, predictionOptions, modelOptions, modelEvaluator, fitDetector, fitnessCalculator, modelCache, gradientCache)
    {
        _options = options ?? new AdagradOptimizerOptions();
        InitializeAdaptiveParameters();
    }

    /// <summary>
    /// Initializes the adaptive parameters for the Adagrad optimizer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method sets up the initial learning rate for the optimizer based on the options.
    /// </para>
    /// <para><b>For Beginners:</b> This is like setting the initial speed at which your assistant learns.
    /// 
    /// The learning rate determines how big the steps are when the optimizer is trying to find the best solution.
    /// A good initial learning rate helps the optimizer start its learning process effectively.
    /// </para>
    /// </remarks>
    protected override void InitializeAdaptiveParameters()
    {
        base.InitializeAdaptiveParameters();
        CurrentLearningRate = NumOps.FromDouble(_options.InitialLearningRate);
    }

    /// <summary>
    /// Performs the optimization process using the Adagrad algorithm.
    /// </summary>
    /// <param name="inputData">The input data for optimization.</param>
    /// <returns>The result of the optimization process.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the main optimization loop of the Adagrad algorithm. It iteratively
    /// updates the solution based on calculated gradients and accumulated squared gradients.
    /// </para>
    /// <para><b>For Beginners:</b> This is the main learning process of the Adagrad optimizer.
    /// 
    /// Here's what happens in each iteration:
    /// 1. Calculate how to improve the current solution (gradient)
    /// 2. Update the memory of past improvements (accumulated squared gradients)
    /// 3. Create a new, hopefully better solution
    /// 4. Check if this new solution is the best so far
    /// 5. Adjust how the optimizer learns (adaptive parameters)
    /// 6. Check if we should stop early (if the solution is good enough)
    /// 
    /// This process repeats until we reach the maximum number of iterations or find a good enough solution.
    /// </para>
    /// </remarks>
    public override OptimizationResult<T> Optimize(OptimizationInputData<T> inputData)
    {
        ValidationHelper<T>.ValidateInputData(inputData);

        var currentSolution = InitializeRandomSolution(inputData.XTrain.Columns);
        var bestStepData = new OptimizationStepData<T>();
        var previousStepData = new OptimizationStepData<T>();

        _accumulatedSquaredGradients = new Vector<T>(currentSolution.Coefficients.Length);
        InitializeAdaptiveParameters();

        for (int iteration = 0; iteration < _options.MaxIterations; iteration++)
        {
            var gradient = CalculateGradient(currentSolution, inputData.XTrain, inputData.YTrain);
            UpdateAccumulatedSquaredGradients(gradient);
            var newSolution = UpdateSolution(currentSolution, gradient);

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
    /// Updates the accumulated squared gradients used in the Adagrad algorithm.
    /// </summary>
    /// <param name="gradient">The current gradient vector.</param>
    /// <remarks>
    /// <para>
    /// This method updates the accumulated squared gradients by adding the square of each gradient component.
    /// These accumulated values are used to adapt the learning rate for each parameter.
    /// </para>
    /// <para><b>For Beginners:</b> This is like updating the optimizer's memory of past improvements.
    /// 
    /// For each piece of the solution:
    /// 1. Square the current improvement (gradient)
    /// 2. Add this square to the memory of past improvements
    /// 
    /// This memory helps the optimizer decide how much to change each part of the solution in future steps.
    /// Parts with a history of larger improvements will get smaller changes, and vice versa.
    /// </para>
    /// </remarks>
    private void UpdateAccumulatedSquaredGradients(Vector<T> gradient)
    {
        for (int i = 0; i < _accumulatedSquaredGradients!.Length; i++)
        {
            _accumulatedSquaredGradients[i] = NumOps.Add(
                _accumulatedSquaredGradients[i],
                NumOps.Multiply(gradient[i], gradient[i])
            );
        }
    }

    /// <summary>
    /// Updates the current solution using the Adagrad update rule.
    /// </summary>
    /// <param name="currentSolution">The current solution model.</param>
    /// <param name="gradient">The calculated gradient.</param>
    /// <returns>A new solution model after applying the Adagrad update.</returns>
    /// <remarks>
    /// <para>
    /// This method applies the Adagrad update rule to each coefficient of the current solution.
    /// It uses the accumulated squared gradients to adapt the learning rate for each parameter.
    /// </para>
    /// <para><b>For Beginners:</b> This is like taking a step towards a better solution.
    /// 
    /// For each part of the solution:
    /// 1. Calculate a custom learning rate based on past improvements
    /// 2. Use this rate to decide how big a step to take
    /// 3. Take the step by updating that part of the solution
    /// 
    /// This adaptive approach allows the optimizer to take larger steps for less frequently updated parts
    /// and smaller steps for more frequently updated parts.
    /// </para>
    /// </remarks>
    private ISymbolicModel<T> UpdateSolution(ISymbolicModel<T> currentSolution, Vector<T> gradient)
    {
        var newCoefficients = new Vector<T>(currentSolution.Coefficients.Length);
        for (int i = 0; i < currentSolution.Coefficients.Length; i++)
        {
            var adaptiveLearningRate = NumOps.Divide(
                CurrentLearningRate,
                NumOps.Add(NumOps.Sqrt(_accumulatedSquaredGradients![i]), NumOps.FromDouble(_options.Epsilon))
            );
            newCoefficients[i] = NumOps.Subtract(
                currentSolution.Coefficients[i],
                NumOps.Multiply(adaptiveLearningRate, gradient[i])
            );
        }
        return new VectorModel<T>(newCoefficients);
    }

    /// <summary>
    /// Updates the adaptive parameters of the Adagrad optimizer.
    /// </summary>
    /// <param name="currentStepData">The optimization step data for the current iteration.</param>
    /// <param name="previousStepData">The optimization step data for the previous iteration.</param>
    /// <remarks>
    /// <para>
    /// This method updates the learning rate if adaptive learning rate is enabled in the options.
    /// It increases or decreases the learning rate based on whether the current solution is better than the previous one.
    /// </para>
    /// <para><b>For Beginners:</b> This is like adjusting how fast the optimizer learns based on its recent progress.
    /// 
    /// If adaptive learning rate is turned on:
    /// - If the current solution is better, slightly increase the learning rate
    /// - If the current solution is worse, slightly decrease the learning rate
    /// - Keep the learning rate within specified limits
    /// 
    /// This helps the optimizer adapt its learning speed based on how well it's doing,
    /// potentially making the learning process more efficient.
    /// </para>
    /// </remarks>
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
    /// Updates the options for the Adagrad optimizer.
    /// </summary>
    /// <param name="options">The new options to be set.</param>
    /// <exception cref="ArgumentException">Thrown when the provided options are not of type AdagradOptimizerOptions.</exception>
    /// <remarks>
    /// <para>
    /// This method updates the optimizer's configuration with new options. It ensures that only
    /// AdagradOptimizerOptions are used to configure this optimizer.
    /// </para>
    /// <para><b>For Beginners:</b> This is like updating the instructions for your learning assistant.
    /// 
    /// - It checks if the new instructions are the right type for this specific assistant (Adagrad)
    /// - If they are, it updates the assistant's settings
    /// - If they're not, it reports an error
    /// 
    /// This helps prevent accidentally using the wrong type of settings, which could cause problems.
    /// </para>
    /// </remarks>
    protected override void UpdateOptions(OptimizationAlgorithmOptions options)
    {
        if (options is AdagradOptimizerOptions adagradOptions)
        {
            _options = adagradOptions;
        }
        else
        {
            throw new ArgumentException("Invalid options type. Expected AdagradOptimizerOptions.");
        }
    }

    /// <summary>
    /// Retrieves the current options of the Adagrad optimizer.
    /// </summary>
    /// <returns>The current AdagradOptimizerOptions.</returns>
    /// <remarks>
    /// <para>
    /// This method returns the current configuration options of the Adagrad optimizer.
    /// </para>
    /// <para><b>For Beginners:</b> This is like asking your learning assistant for its current instructions.
    /// 
    /// It allows you to check:
    /// - What learning rate the optimizer is using
    /// - How many iterations it will run
    /// - Other specific settings for the Adagrad method
    /// 
    /// This can be useful for understanding how the optimizer is currently set up or for saving its configuration.
    /// </para>
    /// </remarks>
    public override OptimizationAlgorithmOptions GetOptions()
    {
        return _options;
    }

    /// <summary>
    /// Serializes the Adagrad optimizer to a byte array.
    /// </summary>
    /// <returns>A byte array representing the serialized state of the optimizer.</returns>
    /// <remarks>
    /// <para>
    /// This method saves the current state of the Adagrad optimizer, including its base class state and specific options,
    /// into a byte array. This allows the optimizer's state to be stored or transmitted.
    /// </para>
    /// <para><b>For Beginners:</b> This is like taking a snapshot of your learning assistant's current state.
    /// 
    /// The process:
    /// 1. Saves the basic information (from the parent class)
    /// 2. Saves the specific Adagrad settings
    /// 3. Combines all this information into a single package (byte array)
    /// 
    /// This snapshot can be used later to recreate the exact same state of the optimizer,
    /// which is useful for saving progress or sharing the optimizer's configuration.
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
    /// Deserializes the Adagrad optimizer from a byte array.
    /// </summary>
    /// <param name="data">The byte array containing the serialized optimizer state.</param>
    /// <exception cref="InvalidOperationException">Thrown when deserialization of optimizer options fails.</exception>
    /// <remarks>
    /// <para>
    /// This method reconstructs the state of the Adagrad optimizer from a byte array, including its base class state
    /// and specific options. It's used to restore a previously serialized optimizer state.
    /// </para>
    /// <para><b>For Beginners:</b> This is like recreating your learning assistant from a saved snapshot.
    /// 
    /// The process:
    /// 1. Reads the basic information (for the parent class)
    /// 2. Recreates the parent class state
    /// 3. Reads and recreates the specific Adagrad settings
    /// 
    /// This allows you to continue using the optimizer from exactly where you left off,
    /// with all its learned information and settings intact.
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
            _options = JsonConvert.DeserializeObject<AdagradOptimizerOptions>(optionsJson)
                ?? throw new InvalidOperationException("Failed to deserialize optimizer options.");
        }
    }

    /// <summary>
    /// Generates a unique key for caching gradients based on the model, input data, and Adagrad-specific parameters.
    /// </summary>
    /// <param name="model">The symbolic model.</param>
    /// <param name="X">The input feature matrix.</param>
    /// <param name="y">The target vector.</param>
    /// <returns>A string representing the unique gradient cache key.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a unique identifier for caching gradients. It combines the base cache key with
    /// Adagrad-specific parameters to ensure that cached gradients are only reused when all relevant factors are identical.
    /// </para>
    /// <para><b>For Beginners:</b> This is like creating a unique label for each set of calculations.
    /// 
    /// The label includes:
    /// - Information about the model and data (from the base class)
    /// - Specific settings of the Adagrad optimizer (initial learning rate and epsilon)
    /// 
    /// This helps the optimizer quickly find and reuse previous calculations when the same situation occurs again,
    /// which can save time and computational resources.
    /// </para>
    /// </remarks>
    protected override string GenerateGradientCacheKey(ISymbolicModel<T> model, Matrix<T> X, Vector<T> y)
    {
        var baseKey = base.GenerateGradientCacheKey(model, X, y);
        return $"{baseKey}_Adagrad_{_options.InitialLearningRate}_{_options.Epsilon}";
    }
}