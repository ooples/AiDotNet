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
public class AdaGradOptimizer<T, TInput, TOutput> : GradientBasedOptimizerBase<T, TInput, TOutput>
{
    /// <summary>
    /// The configuration options specific to the Adagrad optimizer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the configuration parameters that control the behavior of the Adagrad algorithm,
    /// such as learning rate, epsilon value, and convergence criteria.
    /// </para>
    /// <para><b>For Beginners:</b> This is like the instruction manual for our learning assistant.
    /// It contains all the settings that determine how the optimizer behaves, such as how fast it learns
    /// and when it should consider its job complete.
    /// 
    /// These settings can be customized to make the optimizer work better for different types of problems.
    /// </para>
    /// </remarks>
    private AdagradOptimizerOptions<T, TInput, TOutput> _options = default!;

    /// <summary>
    /// Stores the sum of squared gradients for each parameter during optimization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This vector accumulates the squared gradients for each parameter throughout the optimization process.
    /// It's used to adapt the learning rate individually for each parameter - parameters with larger
    /// accumulated gradients will have smaller learning rates and vice versa.
    /// </para>
    /// <para><b>For Beginners:</b> This is like the optimizer's memory of how much each part of the solution
    /// has changed over time.
    /// 
    /// Imagine you're learning different subjects:
    /// - For subjects you've practiced a lot (large accumulated gradients), you take smaller learning steps
    /// - For subjects you've rarely practiced (small accumulated gradients), you take larger learning steps
    /// 
    /// This helps the optimizer focus more on the parts of the solution that haven't received much attention,
    /// which is especially useful when some parameters are more important than others or appear less frequently.
    /// </para>
    /// </remarks>
    private Vector<T>? _accumulatedSquaredGradients;

    /// <summary>
    /// Initializes a new instance of the AdagradOptimizer class.
    /// </summary>
    /// <param name="model">The model to be optimized.</param>
    /// <param name="options">The options for configuring the Adagrad optimizer.</param>
    /// <remarks>
    /// <para>
    /// This constructor sets up the Adagrad optimizer with the specified model and options.
    /// If no options are provided, it uses default AdagradOptimizerOptions.
    /// </para>
    /// <para><b>For Beginners:</b> This is like setting up your learning assistant with specific instructions.
    /// 
    /// You provide:
    /// - model: The specific model you want to improve (like a recipe you want to perfect)
    /// - options: Special settings for Adagrad (like how fast it should learn)
    /// 
    /// If you don't specify options, it will use default settings.
    /// </para>
    /// </remarks>
    public AdaGradOptimizer(
        IFullModel<T, TInput, TOutput> model,
        AdagradOptimizerOptions<T, TInput, TOutput>? options = null)
        : base(model, options ?? new())
    {
        _options = options ?? new AdagradOptimizerOptions<T, TInput, TOutput>();

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
    /// creates and evaluates solutions based on the chosen optimization mode.
    /// </para>
    /// <para><b>For Beginners:</b> This is the main learning process of the Adagrad optimizer.
    /// 
    /// Here's what happens:
    /// 1. It starts with the provided model
    /// 2. In each step (iteration):
    ///    - It creates a potential solution based on the optimization mode (feature selection, parameter adjustment, or both)
    ///    - It evaluates the solution to see how good it is
    ///    - It keeps track of the best solution found so far
    ///    - It adjusts its approach based on how well it's doing
    ///    - It decides whether to stop early if the solution is good enough
    /// 3. It repeats this process until it reaches the maximum number of steps or finds a good enough solution
    /// 
    /// This is like practicing a skill over and over, getting a little better each time, until you're satisfied with your performance.
    /// </para>
    /// </remarks>
    public override OptimizationResult<T, TInput, TOutput> Optimize(OptimizationInputData<T, TInput, TOutput> inputData)
    {
        ValidationHelper<T>.ValidateInputData(inputData);

        var bestStepData = new OptimizationStepData<T, TInput, TOutput>
        {
            Solution = Model.DeepCopy(),
            FitnessScore = FitnessCalculator.IsHigherScoreBetter ? NumOps.MinValue : NumOps.MaxValue
        };
        var previousStepData = new OptimizationStepData<T, TInput, TOutput>();

        var parameters = Model.GetParameters();
        _accumulatedSquaredGradients = new Vector<T>(parameters.Length);
        InitializeAdaptiveParameters();

        for (int iteration = 0; iteration < _options.MaxIterations; iteration++)
        {
            var currentSolution = CreateSolution(inputData.XTrain);
            var currentStepData = EvaluateSolution(currentSolution, inputData);

            UpdateBestSolution(currentStepData, ref bestStepData);
            UpdateAdaptiveParameters(currentStepData, previousStepData);

            if (UpdateIterationHistoryAndCheckEarlyStopping(iteration, bestStepData))
            {
                break;
            }

            if (NumOps.LessThan(NumOps.Abs(NumOps.Subtract(bestStepData.FitnessScore, currentStepData.FitnessScore)), NumOps.FromDouble(_options.Tolerance)))
            {
                break;
            }

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
    protected override IFullModel<T, TInput, TOutput> UpdateSolution(IFullModel<T, TInput, TOutput> currentSolution, Vector<T> gradient)
    {
        var parameters = currentSolution.GetParameters();
        var newCoefficients = new Vector<T>(parameters.Length);

        // Update accumulated squared gradients
        UpdateAccumulatedSquaredGradients(gradient);

        for (int i = 0; i < parameters.Length; i++)
        {
            var adaptiveLearningRate = NumOps.Divide(
                CurrentLearningRate,
                NumOps.Add(NumOps.Sqrt(_accumulatedSquaredGradients![i]), NumOps.FromDouble(_options.Epsilon))
            );
            newCoefficients[i] = NumOps.Subtract(
                parameters[i],
                NumOps.Multiply(adaptiveLearningRate, gradient[i])
            );
        }

        return currentSolution.WithParameters(newCoefficients);
    }

    /// <summary>
    /// Updates the adaptive parameters of the Adagrad optimizer.
    /// </summary>
    /// <param name="currentStepData">The optimization step data for the current iteration.</param>
    /// <param name="previousStepData">The optimization step data for the previous iteration.</param>
    /// <remarks>
    /// <para>
    /// This method updates various adaptive parameters based on the performance of the current solution
    /// compared to the previous one. It can adjust the learning rate, feature selection parameters,
    /// and parameter adjustment settings.
    /// </para>
    /// <para><b>For Beginners:</b> This is like adjusting how the optimizer learns over time.
    /// 
    /// It looks at how well the current solution is doing compared to the previous one:
    /// - If things are improving, it might make smaller, more careful adjustments
    /// - If things are not improving, it might try more dramatic changes
    /// 
    /// This adaptive approach helps the optimizer be more efficient in finding the best solution.
    /// </para>
    /// </remarks>
    protected override void UpdateAdaptiveParameters(OptimizationStepData<T, TInput, TOutput> currentStepData, OptimizationStepData<T, TInput, TOutput> previousStepData)
    {
        // Call the base implementation to update common parameters
        base.UpdateAdaptiveParameters(currentStepData, previousStepData);

        // Skip if previous step data is null (first iteration)
        if (previousStepData.Solution == null)
            return;

        bool isImproving = FitnessCalculator.IsBetterFitness(currentStepData.FitnessScore, previousStepData.FitnessScore);

        // Adaptive feature selection parameters
        if ((_options.OptimizationMode == OptimizationMode.FeatureSelectionOnly ||
             _options.OptimizationMode == OptimizationMode.Both))
        {
            UpdateFeatureSelectionParameters(isImproving);
        }

        // Adaptive parameter adjustment settings
        if ((_options.OptimizationMode == OptimizationMode.ParametersOnly ||
             _options.OptimizationMode == OptimizationMode.Both))
        {
            UpdateParameterAdjustmentSettings(isImproving);
        }

        // Adagrad specific adaptive parameters
        if (_options.UseAdaptiveLearningRate)
        {
            UpdateLearningRate(isImproving);
        }
    }

    /// <summary>
    /// Updates the learning rate based on whether the solution is improving.
    /// </summary>
    /// <param name="isImproving">Indicates whether the solution is improving.</param>
    /// <remarks>
    /// <para>
    /// This method adjusts the learning rate based on performance improvements.
    /// It increases the learning rate when the solution is improving and decreases it when it's not.
    /// </para>
    /// <para><b>For Beginners:</b> This method adjusts how big the steps are that the optimizer takes.
    /// 
    /// If the solution is improving:
    /// - It slightly increases the learning rate, taking slightly bigger steps
    /// - This helps the optimizer move faster toward better solutions
    /// 
    /// If the solution is not improving:
    /// - It slightly decreases the learning rate, taking smaller, more careful steps
    /// - This helps the optimizer be more precise when it's near a good solution
    /// </para>
    /// </remarks>
    private void UpdateLearningRate(bool isImproving)
    {
        if (isImproving)
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

    /// <summary>
    /// Updates the feature selection parameters based on whether the solution is improving.
    /// </summary>
    /// <param name="isImproving">Indicates whether the solution is improving.</param>
    /// <remarks>
    /// <para>
    /// This method adjusts the feature selection parameters, such as the minimum and maximum
    /// number of features to select and the probability of performing feature selection.
    /// </para>
    /// <para><b>For Beginners:</b> This method adjusts how the optimizer selects ingredients for the recipe.
    /// 
    /// If the solution is improving:
    /// - It might expand the range of ingredients it considers
    /// - It might be more likely to try changing ingredients in future steps
    /// 
    /// If the solution is not improving:
    /// - It might narrow its focus to a smaller set of ingredients
    /// - It might be less aggressive about changing ingredients
    /// </para>
    /// </remarks>
    private void UpdateFeatureSelectionParameters(bool isImproving)
    {
        if (isImproving)
        {
            // If improving, gradually expand the range of features to consider
            _options.MinimumFeatures = Math.Max(1, _options.MinimumFeatures - 1);
            _options.MaximumFeatures = Math.Min(_options.MaximumFeatures + 1, _options.AbsoluteMaximumFeatures);

            // Slightly increase the probability of feature selection for future iterations
            _options.FeatureSelectionProbability *= 1.02;
        }
        else
        {
            // If not improving, narrow the range to focus the search
            _options.MinimumFeatures = Math.Min(_options.MinimumFeatures + 1, _options.AbsoluteMaximumFeatures - 1);
            _options.MaximumFeatures = Math.Max(_options.MaximumFeatures - 1, _options.MinimumFeatures + 1);

            // Slightly decrease the probability of feature selection for future iterations
            _options.FeatureSelectionProbability *= 0.98;
        }

        // Ensure probabilities stay within bounds
        _options.FeatureSelectionProbability = MathHelper.Clamp(
            _options.FeatureSelectionProbability,
            _options.MinFeatureSelectionProbability,
            _options.MaxFeatureSelectionProbability);
    }

    /// <summary>
    /// Updates the parameter adjustment settings based on whether the solution is improving.
    /// </summary>
    /// <param name="isImproving">Indicates whether the solution is improving.</param>
    /// <remarks>
    /// <para>
    /// This method adjusts the parameter adjustment settings, such as the adjustment scale,
    /// sign flip probability, and parameter adjustment probability.
    /// </para>
    /// <para><b>For Beginners:</b> This method adjusts how the optimizer changes the amounts of ingredients.
    /// 
    /// If the solution is improving:
    /// - It might make smaller, more precise adjustments
    /// - It might be less likely to make dramatic changes (like flipping signs)
    /// - It might be more likely to adjust parameters in future steps
    /// 
    /// If the solution is not improving:
    /// - It might make larger adjustments to explore more options
    /// - It might be more willing to try dramatic changes
    /// - It might adjust its strategy to try something different
    /// </para>
    /// </remarks>
    private void UpdateParameterAdjustmentSettings(bool isImproving)
    {
        if (isImproving)
        {
            // If improving, make smaller adjustments to fine-tune
            _options.ParameterAdjustmentScale *= 0.95;

            // Decrease the probability of sign flips when things are going well
            _options.SignFlipProbability *= 0.9;

            // Increase the probability of parameter adjustments
            _options.ParameterAdjustmentProbability *= 1.02;
        }
        else
        {
            // If not improving, make larger adjustments to explore more
            _options.ParameterAdjustmentScale *= 1.05;

            // Increase the probability of sign flips to try more dramatic changes
            _options.SignFlipProbability *= 1.1;

            // Slightly decrease the probability of parameter adjustments
            _options.ParameterAdjustmentProbability *= 0.98;
        }

        // Ensure values stay within bounds
        _options.ParameterAdjustmentScale = MathHelper.Clamp(
            _options.ParameterAdjustmentScale,
            _options.MinParameterAdjustmentScale,
            _options.MaxParameterAdjustmentScale);

        _options.SignFlipProbability = MathHelper.Clamp(
            _options.SignFlipProbability,
            _options.MinSignFlipProbability,
            _options.MaxSignFlipProbability);

        _options.ParameterAdjustmentProbability = MathHelper.Clamp(
            _options.ParameterAdjustmentProbability,
            _options.MinParameterAdjustmentProbability,
            _options.MaxParameterAdjustmentProbability);
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
    protected override void UpdateOptions(OptimizationAlgorithmOptions<T, TInput, TOutput> options)
    {
        if (options is AdagradOptimizerOptions<T, TInput, TOutput> adagradOptions)
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
    public override OptimizationAlgorithmOptions<T, TInput, TOutput> GetOptions()
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

            // Serialize optimization mode
            writer.Write((int)_options.OptimizationMode);

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

            // Deserialize optimization mode
            _options.OptimizationMode = (OptimizationMode)reader.ReadInt32();

            string optionsJson = reader.ReadString();
            _options = JsonConvert.DeserializeObject<AdagradOptimizerOptions<T, TInput, TOutput>>(optionsJson)
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
    /// - The optimization mode being used
    /// 
    /// This helps the optimizer quickly find and reuse previous calculations when the same situation occurs again,
    /// which can save time and computational resources.
    /// </para>
    /// </remarks>
    protected override string GenerateGradientCacheKey(IFullModel<T, TInput, TOutput> model, TInput X, TOutput y)
    {
        var baseKey = base.GenerateGradientCacheKey(model, X, y);
        return $"{baseKey}_Adagrad_{_options.InitialLearningRate}_{_options.Epsilon}_{_options.OptimizationMode}";
    }
}