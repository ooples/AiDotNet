namespace AiDotNet.Optimizers;

/// <summary>
/// Implements a normal optimization algorithm with adaptive parameters that can optimize 
/// both feature selection and model parameters.
/// </summary>
/// <remarks>
/// <para>
/// The NormalOptimizer uses a combination of random search and adaptive parameter tuning to find optimal solutions.
/// It can optimize feature selection, model parameters (weights), or both, depending on the configuration.
/// </para>
/// <para><b>For Beginners:</b>
/// Think of this optimizer like a chef trying to perfect a recipe. It can adjust which ingredients to use 
/// (feature selection), the amount of each ingredient (parameter weights), or both. With each attempt, 
/// the optimizer learns from its results and makes smarter adjustments to find the best recipe.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class NormalOptimizer<T, TInput, TOutput> : OptimizerBase<T, TInput, TOutput>
{
    /// <summary>
    /// The specific configuration options for the normal optimization algorithm.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the detailed settings that control how the normal optimizer behaves,
    /// including parameters like adjustment scales, probabilities for different operations,
    /// and bounds for adaptive parameters.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// These are the specific rules the chef follows when experimenting with the recipe.
    /// They control things like how much the chef can change ingredient amounts, how often to try
    /// removing ingredients, and how the chef should adjust their approach based on previous results.
    /// </para>
    /// </remarks>
    private NormalOptimizerOptions<T, TInput, TOutput> _normalOptions = default!;

    /// <summary>
    /// Initializes a new instance of the NormalOptimizer class with specific options.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This constructor sets up the NormalOptimizer with the provided options which should include a model to optimize.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like telling the chef which dish to prepare (the model), what aspects to focus on improving 
    /// (ingredients, quantities, or both), and what cooking techniques to use (the options).
    /// </para>
    /// </remarks>
    /// <param name="options">The optimization options including the model to optimize.</param>
    /// <param name="optimizationMode">The aspects of the model to optimize (features, parameters, or both).</param>
    public NormalOptimizer(IFullModel<T, TInput, TOutput> model,
        NormalOptimizerOptions<T, TInput, TOutput>? options = null)
        : base(model, options ?? new())
    {
        _normalOptions = options ?? new();

        InitializeAdaptiveParameters();
    }

    /// <summary>
    /// Performs the optimization process.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method implements the main optimization loop. Based on the optimization mode,
    /// it may optimize feature selection, model parameters, or both. It generates potential solutions,
    /// evaluates them, and keeps track of the best solution found while adapting its strategy.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like the chef cooking multiple versions of a dish:
    /// 1. Creating variations of the recipe (by changing ingredients or quantities)
    /// 2. Tasting each version and rating it
    /// 3. Remembering the best version found so far
    /// 4. Learning from each attempt to make better adjustments next time
    /// 5. Deciding when the recipe is good enough to stop experimenting
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

        for (int iteration = 0; iteration < Options.MaxIterations; iteration++)
        {
            var currentSolution = CreateSolution(inputData.XTrain);
            var currentStepData = EvaluateSolution(currentSolution, inputData);

            UpdateBestSolution(currentStepData, ref bestStepData);

            // Update adaptive parameters
            UpdateAdaptiveParameters(currentStepData, previousStepData);

            if (UpdateIterationHistoryAndCheckEarlyStopping(iteration, bestStepData))
            {
                break;
            }

            previousStepData = currentStepData;
        }

        return CreateOptimizationResult(bestStepData, inputData);
    }

    /// <summary>
    /// Updates the adaptive parameters of the optimizer based on the current and previous optimization steps.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method adjusts various parameters of the optimization process based on the performance
    /// of the current solution compared to the previous one.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like adjusting your cooking strategy based on how your recent dishes turned out.
    /// If your changes led to improvements, you might continue in that direction. If not,
    /// you might try more dramatic changes to find a better approach.
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
            _normalOptions.MinimumFeatures = Math.Max(1, _normalOptions.MinimumFeatures - 1);
            _normalOptions.MaximumFeatures = Math.Min(_normalOptions.MaximumFeatures + 1, _normalOptions.AbsoluteMaximumFeatures);

            // Slightly increase the probability of feature selection for future iterations
            _normalOptions.FeatureSelectionProbability *= 1.02;
        }
        else
        {
            // If not improving, narrow the range to focus the search
            _normalOptions.MinimumFeatures = Math.Min(_normalOptions.MinimumFeatures + 1, _normalOptions.AbsoluteMaximumFeatures - 1);
            _normalOptions.MaximumFeatures = Math.Max(_normalOptions.MaximumFeatures - 1, _normalOptions.MinimumFeatures + 1);

            // Slightly decrease the probability of feature selection for future iterations
            _normalOptions.FeatureSelectionProbability *= 0.98;
        }

        // Ensure probabilities stay within bounds
        _normalOptions.FeatureSelectionProbability = MathHelper.Clamp(
            _normalOptions.FeatureSelectionProbability,
            _normalOptions.MinFeatureSelectionProbability,
            _normalOptions.MaxFeatureSelectionProbability);
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
            _normalOptions.ParameterAdjustmentScale *= 0.95;

            // Decrease the probability of sign flips when things are going well
            _normalOptions.SignFlipProbability *= 0.9;

            // Increase the probability of parameter adjustments
            _normalOptions.ParameterAdjustmentProbability *= 1.02;
        }
        else
        {
            // If not improving, make larger adjustments to explore more
            _normalOptions.ParameterAdjustmentScale *= 1.05;

            // Increase the probability of sign flips to try more dramatic changes
            _normalOptions.SignFlipProbability *= 1.1;

            // Slightly decrease the probability of parameter adjustments
            _normalOptions.ParameterAdjustmentProbability *= 0.98;
        }

        // Ensure values stay within bounds
        _normalOptions.ParameterAdjustmentScale = MathHelper.Clamp(
            _normalOptions.ParameterAdjustmentScale,
            _normalOptions.MinParameterAdjustmentScale,
            _normalOptions.MaxParameterAdjustmentScale);

        _normalOptions.SignFlipProbability = MathHelper.Clamp(
            _normalOptions.SignFlipProbability,
            _normalOptions.MinSignFlipProbability,
            _normalOptions.MaxSignFlipProbability);

        _normalOptions.ParameterAdjustmentProbability = MathHelper.Clamp(
            _normalOptions.ParameterAdjustmentProbability,
            _normalOptions.MinParameterAdjustmentProbability,
            _normalOptions.MaxParameterAdjustmentProbability);
    }

    /// <summary>
    /// Gets the current optimization algorithm options.
    /// </summary>
    /// <returns>The current optimization algorithm options.</returns>
    public override OptimizationAlgorithmOptions<T, TInput, TOutput> GetOptions()
    {
        return _normalOptions;
    }

    /// <summary>
    /// Updates the optimization algorithm options.
    /// </summary>
    /// <param name="options">The new optimization algorithm options to apply.</param>
    /// <exception cref="ArgumentException">Thrown when the provided options are not of the expected type.</exception>
    protected override void UpdateOptions(OptimizationAlgorithmOptions<T, TInput, TOutput> options)
    {
        if (options is NormalOptimizerOptions<T, TInput, TOutput> normalOptions)
        {
            _normalOptions = normalOptions;
        }
        else
        {
            throw new ArgumentException("Invalid options type. Expected NormalOptimizerOptions.");
        }
    }

    /// <summary>
    /// Serializes the current state of the optimizer into a byte array.
    /// </summary>
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

            // Serialize NormalOptimizerOptions
            string optionsJson = JsonConvert.SerializeObject(_normalOptions);
            writer.Write(optionsJson);

            return ms.ToArray();
        }
    }

    /// <summary>
    /// Deserializes a byte array to restore the optimizer's state.
    /// </summary>
    /// <param name="data">The byte array containing the serialized optimizer state.</param>
    /// <exception cref="InvalidOperationException">Thrown when deserialization fails.</exception>
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

            // Deserialize NormalOptimizerOptions
            string optionsJson = reader.ReadString();
            _normalOptions = JsonConvert.DeserializeObject<NormalOptimizerOptions<T, TInput, TOutput>>(optionsJson)
                ?? throw new InvalidOperationException("Failed to deserialize optimizer options.");
        }
    }
}