namespace AiDotNet.Optimizers;

/// <summary>
/// Represents an AdaMax optimizer, an extension of Adam that uses the infinity norm.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// AdaMax is an adaptive learning rate optimization algorithm that extends the Adam optimizer.
/// It uses the infinity norm to update parameters, which can make it more robust in certain scenarios.
/// </para>
/// <para><b>For Beginners:</b> AdaMax is like a smart learning assistant that adjusts its learning speed
/// for each piece of information it's trying to learn. It's particularly good at handling different
/// scales of information without getting confused.
/// 
/// Key features:
/// - Adapts the learning rate for each parameter
/// - Uses the maximum (infinity norm) of past gradients, which can be more stable
/// - Good for problems where the gradients can be sparse or have different scales
/// </para>
/// </remarks>
public class AdaMaxOptimizer<T> : GradientBasedOptimizerBase<T>
{
    private AdaMaxOptimizerOptions _options;
    private Vector<T>? _m; // First moment vector
    private Vector<T>? _u; // Exponentially weighted infinity norm
    private int _t; // Time step

    /// <summary>
    /// Initializes a new instance of the AdaMaxOptimizer class.
    /// </summary>
    /// <param name="options">The options for configuring the AdaMax optimizer.</param>
    /// <param name="predictionOptions">Options for prediction statistics.</param>
    /// <param name="modelOptions">Options for model statistics.</param>
    /// <param name="modelEvaluator">The model evaluator to use.</param>
    /// <param name="fitDetector">The fit detector to use.</param>
    /// <param name="fitnessCalculator">The fitness calculator to use.</param>
    /// <param name="modelCache">The model cache to use.</param>
    /// <param name="gradientCache">The gradient cache to use.</param>
    /// <remarks>
    /// <para>
    /// This constructor sets up the AdaMax optimizer with the specified options and components.
    /// If no options are provided, it uses default AdaMaxOptimizerOptions.
    /// </para>
    /// <para><b>For Beginners:</b> This is like setting up your smart learning assistant with specific instructions.
    /// 
    /// You can customize:
    /// - How fast it learns (learning rate)
    /// - How it remembers past information (beta parameters)
    /// - How long it should try to learn (max iterations)
    /// - And many other aspects of its learning process
    /// 
    /// If you don't provide custom settings, it will use default settings that work well in many situations.
    /// </para>
    /// </remarks>
    public AdaMaxOptimizer(
        AdaMaxOptimizerOptions? options = null,
        PredictionStatsOptions? predictionOptions = null,
        ModelStatsOptions? modelOptions = null,
        IModelEvaluator<T>? modelEvaluator = null,
        IFitDetector<T>? fitDetector = null,
        IFitnessCalculator<T>? fitnessCalculator = null,
        IModelCache<T>? modelCache = null,
        IGradientCache<T>? gradientCache = null)
        : base(options, predictionOptions, modelOptions, modelEvaluator, fitDetector, fitnessCalculator, modelCache, gradientCache)
    {
        _options = options ?? new AdaMaxOptimizerOptions();
        InitializeAdaptiveParameters();
    }

    /// <summary>
    /// Initializes the adaptive parameters for the AdaMax optimizer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method sets up the initial state of the optimizer, including the learning rate and time step.
    /// </para>
    /// <para><b>For Beginners:</b> This is like resetting your learning assistant to its starting point.
    /// 
    /// It does two main things:
    /// 1. Sets the initial learning speed (learning rate) based on the options you provided
    /// 2. Resets the time step to 0, which is like starting a new learning session
    /// 
    /// This method is called when you first create the optimizer and can be called again if you want to restart the learning process.
    /// </para>
    /// </remarks>
    protected override void InitializeAdaptiveParameters()
    {
        base.InitializeAdaptiveParameters();
        CurrentLearningRate = NumOps.FromDouble(_options.LearningRate);
        _t = 0;
    }

    /// <summary>
    /// Performs the optimization process using the AdaMax algorithm.
    /// </summary>
    /// <param name="inputData">The input data for optimization, including training data and targets.</param>
    /// <returns>The result of the optimization process, including the best solution found.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the core optimization loop of the AdaMax algorithm. It iteratively improves
    /// the solution by calculating gradients, updating parameters, and evaluating the current solution.
    /// </para>
    /// <para><b>For Beginners:</b> This method is like a smart learning process that tries to find the best answer.
    /// 
    /// Here's what it does:
    /// 1. Starts with a random guess (solution)
    /// 2. Repeatedly tries to improve the guess:
    ///    - Calculates how to change the guess to make it better (gradient)
    ///    - Updates the guess based on this information
    ///    - Checks if the new guess is the best one so far
    /// 3. Stops when it has tried a certain number of times or when the improvement becomes very small
    /// 
    /// It's like playing a game where you're trying to find a hidden treasure, and after each step,
    /// you get a hint about which direction to go next.
    /// </para>
    /// </remarks>
    public override OptimizationResult<T> Optimize(OptimizationInputData<T> inputData)
    {
        ValidationHelper<T>.ValidateInputData(inputData);

        var currentSolution = InitializeRandomSolution(inputData.XTrain.Columns);
        var bestStepData = new OptimizationStepData<T>();
        var previousStepData = new OptimizationStepData<T>();

        _m = new Vector<T>(currentSolution.Coefficients.Length);
        _u = new Vector<T>(currentSolution.Coefficients.Length);
        InitializeAdaptiveParameters();

        for (int iteration = 0; iteration < _options.MaxIterations; iteration++)
        {
            _t++;
            var gradient = CalculateGradient(currentSolution, inputData.XTrain, inputData.YTrain);
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
    /// Updates the current solution using the AdaMax update rule.
    /// </summary>
    /// <param name="currentSolution">The current solution being optimized.</param>
    /// <param name="gradient">The calculated gradient for the current solution.</param>
    /// <returns>A new solution with updated parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method applies the AdaMax update rule to adjust the parameters of the current solution.
    /// It uses moment estimates and the infinity norm to adapt the learning rate for each parameter.
    /// </para>
    /// <para><b>For Beginners:</b> This method fine-tunes our current guess to make it better.
    /// 
    /// Imagine you're adjusting the volume and bass on a stereo:
    /// - The current solution is like the current settings
    /// - The gradient tells us how to adjust each knob
    /// - We don't just follow the gradient directly; we use some clever math (AdaMax rules) to decide
    ///   how much to turn each knob
    /// - This clever math helps us avoid overreacting to any single piece of information
    /// 
    /// The result is a new, slightly improved set of stereo settings (or in our case, a better solution).
    /// </para>
    /// </remarks>
    private ISymbolicModel<T> UpdateSolution(ISymbolicModel<T> currentSolution, Vector<T> gradient)
    {
        var newCoefficients = new Vector<T>(currentSolution.Coefficients.Length);
        var beta1 = NumOps.FromDouble(_options.Beta1);
        var oneMinusBeta1 = NumOps.FromDouble(1 - _options.Beta1);
        var beta2 = NumOps.FromDouble(_options.Beta2);

        for (int i = 0; i < currentSolution.Coefficients.Length; i++)
        {
            // Update biased first moment estimate
            _m![i] = NumOps.Add(NumOps.Multiply(beta1, _m[i]), NumOps.Multiply(oneMinusBeta1, gradient[i]));

            // Update the exponentially weighted infinity norm
            _u![i] = MathHelper.Max(NumOps.Multiply(beta2, _u[i]), NumOps.Abs(gradient[i]));

            // Compute the learning rate
            var alpha = NumOps.Divide(CurrentLearningRate, NumOps.FromDouble(1 - Math.Pow(_options.Beta1, _t)));

            // Update parameters
            var update = NumOps.Divide(NumOps.Multiply(alpha, _m[i]), _u[i]);
            newCoefficients[i] = NumOps.Subtract(currentSolution.Coefficients[i], update);
        }

        return new VectorModel<T>(newCoefficients);
    }

    /// <summary>
    /// Updates the adaptive parameters of the optimizer based on the current and previous optimization steps.
    /// </summary>
    /// <param name="currentStepData">Data from the current optimization step.</param>
    /// <param name="previousStepData">Data from the previous optimization step.</param>
    /// <remarks>
    /// <para>
    /// This method adjusts the learning rate based on the performance of the current solution compared to the previous one.
    /// If adaptive learning rate is enabled, it increases or decreases the learning rate accordingly.
    /// </para>
    /// <para><b>For Beginners:</b> This method adjusts how big steps we take in our learning process.
    /// 
    /// It's like learning to ride a bike:
    /// - If you're doing better (not falling as much), you might try to pedal a bit faster (increase learning rate)
    /// - If you're struggling more, you might slow down a bit (decrease learning rate)
    /// - There's a limit to how fast or slow you can go (min and max learning rates)
    /// 
    /// This helps the optimizer to learn efficiently: not too slow, but also not so fast that it becomes unstable.
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

            CurrentLearningRate = MathHelper.Clamp(CurrentLearningRate, 
                NumOps.FromDouble(_options.MinLearningRate), 
                NumOps.FromDouble(_options.MaxLearningRate));
        }
    }

    /// <summary>
    /// Updates the optimizer options with new AdaMax-specific options.
    /// </summary>
    /// <param name="options">The new options to set.</param>
    /// <exception cref="ArgumentException">Thrown when the provided options are not of type AdaMaxOptimizerOptions.</exception>
    /// <remarks>
    /// <para>
    /// This method updates the optimizer's configuration with new options. It ensures that only valid
    /// AdaMax-specific options are applied.
    /// </para>
    /// <para><b>For Beginners:</b> This method is like updating the settings on your learning assistant.
    /// 
    /// Imagine you have a robot helper for studying:
    /// - You can give it new instructions on how to help you (new options)
    /// - But you need to make sure you're giving it the right kind of instructions (AdaMax-specific)
    /// - If you try to give it instructions for a different type of helper, it will let you know there's a mistake
    /// 
    /// This ensures that your optimizer always has the correct and up-to-date settings to work with.
    /// </para>
    /// </remarks>
    protected override void UpdateOptions(OptimizationAlgorithmOptions options)
    {
        if (options is AdaMaxOptimizerOptions adaMaxOptions)
        {
            _options = adaMaxOptions;
        }
        else
        {
            throw new ArgumentException("Invalid options type. Expected AdaMaxOptimizerOptions.");
        }
    }

    /// <summary>
    /// Gets the current options of the AdaMax optimizer.
    /// </summary>
    /// <returns>The current AdaMaxOptimizerOptions.</returns>
    /// <remarks>
    /// <para>
    /// This method returns the current configuration options of the AdaMax optimizer.
    /// </para>
    /// <para><b>For Beginners:</b> This method lets you see the current settings of your learning assistant.
    /// 
    /// It's like checking the current settings on your study robot:
    /// - You can see how fast it's set to work (learning rate)
    /// - How much it remembers from past lessons (beta parameters)
    /// - How long it's supposed to study for (max iterations)
    /// 
    /// This is useful if you want to know exactly how your optimizer is currently configured.
    /// </para>
    /// </remarks>
    public override OptimizationAlgorithmOptions GetOptions()
    {
        return _options;
    }

    /// <summary>
    /// Converts the current state of the optimizer into a byte array for storage or transmission.
    /// </summary>
    /// <returns>A byte array representing the serialized state of the optimizer.</returns>
    /// <remarks>
    /// <para>
    /// This method saves the current state of the optimizer, including its options and internal counters,
    /// into a compact binary format.
    /// </para>
    /// <para><b>For Beginners:</b> This method is like taking a snapshot of your learning assistant's brain.
    /// 
    /// Imagine you could:
    /// - Take a picture of everything your study robot knows and how it's set up
    /// - Turn that picture into a long string of numbers
    /// - Save those numbers so you can perfectly recreate the robot's state later
    /// 
    /// This is useful for:
    /// - Saving your progress so you can continue later
    /// - Sharing your optimizer's exact state with others
    /// - Creating backups in case something goes wrong
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

            writer.Write(_t);

            return ms.ToArray();
        }
    }

    /// <summary>
    /// Restores the optimizer's state from a byte array created by the Serialize method.
    /// </summary>
    /// <param name="data">The byte array containing the serialized optimizer state.</param>
    /// <exception cref="InvalidOperationException">Thrown when deserialization of optimizer options fails.</exception>
    /// <remarks>
    /// <para>
    /// This method reconstructs the optimizer's state, including its options and internal counters,
    /// from a binary format created by the Serialize method.
    /// </para>
    /// <para><b>For Beginners:</b> This method is like rebuilding your learning assistant's brain from a saved picture.
    /// 
    /// Imagine you have a robot helper that you previously "photographed" (serialized):
    /// 1. You give it the "photograph" (byte array)
    /// 2. It reads the photograph piece by piece:
    ///    - First, it rebuilds its basic knowledge (base data)
    ///    - Then, it sets up its specific AdaMax settings (options)
    ///    - Finally, it remembers how long it has been learning (time step)
    /// 3. If anything goes wrong while reading the settings, it lets you know
    /// 
    /// After this process, your robot helper is back to exactly the same state it was in when you took the "photograph".
    /// This is useful for:
    /// - Continuing a learning session that was paused
    /// - Setting up multiple identical helpers
    /// - Recovering from a backup if something goes wrong
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
            _options = JsonConvert.DeserializeObject<AdaMaxOptimizerOptions>(optionsJson)
                ?? throw new InvalidOperationException("Failed to deserialize optimizer options.");

            _t = reader.ReadInt32();
        }
    }

    /// <summary>
    /// Generates a unique key for caching gradients specific to the AdaMax optimizer.
    /// </summary>
    /// <param name="model">The current model being optimized.</param>
    /// <param name="X">The input data matrix.</param>
    /// <param name="y">The target values vector.</param>
    /// <returns>A string that uniquely identifies the gradient for the given model, data, and optimizer state.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a unique identifier for caching gradients. It extends the base gradient cache key
    /// with AdaMax-specific parameters to ensure that cached gradients are only reused when all relevant
    /// conditions are identical.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates a special label for storing and retrieving
    /// calculated gradients.
    /// 
    /// Imagine you're solving a math problem:
    /// - The "base key" is like writing down the problem you're solving
    /// - Adding "AdaMax" tells us we're using this specific method to solve it
    /// - Including Beta1, Beta2, and t (time step) is like noting which specific tools and at what stage we're using them
    /// 
    /// This helps us quickly find the right answer if we've solved a very similar problem before,
    /// saving time and effort.
    /// </para>
    /// </remarks>
    protected override string GenerateGradientCacheKey(ISymbolicModel<T> model, Matrix<T> X, Vector<T> y)
    {
        var baseKey = base.GenerateGradientCacheKey(model, X, y);
        return $"{baseKey}_AdaMax_{_options.Beta1}_{_options.Beta2}_{_t}";
    }
}