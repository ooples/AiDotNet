using Newtonsoft.Json;

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
public class AdaMaxOptimizer<T, TInput, TOutput> : GradientBasedOptimizerBase<T, TInput, TOutput>
{
    /// <summary>
    /// The configuration options specific to the AdaMax optimizer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the configuration parameters that control the behavior of the AdaMax algorithm,
    /// such as learning rate, beta values, and convergence criteria.
    /// </para>
    /// <para><b>For Beginners:</b> This is like the instruction manual for our learning assistant.
    /// It contains all the settings that determine how the optimizer behaves, such as how fast it learns
    /// and how it balances new information with what it already knows.
    /// </para>
    /// </remarks>
    private AdaMaxOptimizerOptions<T, TInput, TOutput> _options;

    /// <summary>
    /// The first moment vector that tracks the exponentially weighted moving average of gradients.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This vector maintains an exponentially decaying average of past gradients, which provides
    /// momentum to the optimization process and helps smooth out noisy gradient updates.
    /// </para>
    /// <para><b>For Beginners:</b> This is like the optimizer's memory of which direction it's been moving.
    /// 
    /// Imagine you're walking through a foggy forest:
    /// - Each step, you get a hint about which way to go (the gradient)
    /// - But instead of just following the latest hint, you remember a weighted average of all past hints
    /// - This helps you stay on a consistent path even if some hints are misleading
    /// 
    /// This "memory" helps the optimizer move more steadily toward the solution.
    /// </para>
    /// </remarks>
    private Vector<T>? _m; // First moment vector

    /// <summary>
    /// The exponentially weighted infinity norm of past gradients.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This vector tracks the maximum magnitude of gradients seen so far (with exponential decay),
    /// which AdaMax uses to scale the learning rate for each parameter individually.
    /// </para>
    /// <para><b>For Beginners:</b> This is like the optimizer's memory of how dramatic the changes have been.
    /// 
    /// Continuing our forest analogy:
    /// - For each direction you might walk, you remember the strongest hint you've ever received
    /// - This helps you decide how big or small your steps should be in each direction
    /// - If you've received strong hints to go north, you might take bigger steps north
    /// - If hints about going east have always been mild, you take smaller steps east
    /// 
    /// This adaptive step sizing helps the optimizer learn efficiently across different parameters.
    /// </para>
    /// </remarks>
    private Vector<T>? _u; // Exponentially weighted infinity norm

    /// <summary>
    /// The current time step or iteration counter.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This counter tracks how many optimization steps have been performed, which is used
    /// to correct the bias in the moment estimates during the early iterations.
    /// </para>
    /// <para><b>For Beginners:</b> This is like a counter that keeps track of how many learning steps we've taken.
    /// 
    /// It serves two important purposes:
    /// 1. It helps us know when to stop (if we've taken too many steps without improvement)
    /// 2. It's used in calculations to make the early learning steps more accurate
    /// 
    /// Without this counter, the optimizer might behave poorly in the first few iterations
    /// because it doesn't have enough history to make good decisions yet.
    /// </para>
    /// </remarks>
    private int _t; // Time step

    /// <summary>
    /// Initializes a new instance of the AdaMaxOptimizer class.
    /// </summary>
    /// <param name="model">The model to optimize.</param>
    /// <param name="options">The options for configuring the AdaMax optimizer.</param>
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
        IFullModel<T, TInput, TOutput> model,
        AdaMaxOptimizerOptions<T, TInput, TOutput>? options = null,
        IEngine? engine = null)
        : base(model, options ?? new())
    {
        _options = options ?? new AdaMaxOptimizerOptions<T, TInput, TOutput>();
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
    /// <para><b>DataLoader Integration:</b> This method uses the DataLoader API for efficient batch processing.
    /// It creates a batcher using <see cref="GradientBasedOptimizerBase{T,TInput,TOutput}.CreateBatcher"/>
    /// and notifies the sampler of epoch starts using
    /// <see cref="GradientBasedOptimizerBase{T,TInput,TOutput}.NotifyEpochStart"/>.
    /// </para>
    /// </remarks>
    public override OptimizationResult<T, TInput, TOutput> Optimize(OptimizationInputData<T, TInput, TOutput> inputData)
    {
        ValidationHelper<T>.ValidateInputData(inputData);

        var currentSolution = InitializeRandomSolution(inputData.XTrain);
        var parameters = currentSolution.GetParameters();
        var bestStepData = new OptimizationStepData<T, TInput, TOutput>();
        var previousStepData = new OptimizationStepData<T, TInput, TOutput>();

        _m = new Vector<T>(parameters.Length);
        _u = new Vector<T>(parameters.Length);
        InitializeAdaptiveParameters();

        for (int epoch = 0; epoch < _options.MaxIterations; epoch++)
        {
            NotifyEpochStart(epoch);
            var batcher = CreateBatcher(inputData, _options.BatchSize);

            foreach (var (xBatch, yBatch, batchIndices) in batcher.GetBatches())
            {
                _t++;
                var gradient = CalculateGradient(currentSolution, xBatch, yBatch);
                var newSolution = UpdateSolution(currentSolution, gradient);
                currentSolution = newSolution;
            }

            var currentStepData = EvaluateSolution(currentSolution, inputData);
            UpdateBestSolution(currentStepData, ref bestStepData);

            UpdateAdaptiveParameters(currentStepData, previousStepData);

            if (UpdateIterationHistoryAndCheckEarlyStopping(epoch, bestStepData))
            {
                return CreateOptimizationResult(bestStepData, inputData);
            }

            if (NumOps.LessThan(NumOps.Abs(NumOps.Subtract(bestStepData.FitnessScore, currentStepData.FitnessScore)), NumOps.FromDouble(_options.Tolerance)))
            {
                return CreateOptimizationResult(bestStepData, inputData);
            }

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
    protected override IFullModel<T, TInput, TOutput> UpdateSolution(IFullModel<T, TInput, TOutput> currentSolution, Vector<T> gradient)
    {
        var parameters = currentSolution.GetParameters();

        // Initialize state vectors if needed
        if (_m == null || _u == null || _m.Length != parameters.Length)
        {
            _m = new Vector<T>(parameters.Length);
            _u = new Vector<T>(parameters.Length);
            _t = 0;
        }

        // === Vectorized AdaMax Update using IEngine (Phase B: US-GPU-015) ===
        T beta1 = NumOps.FromDouble(_options.Beta1);
        T oneMinusBeta1 = NumOps.FromDouble(1 - _options.Beta1);
        T beta2 = NumOps.FromDouble(_options.Beta2);

        // Update biased first moment estimate: m = beta1 * m + (1 - beta1) * gradient
        var mScaled = (Vector<T>)Engine.Multiply(_m, beta1);
        var gradScaled = (Vector<T>)Engine.Multiply(gradient, oneMinusBeta1);
        _m = (Vector<T>)Engine.Add(mScaled, gradScaled);

        // Update exponentially weighted infinity norm: u = max(beta2 * u, |gradient|)
        // Fully vectorized using IEngine (Phase B: US-GPU-015)
        var uScaled = (Vector<T>)Engine.Multiply(_u, beta2);
        var absGradient = (Vector<T>)Engine.Abs(gradient);
        _u = (Vector<T>)Engine.Max(uScaled, absGradient);

        // Compute bias-corrected learning rate
        T alpha = NumOps.Divide(CurrentLearningRate, NumOps.FromDouble(1 - Math.Pow(_options.Beta1, _t)));

        // Update parameters: params = params - (alpha * m) / (u + epsilon)
        // Add epsilon to prevent 0/0 division when gradients are always zero
        T epsilon = NumOps.FromDouble(1e-8);
        var epsilonVec = Vector<T>.CreateDefault(_u.Length, epsilon);
        var uSafe = (Vector<T>)Engine.Add(_u, epsilonVec);
        var alphaMScaled = (Vector<T>)Engine.Multiply(_m, alpha);
        var update = (Vector<T>)Engine.Divide(alphaMScaled, uSafe);
        var newCoefficients = (Vector<T>)Engine.Subtract(parameters, update);

        return currentSolution.WithParameters(newCoefficients);
    }

    /// <summary>
    /// Updates a vector of parameters using the AdaMax optimization algorithm.
    /// </summary>
    /// <param name="parameters">The current parameter vector to be updated.</param>
    /// <param name="gradient">The gradient vector corresponding to the parameters.</param>
    /// <returns>The updated parameter vector.</returns>
    /// <remarks>
    /// <para>
    /// AdaMax is a variant of Adam based on the infinity norm, which can be more stable than Adam for
    /// some problems. It adapts the learning rate using the maximum absolute value of gradients.
    /// </para>
    /// <para><b>For Beginners:</b> AdaMax adjusts step sizes by tracking the largest gradient magnitude
    /// seen so far for each parameter. This makes it robust to large, occasional gradient spikes.
    /// </para>
    /// </remarks>
    public override Vector<T> UpdateParameters(Vector<T> parameters, Vector<T> gradient)
    {
        if (_m == null || _u == null || _m.Length != parameters.Length)
        {
            _m = new Vector<T>(parameters.Length);
            _u = new Vector<T>(parameters.Length);
            _t = 0;
        }

        _t++;

        // === Vectorized AdaMax Update using IEngine (Phase B: US-GPU-015) ===
        T beta1 = NumOps.FromDouble(_options.Beta1);
        T oneMinusBeta1 = NumOps.FromDouble(1 - _options.Beta1);
        T beta2 = NumOps.FromDouble(_options.Beta2);

        // Update biased first moment estimate: m = beta1 * m + (1 - beta1) * gradient
        var mScaled = (Vector<T>)Engine.Multiply(_m, beta1);
        var gradScaled = (Vector<T>)Engine.Multiply(gradient, oneMinusBeta1);
        _m = (Vector<T>)Engine.Add(mScaled, gradScaled);

        // Update exponentially weighted infinity norm: u = max(beta2 * u, |gradient|)
        var uScaled = (Vector<T>)Engine.Multiply(_u, beta2);
        var absGradient = (Vector<T>)Engine.Abs(gradient);
        _u = (Vector<T>)Engine.Max(uScaled, absGradient);

        // Compute bias-corrected learning rate
        T alpha = NumOps.Divide(CurrentLearningRate, NumOps.FromDouble(1 - Math.Pow(_options.Beta1, _t)));

        // Update parameters: params = params - (alpha * m) / (u + epsilon)
        // Add epsilon to prevent 0/0 division when gradients are always zero
        T epsilon = NumOps.FromDouble(1e-8);
        var epsilonVec = Vector<T>.CreateDefault(_u.Length, epsilon);
        var uSafe = (Vector<T>)Engine.Add(_u, epsilonVec);
        var alphaMScaled = (Vector<T>)Engine.Multiply(_m, alpha);
        var update = (Vector<T>)Engine.Divide(alphaMScaled, uSafe);
        var updatedParams = (Vector<T>)Engine.Subtract(parameters, update);

        return updatedParams;
    }

    /// <summary>
    /// Reverses an AdaMax gradient update to recover original parameters.
    /// </summary>
    /// <param name="updatedParameters">Parameters after AdaMax update</param>
    /// <param name="appliedGradients">The gradients that were applied</param>
    /// <returns>Original parameters before the update</returns>
    /// <remarks>
    /// <para>
    /// AdaMax's reverse update requires the optimizer's internal state (_m, _u, _t) from the forward pass.
    /// This method must be called immediately after UpdateParameters while the state is fresh.
    /// It recalculates the bias-corrected learning rate and the infinity-norm-scaled update.
    /// </para>
    /// <para><b>For Beginners:</b> This calculates where parameters were before an AdaMax update.
    /// AdaMax uses the maximum gradient magnitude to scale updates, so we need to remember those
    /// maximum values (_u) and the momentum (_m) to reverse the step accurately.
    /// </para>
    /// </remarks>
    public override Vector<T> ReverseUpdate(Vector<T> updatedParameters, Vector<T> appliedGradients)
    {
        if (updatedParameters == null)
            throw new ArgumentNullException(nameof(updatedParameters));
        if (appliedGradients == null)
            throw new ArgumentNullException(nameof(appliedGradients));

        if (updatedParameters.Length != appliedGradients.Length)
        {
            throw new ArgumentException(
                $"Updated parameters size ({updatedParameters.Length}) must match applied gradients size ({appliedGradients.Length})",
                nameof(appliedGradients));
        }

        if (_m == null || _u == null || _m.Length != updatedParameters.Length)
        {
            throw new InvalidOperationException(
                "AdaMax optimizer state is not initialized. ReverseUpdate must be called after UpdateParameters.");
        }

        // === Vectorized Reverse AdaMax Update using IEngine (Phase B: US-GPU-015) ===
        // Recalculate the bias-corrected learning rate (same for all elements)
        T alpha = NumOps.Divide(CurrentLearningRate, NumOps.FromDouble(1 - Math.Pow(_options.Beta1, _t)));
        var alphaVec = Vector<T>.CreateDefault(updatedParameters.Length, alpha);

        // Recalculate the update that was applied: update = (alpha * m) / (u + epsilon)
        T epsilon = NumOps.FromDouble(1e-8);
        var epsilonVec = Vector<T>.CreateDefault(_u.Length, epsilon);
        var uSafe = (Vector<T>)Engine.Add(_u, epsilonVec);
        var alphaTimes_m = (Vector<T>)Engine.Multiply(alphaVec, _m);
        var update = (Vector<T>)Engine.Divide(alphaTimes_m, uSafe);

        // Reverse: original = updated + update
        return (Vector<T>)Engine.Add(updatedParameters, update);
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
    protected override void UpdateOptions(OptimizationAlgorithmOptions<T, TInput, TOutput> options)
    {
        if (options is AdaMaxOptimizerOptions<T, TInput, TOutput> adaMaxOptions)
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
    public override OptimizationAlgorithmOptions<T, TInput, TOutput> GetOptions()
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
            _options = JsonConvert.DeserializeObject<AdaMaxOptimizerOptions<T, TInput, TOutput>>(optionsJson)
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
    protected override string GenerateGradientCacheKey(IFullModel<T, TInput, TOutput> model, TInput X, TOutput y)
    {
        var baseKey = base.GenerateGradientCacheKey(model, X, y);
        return $"{baseKey}_AdaMax_{_options.Beta1}_{_options.Beta2}_{_t}";
    }
}
