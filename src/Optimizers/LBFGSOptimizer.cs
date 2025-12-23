using Newtonsoft.Json;

namespace AiDotNet.Optimizers;

/// <summary>
/// Implements the Limited-memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS) optimization algorithm.
/// </summary>
/// <remarks>
/// <para>
/// L-BFGS is a quasi-Newton method for solving unconstrained nonlinear optimization problems. It approximates the
/// Broyden-Fletcher-Goldfarb-Shanno (BFGS) algorithm using a limited amount of computer memory, making it suitable 
/// for optimization problems with many variables.
/// </para>
/// <para><b>For Beginners:</b> 
/// L-BFGS is an advanced optimization algorithm that efficiently finds the minimum of a function, especially useful 
/// for problems with many variables. It uses information from previous iterations to make intelligent decisions 
/// about where to search next, while keeping memory usage low.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class LBFGSOptimizer<T, TInput, TOutput> : GradientBasedOptimizerBase<T, TInput, TOutput>
{
    /// <summary>
    /// Options specific to the L-BFGS optimizer.
    /// </summary>
    private LBFGSOptimizerOptions<T, TInput, TOutput> _options;

    /// <summary>
    /// List of position (solution) differences used in the L-BFGS update.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This list stores the differences between consecutive solutions, which are used to approximate the inverse Hessian matrix.
    /// </para>
    /// <para><b>For Beginners:</b> 
    /// Think of this as the optimizer's memory of how the solution has changed over recent iterations.
    /// </para>
    /// </remarks>
    private List<Vector<T>> _s;

    /// <summary>
    /// List of gradient differences used in the L-BFGS update.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This list stores the differences between consecutive gradients, which are used along with the solution differences 
    /// to approximate the inverse Hessian matrix.
    /// </para>
    /// <para><b>For Beginners:</b> 
    /// This represents how the direction of steepest descent has changed over recent iterations.
    /// </para>
    /// </remarks>
    private List<Vector<T>> _y;

    /// <summary>
    /// The current iteration count of the optimization process.
    /// </summary>
    private int _iteration;

    /// <summary>
    /// Initializes a new instance of the LBFGSOptimizer class.
    /// </summary>
    /// <param name="model">The model to optimize.</param>
    /// <param name="options">Options for the L-BFGS optimizer. If null, default options are used.</param>
    /// <param name="engine">The computation engine (CPU or GPU) for vectorized operations.</param>
    public LBFGSOptimizer(
        IFullModel<T, TInput, TOutput> model,
        LBFGSOptimizerOptions<T, TInput, TOutput>? options = null,
        IEngine? engine = null)
        : base(model, options ?? new())
    {
        _options = options ?? new LBFGSOptimizerOptions<T, TInput, TOutput>();
        _s = new List<Vector<T>>();
        _y = new List<Vector<T>>();

        InitializeAdaptiveParameters();
    }

    /// <summary>
    /// Initializes or resets the adaptive parameters used in the optimization process.
    /// </summary>
    protected override void InitializeAdaptiveParameters()
    {
        base.InitializeAdaptiveParameters();

        CurrentLearningRate = NumOps.FromDouble(_options.InitialLearningRate);
        _iteration = 0;
    }

    /// <summary>
    /// Performs the main optimization process using the L-BFGS algorithm.
    /// </summary>
    /// <param name="inputData">The input data for the optimization process.</param>
    /// <returns>The result of the optimization process.</returns>
    public override OptimizationResult<T, TInput, TOutput> Optimize(OptimizationInputData<T, TInput, TOutput> inputData)
    {
        ValidationHelper<T>.ValidateInputData(inputData);

        var currentSolution = InitializeRandomSolution(inputData.XTrain);
        var bestStepData = new OptimizationStepData<T, TInput, TOutput>();
        var previousStepData = new OptimizationStepData<T, TInput, TOutput>();

        _s.Clear();
        _y.Clear();
        InitializeAdaptiveParameters();

        Vector<T> previousGradient = Vector<T>.Empty();

        for (int iteration = 0; iteration < _options.MaxIterations; iteration++)
        {
            _iteration++;
            var gradient = CalculateGradient(currentSolution, inputData.XTrain, inputData.YTrain);
            var direction = CalculateDirection(gradient);
            var newSolution = UpdateSolution(currentSolution, direction, gradient, inputData);

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

            UpdateLBFGSMemory(currentSolution.GetParameters(), newSolution.GetParameters(), gradient, previousGradient);

            previousGradient = gradient;
            currentSolution = newSolution;
            previousStepData = currentStepData;
        }

        return CreateOptimizationResult(bestStepData, inputData);
    }

    /// <summary>
    /// Calculates the search direction using the L-BFGS algorithm.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method computes the search direction using the L-BFGS two-loop recursion algorithm. It uses the stored 
    /// solution and gradient differences to approximate the inverse Hessian matrix.
    /// </para>
    /// <para><b>For Beginners:</b> 
    /// This method determines the best direction to move in the solution space, using information from previous iterations 
    /// to make a more informed decision than just following the steepest descent.
    /// </para>
    /// </remarks>
    /// <param name="gradient">The current gradient.</param>
    /// <returns>The calculated search direction.</returns>
    private Vector<T> CalculateDirection(Vector<T> gradient)
    {
        // === Partially Vectorized L-BFGS Two-Loop Recursion using IEngine (Phase B: US-GPU-015) ===

        if (_s.Count == 0 || _y.Count == 0)
        {
            // First iteration: direction = -gradient
            return (Vector<T>)Engine.Multiply(gradient, NumOps.Negate(NumOps.One));
        }

        var q = new Vector<T>(gradient);
        var alphas = new T[_s.Count];

        // First loop (backward)
        for (int i = _s.Count - 1; i >= 0; i--)
        {
            alphas[i] = NumOps.Divide(_s[i].DotProduct(q), _y[i].DotProduct(_s[i]));
            // Vectorized: q = q - alpha * y[i]
            var alphaTimesY = (Vector<T>)Engine.Multiply(_y[i], alphas[i]);
            q = (Vector<T>)Engine.Subtract(q, alphaTimesY);
        }

        var gamma = NumOps.Divide(_s[_s.Count - 1].DotProduct(_y[_s.Count - 1]), _y[_s.Count - 1].DotProduct(_y[_s.Count - 1]));
        // Vectorized: z = gamma * q
        var z = (Vector<T>)Engine.Multiply(q, gamma);

        // Second loop (forward)
        for (int i = 0; i < _s.Count; i++)
        {
            var beta = NumOps.Divide(_y[i].DotProduct(z), _y[i].DotProduct(_s[i]));
            var alphaMinusBeta = NumOps.Subtract(alphas[i], beta);
            // Vectorized: z = z + (alpha - beta) * s[i]
            var scaledS = (Vector<T>)Engine.Multiply(_s[i], alphaMinusBeta);
            z = (Vector<T>)Engine.Add(z, scaledS);
        }

        // Vectorized negation
        return (Vector<T>)Engine.Multiply(z, NumOps.Negate(NumOps.One));
    }

    /// <summary>
    /// Updates the L-BFGS memory with the latest step information.
    /// </summary>
    /// <param name="oldSolution">The previous solution vector.</param>
    /// <param name="newSolution">The current solution vector.</param>
    /// <param name="gradient">The current gradient.</param>
    /// <param name="previousGradient">The previous gradient.</param>
    private void UpdateLBFGSMemory(Vector<T> oldSolution, Vector<T> newSolution, Vector<T> gradient, Vector<T> previousGradient)
    {
        // === Vectorized Memory Update using IEngine (Phase B: US-GPU-015) ===
        // s = new_solution - old_solution
        // y = current_gradient - previous_gradient

        // Skip first iteration when previousGradient is empty
        if (previousGradient.Length == 0)
        {
            return;
        }

        var s = (Vector<T>)Engine.Subtract(newSolution, oldSolution);
        var y = (Vector<T>)Engine.Subtract(gradient, previousGradient);

        if (_s.Count >= _options.MemorySize)
        {
            _s.RemoveAt(0);
            _y.RemoveAt(0);
        }

        _s.Add(s);
        _y.Add(y);
    }

    /// <summary>
    /// Updates the current solution based on the calculated direction.
    /// </summary>
    /// <param name="currentSolution">The current solution.</param>
    /// <param name="direction">The search direction.</param>
    /// <param name="gradient">The current gradient.</param>
    /// <param name="inputData">The input data for the optimization process.</param>
    /// <returns>The updated solution.</returns>
    private IFullModel<T, TInput, TOutput> UpdateSolution(IFullModel<T, TInput, TOutput> currentSolution, Vector<T> direction, Vector<T> gradient, OptimizationInputData<T, TInput, TOutput> inputData)
    {
        var step = LineSearch(currentSolution, direction, gradient, inputData);
        var scaledDirection = direction.Transform(x => NumOps.Multiply(x, step));
        var newCoefficients = currentSolution.GetParameters().Add(scaledDirection);

        return currentSolution.WithParameters(newCoefficients);
    }

    /// <summary>
    /// Updates the adaptive parameters of the optimizer based on the current and previous optimization steps.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method adjusts the learning rate based on the performance of the current step compared to the previous step.
    /// If the adaptive learning rate option is enabled, it increases or decreases the learning rate accordingly.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This method helps the optimizer learn more efficiently by adjusting how big its steps are.
    /// If the current step improved the solution, it takes slightly bigger steps.
    /// If not, it takes smaller steps to be more careful.
    /// </para>
    /// </remarks>
    /// <param name="currentStepData">Data from the current optimization step.</param>
    /// <param name="previousStepData">Data from the previous optimization step.</param>
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
    /// Updates the optimizer's options with new settings.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method updates the optimizer's configuration with new options. It ensures that only valid
    /// LBFGSOptimizerOptions are applied to this optimizer.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like changing the settings on the optimizer. It makes sure you're using the right kind of settings
    /// for this specific type of optimizer.
    /// </para>
    /// </remarks>
    /// <param name="options">The new options to apply to the optimizer.</param>
    /// <exception cref="ArgumentException">Thrown when the provided options are not of type LBFGSOptimizerOptions.</exception>
    protected override void UpdateOptions(OptimizationAlgorithmOptions<T, TInput, TOutput> options)
    {
        if (options is LBFGSOptimizerOptions<T, TInput, TOutput> lbfgsOptions)
        {
            _options = lbfgsOptions;
        }
        else
        {
            throw new ArgumentException("Invalid options type. Expected LBFGSOptimizerOptions.");
        }
    }

    /// <summary>
    /// Retrieves the current options of the optimizer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method returns the current configuration options of the L-BFGS optimizer.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This lets you see what settings the optimizer is currently using.
    /// </para>
    /// </remarks>
    /// <returns>The current options of the optimizer.</returns>
    public override OptimizationAlgorithmOptions<T, TInput, TOutput> GetOptions()
    {
        return _options;
    }

    /// <summary>
    /// Serializes the optimizer's state into a byte array.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method converts the current state of the optimizer, including its options and internal memory,
    /// into a byte array. This allows the optimizer's state to be saved or transmitted.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like taking a snapshot of the optimizer's current state so it can be saved or sent somewhere else.
    /// It includes all the important information about what the optimizer has learned so far.
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

            writer.Write(_iteration);
            writer.Write(_s.Count);
            foreach (var vector in _s)
            {
                writer.Write(vector.Serialize());
            }
            writer.Write(_y.Count);
            foreach (var vector in _y)
            {
                writer.Write(vector.Serialize());
            }

            return ms.ToArray();
        }
    }

    /// <summary>
    /// Deserializes a byte array to restore the optimizer's state.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method takes a byte array (previously created by the Serialize method) and uses it to restore
    /// the optimizer's state, including its options and internal memory.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like loading a saved snapshot of the optimizer's state. It rebuilds the optimizer's memory
    /// and settings from the saved data, allowing it to continue from where it left off.
    /// </para>
    /// </remarks>
    /// <param name="data">The byte array containing the serialized optimizer state.</param>
    /// <exception cref="InvalidOperationException">Thrown when deserialization of optimizer options fails.</exception>
    public override void Deserialize(byte[] data)
    {
        using (MemoryStream ms = new MemoryStream(data))
        using (BinaryReader reader = new BinaryReader(ms))
        {
            int baseDataLength = reader.ReadInt32();
            byte[] baseData = reader.ReadBytes(baseDataLength);
            base.Deserialize(baseData);

            string optionsJson = reader.ReadString();
            _options = JsonConvert.DeserializeObject<LBFGSOptimizerOptions<T, TInput, TOutput>>(optionsJson)
                ?? throw new InvalidOperationException("Failed to deserialize optimizer options.");

            _iteration = reader.ReadInt32();

            int sCount = reader.ReadInt32();
            _s = new List<Vector<T>>(sCount);
            for (int i = 0; i < sCount; i++)
            {
                int vectorLength = reader.ReadInt32();
                byte[] vectorData = reader.ReadBytes(vectorLength);
                _s.Add(Vector<T>.Deserialize(vectorData));
            }

            int yCount = reader.ReadInt32();
            _y = new List<Vector<T>>(yCount);
            for (int i = 0; i < yCount; i++)
            {
                int vectorLength = reader.ReadInt32();
                byte[] vectorData = reader.ReadBytes(vectorLength);
                _y.Add(Vector<T>.Deserialize(vectorData));
            }
        }
    }
}
