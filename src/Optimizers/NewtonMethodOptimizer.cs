using AiDotNet.Tensors.Engines.DirectGpu;
using Newtonsoft.Json;

namespace AiDotNet.Optimizers;

/// <summary>
/// Implements the Newton's Method optimization algorithm.
/// </summary>
/// <remarks>
/// <para>
/// Newton's Method is a powerful optimization algorithm that uses both first and second derivatives of the objective function.
/// It often converges faster than first-order methods, especially near the optimum, but can be computationally expensive due to the need to compute and invert the Hessian matrix.
/// </para>
/// <para><b>For Beginners:</b>
/// Imagine you're trying to find the lowest point in a valley. Gradient descent is like rolling a ball and letting it follow the slope.
/// Newton's Method is like using a telescope to look at the whole valley, predicting where the lowest point is, and jumping directly there.
/// It's often faster but requires more complex calculations at each step.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class NewtonMethodOptimizer<T, TInput, TOutput> : GradientBasedOptimizerBase<T, TInput, TOutput>
{
    /// <summary>
    /// The options specific to the Newton's Method optimizer.
    /// </summary>
    private NewtonMethodOptimizerOptions<T, TInput, TOutput> _options;

    /// <summary>
    /// The current iteration count of the optimization process.
    /// </summary>
    private int _iteration;

    /// <summary>
    /// Initializes a new instance of the NewtonMethodOptimizer class.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This constructor sets up the Newton's Method optimizer with the provided options and dependencies.
    /// If no options are provided, it uses default settings.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like preparing all your tools and maps before starting your journey to find the lowest point in the valley.
    /// You're setting up how you'll make decisions and what information you'll use along the way.
    /// </para>
    /// </remarks>
    /// <param name="model">The model to optimize.</param>
    /// <param name="options">The Newton's Method-specific optimization options.</param>
    public NewtonMethodOptimizer(
        IFullModel<T, TInput, TOutput> model,
        NewtonMethodOptimizerOptions<T, TInput, TOutput>? options = null)
        : base(model, options ?? new())
    {
        _options = options ?? new NewtonMethodOptimizerOptions<T, TInput, TOutput>();

        InitializeAdaptiveParameters();
    }

    /// <summary>
    /// Initializes the adaptive parameters for the Newton's Method optimizer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method sets up the initial values for the learning rate and resets the iteration count.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like setting your initial step size and resetting your step counter before you start your journey.
    /// The learning rate determines how big your steps will be, and the iteration count keeps track of how many steps you've taken.
    /// </para>
    /// </remarks>
    protected override void InitializeAdaptiveParameters()
    {
        base.InitializeAdaptiveParameters();
        CurrentLearningRate = NumOps.FromDouble(_options.InitialLearningRate);
        _iteration = 0;
    }

    /// <summary>
    /// Performs the optimization process using Newton's Method algorithm.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method implements the main optimization loop. It uses Newton's Method to update the solution iteratively,
    /// aiming to find the optimal set of parameters that minimize the loss function.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is your actual journey through the valley. At each step:
    /// 1. You look at the slope (gradient) and curvature (Hessian) of the valley around you.
    /// 2. Based on this information, you calculate the best direction to move.
    /// 3. You take a step in that direction.
    /// 4. You check if you've found a better spot than any you've seen before.
    /// 5. You decide whether to keep going or stop if you think you've found the lowest point.
    /// </para>
    /// <para><b>DataLoader Integration:</b> This method uses the DataLoader API for epoch management.
    /// Newton's Method typically operates on the full dataset because it requires computing the Hessian
    /// matrix that needs consistent second derivative information. The method notifies the
    /// sampler of epoch starts using <see cref="GradientBasedOptimizerBase{T,TInput,TOutput}.NotifyEpochStart"/>
    /// for compatibility with curriculum learning and sampling strategies.
    /// </para>
    /// </remarks>
    /// <param name="inputData">The input data for the optimization process.</param>
    /// <returns>The result of the optimization process.</returns>
    public override OptimizationResult<T, TInput, TOutput> Optimize(OptimizationInputData<T, TInput, TOutput> inputData)
    {
        ValidationHelper<T>.ValidateInputData(inputData);

        var currentSolution = InitializeRandomSolution(inputData.XTrain);
        var bestStepData = new OptimizationStepData<T, TInput, TOutput>();
        var previousStepData = new OptimizationStepData<T, TInput, TOutput>();

        InitializeAdaptiveParameters();

        for (int epoch = 0; epoch < _options.MaxIterations; epoch++)
        {
            NotifyEpochStart(epoch);
            _iteration++;

            var gradient = CalculateGradient(currentSolution, inputData.XTrain, inputData.YTrain);
            // Use efficient Hessian computation (automatically uses IGradientComputable if available)
            var hessian = ComputeHessianEfficiently(currentSolution, inputData);
            var direction = CalculateDirection(gradient, hessian);
            var newSolution = UpdateSolution(currentSolution, direction);

            var currentStepData = EvaluateSolution(newSolution, inputData);
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

            currentSolution = newSolution;
            previousStepData = currentStepData;
        }

        return CreateOptimizationResult(bestStepData, inputData);
    }

    /// <summary>
    /// Calculates the direction for the next step in Newton's Method.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method computes the direction by multiplying the inverse of the Hessian matrix with the gradient.
    /// If the Hessian is not invertible, it falls back to the negative gradient direction.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like using your telescope and map to decide which way to go next. You're looking at both the steepness (gradient)
    /// and the shape (Hessian) of the valley to make the best guess about where the lowest point is. If your calculations get too
    /// complicated, you simply decide to go downhill (like in regular gradient descent).
    /// </para>
    /// </remarks>
    /// <param name="gradient">The gradient vector at the current point.</param>
    /// <param name="hessian">The Hessian matrix at the current point.</param>
    /// <returns>The direction vector for the next step.</returns>
    private Vector<T> CalculateDirection(Vector<T> gradient, Matrix<T> hessian)
    {
        // === Vectorized Direction Calculation using IEngine (Phase B: US-GPU-015) ===
        // direction = -H^{-1} * gradient (or -gradient if H is singular)

        try
        {
            var inverseHessian = hessian.Inverse();
            var direction = inverseHessian.Multiply(gradient);
            return (Vector<T>)Engine.Multiply(direction, NumOps.Negate(NumOps.One));
        }
        catch (InvalidOperationException)
        {
            // If Hessian is not invertible, fall back to gradient descent
            return (Vector<T>)Engine.Multiply(gradient, NumOps.Negate(NumOps.One));
        }
    }

    /// <summary>
    /// Calculates the Hessian matrix for the current model and input data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method computes the Hessian matrix, which represents the second-order partial derivatives of the loss function.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// Imagine creating a detailed topographic map of the valley around your current position. You're measuring how the slope changes
    /// in every direction, which gives you a complete picture of the valley's shape at your location.
    /// </para>
    /// </remarks>
    /// <param name="model">The current model.</param>
    /// <param name="inputData">The input data for optimization.</param>
    /// <returns>The Hessian matrix at the current point.</returns>
    private Matrix<T> CalculateHessian(IFullModel<T, TInput, TOutput> model, OptimizationInputData<T, TInput, TOutput> inputData)
    {
        int n = model.GetParameters().Length;
        var hessian = new Matrix<T>(n, n);

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                hessian[i, j] = CalculateSecondPartialDerivative(model, inputData, i, j);
            }
        }

        return hessian;
    }

    /// <summary>
    /// Calculates the second partial derivative of the loss function with respect to two parameters.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method uses finite differences to approximate the second partial derivative.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like measuring how the slope of the valley changes when you take tiny steps in two different directions.
    /// It helps you understand the curvature of the valley at your current position.
    /// </para>
    /// </remarks>
    /// <param name="model">The current model.</param>
    /// <param name="inputData">The input data for optimization.</param>
    /// <param name="i">The index of the first parameter.</param>
    /// <param name="j">The index of the second parameter.</param>
    /// <returns>The approximated second partial derivative.</returns>
    private T CalculateSecondPartialDerivative(IFullModel<T, TInput, TOutput> model, OptimizationInputData<T, TInput, TOutput> inputData, int i, int j)
    {
        var parameters = model.GetParameters();
        var epsilon = NumOps.FromDouble(1e-5);
        var originalI = parameters[i];
        var originalJ = parameters[j];

        // f(x+h, y+h)
        parameters[i] = NumOps.Add(originalI, epsilon);
        parameters[j] = NumOps.Add(originalJ, epsilon);
        var fhh = CalculateLoss(model, inputData);

        // f(x+h, y-h)
        parameters[j] = NumOps.Subtract(originalJ, epsilon);
        var fhm = CalculateLoss(model, inputData);

        // f(x-h, y+h)
        parameters[i] = NumOps.Subtract(originalI, epsilon);
        parameters[j] = NumOps.Add(originalJ, epsilon);
        var fmh = CalculateLoss(model, inputData);

        // f(x-h, y-h)
        parameters[j] = NumOps.Subtract(originalJ, epsilon);
        var fmm = CalculateLoss(model, inputData);

        // Reset coefficients
        parameters[i] = originalI;
        parameters[j] = originalJ;

        // Calculate second partial derivative
        var numerator = NumOps.Subtract(NumOps.Add(fhh, fmm), NumOps.Add(fhm, fmh));
        var denominator = NumOps.Multiply(NumOps.FromDouble(4), NumOps.Multiply(epsilon, epsilon));

        return NumOps.Divide(numerator, denominator);
    }

    /// <summary>
    /// Updates the current solution based on the calculated direction.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method applies the Newton's Method update rule to the current solution using the calculated direction and learning rate.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like taking a step in the direction you've calculated. The size of your step is determined by the learning rate,
    /// and the direction is based on both the slope and the curvature of the valley at your current position.
    /// </para>
    /// </remarks>
    /// <param name="currentSolution">The current solution (model parameters).</param>
    /// <param name="direction">The direction to move in the parameter space.</param>
    /// <returns>A new ISymbolicModel with updated coefficients.</returns>
    protected override IFullModel<T, TInput, TOutput> UpdateSolution(IFullModel<T, TInput, TOutput> currentSolution, Vector<T> direction)
    {
        // === Vectorized Solution Update using IEngine (Phase B: US-GPU-015) ===
        // newCoefficients = parameters + learningRate * direction
        // Note: direction is already negated (-H^{-1} * g), so adding moves downhill

        var parameters = currentSolution.GetParameters();
        var scaledDirection = (Vector<T>)Engine.Multiply(direction, CurrentLearningRate);
        var newCoefficients = (Vector<T>)Engine.Add(parameters, scaledDirection);

        return currentSolution.WithParameters(newCoefficients);
    }

    /// <summary>
    /// Updates the adaptive parameters of the optimizer based on the current and previous optimization steps.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method adjusts the learning rate based on the performance of the current step compared to the previous step.
    /// If the fitness score improves, the learning rate is increased; otherwise, it is decreased.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like adjusting your step size based on how well you're doing. If you're making good progress (better fitness score),
    /// you might take slightly larger steps. If you're not improving, you'll take smaller, more cautious steps.
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
    /// This method ensures that only compatible option types are used with this optimizer.
    /// It updates the internal options if the provided options are of the correct type.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like changing the rules for how you navigate the valley. It makes sure you're only using rules that work for
    /// Newton's Method of exploring the valley.
    /// </para>
    /// </remarks>
    /// <param name="options">The new options to be applied to the optimizer.</param>
    /// <exception cref="ArgumentException">Thrown when the provided options are not of the correct type.</exception>
    protected override void UpdateOptions(OptimizationAlgorithmOptions<T, TInput, TOutput> options)
    {
        if (options is NewtonMethodOptimizerOptions<T, TInput, TOutput> newtonOptions)
        {
            _options = newtonOptions;
        }
        else
        {
            throw new ArgumentException("Invalid options type. Expected NewtonMethodOptimizerOptions.");
        }
    }

    /// <summary>
    /// Gets the current options of the optimizer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method returns the current settings used by the Newton's Method optimizer.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like checking your current rulebook for exploring the valley. It tells you what settings and strategies
    /// you're currently using in your search for the lowest point.
    /// </para>
    /// </remarks>
    /// <returns>The current optimization algorithm options.</returns>
    public override OptimizationAlgorithmOptions<T, TInput, TOutput> GetOptions()
    {
        return _options;
    }

    /// <summary>
    /// Updates parameters using GPU-accelerated Newton's Method.
    /// </summary>
    /// <remarks>
    /// Newton's method requires computing the full Hessian matrix (second derivatives).
    /// GPU implementation is not yet available due to the O(n^2) memory requirements
    /// and matrix inversion needed for the Hessian.
    /// </remarks>
    public override void UpdateParametersGpu(IGpuBuffer parameters, IGpuBuffer gradients, int parameterCount, IDirectGpuBackend backend)
    {
        throw new NotSupportedException(
            "GPU-accelerated Newton's method is not yet implemented. " +
            "Newton's method requires computing and inverting the Hessian matrix which is " +
            "memory-intensive (O(n^2)) and complex on GPU. " +
            "Use CPU-based UpdateParameters or consider using Adam/AdamW for GPU-resident training.");
    }

    /// <summary>
    /// Serializes the current state of the optimizer into a byte array.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method saves the current state of the optimizer, including its base class state, options, and iteration count.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like taking a snapshot of your current position, all your tools, and your strategy for exploring the valley.
    /// You can use this snapshot later to continue your exploration from exactly where you left off.
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

            return ms.ToArray();
        }
    }

    /// <summary>
    /// Deserializes a byte array to restore the optimizer's state.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method restores the optimizer's state from a previously serialized byte array, including its base class state, options, and iteration count.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like using a snapshot you took earlier to set up your exploration exactly as it was at that point.
    /// You're restoring all your tools, your position in the valley, and your strategy to continue your search from where you left off.
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
            _options = JsonConvert.DeserializeObject<NewtonMethodOptimizerOptions<T, TInput, TOutput>>(optionsJson)
                ?? throw new InvalidOperationException("Failed to deserialize optimizer options.");

            _iteration = reader.ReadInt32();
        }
    }
}
