using AiDotNet.Helpers;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Autodiff;
using Newtonsoft.Json;

using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.Optimizers;

/// <summary>
/// Implements the Coordinate Descent optimization algorithm for numerical optimization problems.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Coordinate Descent is an optimization algorithm that minimizes a multivariable function by solving a series of 
/// single-variable optimization problems. It cycles through each variable (coordinate) and optimizes it while holding 
/// the others constant.
/// </para>
/// <para><b>For Beginners:</b> This optimizer is like adjusting the knobs on a complex machine one at a time. 
/// It focuses on improving one aspect of the solution at a time, which can be more manageable and sometimes 
/// more effective than trying to adjust everything at once.
/// </para>
/// </remarks>
[ComponentType(ComponentType.Optimizer)]
[PipelineStage(PipelineStage.Training)]
public class CoordinateDescentOptimizer<T, TInput, TOutput> : GradientBasedOptimizerBase<T, TInput, TOutput>, Fused.IFusedOptimizerSpec
{
    /// <summary>
    /// Describes this optimizer for the compiled fused-training kernel.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Maps to <see cref="Tensors.Engines.Compilation.OptimizerType.SGDMomentum"/>, which is this optimizer's
    /// update EXACTLY rather than an approximation of it. The sweep applies
    /// <c>u_t = lr*g_t + m*u_{t-1}; x -= u_t</c>; substituting <c>u = lr*v</c> gives
    /// <c>v_t = g_t + m*v_{t-1}</c> and <c>x -= lr*v_t</c>, which is precisely the kernel's
    /// <c>v = mu*v + g; p -= lr*v</c> with <c>mu = m</c>. The two differ only in whether the stored state is
    /// scaled by lr, and both start that state at zero.
    /// </para>
    /// <para>
    /// The per-coordinate rate vectors do not break the mapping. They are seeded uniformly from the options
    /// and <c>UpdateAdaptiveParameters</c> only ever scales the whole vector by one factor and clamps it to
    /// uniform bounds, so every coordinate carries the same value for the life of the optimizer. Were that to
    /// change — a genuinely per-coordinate rate — this mapping would no longer hold and the spec would have
    /// to decline.
    /// </para>
    /// <para>
    /// Declines on a non-constant learning-rate schedule, and that guard is load-bearing rather than
    /// defensive. The <c>u = lr*v</c> identity holds across steps only while lr is fixed: with a changing
    /// rate the eager recurrence carries <c>m*u_{t-1}</c> while the kernel carries
    /// <c>(lr_t/lr_{t-1})*m*u_{t-1}</c>, so the two paths would diverge silently the moment a schedule moved.
    /// </para>
    /// </remarks>
    bool Fused.IFusedOptimizerSpec.TryGetFusedOptimizerConfig(out Fused.FusedOptimizerConfig config)
    {
        config = default;
        if (_options.UseAdaptiveLearningRate) return false;
        if (!TryGetFusedLrSchedule(out var schedule)) return false;
        if (schedule is not null) return false;

        // Deliberately the OPTION value, not CurrentLearningRate: the sweep reads _learningRates, which is
        // seeded from InitialLearningRate. With the guards above the two agree, but naming the one the eager
        // path actually uses keeps that true if the base class's notion of "current" ever drifts.
        config = new Fused.FusedOptimizerConfig(
            Tensors.Engines.Compilation.OptimizerType.SGDMomentum,
            (float)_options.InitialLearningRate,
            (float)_options.InitialMomentum,   // Beta1 carries the momentum coefficient
            0f, 0f, 0f, schedule);
        return true;
    }

    /// <summary>
    /// The options specific to the Coordinate Descent optimization algorithm.
    /// </summary>
    private CoordinateDescentOptimizerOptions<T, TInput, TOutput> _options;

    /// <summary>
    /// Vector of learning rates for each coordinate (variable) in the optimization problem.
    /// </summary>
    private Vector<T> _learningRates;

    /// <summary>
    /// Vector of momentum values for each coordinate (variable) in the optimization problem.
    /// </summary>
    private Vector<T> _momentums;

    /// <summary>
    /// Vector of previous update values for each coordinate (variable) in the optimization problem.
    /// </summary>
    private Vector<T> _previousUpdate;

    /// <summary>
    /// Initializes a new instance of the CoordinateDescentOptimizer class.
    /// </summary>
    /// <param name="model">The model to optimize.</param>
    /// <param name="options">The options for configuring the Coordinate Descent algorithm.</param>
    /// <param name="engine">The computation engine (CPU or GPU) for vectorized operations.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This constructor sets up the Coordinate Descent optimizer with its initial configuration.
    /// You can customize various aspects of how it works, or use default settings.
    /// </para>
    /// </remarks>
    public CoordinateDescentOptimizer(
        IFullModel<T, TInput, TOutput> model,
        CoordinateDescentOptimizerOptions<T, TInput, TOutput>? options = null,
        IEngine? engine = null)
        : base(model, options ?? new())
    {
        _options = options ?? new CoordinateDescentOptimizerOptions<T, TInput, TOutput>();
        _learningRates = Vector<T>.Empty();
        _momentums = Vector<T>.Empty();
        _previousUpdate = Vector<T>.Empty();
    }

    /// <summary>
    /// Creates a coordinate descent optimizer for minimizing a plain function, with no model attached.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Use this with <see cref="GradientBasedOptimizerBase{T, TInput, TOutput}.Minimize(Vector{T}, Func{Vector{T}, ValueTuple{T, Vector{T}}}, int, T)"/>
    /// when you want to minimize a mathematical function directly rather than train a model.
    /// <see cref="Optimize"/> requires a model and is not available on an instance created
    /// this way.
    /// </para>
    /// <para><b>For Beginners:</b> The constructor above asks for a model because it is set up
    /// to tune that model against training data. If all you have is a formula you want to make
    /// as small as possible, there is no model to hand over — use this factory instead.
    /// </para>
    /// </remarks>
    /// <param name="options">The optimizer-specific options. If null, defaults are used.</param>
    public static CoordinateDescentOptimizer<T, TInput, TOutput> CreateForFunction(
        CoordinateDescentOptimizerOptions<T, TInput, TOutput>? options = null)
        => new(options);

    /// <summary>
    /// Backs <see cref="CreateForFunction"/>: the same setup with no model.
    /// </summary>
    private CoordinateDescentOptimizer(CoordinateDescentOptimizerOptions<T, TInput, TOutput>? options)
        : base(null, options ?? new())
    {
        _options = options ?? new CoordinateDescentOptimizerOptions<T, TInput, TOutput>();
        _learningRates = Vector<T>.Empty();
        _momentums = Vector<T>.Empty();
        _previousUpdate = Vector<T>.Empty();
    }

    /// <summary>
    /// Initializes the adaptive parameters used in the Coordinate Descent algorithm.
    /// </summary>
    /// <param name="currentSolution">The current solution model.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method sets up the initial state for the optimizer,
    /// including learning rates, momentums, and previous updates for each coordinate (variable).
    /// </para>
    /// </remarks>
    private void InitializeAdaptiveParameters(IFullModel<T, TInput, TOutput> currentSolution)
    {
        base.InitializeAdaptiveParameters();
        int dimensions = InterfaceGuard.Parameterizable(currentSolution).GetParameters().Length;
        _learningRates = Vector<T>.CreateDefault(dimensions, NumOps.FromDouble(_options.InitialLearningRate));
        _momentums = Vector<T>.CreateDefault(dimensions, NumOps.FromDouble(_options.InitialMomentum));
        _previousUpdate = Vector<T>.CreateDefault(dimensions, NumOps.Zero);
    }

    /// <summary>
    /// Performs the main optimization process using the Coordinate Descent algorithm.
    /// </summary>
    /// <param name="inputData">The input data for the optimization process.</param>
    /// <returns>The result of the optimization process.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the heart of the Coordinate Descent algorithm. It iteratively improves the solution
    /// by updating one coordinate (variable) at a time. The process continues until it reaches the maximum number of iterations
    /// or meets the stopping criteria.
    /// </para>
    /// <para><b>DataLoader Integration:</b> This method uses the DataLoader API for epoch management.
    /// Coordinate Descent typically operates on the full dataset for derivative estimation,
    /// but notifies the sampler of epoch starts using
    /// <see cref="GradientBasedOptimizerBase{T,TInput,TOutput}.NotifyEpochStart"/> for compatibility with
    /// curriculum learning and sampling strategies.
    /// </para>
    /// </remarks>
    public override OptimizationResult<T, TInput, TOutput> Optimize(OptimizationInputData<T, TInput, TOutput> inputData)
    {
        ValidationHelper<T>.ValidateInputData(inputData);

        var currentSolution = InitializeWorkingSolution(inputData.XTrain);
        var bestStepData = new OptimizationStepData<T, TInput, TOutput>();
        var previousStepData = new OptimizationStepData<T, TInput, TOutput>();

        InitializeAdaptiveParameters(currentSolution);

        for (int epoch = 0; epoch < _options.MaxIterations; epoch++)
        {
            NotifyEpochStart(epoch);

            var newSolution = UpdateSolution(currentSolution, inputData);
            var currentStepData = EvaluateSolution(newSolution, inputData);

            UpdateBestSolution(currentStepData, ref bestStepData);
            UpdateAdaptiveParameters(currentStepData, previousStepData);

            if (UpdateIterationHistoryAndCheckEarlyStopping(epoch, bestStepData))
            {
                return CreateOptimizationResult(bestStepData, inputData);
            }

            if (IsConvergedAgainstPreviousEpoch(epoch, currentStepData, previousStepData, _options.Tolerance))
            {
                // H6 convergence fix (PR #1364): compare CURRENT vs PREVIOUS
                // epoch (not bestStepData — UpdateBestSolution copies
                // currentStepData into bestStepData on epoch 0, so |best -
                // current| = 0 < tolerance would falsely converge). Skip
                // check on epoch 0 where previousStepData is the pre-training
                // baseline. Helper is on GradientBasedOptimizerBase.
                return CreateOptimizationResult(bestStepData, inputData);
            }

            currentSolution = newSolution;
            previousStepData = currentStepData;
        }

        return CreateOptimizationResult(bestStepData, inputData);
    }

    /// <summary>
    /// Updates the current solution by optimizing each coordinate (variable) individually.
    /// </summary>
    /// <param name="currentSolution">The current solution model.</param>
    /// <param name="inputData">The input data for the optimization process.</param>
    /// <returns>The updated solution model.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method goes through each variable in the solution and tries to improve it individually.
    /// It's like fine-tuning each knob on a machine one at a time to get the best overall performance.
    /// </para>
    /// </remarks>
    private IFullModel<T, TInput, TOutput> UpdateSolution(IFullModel<T, TInput, TOutput> currentSolution, OptimizationInputData<T, TInput, TOutput> inputData)
    {
        var newCoefficients = InterfaceGuard.Parameterizable(currentSolution).GetParameters().Clone();

        for (int i = 0; i < newCoefficients.Length; i++)
        {
            var gradient = CalculatePartialDerivative(currentSolution, inputData, i);
            var update = CalculateUpdate(gradient, i);
            newCoefficients[i] = NumOps.Add(newCoefficients[i], update);
        }

        return InterfaceGuard.Parameterizable(currentSolution).WithParameters(newCoefficients);
    }

    /// <summary>
    /// Calculates the partial derivative (gradient) for a specific coordinate (variable).
    /// </summary>
    /// <param name="model">The current solution model.</param>
    /// <param name="inputData">The input data for the optimization process.</param>
    /// <param name="index">The index of the coordinate to calculate the partial derivative for.</param>
    /// <returns>The calculated partial derivative.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method estimates how much the overall performance would change if we slightly adjust
    /// one specific variable. It helps determine which direction to move that variable to improve the solution.
    /// </para>
    /// </remarks>
    private T CalculatePartialDerivative(IFullModel<T, TInput, TOutput> model, OptimizationInputData<T, TInput, TOutput> inputData, int index)
    {
        var epsilon = NumOps.FromDouble(1e-6);
        var parameters = InterfaceGuard.Parameterizable(model).GetParameters();
        var originalCoeff = parameters[index];

        var coefficientsPlus = parameters.Clone();
        coefficientsPlus[index] = NumOps.Add(originalCoeff, epsilon);
        var modelPlus = InterfaceGuard.Parameterizable(model).WithParameters(coefficientsPlus);

        var coefficientsMinus = parameters.Clone();
        coefficientsMinus[index] = NumOps.Subtract(originalCoeff, epsilon);
        var modelMinus = InterfaceGuard.Parameterizable(model).WithParameters(coefficientsMinus);

        var lossPlus = CalculateLoss(modelPlus, inputData);
        var lossMinus = CalculateLoss(modelMinus, inputData);

        return NumOps.Divide(NumOps.Subtract(lossPlus, lossMinus), NumOps.Multiply(NumOps.FromDouble(2.0), epsilon));
    }

    /// <summary>
    /// Calculates the update for a specific coordinate based on its gradient and momentum.
    /// </summary>
    /// <param name="gradient">The calculated gradient for the coordinate.</param>
    /// <param name="index">The index of the coordinate.</param>
    /// <returns>The calculated update value.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method determines how much to change a specific variable. It considers both
    /// the current gradient (which suggests the best direction to move) and momentum (which helps maintain consistent movement).
    /// </para>
    /// </remarks>
    private T CalculateUpdate(T gradient, int index)
    {
        var update = NumOps.Add(
            NumOps.Multiply(_learningRates[index], gradient),
            NumOps.Multiply(_momentums[index], _previousUpdate[index])
        );
        _previousUpdate[index] = update;

        return NumOps.Negate(update);
    }

    /// <summary>
    /// Updates parameters using GPU-accelerated coordinate descent.
    /// </summary>
    /// <remarks>
    /// Coordinate descent optimizes one coordinate at a time sequentially.
    /// GPU implementation is not yet available because coordinate descent's sequential
    /// nature doesn't parallelize well on GPU.
    /// </remarks>
    public override void UpdateParametersGpu(IGpuBuffer parameters, IGpuBuffer gradients, int parameterCount, IDirectGpuBackend backend)
    {
        throw new NotSupportedException(
            "GPU-accelerated coordinate descent is not yet implemented. " +
            "Coordinate descent's sequential per-coordinate updates don't parallelize well on GPU. " +
            "Use CPU-based UpdateParameters or consider using Adam/AdamW for GPU-resident training.");
    }

    /// <summary>
    /// Updates the adaptive parameters (learning rates and momentums) based on the optimization progress.
    /// </summary>
    /// <param name="currentStepData">Data from the current optimization step.</param>
    /// <param name="previousStepData">Data from the previous optimization step.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method adjusts how big of steps the optimizer takes for each variable.
    /// If the solution is improving, it might increase the step sizes to progress faster. If not, it might decrease
    /// them to be more careful.
    /// </para>
    /// </remarks>
    protected override void UpdateAdaptiveParameters(OptimizationStepData<T, TInput, TOutput> currentStepData, OptimizationStepData<T, TInput, TOutput> previousStepData)
    {
        // === Vectorized Adaptive Parameter Update using IEngine (Phase B: US-GPU-015) ===
        // All learning rates and momentums updated in parallel

        base.UpdateAdaptiveParameters(currentStepData, previousStepData);

        var improvement = NumOps.Subtract(currentStepData.FitnessScore, previousStepData.FitnessScore);

        if (NumOps.GreaterThan(improvement, NumOps.Zero))
        {
            // Improvement: increase learning rates and momentums
            var lrIncreaseFactor = NumOps.Add(NumOps.One, NumOps.FromDouble(_options.LearningRateIncreaseRate));
            var momentumIncreaseFactor = NumOps.Add(NumOps.One, NumOps.FromDouble(_options.MomentumIncreaseRate));

            _learningRates = (Vector<T>)Engine.Multiply(_learningRates, lrIncreaseFactor);
            _momentums = (Vector<T>)Engine.Multiply(_momentums, momentumIncreaseFactor);
        }
        else
        {
            // No improvement: decrease learning rates and momentums
            var lrDecreaseFactor = NumOps.Subtract(NumOps.One, NumOps.FromDouble(_options.LearningRateDecreaseRate));
            var momentumDecreaseFactor = NumOps.Subtract(NumOps.One, NumOps.FromDouble(_options.MomentumDecreaseRate));

            _learningRates = (Vector<T>)Engine.Multiply(_learningRates, lrDecreaseFactor);
            _momentums = (Vector<T>)Engine.Multiply(_momentums, momentumDecreaseFactor);
        }

        // Clamp values to configured ranges (per-element still needed for now)
        var minLr = NumOps.FromDouble(_options.MinLearningRate);
        // === Vectorized Parameter Clamping (Phase B: US-GPU-015) ===
        var maxLr = NumOps.FromDouble(_options.MaxLearningRate);
        var minMom = NumOps.FromDouble(_options.MinMomentum);
        var maxMom = NumOps.FromDouble(_options.MaxMomentum);

        // Clamp all learning rates and momentums at once using Transform
        _learningRates = _learningRates.Transform(lr => MathHelper.Clamp(lr, minLr, maxLr));
        _momentums = _momentums.Transform(mom => MathHelper.Clamp(mom, minMom, maxMom));
    }

    /// <summary>
    /// Updates the options for the Coordinate Descent optimizer.
    /// </summary>
    /// <param name="options">The new options to be set.</param>
    /// <exception cref="ArgumentException">Thrown when the provided options are not of type CoordinateDescentOptimizerOptions.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method allows you to change the settings of the optimizer during runtime.
    /// It ensures that only the correct type of options (specific to Coordinate Descent) can be used.
    /// </para>
    /// </remarks>
    protected override void UpdateOptions(OptimizationAlgorithmOptions<T, TInput, TOutput> options)
    {
        if (options is CoordinateDescentOptimizerOptions<T, TInput, TOutput> cdOptions)
        {
            _options = cdOptions;
        }
        else
        {
            throw new ArgumentException("Options must be of type CoordinateDescentOptimizerOptions", nameof(options));
        }
    }

    /// <summary>
    /// Retrieves the current options of the Coordinate Descent optimizer.
    /// </summary>
    /// <returns>The current optimization algorithm options.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method allows you to check the current settings of the optimizer.
    /// It's useful if you need to inspect or copy the current configuration.
    /// </para>
    /// </remarks>
    public override OptimizationAlgorithmOptions<T, TInput, TOutput> GetOptions()
    {
        return _options;
    }

    /// <summary>
    /// Serializes the Coordinate Descent optimizer to a byte array.
    /// </summary>
    /// <returns>A byte array representing the serialized state of the optimizer.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method converts the current state of the optimizer into a series of bytes.
    /// This is useful for saving the optimizer's state to a file or sending it over a network.
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

            // Serialize _learningRates
            byte[] learningRatesData = _learningRates.Serialize();
            writer.Write(learningRatesData.Length);
            writer.Write(learningRatesData);

            // Serialize _momentums
            byte[] momentumsData = _momentums.Serialize();
            writer.Write(momentumsData.Length);
            writer.Write(momentumsData);

            // Serialize _previousUpdate
            byte[] previousUpdateData = _previousUpdate.Serialize();
            writer.Write(previousUpdateData.Length);
            writer.Write(previousUpdateData);

            return ms.ToArray();
        }
    }

    /// <summary>
    /// Deserializes the Coordinate Descent optimizer from a byte array.
    /// </summary>
    /// <param name="data">The byte array containing the serialized optimizer state.</param>
    /// <exception cref="InvalidOperationException">Thrown when deserialization of optimizer options fails.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method reconstructs the optimizer's state from a series of bytes.
    /// It's used to restore a previously saved state of the optimizer, allowing you to continue from where you left off.
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
            _options = JsonConvert.DeserializeObject<CoordinateDescentOptimizerOptions<T, TInput, TOutput>>(optionsJson)
                ?? throw new InvalidOperationException("Failed to deserialize optimizer options.");

            // Deserialize _learningRates
            int learningRatesLength = reader.ReadInt32();
            byte[] learningRatesData = reader.ReadBytes(learningRatesLength);
            _learningRates = Vector<T>.Deserialize(learningRatesData);

            // Deserialize _momentums
            int momentumsLength = reader.ReadInt32();
            byte[] momentumsData = reader.ReadBytes(momentumsLength);
            _momentums = Vector<T>.Deserialize(momentumsData);

            // Deserialize _previousUpdate
            int previousUpdateLength = reader.ReadInt32();
            byte[] previousUpdateData = reader.ReadBytes(previousUpdateLength);
            _previousUpdate = Vector<T>.Deserialize(previousUpdateData);
        }
    }

    /// <inheritdoc />
    public override void Step(TapeStepContext<T> context)
    {
        // Sparse-by-default: walk any embedding params whose gradient lives
        // only in the sparse list (Tensors stopped seeding dense alongside) and
        // materialise via ToDense into context.Gradients before GetFlatGradients
        // / Hessian assembly reads from the dict. No-op for non-embedding params
        // and for embedding params that already have a dense entry.
        SparseEmbeddingOptimizerHelpers.MaterializeSparseIntoGradientsDict(context, Engine);

        var updated = UpdateParameters(context.GetFlatParameters(), context.GetFlatGradients());
        context.SetFlatParameters(updated);
    }

    /// <summary>
    /// Applies one full coordinate sweep to a flat parameter vector, using per-coordinate learning rates and
    /// per-coordinate momentum.
    /// </summary>
    /// <param name="parameters">The current parameters.</param>
    /// <param name="gradient">The gradient at those parameters.</param>
    /// <returns>The updated parameters.</returns>
    /// <remarks>
    /// <para>
    /// Without this override, <see cref="Step"/> resolved to
    /// <see cref="GradientBasedOptimizerBase{T, TInput, TOutput}"/>'s default of plain
    /// <c>theta -= lr * g</c>, so training a neural network with this optimizer silently produced gradient
    /// descent with a single global learning rate — discarding the per-coordinate rates and momentum that are
    /// the entire reason to choose it.
    /// </para>
    /// <para>
    /// This reproduces exactly what <see cref="Optimize"/>'s sweep does, coordinate by coordinate:
    /// </para>
    /// <code>
    /// update_i = -(lr_i * g_i + momentum_i * previousUpdate_i)
    /// theta_i += update_i
    /// </code>
    /// <para>
    /// One difference, and it is an improvement rather than a deviation: <see cref="Optimize"/> obtains each
    /// partial derivative from <c>CalculatePartialDerivative</c>, a one-sided finite difference with a fixed
    /// 1e-6 epsilon costing one model evaluation per coordinate. The tape hands us every partial exactly, in
    /// one backward pass. Same update rule, exact derivatives, and O(1) rather than O(n) evaluations.
    /// </para>
    /// <para>
    /// The per-coordinate state is sized lazily here because the tape path never calls the private
    /// <c>InitializeAdaptiveParameters(IFullModel)</c> overload — that one needs a model to read the
    /// parameter count from, and <see cref="Step"/> has only the flat vector.
    /// </para>
    /// </remarks>
    public override Vector<T> UpdateParameters(Vector<T> parameters, Vector<T> gradient)
    {
        if (parameters.Length != gradient.Length)
        {
            throw new ArgumentException(
                $"Parameter vector length ({parameters.Length}) must match gradient vector length ({gradient.Length}).",
                nameof(gradient));
        }

        int n = parameters.Length;

        // Size (or resize) the per-coordinate state against the vector we were actually handed.
        if (_learningRates is null || _learningRates.Length != n)
        {
            _learningRates = Vector<T>.CreateDefault(n, NumOps.FromDouble(_options.InitialLearningRate));
            _momentums = Vector<T>.CreateDefault(n, NumOps.FromDouble(_options.InitialMomentum));
            _previousUpdate = Vector<T>.CreateDefault(n, NumOps.Zero);
        }

        var updated = parameters.Clone();
        for (int i = 0; i < n; i++)
        {
            var update = CalculateUpdate(gradient[i], i);
            updated[i] = NumOps.Add(updated[i], update);
        }

        return updated;
    }
}
