using System.Collections.Concurrent;
using AiDotNet.Tensors.Engines.Autodiff;
using Newtonsoft.Json;
using AiDotNet.Helpers;

using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.Optimizers;

/// <summary>
/// Implements Rprop (resilient backpropagation): a per-weight adaptive step size driven by the SIGN of the
/// gradient, with the magnitude discarded entirely.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type of the model being optimized.</typeparam>
/// <typeparam name="TOutput">The output data type of the model being optimized.</typeparam>
/// <remarks>
/// <para>
/// Faithful to the pseudocode of Riedmiller and Braun, "A Direct Adaptive Method for Faster Backpropagation
/// Learning: The RPROP Algorithm" (Proc. IEEE International Conference on Neural Networks, 1993, pp. 586-591,
/// doi:10.1109/ICNN.1993.298623). For each weight, with g_prev the gradient stored from the previous step:
/// </para>
/// <code>
/// if g_prev * g &gt; 0:   Delta = min(Delta * eta_plus,  Delta_max);  w -= sign(g) * Delta;  g_prev = g
/// if g_prev * g &lt; 0:   Delta = max(Delta * eta_minus, Delta_min);  (no move);             g_prev = 0
/// if g_prev * g == 0:  w -= sign(g) * Delta;                                               g_prev = g
/// </code>
/// <para>
/// The sign-reversal branch is the one that is easy to get wrong. The paper takes NO step when the gradient
/// reverses -- it shrinks the step size and waits -- and it zeroes the stored gradient so the following step is
/// treated as a fresh sign rather than as a second reversal. Both halves matter: without the zeroing, a single
/// oscillation would be counted twice and the step size would halve twice for one overshoot. This is the variant
/// Igel and Huesken later labelled Rprop-minus (no weight backtracking), and it is what the paper's published
/// pseudocode and the fused Rprop kernel both implement.
/// </para>
/// <para>
/// Both eager paths are written branchlessly over whole vectors/tensors rather than as elementwise loops with
/// data-dependent branches. The three cases are recovered from sign(g_prev * g) as three disjoint indicator
/// vectors, which keeps the update on the vectorized engine while remaining exactly the three-case rule above.
/// </para>
/// <para>
/// <b>Rprop requires full-batch gradients.</b> The paper is explicit that this is a batch method, and the reason
/// is structural: the algorithm interprets a sign flip as "I overshot the minimum", but on a mini-batch the sign
/// flips constantly from sampling noise alone, so every step size is driven down to Delta_min and training
/// stalls. <see cref="Optimize"/> therefore evaluates the gradient over the entire training set each iteration
/// and this optimizer exposes no batch size. On the tape path the caller controls batching, so the same
/// requirement applies to the caller.
/// </para>
/// <para><b>For Beginners:</b> Nearly every other optimizer moves further when the gradient is large. Rprop
/// throws the size away and keeps only the direction. Each weight carries its own step size: keep pushing the
/// same way and the step grows by 20%, reverse direction and you must have gone too far, so the step is halved
/// and this turn is skipped. That makes it completely immune to gradients that are vanishingly small or
/// explosively large -- but it also means it needs an honest gradient computed over all your data, not a noisy
/// one from a small batch.</para>
/// </remarks>
[ComponentType(ComponentType.Optimizer)]
[PipelineStage(PipelineStage.Training)]
public class RpropOptimizer<T, TInput, TOutput> : GradientBasedOptimizerBase<T, TInput, TOutput>, Fused.IFusedOptimizerSpec
{
    /// <summary>
    /// The options specific to the Rprop optimizer.
    /// </summary>
    private RpropOptimizerOptions<T, TInput, TOutput> _options;

    /// <summary>
    /// The gradient carried over from the previous step, zeroed on each weight whose sign reversed.
    /// </summary>
    private Vector<T>? _prevGradient;

    /// <summary>
    /// The per-weight step size Delta.
    /// </summary>
    private Vector<T>? _stepSize;

    /// <summary>
    /// The current time step. Not used by the update itself -- Rprop's state is entirely in the step sizes -- but
    /// tracked for serialization and cache keys.
    /// </summary>
    private int _t;

    /// <summary>
    /// Initializes a new instance of the RpropOptimizer class.
    /// </summary>
    /// <param name="model">The model whose parameters this optimizer updates.</param>
    /// <param name="options">The options for configuring Rprop, or null for the paper's defaults.</param>
    /// <exception cref="ArgumentOutOfRangeException">
    /// Thrown when a hyperparameter falls outside the range the algorithm is defined for.
    /// </exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This sets up Rprop with the values from the 1993 paper (grow by 1.2, shrink by
    /// 0.5, first step 0.1). Those defaults are well tested and rarely worth changing.</para>
    /// </remarks>
    public RpropOptimizer(
        IFullModel<T, TInput, TOutput> model,
        RpropOptimizerOptions<T, TInput, TOutput>? options = null)
        : base(model, options ?? new())
    {
        _options = options ?? new RpropOptimizerOptions<T, TInput, TOutput>();

        ValidateHyperparameters(_options);
        InitializeAdaptiveParameters();
    }

    /// <summary>
    /// Rejects hyperparameters the algorithm is not defined for, at construction rather than mid-training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The Delta_0 range check is not pedantry. The eager paths clamp the step size into
    /// [MinStepSize, MaxStepSize] on every step, which is equivalent to the kernel's one-sided clamps ONLY while
    /// the step size is already inside that interval -- an invariant established by Delta_0 and preserved by every
    /// later step. Starting outside the interval would break the equivalence on the first step, so it is refused.
    /// </para>
    /// </remarks>
    private static void ValidateHyperparameters(RpropOptimizerOptions<T, TInput, TOutput> options)
    {
        if (!(options.EtaPlus > 1.0))
            throw new ArgumentOutOfRangeException(nameof(options.EtaPlus), options.EtaPlus,
                "Rprop's EtaPlus must be greater than 1, otherwise the step size can never grow.");

        if (!(options.EtaMinus > 0.0) || !(options.EtaMinus < 1.0))
            throw new ArgumentOutOfRangeException(nameof(options.EtaMinus), options.EtaMinus,
                "Rprop's EtaMinus must lie strictly between 0 and 1, otherwise a sign reversal cannot shrink the step size.");

        if (!(options.MinStepSize > 0.0))
            throw new ArgumentOutOfRangeException(nameof(options.MinStepSize), options.MinStepSize,
                "Rprop's MinStepSize must be greater than 0.");

        if (!(options.MaxStepSize > options.MinStepSize))
            throw new ArgumentOutOfRangeException(nameof(options.MaxStepSize), options.MaxStepSize,
                $"Rprop's MaxStepSize must exceed MinStepSize ({options.MinStepSize}).");

        if (options.InitialStepSize < options.MinStepSize || options.InitialStepSize > options.MaxStepSize)
            throw new ArgumentOutOfRangeException(nameof(options.InitialStepSize), options.InitialStepSize,
                $"Rprop's InitialStepSize must lie within [{options.MinStepSize}, {options.MaxStepSize}].");
    }

    /// <summary>
    /// Describes this Rprop optimizer for the compiled fused-training kernel.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Maps to <see cref="Tensors.Engines.Compilation.OptimizerType.Rprop"/>. All five hyperparameters travel in
    /// <see cref="Tensors.Engines.Compilation.FusedOptimizerExtras"/>, and all five are set explicitly rather
    /// than left to the extras' own defaults -- the extras default Delta_0 to 0.01 while the paper (and this
    /// optimizer) use 0.1, so relying on the default would have started the fused path from a different step size
    /// than the eager path.
    /// </para>
    /// <para>
    /// Declines whenever ANY learning-rate schedule is configured, not merely an unsupported one. Rprop has no
    /// learning rate for a schedule to act on and the fused kernel takes no lr argument, so a scheduled run would
    /// have the schedule quietly ignored on the compiled path while still being visible in configuration.
    /// Refusing to fuse is the honest response.
    /// </para>
    /// </remarks>
    bool Fused.IFusedOptimizerSpec.TryGetFusedOptimizerConfig(out Fused.FusedOptimizerConfig config)
    {
        config = default;
        if (_options.UseAdaptiveLearningRate) return false;
        if (!TryGetFusedLrSchedule(out var schedule)) return false;
        if (schedule is not null) return false;

        config = new Fused.FusedOptimizerConfig(
            Tensors.Engines.Compilation.OptimizerType.Rprop,
            0f, 0f, 0f, 0f, 0f, null)
        {
            Extras = new Tensors.Engines.Compilation.FusedOptimizerExtras
            {
                RpropEtaPlus = (float)_options.EtaPlus,
                RpropEtaMinus = (float)_options.EtaMinus,
                RpropStepMin = (float)_options.MinStepSize,
                RpropStepMax = (float)_options.MaxStepSize,
                RpropInitialStep = (float)_options.InitialStepSize,
            }
        };
        return true;
    }

    /// <summary>
    /// Initializes the adaptive parameters used by the Rprop optimizer.
    /// </summary>
    protected override void InitializeAdaptiveParameters()
    {
        base.InitializeAdaptiveParameters();
        _t = 0;
    }

    /// <summary>
    /// Performs the optimization process using the Rprop algorithm.
    /// </summary>
    /// <param name="inputData">The input data for optimization, including training data and targets.</param>
    /// <returns>The result of the optimization process, including the best solution found.</returns>
    /// <remarks>
    /// <para>
    /// Unlike every other gradient optimizer here, this loop does NOT create a batcher. The gradient is computed
    /// over the whole training set on each iteration, because Rprop reads meaning into gradient sign changes and
    /// mini-batch noise would flip those signs for reasons unrelated to the loss surface, collapsing every step
    /// size to Delta_min. That is the paper's stated requirement, not a simplification.
    /// </para>
    /// </remarks>
    public override OptimizationResult<T, TInput, TOutput> Optimize(OptimizationInputData<T, TInput, TOutput> inputData)
    {
        ValidationHelper<T>.ValidateInputData(inputData);

        var currentSolution = InitializeWorkingSolution(inputData.XTrain);
        var bestStepData = new OptimizationStepData<T, TInput, TOutput>();
        var previousStepData = new OptimizationStepData<T, TInput, TOutput>();
        var parameters = InterfaceGuard.Parameterizable(currentSolution).GetParameters();
        _prevGradient = new Vector<T>(parameters.Length);
        _stepSize = Vector<T>.CreateDefault(parameters.Length, NumOps.FromDouble(_options.InitialStepSize));
        // Reset the tape-side state, which persists across Optimize calls on the same instance. Carrying over
        // grown or collapsed step sizes from a previous run would start the next one mid-adaptation.
        _tapePrevGradient.Clear();
        _tapeStepSize.Clear();
        _tapeStep = 0;
        InitializeAdaptiveParameters();

        for (int epoch = 0; epoch < _options.MaxIterations; epoch++)
        {
            NotifyEpochStart(epoch);

            // Full-batch gradient: one honest gradient over all the data, exactly once per iteration.
            var gradient = CalculateGradient(currentSolution, inputData.XTrain, inputData.YTrain);
            currentSolution = UpdateSolution(currentSolution, gradient);

            var currentStepData = EvaluateSolution(currentSolution, inputData);
            UpdateBestSolution(currentStepData, ref bestStepData);

            UpdateAdaptiveParameters(currentStepData, previousStepData);

            if (UpdateIterationHistoryAndCheckEarlyStopping(epoch, bestStepData))
            {
                return CreateOptimizationResult(bestStepData, inputData);
            }

            if (IsConvergedAgainstPreviousEpoch(epoch, currentStepData, previousStepData, _options.Tolerance))
            {
                return CreateOptimizationResult(bestStepData, inputData);
            }

            previousStepData = currentStepData;
        }

        return CreateOptimizationResult(bestStepData, inputData);
    }

    /// <summary>
    /// Updates the current solution using the Rprop update rule.
    /// </summary>
    /// <param name="currentSolution">The current solution being optimized.</param>
    /// <param name="gradient">The gradient of the current solution.</param>
    /// <returns>A new solution with updated parameters.</returns>
    protected override IFullModel<T, TInput, TOutput> UpdateSolution(IFullModel<T, TInput, TOutput> currentSolution, Vector<T> gradient)
    {
        // #1413 CONSOLIDATION: NN solutions go through base.UpdateSolution which synthesizes a
        // TapeStepContext and delegates to Step (one source of truth, matches PyTorch/TF/JAX).
        if (currentSolution is AiDotNet.Interfaces.INeuralNetwork<T>)
        {
            return base.UpdateSolution(currentSolution, gradient);
        }

        var parameters = InterfaceGuard.Parameterizable(currentSolution).GetParameters();
        var newParameters = UpdateParameters(parameters, gradient);

        return InterfaceGuard.Parameterizable(currentSolution).WithParameters(newParameters);
    }

    /// <summary>
    /// Updates a vector of parameters using the Rprop algorithm.
    /// </summary>
    /// <param name="parameters">The current parameter vector to be updated.</param>
    /// <param name="gradient">The gradient vector corresponding to the parameters.</param>
    /// <returns>The updated parameter vector.</returns>
    /// <remarks>
    /// <para>
    /// The three cases of the paper's rule are selected branchlessly. With s = sign(g_prev * g), the indicators
    /// <c>grew = max(s, 0)</c>, <c>shrank = max(-s, 0)</c> and <c>held = 1 - grew - shrank</c> are one-hot across
    /// the three cases, so multiplying by them and summing reproduces the if/else exactly while keeping the whole
    /// update on the vectorized engine.
    /// </para>
    /// </remarks>
    public override Vector<T> UpdateParameters(Vector<T> parameters, Vector<T> gradient)
    {
        if (_prevGradient == null || _stepSize == null || _prevGradient.Length != parameters.Length)
        {
            _prevGradient = new Vector<T>(parameters.Length);
            _stepSize = Vector<T>.CreateDefault(parameters.Length, NumOps.FromDouble(_options.InitialStepSize));
            _t = 0;
        }

        _t++;

        int n = parameters.Length;
        var zeros = new Vector<T>(n);
        var ones = Vector<T>.CreateDefault(n, NumOps.One);

        // s = sign(g_prev * g): +1 same direction, -1 reversed, 0 if either gradient was exactly zero.
        var signOfProduct = (Vector<T>)Engine.Sign((Vector<T>)Engine.Multiply(_prevGradient, gradient));

        // One-hot indicators for the paper's three cases.
        var grew = (Vector<T>)Engine.Max(signOfProduct, zeros);
        var shrank = (Vector<T>)Engine.Max((Vector<T>)Engine.Multiply(signOfProduct, NumOps.FromDouble(-1.0)), zeros);
        var held = (Vector<T>)Engine.Subtract(Engine.Subtract(ones, grew), shrank);

        // Delta *= eta_plus | eta_minus | 1, then clamp into [Delta_min, Delta_max].
        var factor = (Vector<T>)Engine.Add(
            Engine.Add(
                Engine.Multiply(grew, NumOps.FromDouble(_options.EtaPlus)),
                Engine.Multiply(shrank, NumOps.FromDouble(_options.EtaMinus))),
            held);
        _stepSize = (Vector<T>)Engine.Clamp(
            (Vector<T>)Engine.Multiply(_stepSize, factor),
            NumOps.FromDouble(_options.MinStepSize),
            NumOps.FromDouble(_options.MaxStepSize));

        // On a reversal the paper takes no step and forgets the gradient. Zeroing the gradient there delivers
        // both at once: sign(0) is 0, so the move below is a no-op, and the stored gradient becomes 0.
        var effectiveGradient = (Vector<T>)Engine.Multiply(gradient, (Vector<T>)Engine.Subtract(ones, shrank));

        var move = (Vector<T>)Engine.Multiply((Vector<T>)Engine.Sign(effectiveGradient), _stepSize);
        _prevGradient = effectiveGradient;

        return (Vector<T>)Engine.Subtract(parameters, move);
    }

    // Per-parameter Rprop state for tape-based (neural-network) training.
    private readonly ConcurrentDictionary<Tensor<T>, Tensor<T>> _tapePrevGradient = new(TensorReferenceComparer<Tensor<T>>.Instance);
    private readonly ConcurrentDictionary<Tensor<T>, Tensor<T>> _tapeStepSize = new(TensorReferenceComparer<Tensor<T>>.Instance);
    private int _tapeStep;

    /// <inheritdoc />
    public override void Step(TapeStepContext<T> context)
    {
        PrepareTapeState(context);

        _tapeStep++;

        T etaPlus = NumOps.FromDouble(_options.EtaPlus);
        T etaMinus = NumOps.FromDouble(_options.EtaMinus);
        T minStep = NumOps.FromDouble(_options.MinStepSize);
        T maxStep = NumOps.FromDouble(_options.MaxStepSize);
        T initialStep = NumOps.FromDouble(_options.InitialStepSize);
        T negativeOne = NumOps.FromDouble(-1.0);

        foreach (var param in context.Parameters)
        {
            if (!SparseEmbeddingOptimizerHelpers.TryGetEffectiveGradient(context, param, Engine, out var grad))
                continue;

            if (!_tapePrevGradient.TryGetValue(param, out var prevGrad))
            {
                prevGrad = new Tensor<T>(param._shape);
                _tapePrevGradient[param] = prevGrad;
            }
            if (!_tapeStepSize.TryGetValue(param, out var stepSize))
            {
                stepSize = new Tensor<T>(param._shape);
                stepSize.Fill(initialStep);
                _tapeStepSize[param] = stepSize;
            }

            var signOfProduct = Engine.TensorSign(Engine.TensorMultiply(prevGrad, grad));

            // The sign is exactly -1, 0, or +1. Scalar clamps create the one-hot masks without
            // allocating and filling full-size ones/zeros tensors for every parameter on every step.
            var grew = Engine.TensorClamp(signOfProduct, NumOps.Zero, NumOps.One);
            var shrank = Engine.TensorClamp(
                Engine.TensorMultiplyScalar(signOfProduct, negativeOne),
                NumOps.Zero,
                NumOps.One);
            var held = Engine.TensorNegate(
                Engine.TensorSubtractScalar(Engine.TensorAbs(signOfProduct), NumOps.One));

            var factor = Engine.TensorAdd(
                Engine.TensorAdd(
                    Engine.TensorMultiplyScalar(grew, etaPlus),
                    Engine.TensorMultiplyScalar(shrank, etaMinus)),
                held);
            Engine.TensorCopy(
                Engine.TensorClamp(Engine.TensorMultiply(stepSize, factor), minStep, maxStep),
                stepSize);

            var notShrank = Engine.TensorNegate(Engine.TensorSubtractScalar(shrank, NumOps.One));
            var effectiveGradient = Engine.TensorMultiply(grad, notShrank);

            Engine.TensorSubtractInPlace(
                param,
                Engine.TensorMultiply(Engine.TensorSign(effectiveGradient), stepSize));

            Engine.TensorCopy(effectiveGradient, prevGrad);
        }
    }

    /// <summary>
    /// Reverses an Rprop update to recover the parameters from before the step.
    /// </summary>
    /// <param name="updatedParameters">Parameters after the Rprop update.</param>
    /// <param name="appliedGradients">The gradients that were applied. Validated but not read -- see remarks.</param>
    /// <returns>The parameters as they were before the update.</returns>
    /// <remarks>
    /// <para>
    /// The move was <c>-sign(g_effective) * Delta</c>, and after the update the optimizer holds exactly those two
    /// quantities: <c>_prevGradient</c> IS g_effective (already zeroed on the weights that reversed) and
    /// <c>_stepSize</c> IS the Delta that was applied. So the inverse is exact and needs no gradient argument --
    /// the parameter is accepted for signature compatibility and validated, but the stored state is more
    /// authoritative than a caller-supplied gradient would be, because it already encodes the reversal masking.
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

        if (_prevGradient == null || _stepSize == null || _prevGradient.Length != updatedParameters.Length)
        {
            throw new InvalidOperationException(
                "Rprop optimizer state is not initialized. ReverseUpdate must be called after UpdateParameters.");
        }

        var move = (Vector<T>)Engine.Multiply((Vector<T>)Engine.Sign(_prevGradient), _stepSize);

        return (Vector<T>)Engine.Add(updatedParameters, move);
    }

    /// <summary>
    /// Gets the current per-weight step sizes, or null before the first update.
    /// </summary>
    /// <returns>The step-size vector Delta.</returns>
    /// <remarks>
    /// <para>
    /// Exposed because these ARE Rprop's learned state, and because they diagnose the classic misuse: step sizes
    /// sitting at <see cref="RpropOptimizerOptions{T, TInput, TOutput}.MinStepSize"/> across the board means the
    /// gradient signs are flipping every step, which in practice means mini-batch noise rather than a genuinely
    /// converged model.
    /// </para>
    /// <para><b>For Beginners:</b> Returns each weight's current step size. If they have all collapsed to the
    /// minimum, Rprop is being fed noisy gradients and is not the right optimizer for that setup.</para>
    /// </remarks>
    public Vector<T>? GetStepSizes() => _stepSize;

    /// <summary>
    /// Updates the adaptive parameters of the optimizer.
    /// </summary>
    /// <param name="currentStepData">Data from the current optimization step.</param>
    /// <param name="previousStepData">Data from the previous optimization step.</param>
    /// <remarks>
    /// <para>
    /// Only the base bookkeeping runs. Rprop has no learning rate to adapt -- the step sizes already adapt
    /// themselves, per weight, which is the entire algorithm -- so the outer increase/decrease machinery the other
    /// optimizers share deliberately has nothing to do here.
    /// </para>
    /// </remarks>
    protected override void UpdateAdaptiveParameters(OptimizationStepData<T, TInput, TOutput> currentStepData, OptimizationStepData<T, TInput, TOutput> previousStepData)
    {
        base.UpdateAdaptiveParameters(currentStepData, previousStepData);
    }

    /// <summary>
    /// Updates the optimizer's options with new settings.
    /// </summary>
    /// <param name="options">The new options to be applied to the optimizer.</param>
    /// <exception cref="ArgumentException">Thrown when the provided options are not of the correct type.</exception>
    protected override void UpdateOptions(OptimizationAlgorithmOptions<T, TInput, TOutput> options)
    {
        if (options is RpropOptimizerOptions<T, TInput, TOutput> rpropOptions)
        {
            ValidateHyperparameters(rpropOptions);
            _options = rpropOptions;
        }
        else
        {
            throw new ArgumentException("Invalid options type. Expected RpropOptimizerOptions.");
        }
    }

    /// <summary>
    /// Retrieves the current options of the optimizer.
    /// </summary>
    /// <returns>The current optimization algorithm options.</returns>
    public override OptimizationAlgorithmOptions<T, TInput, TOutput> GetOptions()
    {
        return _options;
    }

    /// <summary>
    /// Converts the current state of the optimizer into a byte array for storage or transmission.
    /// </summary>
    /// <returns>A byte array representing the serialized state of the optimizer.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This saves the per-weight step sizes and remembered gradients, which are all
    /// of Rprop's learned state -- resuming without them would throw away everything the optimizer had worked out
    /// about the loss surface.</para>
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

            SerializeVector(writer, _prevGradient);
            SerializeVector(writer, _stepSize);

            return ms.ToArray();
        }
    }

    private void SerializeVector(BinaryWriter writer, Vector<T>? vector)
    {
        writer.Write(vector is not null);
        if (vector is not null)
        {
            writer.Write(vector.Length);
            for (int i = 0; i < vector.Length; i++)
            {
                writer.Write(NumOps.ToDouble(vector[i]));
            }
        }
    }

    private Vector<T>? DeserializeVector(BinaryReader reader)
    {
        bool hasVector = reader.ReadBoolean();
        if (hasVector)
        {
            int length = reader.ReadInt32();
            T[] data = new T[length];
            for (int i = 0; i < length; i++)
            {
                data[i] = NumOps.FromDouble(reader.ReadDouble());
            }
            return new Vector<T>(data);
        }
        return null;
    }

    /// <summary>
    /// Restores the optimizer's state from a byte array previously created by <see cref="Serialize"/>.
    /// </summary>
    /// <param name="data">The byte array containing the serialized optimizer state.</param>
    public override void Deserialize(byte[] data)
    {
        using (MemoryStream ms = new MemoryStream(data))
        using (BinaryReader reader = new BinaryReader(ms))
        {
            int baseDataLength = reader.ReadInt32();
            byte[] baseData = reader.ReadBytes(baseDataLength);
            base.Deserialize(baseData);

            string optionsJson = reader.ReadString();
            _options = JsonConvert.DeserializeObject<RpropOptimizerOptions<T, TInput, TOutput>>(optionsJson)
                ?? throw new InvalidOperationException("Failed to deserialize optimizer options.");

            _t = reader.ReadInt32();

            _prevGradient = DeserializeVector(reader);
            _stepSize = DeserializeVector(reader);
        }
    }

    /// <summary>
    /// Generates a unique key for caching gradients based on the current state of the optimizer and input data.
    /// </summary>
    /// <param name="model">The model being optimized.</param>
    /// <param name="X">The input data.</param>
    /// <param name="y">The target values.</param>
    /// <returns>A string that uniquely identifies the current optimization state for gradient caching.</returns>
    protected override string GenerateGradientCacheKey(IFullModel<T, TInput, TOutput> model, TInput X, TOutput y)
    {
        var baseKey = base.GenerateGradientCacheKey(model, X, y);
        return $"{baseKey}_Rprop_{_options.EtaPlus}_{_options.EtaMinus}_{_t}";
    }
}
