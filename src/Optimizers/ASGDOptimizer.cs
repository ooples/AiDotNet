using System.Collections.Concurrent;
using AiDotNet.Tensors.Engines.Autodiff;
using Newtonsoft.Json;
using AiDotNet.Helpers;

using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.Optimizers;

/// <summary>
/// Implements ASGD (Averaged Stochastic Gradient Descent): decayed SGD that additionally maintains a running
/// average of the iterates, which is the quantity the method is named for and the one worth reading out.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type of the model being optimized.</typeparam>
/// <typeparam name="TOutput">The output data type of the model being optimized.</typeparam>
/// <remarks>
/// <para>
/// The averaging result is Ruppert (1988) and Polyak and Juditsky, "Acceleration of Stochastic Approximation by
/// Averaging" (SIAM J. Control Optim. 30(4):838-855, 1992, doi:10.1137/0330046): with a slowly decaying step
/// size, the average of the iterates converges at the optimal asymptotic rate even though no individual iterate
/// does. The schedule and decay parameterization used here follows Xu, "Towards Optimal One Pass Large Scale
/// Learning with Averaged Stochastic Gradient Descent" (arXiv:1107.2490, 2011). Per step t:
/// </para>
/// <code>
/// eta_t = gamma_0 / (1 + lambda * gamma_0 * t)^alpha
/// mu_t  = 1 / max(1, t - t0)
/// g_t   = grad + weightDecay * theta
/// theta = theta * (1 - eta_t * lambda) - eta_t * g_t
/// ax    = ax + mu_t * (theta - ax)
/// </code>
/// <para>
/// Note what mu_t does at the two ends of its range. While t is at or below t0 it equals 1, so <c>ax</c> simply
/// copies the current iterate and no averaging happens; past t0 it becomes 1/(t - t0), which is exactly the
/// incremental form of a running mean over the tail of the trajectory. This is why t0 defaults to 1e6: averaging
/// from step 1 would drag the answer back toward the initialization, so the average is deliberately started only
/// once the iterates are fluctuating around the solution rather than still travelling toward it.
/// </para>
/// <para>
/// <c>eta_t</c> is recomputed from <c>gamma_0</c> and the step count on every step rather than being decayed in
/// place, so the schedule is exact and does not accumulate rounding drift over a long run. That also matches the
/// fused ASGD kernel, which is handed a freshly computed <c>eta_t</c> per step.
/// </para>
/// <para><b>For Beginners:</b> Ordinary SGD never really settles down -- each step reacts to one noisy batch, so
/// the model keeps jittering around the right answer instead of stopping on it. ASGD does not try to stop the
/// jitter. It runs ordinary SGD and quietly keeps a running average of everywhere the model has been, because
/// the average of many jittery positions is far closer to the truth than any one of them. It is the same reason
/// the average of many dart throws finds the bullseye better than any single dart.</para>
/// </remarks>
[ComponentType(ComponentType.Optimizer)]
[PipelineStage(PipelineStage.Training)]
public class ASGDOptimizer<T, TInput, TOutput> : GradientBasedOptimizerBase<T, TInput, TOutput>, Fused.IFusedOptimizerSpec
{
    /// <summary>
    /// The options specific to the ASGD optimizer.
    /// </summary>
    private ASGDOptimizerOptions<T, TInput, TOutput> _options;

    /// <summary>
    /// The running average of the iterates (the <c>ax</c> of the reference implementation).
    /// </summary>
    private Vector<T>? _ax;

    /// <summary>
    /// The current time step. Drives both the learning-rate schedule and the averaging weight.
    /// </summary>
    private int _t;

    /// <summary>
    /// Initializes a new instance of the ASGDOptimizer class.
    /// </summary>
    /// <param name="model">The model whose parameters this optimizer updates.</param>
    /// <param name="options">The options for configuring the ASGD optimizer, or null for reference defaults.</param>
    /// <param name="engine">The compute engine to run tensor operations on, or null for the ambient engine.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This sets up the optimizer. With no options you get the standard defaults
    /// (step size 0.01, decay 1e-4, exponent 0.75). Remember that averaging does not start until step
    /// <see cref="ASGDOptimizerOptions{T, TInput, TOutput}.T0"/>, which defaults to a million -- lower it if you
    /// want the averaging to actually engage.</para>
    /// </remarks>
    public ASGDOptimizer(
        IFullModel<T, TInput, TOutput> model,
        ASGDOptimizerOptions<T, TInput, TOutput>? options = null,
        IEngine? engine = null)
        : base(model, options ?? new())
    {
        _options = options ?? new ASGDOptimizerOptions<T, TInput, TOutput>();

        InitializeAdaptiveParameters();
    }

    /// <summary>
    /// Describes this ASGD optimizer for the compiled fused-training kernel.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Maps to <see cref="Tensors.Engines.Compilation.OptimizerType.ASGD"/>. Lambda, Alpha and T0 travel in
    /// <see cref="Tensors.Engines.Compilation.FusedOptimizerExtras"/> rather than the beta/epsilon slots, because
    /// they are not moment decay rates; the plan recomputes eta_t and mu_t from them per step exactly as the
    /// eager path does.
    /// </para>
    /// <para>
    /// Declines when an outer adaptive learning rate or an unsupported LR scheduler is configured. That guard
    /// matters more here than for most optimizers: ASGD already owns a learning-rate schedule of its own, and the
    /// plan bakes gamma_0 in at build time, so an outer schedule that moved gamma_0 afterwards would silently
    /// give the fused path a different trajectory from the eager one.
    /// </para>
    /// </remarks>
    bool Fused.IFusedOptimizerSpec.TryGetFusedOptimizerConfig(out Fused.FusedOptimizerConfig config)
    {
        config = default;
        if (_options.UseAdaptiveLearningRate) return false;
        if (!TryGetFusedLrSchedule(out var schedule)) return false;
        config = new Fused.FusedOptimizerConfig(
            Tensors.Engines.Compilation.OptimizerType.ASGD,
            (float)GetCurrentLearningRate(),
            0f, 0f, 0f,
            (float)_options.WeightDecay, schedule)
        {
            Extras = new Tensors.Engines.Compilation.FusedOptimizerExtras
            {
                Lambd = (float)_options.Lambda,
                Alpha = (float)_options.Alpha,
                T0 = (float)_options.T0,
            }
        };
        return true;
    }

    /// <summary>
    /// Initializes the adaptive parameters used by the ASGD optimizer.
    /// </summary>
    protected override void InitializeAdaptiveParameters()
    {
        base.InitializeAdaptiveParameters();
        // Learning rate is set by the base class from options.InitialLearningRate.
        _t = 0;
    }

    /// <summary>
    /// Computes the two per-step scalars of the ASGD update.
    /// </summary>
    /// <param name="step">The 1-based time step t.</param>
    /// <param name="gamma0">The base learning rate gamma_0.</param>
    /// <param name="eta">The decayed step size eta_t.</param>
    /// <param name="mu">The averaging weight mu_t.</param>
    /// <remarks>
    /// <para>
    /// Shared by the flat-vector path and the tape path so the two cannot drift apart, and deliberately mirrors
    /// the arithmetic the fused plan performs before calling the kernel.
    /// </para>
    /// </remarks>
    private void ComputeSchedule(int step, double gamma0, out double eta, out double mu)
    {
        eta = gamma0 / Math.Pow(1.0 + _options.Lambda * gamma0 * step, _options.Alpha);
        mu = 1.0 / Math.Max(1.0, step - _options.T0);
    }

    /// <summary>
    /// Performs the optimization process using the ASGD algorithm.
    /// </summary>
    /// <param name="inputData">The input data for optimization, including training data and targets.</param>
    /// <returns>The result of the optimization process, including the best solution found.</returns>
    /// <remarks>
    /// <para>
    /// Once averaging has actually engaged (t has passed t0), the averaged iterate is evaluated alongside the
    /// final one and kept if it scores better. That is the payoff the method exists for -- the average is
    /// supposed to beat the last iterate -- but it is checked rather than assumed, so enabling averaging can
    /// never make the returned model worse.
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

        var currentSolution = InitializeWorkingSolution(inputData.XTrain);
        var bestStepData = new OptimizationStepData<T, TInput, TOutput>();
        var previousStepData = new OptimizationStepData<T, TInput, TOutput>();
        var parameters = InterfaceGuard.Parameterizable(currentSolution).GetParameters();
        // ax starts at the initial iterate, not at zero: mu_1 = 1 makes the first step overwrite it anyway, but
        // starting from the parameters keeps the invariant "ax is a convex combination of visited iterates" true
        // at every point, including before the first step.
        _ax = parameters.Clone();
        // Reset the tape-side state (see RAdam/AMSGrad for why this is not optional): both the per-parameter
        // averages and the step counter persist across Optimize calls on the same instance, and a carried-over
        // counter would put the schedule and the averaging window in the wrong place from iteration 1.
        _tapeAx.Clear();
        _tapeStep = 0;
        InitializeAdaptiveParameters();

        for (int epoch = 0; epoch < _options.MaxIterations; epoch++)
        {
            NotifyEpochStart(epoch);
            var batcher = CreateBatcher(inputData, _options.BatchSize, epoch);

            foreach (var (xBatch, yBatch, batchIndices) in batcher.GetBatches())
            {
                // Note: _t is incremented inside UpdateParameters, not here.
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

            if (IsConvergedAgainstPreviousEpoch(epoch, currentStepData, previousStepData, _options.Tolerance))
            {
                return CreateOptimizationResult(bestStepData, inputData);
            }

            previousStepData = currentStepData;
        }

        TryPromoteAveragedSolution(currentSolution, inputData, ref bestStepData);

        return CreateOptimizationResult(bestStepData, inputData);
    }

    /// <summary>
    /// Evaluates the averaged iterate and adopts it if it scores better than the best solution seen.
    /// </summary>
    /// <remarks>
    /// <para>
    /// A no-op unless averaging has genuinely started (t &gt; t0) -- below that threshold mu_t is 1 and <c>ax</c>
    /// is just a copy of the current iterate, so there would be nothing to compare.
    /// </para>
    /// </remarks>
    private void TryPromoteAveragedSolution(
        IFullModel<T, TInput, TOutput> currentSolution,
        OptimizationInputData<T, TInput, TOutput> inputData,
        ref OptimizationStepData<T, TInput, TOutput> bestStepData)
    {
        if (_ax == null || _t <= _options.T0)
        {
            return;
        }

        var averagedSolution = InterfaceGuard.Parameterizable(currentSolution).WithParameters(_ax);
        var averagedStepData = EvaluateSolution(averagedSolution, inputData);
        UpdateBestSolution(averagedStepData, ref bestStepData);
    }

    /// <summary>
    /// Gets the running average of the iterates, or null if no step has been taken yet.
    /// </summary>
    /// <returns>The averaged parameter vector, or null.</returns>
    /// <remarks>
    /// <para>
    /// This is the quantity ASGD exists to produce. It is exposed rather than substituted automatically because
    /// the caller owns the parameter vector; <see cref="Optimize"/> does consider it on your behalf. Below the
    /// averaging start step it is equal to the current iterate by construction.
    /// </para>
    /// <para><b>For Beginners:</b> Returns the running average of everywhere the model has been. Once averaging
    /// has started, this is usually a better model than the current one.</para>
    /// </remarks>
    public Vector<T>? GetAveragedParameters() => _ax;

    /// <summary>
    /// Updates the current solution using the ASGD update rule.
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
    /// Updates a vector of parameters using the ASGD optimization algorithm.
    /// </summary>
    /// <param name="parameters">The current parameter vector to be updated.</param>
    /// <param name="gradient">The gradient vector corresponding to the parameters.</param>
    /// <returns>The updated parameter vector -- the raw iterate, not the average.</returns>
    /// <remarks>
    /// <para>
    /// Returns the raw iterate deliberately. The recursion needs theta_{t-1} on the next call, so feeding the
    /// average back in would corrupt the trajectory the average is taken over. Read the average with
    /// <see cref="GetAveragedParameters"/>.
    /// </para>
    /// </remarks>
    public override Vector<T> UpdateParameters(Vector<T> parameters, Vector<T> gradient)
    {
        if (_ax == null || _ax.Length != parameters.Length)
        {
            _ax = parameters.Clone();
            _t = 0;
        }

        _t++;

        double gamma0 = NumOps.ToDouble(CurrentLearningRate);
        ComputeSchedule(_t, gamma0, out double eta, out double mu);

        // g = grad + weightDecay * theta
        var effectiveGradient = _options.WeightDecay != 0.0
            ? (Vector<T>)Engine.Add(gradient, Engine.Multiply(parameters, NumOps.FromDouble(_options.WeightDecay)))
            : gradient;

        // theta = theta * (1 - eta * lambda) - eta * g
        var decayed = (Vector<T>)Engine.Multiply(parameters, NumOps.FromDouble(1.0 - eta * _options.Lambda));
        var updated = (Vector<T>)Engine.Subtract(
            decayed,
            Engine.Multiply(effectiveGradient, NumOps.FromDouble(eta)));

        // ax = ax + mu * (theta - ax)
        _ax = (Vector<T>)Engine.Add(
            _ax,
            Engine.Multiply((Vector<T>)Engine.Subtract(updated, _ax), NumOps.FromDouble(mu)));

        return updated;
    }

    // Per-parameter averaged iterate for tape-based (neural-network) training.
    private readonly ConcurrentDictionary<Tensor<T>, Tensor<T>> _tapeAx = new(TensorReferenceComparer<Tensor<T>>.Instance);
    private int _tapeStep;

    /// <inheritdoc />
    public override void Step(TapeStepContext<T> context)
    {
        PrepareTapeState(context);

        _tapeStep++;

        double gamma0 = NumOps.ToDouble(CurrentLearningRate);
        ComputeSchedule(_tapeStep, gamma0, out double eta, out double mu);

        T decayFactor = NumOps.FromDouble(1.0 - eta * _options.Lambda);
        T negEta = NumOps.FromDouble(-eta);
        T muT = NumOps.FromDouble(mu);
        T weightDecay = NumOps.FromDouble(_options.WeightDecay);

        foreach (var param in context.Parameters)
        {
            if (!SparseEmbeddingOptimizerHelpers.TryGetEffectiveGradient(context, param, Engine, out var grad))
                continue;

            // ax seeds from the parameter itself, matching the flat path's Clone() of the initial iterate.
            if (!_tapeAx.TryGetValue(param, out var ax))
            {
                ax = new Tensor<T>(param._shape);
                Engine.TensorCopy(param, ax);
                _tapeAx[param] = ax;
            }

            // g = grad + weightDecay * theta
            var effectiveGradient = _options.WeightDecay != 0.0
                ? Engine.TensorAdd(grad, Engine.TensorMultiplyScalar(param, weightDecay))
                : grad;

            // theta = theta * (1 - eta * lambda) - eta * g
            Engine.TensorCopy(
                Engine.TensorAdd(
                    Engine.TensorMultiplyScalar(param, decayFactor),
                    Engine.TensorMultiplyScalar(effectiveGradient, negEta)),
                param);

            // ax = ax + mu * (theta - ax)
            Engine.TensorCopy(
                Engine.TensorAdd(ax, Engine.TensorMultiplyScalar(Engine.TensorSubtract(param, ax), muT)),
                ax);
        }
    }

    /// <summary>
    /// Reverses an ASGD gradient update to recover the parameters from before the step.
    /// </summary>
    /// <param name="updatedParameters">Parameters after the ASGD update.</param>
    /// <param name="appliedGradients">The gradients that were applied.</param>
    /// <returns>The parameters as they were before the update.</returns>
    /// <remarks>
    /// <para>
    /// The ASGD step is theta_t = theta_{t-1}*(1 - eta*lambda) - eta*(g + wd*theta_{t-1}), which is affine in
    /// theta_{t-1} and therefore exactly invertible:
    /// theta_{t-1} = (theta_t + eta*g) / (1 - eta*lambda - eta*wd).
    /// Unlike the moment-based optimizers this needs no stored state beyond the step count, so it is exact
    /// rather than a reconstruction.
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

        if (_t <= 0)
        {
            throw new InvalidOperationException(
                "ASGD optimizer state is not initialized. ReverseUpdate must be called after UpdateParameters.");
        }

        double gamma0 = NumOps.ToDouble(CurrentLearningRate);
        ComputeSchedule(_t, gamma0, out double eta, out _);

        double scale = 1.0 - eta * _options.Lambda - eta * _options.WeightDecay;
        if (Math.Abs(scale) < 1e-15)
        {
            throw new InvalidOperationException(
                $"ASGD step at t={_t} is not invertible: the combined decay factor (1 - eta*Lambda - eta*WeightDecay) " +
                $"is {scale}, which collapses the parameters onto a single point. Reduce Lambda, WeightDecay or the " +
                "learning rate.");
        }

        // original = (updated + eta * g) / (1 - eta*lambda - eta*wd)
        var restored = (Vector<T>)Engine.Add(
            updatedParameters,
            Engine.Multiply(appliedGradients, NumOps.FromDouble(eta)));

        return (Vector<T>)Engine.Divide(restored, NumOps.FromDouble(scale));
    }

    /// <summary>
    /// Updates the adaptive parameters of the optimizer based on the current and previous optimization steps.
    /// </summary>
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
    /// <param name="options">The new options to be applied to the optimizer.</param>
    /// <exception cref="ArgumentException">Thrown when the provided options are not of the correct type.</exception>
    protected override void UpdateOptions(OptimizationAlgorithmOptions<T, TInput, TOutput> options)
    {
        if (options is ASGDOptimizerOptions<T, TInput, TOutput> asgdOptions)
        {
            _options = asgdOptions;
        }
        else
        {
            throw new ArgumentException("Invalid options type. Expected ASGDOptimizerOptions.");
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
    /// <para><b>For Beginners:</b> This saves the optimizer so training can resume exactly where it left off,
    /// including the running average and the step count that decides when averaging starts.</para>
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

            SerializeVector(writer, _ax);

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
            _options = JsonConvert.DeserializeObject<ASGDOptimizerOptions<T, TInput, TOutput>>(optionsJson)
                ?? throw new InvalidOperationException("Failed to deserialize optimizer options.");

            _t = reader.ReadInt32();

            _ax = DeserializeVector(reader);
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
        return $"{baseKey}_ASGD_{_options.Lambda}_{_options.Alpha}_{_options.T0}_{_t}";
    }
}
