using System.Collections.Concurrent;
using AiDotNet.Tensors.Engines.Autodiff;
using Newtonsoft.Json;
using AiDotNet.Helpers;

using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.Optimizers;

/// <summary>
/// Implements RAdam (Rectified Adam), the variant of Adam that rectifies the variance of the adaptive
/// learning rate instead of relying on a hand-tuned warmup schedule.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type of the model being optimized.</typeparam>
/// <typeparam name="TOutput">The output data type of the model being optimized.</typeparam>
/// <remarks>
/// <para>
/// Faithful to Algorithm 2 of "On the Variance of the Adaptive Learning Rate and Beyond"
/// (Liu, Jiang, He, Chen, Liu, Gao and Han; ICLR 2020; arXiv:1908.03265). Per step t:
/// </para>
/// <code>
/// rho_inf = 2 / (1 - beta2) - 1
/// m_t     = beta1 * m_{t-1} + (1 - beta1) * g_t
/// v_t     = beta2 * v_{t-1} + (1 - beta2) * g_t^2
/// mHat_t  = m_t / (1 - beta1^t)
/// rho_t   = rho_inf - 2 * t * beta2^t / (1 - beta2^t)
/// if rho_t &gt; 4:
///     l_t   = sqrt((1 - beta2^t) / v_t)
///     r_t   = sqrt(((rho_t - 4)(rho_t - 2) rho_inf) / ((rho_inf - 4)(rho_inf - 2) rho_t))
///     theta = theta - alpha * r_t * mHat_t * l_t
/// else:
///     theta = theta - alpha * mHat_t
/// </code>
/// <para>
/// Both branches matter. The <c>else</c> branch is not a degenerate case to be optimized away: for the first
/// few steps the variance of the adaptive term is not merely large but undefined, so the algorithm deliberately
/// takes a plain (bias-corrected) momentum step with NO adaptive scaling at all. At the default beta2 = 0.999
/// that covers roughly the first four steps. Collapsing the two branches into one is the single easiest way to
/// get RAdam subtly wrong, because the resulting optimizer still trains -- just without the property the paper
/// exists to provide.
/// </para>
/// <para>
/// Epsilon is a numerical guard on the rectified branch only, matching the reference implementation. The paper's
/// l_t has no epsilon, and the un-rectified branch has no denominator to guard.
/// </para>
/// <para><b>For Beginners:</b> Adam decides how big a step to take for each individual parameter by looking at
/// how much that parameter's gradient has been bouncing around. Early in training there are only a few gradients
/// to judge from, so that measurement is unreliable -- and acting on an unreliable measurement is what makes Adam
/// unstable at the start. The usual fix is "warmup", where you manually start with a tiny learning rate and ramp
/// it up. RAdam works out mathematically how trustworthy the measurement currently is and scales the step by
/// that, and for the first few steps -- when it is not trustworthy at all -- it just takes an ordinary momentum
/// step instead. The result behaves like Adam-with-warmup without you having to tune a warmup schedule.</para>
/// </remarks>
[ComponentType(ComponentType.Optimizer)]
[PipelineStage(PipelineStage.Training)]
public class RAdamOptimizer<T, TInput, TOutput> : GradientBasedOptimizerBase<T, TInput, TOutput>, Fused.IFusedOptimizerSpec
{
    /// <summary>
    /// The options specific to the RAdam optimizer.
    /// </summary>
    private RAdamOptimizerOptions<T, TInput, TOutput> _options;

    /// <summary>
    /// The first moment vector (exponential moving average of gradients).
    /// </summary>
    private Vector<T>? _m;

    /// <summary>
    /// The second moment vector (exponential moving average of squared gradients).
    /// </summary>
    private Vector<T>? _v;

    /// <summary>
    /// The current time step. Drives both bias correction and the rectification term, so it must
    /// start at 1 on the first update.
    /// </summary>
    private int _t;

    /// <summary>
    /// Initializes a new instance of the RAdamOptimizer class.
    /// </summary>
    /// <param name="model">The model whose parameters this optimizer updates.</param>
    /// <param name="options">The options for configuring the RAdam optimizer, or null for paper defaults.</param>
    /// <param name="engine">The compute engine to run tensor operations on, or null for the ambient engine.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This sets up the RAdam optimizer. Passing no options gives you the settings
    /// from the paper (learning rate 0.001, beta1 0.9, beta2 0.999), which are a good default for most models.
    /// </para>
    /// </remarks>
    public RAdamOptimizer(
        IFullModel<T, TInput, TOutput> model,
        RAdamOptimizerOptions<T, TInput, TOutput>? options = null,
        IEngine? engine = null)
        : base(model, options ?? new())
    {
        _options = options ?? new RAdamOptimizerOptions<T, TInput, TOutput>();

        InitializeAdaptiveParameters();
    }

    /// <summary>
    /// Describes this RAdam optimizer for the compiled fused-training kernel.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The Tensors fused kernel implements RAdam directly as
    /// <see cref="Tensors.Engines.Compilation.OptimizerType.RAdam"/>, including the rectification term and the
    /// un-rectified momentum branch, so the mapping is one-to-one with the eager
    /// <see cref="UpdateParameters"/> / <see cref="Step"/> paths above -- same formula, same epsilon placement,
    /// same step counter semantics.
    /// </para>
    /// <para>
    /// Declines (returns false) when an outer adaptive learning rate or an unsupported LR scheduler is
    /// configured, because those change the learning rate after the plan has baked it in -- exactly like the
    /// Adam, AdaMax and AMSGrad specs.
    /// </para>
    /// </remarks>
    bool Fused.IFusedOptimizerSpec.TryGetFusedOptimizerConfig(out Fused.FusedOptimizerConfig config)
    {
        config = default;
        if (_options.UseAdaptiveLearningRate) return false;
        if (!TryGetFusedLrSchedule(out var schedule)) return false;
        config = new Fused.FusedOptimizerConfig(
            Tensors.Engines.Compilation.OptimizerType.RAdam,
            (float)GetCurrentLearningRate(),
            (float)_options.Beta1, (float)_options.Beta2, (float)_options.Epsilon,
            0f, schedule);
        return true;
    }

    /// <summary>
    /// Initializes the adaptive parameters used by the RAdam optimizer.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This resets the learning rate and time step to their starting values,
    /// preparing the optimizer for a new optimization run.
    /// </para>
    /// </remarks>
    protected override void InitializeAdaptiveParameters()
    {
        base.InitializeAdaptiveParameters();
        // Learning rate is set by the base class from options.InitialLearningRate.
        _t = 0;
    }

    /// <summary>
    /// Performs the optimization process using the RAdam algorithm.
    /// </summary>
    /// <param name="inputData">The input data for optimization, including training data and targets.</param>
    /// <returns>The result of the optimization process, including the best solution found.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the main optimization loop. It repeatedly updates the solution using
    /// RAdam steps until it converges or hits a stopping condition.
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
        _m = new Vector<T>(parameters.Length);
        _v = new Vector<T>(parameters.Length);
        // Reset the NN tape-side state. The flat-vector path got fresh Vectors above; the tape path uses
        // parameter-tensor-keyed dictionaries plus a separate _tapeStep counter that PERSIST across Optimize
        // calls on the same optimizer instance. Without this clear, a second Optimize call would carry the
        // prior run's moments AND -- critically for RAdam -- a pre-advanced step counter, which drives the
        // rectification term. A carried-over counter would silently skip the un-rectified warmup phase that
        // is the entire point of this optimizer.
        _tapeM.Clear();
        _tapeV.Clear();
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

        return CreateOptimizationResult(bestStepData, inputData);
    }

    /// <summary>
    /// Updates the current solution using the RAdam update rule.
    /// </summary>
    /// <param name="currentSolution">The current solution being optimized.</param>
    /// <param name="gradient">The gradient of the current solution.</param>
    /// <returns>A new solution with updated parameters.</returns>
    protected override IFullModel<T, TInput, TOutput> UpdateSolution(IFullModel<T, TInput, TOutput> currentSolution, Vector<T> gradient)
    {
        // #1413 CONSOLIDATION: NN solutions go through base.UpdateSolution which synthesizes a
        // TapeStepContext and delegates to Step (one source of truth, matches PyTorch/TF/JAX).
        // Non-NN solutions (regression, clustering, classical models) keep the flat-vector path below.
        if (currentSolution is AiDotNet.Interfaces.INeuralNetwork<T>)
        {
            return base.UpdateSolution(currentSolution, gradient);
        }

        var parameters = InterfaceGuard.Parameterizable(currentSolution).GetParameters();
        var newParameters = UpdateParameters(parameters, gradient);

        return InterfaceGuard.Parameterizable(currentSolution).WithParameters(newParameters);
    }

    /// <summary>
    /// Computes the rectification scalars for the current step.
    /// </summary>
    /// <param name="step">The 1-based time step t.</param>
    /// <param name="rectified">True when rho_t &gt; 4 and the adaptive term may be used.</param>
    /// <param name="rt">The rectification factor r_t; meaningless (and unused) when not rectified.</param>
    /// <remarks>
    /// <para>
    /// Shared by the flat-vector path, the tape path and <see cref="ReverseUpdate"/> so all three agree on when
    /// the adaptive term switches on. These are the rho_inf / rho_t / r_t lines of Algorithm 2 and depend only
    /// on beta2 and the step count -- never on the parameters -- so they are computed once per step, not per
    /// element.
    /// </para>
    /// </remarks>
    private void ComputeRectification(int step, out bool rectified, out double rt)
    {
        double beta2 = _options.Beta2;
        double bc2 = 1.0 - Math.Pow(beta2, step);
        double rhoInf = 2.0 / (1.0 - beta2) - 1.0;
        double rhoT = rhoInf - 2.0 * step * Math.Pow(beta2, step) / bc2;

        rectified = rhoT > 4.0;
        rt = rectified
            ? Math.Sqrt(((rhoT - 4.0) * (rhoT - 2.0) * rhoInf) /
                        ((rhoInf - 4.0) * (rhoInf - 2.0) * rhoT))
            : 0.0;
    }

    /// <summary>
    /// Updates a vector of parameters using the RAdam optimization algorithm.
    /// </summary>
    /// <param name="parameters">The current parameter vector to be updated.</param>
    /// <param name="gradient">The gradient vector corresponding to the parameters.</param>
    /// <returns>The updated parameter vector.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This applies one RAdam step to a flat list of parameters. Early on it takes a
    /// plain momentum step; once enough gradient history has accumulated it switches to the adaptive step, scaled
    /// down by how reliable that history currently is.</para>
    /// </remarks>
    public override Vector<T> UpdateParameters(Vector<T> parameters, Vector<T> gradient)
    {
        if (_m == null || _v == null || _m.Length != parameters.Length)
        {
            _m = new Vector<T>(parameters.Length);
            _v = new Vector<T>(parameters.Length);
            _t = 0;
        }

        _t++;

        T beta1 = NumOps.FromDouble(_options.Beta1);
        T beta2 = NumOps.FromDouble(_options.Beta2);
        T oneMinusBeta1 = NumOps.FromDouble(1 - _options.Beta1);
        T oneMinusBeta2 = NumOps.FromDouble(1 - _options.Beta2);
        T biasCorrection1 = NumOps.FromDouble(1 - Math.Pow(_options.Beta1, _t));

        // m = beta1 * m + (1 - beta1) * gradient
        _m = (Vector<T>)Engine.Add(
            Engine.Multiply(_m, beta1),
            Engine.Multiply(gradient, oneMinusBeta1));

        // v = beta2 * v + (1 - beta2) * gradient^2
        _v = (Vector<T>)Engine.Add(
            Engine.Multiply(_v, beta2),
            Engine.Multiply((Vector<T>)Engine.Multiply(gradient, gradient), oneMinusBeta2));

        // mHat = m / (1 - beta1^t)
        var mHat = (Vector<T>)Engine.Divide(_m, biasCorrection1);

        ComputeRectification(_t, out bool rectified, out double rt);

        Vector<T> update;
        if (rectified)
        {
            // update = lr * r_t * mHat / (sqrt(v / (1 - beta2^t)) + epsilon)
            T biasCorrection2 = NumOps.FromDouble(1 - Math.Pow(_options.Beta2, _t));
            var vHat = (Vector<T>)Engine.Divide(_v, biasCorrection2);
            var denominator = (Vector<T>)Engine.Add(
                Engine.Sqrt(vHat),
                Vector<T>.CreateDefault(vHat.Length, NumOps.FromDouble(_options.Epsilon)));

            var scaledLr = NumOps.Multiply(CurrentLearningRate, NumOps.FromDouble(rt));
            update = (Vector<T>)Engine.Divide(Engine.Multiply(mHat, scaledLr), denominator);
        }
        else
        {
            // Un-rectified phase: plain bias-corrected momentum step, no adaptive scaling.
            update = (Vector<T>)Engine.Multiply(mHat, CurrentLearningRate);
        }

        return (Vector<T>)Engine.Subtract(parameters, update);
    }

    // Per-parameter RAdam state for tape-based (neural-network) training.
    private readonly ConcurrentDictionary<Tensor<T>, Tensor<T>> _tapeM = new(TensorReferenceComparer<Tensor<T>>.Instance);
    private readonly ConcurrentDictionary<Tensor<T>, Tensor<T>> _tapeV = new(TensorReferenceComparer<Tensor<T>>.Instance);
    private int _tapeStep;

    /// <inheritdoc />
    public override void Step(TapeStepContext<T> context)
    {
        PrepareTapeState(context);

        _tapeStep++;

        T beta1 = NumOps.FromDouble(_options.Beta1);
        T beta2 = NumOps.FromDouble(_options.Beta2);
        T oneMinusBeta1 = NumOps.FromDouble(1 - _options.Beta1);
        T oneMinusBeta2 = NumOps.FromDouble(1 - _options.Beta2);
        T epsilon = NumOps.FromDouble(_options.Epsilon);
        T biasCorrection1 = NumOps.FromDouble(1 - Math.Pow(_options.Beta1, _tapeStep));
        T biasCorrection2 = NumOps.FromDouble(1 - Math.Pow(_options.Beta2, _tapeStep));

        ComputeRectification(_tapeStep, out bool rectified, out double rt);
        T scaledLr = rectified
            ? NumOps.Multiply(CurrentLearningRate, NumOps.FromDouble(rt))
            : CurrentLearningRate;

        foreach (var param in context.Parameters)
        {
            if (!SparseEmbeddingOptimizerHelpers.TryGetEffectiveGradient(context, param, Engine, out var grad))
                continue;

            if (!_tapeM.TryGetValue(param, out var m)) { m = new Tensor<T>(param._shape); _tapeM[param] = m; }
            if (!_tapeV.TryGetValue(param, out var v)) { v = new Tensor<T>(param._shape); _tapeV[param] = v; }

            // m = beta1 * m + (1 - beta1) * grad
            Engine.TensorCopy(
                Engine.TensorAdd(Engine.TensorMultiplyScalar(m, beta1), Engine.TensorMultiplyScalar(grad, oneMinusBeta1)),
                m);

            // v = beta2 * v + (1 - beta2) * grad^2
            Engine.TensorCopy(
                Engine.TensorAdd(
                    Engine.TensorMultiplyScalar(v, beta2),
                    Engine.TensorMultiplyScalar(Engine.TensorMultiply(grad, grad), oneMinusBeta2)),
                v);

            // mHat = m / (1 - beta1^t)
            var mHat = Engine.TensorDivideScalar(m, biasCorrection1);

            Tensor<T> update;
            if (rectified)
            {
                // update = lr * r_t * mHat / (sqrt(v / (1 - beta2^t)) + epsilon)
                var denom = Engine.TensorAddScalar(
                    Engine.TensorSqrt(Engine.TensorDivideScalar(v, biasCorrection2)),
                    epsilon);
                update = Engine.TensorMultiplyScalar(Engine.TensorDivide(mHat, denom), scaledLr);
            }
            else
            {
                // Un-rectified phase: plain bias-corrected momentum step, no adaptive scaling.
                update = Engine.TensorMultiplyScalar(mHat, scaledLr);
            }

            Engine.TensorSubtractInPlace(param, update);
        }
    }

    /// <summary>
    /// Reverses a RAdam gradient update to recover the parameters from before the step.
    /// </summary>
    /// <param name="updatedParameters">Parameters after the RAdam update.</param>
    /// <param name="appliedGradients">The gradients that were applied.</param>
    /// <returns>The parameters as they were before the update.</returns>
    /// <remarks>
    /// <para>
    /// Recomputes the update from the optimizer's post-step state (_m, _v, _t), so it must be called immediately
    /// after <see cref="UpdateParameters"/> while that state still corresponds to the step being reversed. Both
    /// RAdam branches are reproduced -- reversing an un-rectified step with the rectified formula would be wrong
    /// by the whole adaptive factor.
    /// </para>
    /// <para><b>For Beginners:</b> This works out where the parameters were before the last step, by rebuilding
    /// exactly the step that was taken and adding it back on.</para>
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

        if (_m == null || _v == null || _m.Length != updatedParameters.Length)
        {
            throw new InvalidOperationException(
                "RAdam optimizer state is not initialized. ReverseUpdate must be called after UpdateParameters.");
        }

        T biasCorrection1 = NumOps.FromDouble(1 - Math.Pow(_options.Beta1, _t));
        var mHat = (Vector<T>)Engine.Divide(_m, biasCorrection1);

        ComputeRectification(_t, out bool rectified, out double rt);

        Vector<T> update;
        if (rectified)
        {
            T biasCorrection2 = NumOps.FromDouble(1 - Math.Pow(_options.Beta2, _t));
            var vHat = (Vector<T>)Engine.Divide(_v, biasCorrection2);
            var denominator = (Vector<T>)Engine.Add(
                Engine.Sqrt(vHat),
                Vector<T>.CreateDefault(vHat.Length, NumOps.FromDouble(_options.Epsilon)));

            var scaledLr = NumOps.Multiply(CurrentLearningRate, NumOps.FromDouble(rt));
            update = (Vector<T>)Engine.Divide(Engine.Multiply(mHat, scaledLr), denominator);
        }
        else
        {
            update = (Vector<T>)Engine.Multiply(mHat, CurrentLearningRate);
        }

        return (Vector<T>)Engine.Add(updatedParameters, update);
    }

    /// <summary>
    /// Updates the adaptive parameters of the optimizer based on the current and previous optimization steps.
    /// </summary>
    /// <param name="currentStepData">Data from the current optimization step.</param>
    /// <param name="previousStepData">Data from the previous optimization step.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This adjusts the learning rate between epochs based on whether the model is
    /// improving. It is switched off by default, because RAdam's own rectification already handles the part of
    /// training that most needs learning-rate care.</para>
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
    /// Updates the optimizer's options with new settings.
    /// </summary>
    /// <param name="options">The new options to be applied to the optimizer.</param>
    /// <exception cref="ArgumentException">Thrown when the provided options are not of the correct type.</exception>
    protected override void UpdateOptions(OptimizationAlgorithmOptions<T, TInput, TOutput> options)
    {
        if (options is RAdamOptimizerOptions<T, TInput, TOutput> radamOptions)
        {
            _options = radamOptions;
        }
        else
        {
            throw new ArgumentException("Invalid options type. Expected RAdamOptimizerOptions.");
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
    /// <para><b>For Beginners:</b> This saves a snapshot of the optimizer so training can be resumed later from
    /// exactly where it left off -- including the step count, which RAdam needs in order to know whether it is
    /// still in its un-rectified warmup phase.</para>
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

            SerializeVector(writer, _m);
            SerializeVector(writer, _v);

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
            _options = JsonConvert.DeserializeObject<RAdamOptimizerOptions<T, TInput, TOutput>>(optionsJson)
                ?? throw new InvalidOperationException("Failed to deserialize optimizer options.");

            _t = reader.ReadInt32();

            _m = DeserializeVector(reader);
            _v = DeserializeVector(reader);
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
        return $"{baseKey}_RAdam_{_options.Beta1}_{_options.Beta2}_{_t}";
    }
}
