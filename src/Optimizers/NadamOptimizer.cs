using AiDotNet.Tensors.Engines.DirectGpu;
using System.Collections.Concurrent;
using AiDotNet.Tensors.Engines.Autodiff;
using Newtonsoft.Json;
using AiDotNet.Helpers;

using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.Optimizers;

/// <summary>
/// Implements the Nesterov-accelerated Adaptive Moment Estimation (Nadam) optimization algorithm.
/// </summary>
/// <remarks>
/// <para>
/// Nadam combines the ideas of Adam (adaptive learning rates) and Nesterov accelerated gradient (NAG).
/// It adapts the learning rates of each parameter and incorporates momentum using Nesterov's method.
/// </para>
/// <para><b>For Beginners:</b>
/// Imagine you're rolling a smart ball down a hill. This ball can adjust its speed for different parts of the hill (adaptive learning rates),
/// and it can look ahead to anticipate slopes (Nesterov's method). This combination helps it find the lowest point more efficiently.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
[ComponentType(ComponentType.Optimizer)]
[PipelineStage(PipelineStage.Training)]
public partial class NadamOptimizer<T, TInput, TOutput> : GradientBasedOptimizerBase<T, TInput, TOutput>, Fused.IFusedOptimizerSpec
{
    /// <summary>
    /// Describes this Nadam instance for the fused-compiled training kernel
    /// (Tensors <c>OptimizerType.Nadam</c> — Nesterov-accelerated Adam). No
    /// decoupled weight decay, so WeightDecay is 0. Declines (falls back to
    /// eager) when an adaptive LR or an unmappable scheduler is configured.
    /// Verified fused-vs-eager parity in FusedOptimizerParityTests.
    /// </summary>
    bool Fused.IFusedOptimizerSpec.TryGetFusedOptimizerConfig(out Fused.FusedOptimizerConfig config)
    {
        config = default;
        if (_options.UseAdaptiveLearningRate) return false;
        if (!TryGetFusedLrSchedule(out var schedule)) return false;
        config = new Fused.FusedOptimizerConfig(
            Tensors.Engines.Compilation.OptimizerType.Nadam,
            (float)GetCurrentLearningRate(),
            (float)_options.Beta1, (float)_options.Beta2, (float)_options.Epsilon,
            0f, schedule);
        return true;
    }

    /// <summary>
    /// The options specific to the Nadam optimizer.
    /// </summary>
    private NadamOptimizerOptions<T, TInput, TOutput> _options;

    /// <summary>
    /// The first moment vector (momentum).
    /// </summary>
    [AiDotNet.Attributes.Buffer]
    private Vector<T>? _m;

    /// <summary>
    /// The second moment vector (adaptive learning rates).
    /// </summary>
    [AiDotNet.Attributes.Buffer]
    private Vector<T>? _v;

    /// <summary>
    /// The current time step.
    /// </summary>
    private int _t;

    /// <summary>
    /// Stores the pre-update snapshot of first moment vector for accurate reverse updates.
    /// </summary>
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T>? _previousM;

    /// <summary>
    /// Stores the pre-update snapshot of second moment vector for accurate reverse updates.
    /// </summary>
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T>? _previousV;

    /// <summary>
    /// Stores the pre-update snapshot of the time step for accurate reverse updates.
    /// </summary>
    private int _previousT;

    /// <summary>
    /// GPU buffer for first moment estimates (m).
    /// </summary>
    private IGpuBuffer? _gpuM;

    /// <summary>
    /// GPU buffer for second moment estimates (v).
    /// </summary>
    private IGpuBuffer? _gpuV;

    /// <summary>
    /// Initializes a new instance of the NadamOptimizer class.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This constructor sets up the Nadam optimizer with the provided options and dependencies.
    /// If no options are provided, it uses default settings.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like preparing your smart ball for the hill-rolling experiment. You're setting up its initial properties
    /// and deciding how it will adapt during its journey.
    /// </para>
    /// </remarks>
    /// <param name="model">The model to optimize.</param>
    /// <param name="options">The Nadam-specific optimization options.</param>
    public NadamOptimizer(
        IFullModel<T, TInput, TOutput> model,
        NadamOptimizerOptions<T, TInput, TOutput>? options = null)
        : base(model, options ?? new())
    {
        _options = options ?? new NadamOptimizerOptions<T, TInput, TOutput>();

        InitializeAdaptiveParameters();
    }

    /// <summary>
    /// Initializes the adaptive parameters for the Nadam optimizer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method sets up the initial learning rate and resets the time step counter.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like setting the initial speed of your smart ball and resetting its internal clock before it starts rolling.
    /// </para>
    /// </remarks>
    protected override void InitializeAdaptiveParameters()
    {
        base.InitializeAdaptiveParameters();
        // Learning rate is now set by base class from options.InitialLearningRate
        _t = 0;
    }

    /// <summary>
    /// Performs the optimization process using the Nadam algorithm.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method implements the main optimization loop. It iterates through the data, calculating gradients,
    /// updating the momentum and adaptive learning rates, and adjusting the model parameters accordingly.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is the actual process of rolling your smart ball down the hill. In each step, you're calculating which way
    /// the ball should roll (gradient), how fast it's moving (momentum), and how it should adapt its speed (adaptive learning rates).
    /// You keep doing this until the ball finds the lowest point or you've rolled it enough times.
    /// </para>
    /// <para><b>DataLoader Integration:</b> This method uses the DataLoader API for efficient batch processing.
    /// It creates a batcher using <see cref="GradientBasedOptimizerBase{T,TInput,TOutput}.CreateBatcher"/>
    /// and notifies the sampler of epoch starts using
    /// <see cref="GradientBasedOptimizerBase{T,TInput,TOutput}.NotifyEpochStart"/>.
    /// </para>
    /// </remarks>
    /// <param name="inputData">The input data for the optimization process.</param>
    /// <returns>The result of the optimization process.</returns>
    public override OptimizationResult<T, TInput, TOutput> Optimize(OptimizationInputData<T, TInput, TOutput> inputData)
    {
        ValidationHelper<T>.ValidateInputData(inputData);

        var currentSolution = InitializeWorkingSolution(inputData.XTrain);
        var bestStepData = new OptimizationStepData<T, TInput, TOutput>();
        var previousStepData = new OptimizationStepData<T, TInput, TOutput>();
        var parameters = InterfaceGuard.Parameterizable(currentSolution).GetParameters();
        _m = new Vector<T>(parameters.Length);
        _v = new Vector<T>(parameters.Length);
        // Clear the tape-side moments / step so a reused optimizer instance starts a
        // new run with fresh Nadam state instead of resuming the previous run's.
        _tapeM.Clear();
        _tapeV2.Clear();
        _tapeStep = 0;

        InitializeAdaptiveParameters();

        for (int epoch = 0; epoch < _options.MaxIterations; epoch++)
        {
            NotifyEpochStart(epoch);
            var batcher = CreateBatcher(inputData, _options.BatchSize, epoch);

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

            previousStepData = currentStepData;
        }

        return CreateOptimizationResult(bestStepData, inputData);
    }

    /// <summary>
    /// Updates the current solution based on the calculated gradient using the Nadam algorithm.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method applies the Nadam update rule to adjust the model parameters. It uses both momentum
    /// and adaptive learning rates, incorporating Nesterov's accelerated gradient.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like adjusting the ball's position based on its current speed, the slope it's on, and its ability
    /// to look ahead. It's a complex calculation that helps the ball move more efficiently towards the lowest point.
    /// </para>
    /// </remarks>
    /// <param name="currentSolution">The current model solution.</param>
    /// <param name="gradient">The calculated gradient.</param>
    /// <returns>An updated symbolic model with improved coefficients.</returns>
    protected override IFullModel<T, TInput, TOutput> UpdateSolution(IFullModel<T, TInput, TOutput> currentSolution, Vector<T> gradient)
    {
        // #1413 CONSOLIDATION: NN solutions go through base.UpdateSolution
        // which synthesizes a TapeStepContext and delegates to Step
        // (one source of truth, matches PyTorch/TF/JAX). Non-NN solutions
        // (regression, clustering, classical models) keep the legacy
        // flat-vector path below for backward compatibility.
        if (currentSolution is AiDotNet.Interfaces.INeuralNetwork<T>)
        {
            return base.UpdateSolution(currentSolution, gradient);
        }
        var parameters = InterfaceGuard.Parameterizable(currentSolution).GetParameters();

        // === Vectorized Nadam Update using IEngine (Phase B: US-GPU-015) ===
        T beta1 = NumOps.FromDouble(_options.Beta1);
        T beta2 = NumOps.FromDouble(_options.Beta2);
        T oneMinusBeta1 = NumOps.FromDouble(1 - _options.Beta1);
        T oneMinusBeta2 = NumOps.FromDouble(1 - _options.Beta2);
        T epsilon = NumOps.FromDouble(_options.Epsilon);
        T biasCorrectionM = NumOps.FromDouble(1 - Math.Pow(_options.Beta1, _t));
        T biasCorrectionV = NumOps.FromDouble(1 - Math.Pow(_options.Beta2, _t));
        T nesterovFactor = NumOps.Divide(oneMinusBeta1, biasCorrectionM);

        // Update biased first moment estimate: m = beta1 * m + (1 - beta1) * gradient
        var beta1TimesM = (Vector<T>)Engine.Multiply(_m!, beta1);
        var oneMinusBeta1TimesGrad = (Vector<T>)Engine.Multiply(gradient, oneMinusBeta1);
        _m = (Vector<T>)Engine.Add(beta1TimesM, oneMinusBeta1TimesGrad);

        // Update biased second raw moment estimate: v = beta2 * v + (1 - beta2) * gradient^2
        var gradSquared = (Vector<T>)Engine.Multiply(gradient, gradient);
        var beta2TimesV = (Vector<T>)Engine.Multiply(_v!, beta2);
        var oneMinusBeta2TimesGradSq = (Vector<T>)Engine.Multiply(gradSquared, oneMinusBeta2);
        _v = (Vector<T>)Engine.Add(beta2TimesV, oneMinusBeta2TimesGradSq);

        // Compute bias-corrected first moment estimate: mHat = m / (1 - beta1^t)
        var mHat = (Vector<T>)Engine.Divide(_m, biasCorrectionM);

        // Compute bias-corrected second raw moment estimate: vHat = v / (1 - beta2^t)
        var vHat = (Vector<T>)Engine.Divide(_v, biasCorrectionV);

        // Compute the Nesterov momentum term: mHatNesterov = beta1 * mHat + nesterovFactor * gradient
        var beta1TimesMHat = (Vector<T>)Engine.Multiply(mHat, beta1);
        var nesterovGrad = (Vector<T>)Engine.Multiply(gradient, nesterovFactor);
        var mHatNesterov = (Vector<T>)Engine.Add(beta1TimesMHat, nesterovGrad);

        // Update parameters: update = (lr * mHatNesterov) / (sqrt(vHat) + epsilon)
        var sqrtVHat = (Vector<T>)Engine.Sqrt(vHat);
        var epsilonVec = new Vector<T>(Enumerable.Repeat(epsilon, sqrtVHat.Length));
        var denominator = (Vector<T>)Engine.Add(sqrtVHat, epsilonVec);
        var lrTimesMHatNesterov = (Vector<T>)Engine.Multiply(mHatNesterov, CurrentLearningRate);
        var update = (Vector<T>)Engine.Divide(lrTimesMHatNesterov, denominator);

        // params = params - update
        var newCoefficients = (Vector<T>)Engine.Subtract(parameters, update);

        return InterfaceGuard.Parameterizable(currentSolution).WithParameters(newCoefficients);
    }

    /// <summary>
    /// Updates a vector of parameters using the Nadam optimization algorithm.
    /// </summary>
    /// <param name="parameters">The current parameter vector to be updated.</param>
    /// <param name="gradient">The gradient vector corresponding to the parameters.</param>
    /// <returns>The updated parameter vector.</returns>
    /// <remarks>
    /// <para>
    /// Nadam combines Adam's adaptive learning rates with Nesterov's accelerated gradient, providing
    /// the benefits of both techniques: adaptive per-parameter learning rates and lookahead momentum.
    /// </para>
    /// <para><b>For Beginners:</b> Nadam is like a smart ball that not only adapts its speed for
    /// different parts of the hill (Adam) but also looks ahead to anticipate slopes (Nesterov).
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

        if (_m == null || _v == null || _m.Length != parameters.Length || _v.Length != parameters.Length)
        {
            _m = new Vector<T>(parameters.Length);
            _v = new Vector<T>(parameters.Length);
            _t = 0;
        }

        // Save previous state BEFORE updating for ReverseUpdate. Buffers are allocated once and
        // copied into IN PLACE: Clone() allocated two FULL-LENGTH vectors every step (measured as
        // 32 MB/step of AdamW's cost at 2,000,000 parameters before the same fix there).
        if (_previousM == null || _previousM.Length != _m.Length)
        {
            _previousM = new Vector<T>(_m.Length, skipZeroInit: true);
        }
        if (_previousV == null || _previousV.Length != _v.Length)
        {
            _previousV = new Vector<T>(_v.Length, skipZeroInit: true);
        }
        _m.AsSpan().CopyTo(_previousM.AsWritableSpan());
        _v.AsSpan().CopyTo(_previousV.AsWritableSpan());
        _previousT = _t;

        _t++;

        // === Vectorized Nadam Update using IEngine (Phase B: US-GPU-015) ===
        T beta1 = NumOps.FromDouble(_options.Beta1);
        T beta2 = NumOps.FromDouble(_options.Beta2);
        T oneMinusBeta1 = NumOps.FromDouble(1 - _options.Beta1);
        T oneMinusBeta2 = NumOps.FromDouble(1 - _options.Beta2);
        T epsilon = NumOps.FromDouble(_options.Epsilon);
        T biasCorrectionM = NumOps.FromDouble(1 - Math.Pow(_options.Beta1, _t));
        T biasCorrectionV = NumOps.FromDouble(1 - Math.Pow(_options.Beta2, _t));
        T nesterovFactor = NumOps.Divide(oneMinusBeta1, biasCorrectionM);

        // Update biased first moment estimate: m = beta1 * m + (1 - beta1) * gradient
        // ONE FUSED IN-PLACE PASS -- same rewrite as Adam/AdamW/Adagrad/Lion, same reason.
        //
        // Replaces 16 Engine calls that each RETURNED a fresh full-length vector, including
        // `new Vector<T>(Enumerable.Repeat(epsilon, sqrtVHat.Length))` -- a full-length constant
        // vector built through an ENUMERATOR, one element at a time, to hold a single scalar.
        // Measured on Adam's near-identical chain at 2,000,000 double parameters: 701.9 MB/step and
        // ~290 ms/step before, 15.3 MB and 17.0 ms after.
        //
        // Per-element operand and association order preserved exactly (Dozat 2016), including the
        // Nesterov term built from the bias-corrected mHat and the raw gradient:
        //   m = b1*m + (1-b1)*g ;  v = b2*v + ((g*g)*(1-b2))
        //   mHatNesterov = b1*(m/bcM) + nesterovFactor*g
        //   out = p - (mHatNesterov*lr) / (sqrt(v/bcV) + eps)
        var updatedParams = new Vector<T>(parameters.Length, skipZeroInit: true);
        var pSpan = parameters.AsSpan();
        var gSpan = gradient.AsSpan();
        var mSpan = _m.AsWritableSpan();
        var vSpan = _v.AsWritableSpan();
        var outSpan = updatedParams.AsWritableSpan();
        T learningRate = CurrentLearningRate;

        for (int i = 0; i < pSpan.Length; i++)
        {
            T g = gSpan[i];

            T m = NumOps.Add(NumOps.Multiply(mSpan[i], beta1), NumOps.Multiply(g, oneMinusBeta1));
            mSpan[i] = m;

            T v = NumOps.Add(
                NumOps.Multiply(vSpan[i], beta2),
                NumOps.Multiply(NumOps.Multiply(g, g), oneMinusBeta2));
            vSpan[i] = v;

            T mHat = NumOps.Divide(m, biasCorrectionM);
            T vHat = NumOps.Divide(v, biasCorrectionV);

            T mHatNesterov = NumOps.Add(
                NumOps.Multiply(mHat, beta1),
                NumOps.Multiply(g, nesterovFactor));

            T denominator = NumOps.Add(NumOps.Sqrt(vHat), epsilon);
            T update = NumOps.Divide(NumOps.Multiply(mHatNesterov, learningRate), denominator);
            outSpan[i] = NumOps.Subtract(pSpan[i], update);
        }

        return updatedParams;
    }

    // Per-parameter Nadam state for tape-based training
    private readonly ConcurrentDictionary<Tensor<T>, Tensor<T>> _tapeM = new(TensorReferenceComparer<Tensor<T>>.Instance);
    private readonly ConcurrentDictionary<Tensor<T>, Tensor<T>> _tapeV2 = new(TensorReferenceComparer<Tensor<T>>.Instance);
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
        T biasCorrectionM = NumOps.FromDouble(1 - Math.Pow(_options.Beta1, _tapeStep));
        T biasCorrectionV = NumOps.FromDouble(1 - Math.Pow(_options.Beta2, _tapeStep));
        T nesterovFactor = NumOps.Divide(oneMinusBeta1, biasCorrectionM);

        // GPU-resident step (AIDOTNET_GPU_ADAM=1); gated off, CPU fallback per-param when not GPU-resident.
        bool gpuAdam = typeof(T) == typeof(float)
            && System.Environment.GetEnvironmentVariable("AIDOTNET_GPU_ADAM") == "1"
            && AiDotNet.Tensors.Engines.AiDotNetEngine.Current is AiDotNet.Tensors.Engines.DirectGpuTensorEngine;

        foreach (var param in context.Parameters)
        {
            // True sparse scatter NAdam: m + v + param at touched indices only.
            if (!gpuAdam && SparseEmbeddingOptimizerHelpers.HasSparseEmbeddingGrad(param))
            {
                if (!_tapeM.TryGetValue(param, out var mSp)) { mSp = new Tensor<T>(param._shape); _tapeM[param] = mSp; }
                if (!_tapeV2.TryGetValue(param, out var vSp)) { vSp = new Tensor<T>(param._shape); _tapeV2[param] = vSp; }
                double bc1 = 1.0 - Math.Pow(_options.Beta1, _tapeStep);
                double bc2 = 1.0 - Math.Pow(_options.Beta2, _tapeStep);
                if (SparseEmbeddingOptimizerHelpers.TryApplyNadamSparse(
                        param, mSp, vSp,
                        NumOps.ToDouble(CurrentLearningRate),
                        _options.Beta1, _options.Beta2, bc1, bc2,
                        _options.Epsilon, weightDecay: 0.0))
                {
                    continue;
                }
            }

            if (!SparseEmbeddingOptimizerHelpers.TryGetEffectiveGradient(context, param, Engine, out var grad))
                continue;

            if (!_tapeM.TryGetValue(param, out var m)) { m = gpuAdam ? AiDotNet.Tensors.Helpers.TensorAllocator.RentPinnedOnGpu<T>(param._shape) : new Tensor<T>(param._shape); if (gpuAdam) m.AsWritableSpan().Clear(); _tapeM[param] = m; }
            if (!_tapeV2.TryGetValue(param, out var v)) { v = gpuAdam ? AiDotNet.Tensors.Helpers.TensorAllocator.RentPinnedOnGpu<T>(param._shape) : new Tensor<T>(param._shape); if (gpuAdam) v.AsWritableSpan().Clear(); _tapeV2[param] = v; }

            if (gpuAdam && param.Length == grad.Length
                && AiDotNet.Tensors.Engines.Gpu.GpuOptimizer.TryNadamStep((Tensor<float>)(object)param, (Tensor<float>)(object)grad, (Tensor<float>)(object)m, (Tensor<float>)(object)v,
                    (float)NumOps.ToDouble(CurrentLearningRate), (float)_options.Beta1, (float)_options.Beta2, (float)_options.Epsilon, 0f, _tapeStep))
                continue;

            // m = beta1 * m + (1 - beta1) * grad
            Engine.TensorCopy(Engine.TensorAdd(Engine.TensorMultiplyScalar(m, beta1), Engine.TensorMultiplyScalar(grad, oneMinusBeta1)), m);

            // v = beta2 * v + (1 - beta2) * grad^2
            Engine.TensorCopy(Engine.TensorAdd(Engine.TensorMultiplyScalar(v, beta2), Engine.TensorMultiplyScalar(Engine.TensorMultiply(grad, grad), oneMinusBeta2)), v);

            // Bias-corrected estimates
            var mHat = Engine.TensorDivideScalar(m, biasCorrectionM);
            var vHat = Engine.TensorDivideScalar(v, biasCorrectionV);

            // Nesterov momentum term: mHatNesterov = beta1 * mHat + nesterovFactor * grad
            var mHatNesterov = Engine.TensorAdd(Engine.TensorMultiplyScalar(mHat, beta1), Engine.TensorMultiplyScalar(grad, nesterovFactor));

            // param -= lr * mHatNesterov / (sqrt(vHat) + epsilon)
            var denom = Engine.TensorAddScalar(Engine.TensorSqrt(vHat), epsilon);
            var update = Engine.TensorMultiplyScalar(Engine.TensorDivide(mHatNesterov, denom), CurrentLearningRate);
            Engine.TensorSubtractInPlace(param, update);
        }
    }

    /// <summary>
    /// Reverses a Nadam gradient update to recover original parameters.
    /// </summary>
    /// <param name="updatedParameters">Parameters after Nadam update</param>
    /// <param name="appliedGradients">The gradients that were applied</param>
    /// <returns>Original parameters before the update</returns>
    /// <remarks>
    /// <para>
    /// Nadam's reverse update requires the optimizer's internal state (_m, _v, _t) from the forward pass.
    /// This method must be called immediately after UpdateParameters while the state is fresh.
    /// It recalculates the Nesterov-accelerated adaptive update that was applied.
    /// </para>
    /// <para><b>For Beginners:</b> This calculates where parameters were before a Nadam update.
    /// Nadam combines lookahead (Nesterov) with adaptive learning (Adam), so reversing requires
    /// both the momentum history (_m) and variance history (_v) to reconstruct the lookahead step.
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

        if (_m == null || _v == null || _m.Length != updatedParameters.Length || _v.Length != updatedParameters.Length || _t == 0)
        {
            throw new InvalidOperationException(
                "Nadam optimizer state is not initialized or timestep is zero. ReverseUpdate must be called after UpdateParameters.");
        }

        if (_previousM == null || _previousV == null || _previousM.Length != updatedParameters.Length || _previousV.Length != updatedParameters.Length)
        {
            throw new InvalidOperationException(
                "Nadam optimizer previous state is not available. ReverseUpdate must be called after UpdateParameters.");
        }

        // === Vectorized Reverse Nadam Update (Phase B: US-GPU-015) ===
        T beta1 = NumOps.FromDouble(_options.Beta1);
        T beta2 = NumOps.FromDouble(_options.Beta2);
        T oneMinusBeta1 = NumOps.FromDouble(1 - _options.Beta1);
        T biasCorrection1 = NumOps.FromDouble(1 - Math.Pow(_options.Beta1, _t));
        T biasCorrection2 = NumOps.FromDouble(1 - Math.Pow(_options.Beta2, _t));

        // CRITICAL: Use UPDATED moments (current _m and _v), not previous moments
        // Bias-corrected moments
        var biasCorr1Vec = Vector<T>.CreateDefault(_m.Length, biasCorrection1);
        var biasCorr2Vec = Vector<T>.CreateDefault(_v.Length, biasCorrection2);
        var mHat = (Vector<T>)Engine.Divide(_m, biasCorr1Vec);
        var vHat = (Vector<T>)Engine.Divide(_v, biasCorr2Vec);

        // Recalculate the Nesterov momentum term
        var beta1Vec = Vector<T>.CreateDefault(_m.Length, beta1);
        var beta1_mHat = (Vector<T>)Engine.Multiply(beta1Vec, mHat);
        var gradCoeff = NumOps.Divide(oneMinusBeta1, biasCorrection1);
        var gradCoeffVec = Vector<T>.CreateDefault(appliedGradients.Length, gradCoeff);
        var gradTerm = (Vector<T>)Engine.Multiply(gradCoeffVec, appliedGradients);
        var mHatNesterov = (Vector<T>)Engine.Add(beta1_mHat, gradTerm);

        // Recalculate the update that was applied
        var vHatSqrt = (Vector<T>)Engine.Sqrt(vHat);
        var epsilonVec = Vector<T>.CreateDefault(vHatSqrt.Length, NumOps.FromDouble(_options.Epsilon));
        var denominator = (Vector<T>)Engine.Add(vHatSqrt, epsilonVec);
        var currentLrVec = Vector<T>.CreateDefault(mHatNesterov.Length, CurrentLearningRate);
        var numerator = (Vector<T>)Engine.Multiply(currentLrVec, mHatNesterov);
        var update = (Vector<T>)Engine.Divide(numerator, denominator);

        // Reverse: original = updated + update
        var original = (Vector<T>)Engine.Add(updatedParameters, update);

        // Restore state so the rollback fully reverts the step
        _m = new Vector<T>(_previousM);
        _v = new Vector<T>(_previousV);

        // Restore time step to complete the rollback
        _t = _previousT;

        return original;
    }

    /// <summary>
    /// Updates the adaptive parameters of the optimizer based on the current and previous optimization steps.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method adjusts the learning rate based on the performance of the current step compared to the previous step.
    /// If improvement is seen, the learning rate may be increased, otherwise it may be decreased.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like adjusting how fast your ball rolls based on whether it's getting closer to the bottom of the hill.
    /// If it's improving, you might let it roll a bit faster. If not, you might slow it down to be more careful.
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
    /// This is like changing the rules of how your smart ball rolls mid-experiment. It makes sure you're only
    /// using rules that work for this specific type of smart ball (Nadam optimization).
    /// </para>
    /// </remarks>
    /// <param name="options">The new options to be applied to the optimizer.</param>
    /// <exception cref="ArgumentException">Thrown when the provided options are not of the correct type.</exception>
    protected override void UpdateOptions(OptimizationAlgorithmOptions<T, TInput, TOutput> options)
    {
        if (options is NadamOptimizerOptions<T, TInput, TOutput> nadamOptions)
        {
            _options = nadamOptions;
        }
        else
        {
            throw new ArgumentException("Invalid options type. Expected NadamOptimizerOptions.");
        }
    }

    /// <summary>
    /// Gets the current optimization algorithm options.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method returns the current options used by the Nadam optimizer.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like checking your current smart ball rolling rules. It lets you see all the settings and strategies 
    /// you're currently using in your experiment.
    /// </para>
    /// </remarks>
    /// <returns>The current NadamOptimizerOptions object.</returns>
    public override OptimizationAlgorithmOptions<T, TInput, TOutput> GetOptions()
    {
        return _options;
    }

    private void SerializeVector(BinaryWriter writer, Vector<T>? vector)
    {
        bool hasVector = vector is not null;
        writer.Write(hasVector);
        if (hasVector)
        {
            writer.Write((vector ?? throw new InvalidOperationException("vector has not been initialized.")).Length);
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
    /// Initializes Nadam optimizer state on the GPU.
    /// </summary>
    /// <param name="parameterCount">Number of parameters.</param>
    /// <param name="backend">GPU backend for memory allocation.</param>
    public override void InitializeGpuState(int parameterCount, IDirectGpuBackend backend)
    {
        if (_gpuStateInitialized && _gpuM != null && _gpuV != null)
            return;

        // Allocate GPU buffers for first and second moment estimates (initialized to zero)
        var zeros = new float[parameterCount];
        _gpuM = backend.AllocateBuffer(zeros);
        _gpuV = backend.AllocateBuffer(zeros);

        _t = 0;
        _gpuStateInitialized = true;
    }

    /// <summary>
    /// Updates parameters on GPU using Nadam optimization.
    /// </summary>
    public override void UpdateParametersGpu(IGpuBuffer parameters, IGpuBuffer gradients, int parameterCount, IDirectGpuBackend backend)
    {
        if (!_gpuStateInitialized || _gpuM == null || _gpuV == null)
        {
            InitializeGpuState(parameterCount, backend);
        }

        _t++;

        // Call the Nadam GPU kernel
        // Note: Nadam doesn't have weight decay option, passing 0.0f
        backend.NadamUpdate(
            parameters,
            gradients,
            _gpuM!,
            _gpuV!,
            (float)_options.InitialLearningRate,
            (float)_options.Beta1,
            (float)_options.Beta2,
            (float)_options.Epsilon,
            0.0f, // Nadam doesn't use weight decay
            _t,
            parameterCount
        );
    }

    /// <summary>
    /// Disposes GPU-allocated optimizer state.
    /// </summary>
    public override void DisposeGpuState()
    {
        _gpuM?.Dispose();
        _gpuM = null;
        _gpuV?.Dispose();
        _gpuV = null;
        _gpuStateInitialized = false;
    }

    /// <summary>
    /// Generates a unique key for caching gradients.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method creates a unique identifier for caching gradients based on the current model, input data,
    /// and Nadam-specific parameters. This helps in efficiently reusing previously calculated gradients when possible.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like creating a special label for each unique situation your smart ball encounters. It helps the ball
    /// remember and quickly recall how it should move in similar situations, making the whole process more efficient.
    /// </para>
    /// </remarks>
    /// <param name="model">The current symbolic model.</param>
    /// <param name="X">The input feature matrix.</param>
    /// <param name="y">The target vector.</param>
    /// <returns>A string that uniquely identifies the current gradient calculation scenario.</returns>
    protected override string GenerateGradientCacheKey(IFullModel<T, TInput, TOutput> model, TInput X, TOutput y)
    {
        var baseKey = base.GenerateGradientCacheKey(model, X, y);
        return $"{baseKey}_Nadam_{_options.Beta1}_{_options.Beta2}_{_t}";
    }
}
