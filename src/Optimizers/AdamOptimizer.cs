using AiDotNet.Tensors.Engines.DirectGpu;
using System.Collections.Concurrent;
using AiDotNet.Tensors.Engines.Autodiff;
using Newtonsoft.Json;
using AiDotNet.Helpers;

using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.Optimizers;

/// <summary>
/// Implements the Adam (Adaptive Moment Estimation) optimization algorithm for gradient-based optimization.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Adam is an advanced optimization algorithm that combines ideas from RMSprop and Momentum optimization methods.
/// It adapts the learning rates for each parameter individually and is well-suited for problems with noisy or sparse gradients.
/// </para>
/// <para><b>For Beginners:</b> Adam is like a smart personal trainer for your machine learning model.
/// It helps your model learn efficiently by adjusting how it learns based on past experiences.
/// </para>
/// </remarks>
[ComponentType(ComponentType.Optimizer)]
[PipelineStage(PipelineStage.Training)]
public class AdamOptimizer<T, TInput, TOutput> : GradientBasedOptimizerBase<T, TInput, TOutput>, Fused.IFusedOptimizerSpec
{
    /// <summary>
    /// The options specific to the Adam optimizer.
    /// </summary>
    private AdamOptimizerOptions<T, TInput, TOutput> _options;

    /// <summary>
    /// The first moment vector (moving average of gradients).
    /// </summary>
    private Vector<T> _m;

    /// <summary>
    /// The second moment vector (moving average of squared gradients).
    /// </summary>
    private Vector<T> _v;

    /// <summary>
    /// Running maximum of v̂ when AMSGrad is enabled, for the
    /// Vector-based UpdateParameters / UpdateSolution paths (the tape-based
    /// Step path tracks vMax per-tensor in <see cref="_tapeVMax"/>).
    /// Reddi, Kale, Kumar 2018 §4 — non-decreasing v̂_max prevents the
    /// post-convergence m / sqrt(v) drift Adam exhibits on fixed-input
    /// regression. Issue #1332 cluster 6.
    /// </summary>
    private Vector<T>? _vMaxVector;

    /// <summary>
    /// The current time step (iteration count).
    /// </summary>
    private int _t;


    /// <summary>
    /// The current value of beta1 (exponential decay rate for first moment estimates).
    /// </summary>
    private T _currentBeta1;

    /// <summary>
    /// The current value of beta2 (exponential decay rate for second moment estimates).
    /// </summary>
    private T _currentBeta2;

    /// <summary>
    /// Stores the pre-update snapshot of first moment vector for accurate reverse updates.
    /// </summary>
    private Vector<T>? _previousM;

    /// <summary>
    /// Stores the pre-update snapshot of second moment vector for accurate reverse updates.
    /// </summary>
    private Vector<T>? _previousV;

    /// <summary>
    /// Stores the pre-update timestep for accurate reverse updates.
    /// </summary>
    private int _previousT;

    /// <summary>
    /// Initializes a new instance of the AdamOptimizer class.
    /// </summary>
    /// <param name="model">The model to optimize.</param>
    /// <param name="options">The options for configuring the Adam optimizer.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This sets up the Adam optimizer with its initial configuration.
    /// You can customize various aspects of how it learns, or use default settings that work well for many problems.
    /// </para>
    /// </remarks>
    public AdamOptimizer(
        IFullModel<T, TInput, TOutput>? model,
        AdamOptimizerOptions<T, TInput, TOutput>? options = null)
        : base(model, options ?? new())
    {
        _m = Vector<T>.Empty();
        _v = Vector<T>.Empty();
        _t = 0;
        _options = options ?? new();
        _currentBeta1 = NumOps.Zero;
        _currentBeta2 = NumOps.Zero;

        InitializeAdaptiveParameters();
    }

    /// <summary>
    /// Initializes the adaptive parameters used by the Adam optimizer.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This sets up the initial learning rate and momentum factors.
    /// These values will be adjusted as the optimizer learns more about the problem.
    /// </para>
    /// </remarks>
    protected override void InitializeAdaptiveParameters()
    {
        // Note: Learning rate is handled by the base class (GradientBasedOptimizerBase)
        // which syncs CurrentLearningRate with the scheduler. We don't set it here.
        _currentBeta1 = NumOps.FromDouble(_options.Beta1);
        _currentBeta2 = NumOps.FromDouble(_options.Beta2);
    }

    /// <inheritdoc/>
    bool Fused.IFusedOptimizerSpec.TryGetFusedOptimizerConfig(out Fused.FusedOptimizerConfig config)
    {
        config = default;
        // Adaptive LR mutates the rate between steps; the fused kernel bakes a
        // constant rate, so it can't reproduce that — fall back to eager.
        if (_options.UseAdaptiveLearningRate) return false;
        if (!TryGetFusedLrSchedule(out var schedule)) return false;
        // AMSGrad opt-in selects the AMSGrad kernel variant (max-second-moment),
        // which keeps the fast path instead of falling back.
        config = new Fused.FusedOptimizerConfig(
            _options.UseAMSGrad
                ? Tensors.Engines.Compilation.OptimizerType.AMSGrad
                : Tensors.Engines.Compilation.OptimizerType.Adam,
            (float)GetCurrentLearningRate(),
            (float)_options.Beta1, (float)_options.Beta2, (float)_options.Epsilon,
            0f, schedule);
        return true;
    }

    /// <summary>
    /// Performs the optimization process using the Adam algorithm.
    /// </summary>
    /// <param name="inputData">The input data for optimization, including training data and targets.</param>
    /// <returns>The result of the optimization process, including the best solution found.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the main learning process. It repeatedly tries to improve
    /// the model's parameters, using the Adam algorithm to decide how to change them.
    /// </para>
    /// <para><b>DataLoader Integration:</b>
    /// This optimizer now uses the DataLoader batching infrastructure which supports:
    /// - Custom samplers (weighted, stratified, curriculum, importance, active learning)
    /// - Reproducible shuffling via RandomSeed
    /// - Option to drop incomplete final batches
    /// Set these options via GradientBasedOptimizerOptions.DataSampler, ShuffleData, DropLastBatch, and RandomSeed.
    /// </para>
    /// </remarks>
    public override OptimizationResult<T, TInput, TOutput> Optimize(OptimizationInputData<T, TInput, TOutput> inputData)
    {
        // Initialize with random solution
        var currentSolution = InitializeWorkingSolution(inputData.XTrain);
        var bestStepData = new OptimizationStepData<T, TInput, TOutput>();

        // Issue #1221: defer _m/_v allocation. Lazy-shape models report a
        // truncated parameter count before the first Forward; UpdateSolution
        // right-sizes against the actual gradient.
        _m = Vector<T>.Empty();
        _v = Vector<T>.Empty();
        _t = 0;
        // Also reset the AMSGrad v̂_max buffer — reusing the optimizer
        // instance for a second Optimize() would otherwise carry the
        // previous run's running maximum as a lower bound and suppress
        // early updates in the new run. (PR #1350 round-2 review.)
        _vMaxVector = null;
        // Reset the NN tape-side state. The flat-vector path got reset
        // above; the tape path uses parameter-tensor-keyed dictionaries
        // (_tapeM, _tapeV, _tapeVMax) and a separate _tapeStep counter
        // that PERSIST across Optimize calls on the same optimizer
        // instance. Without this clear, a second Optimize call on the
        // same optimizer would carry the prior run's first/second moments
        // (and AMSGrad's running maximum) plus a pre-advanced bias-
        // correction counter, biasing every per-parameter step from
        // iteration 1.
        _tapeM.Clear();
        _tapeV.Clear();
        _tapeVMax.Clear();
        _tapeStep = 0;

        // Initialize parameters
        InitializeAdaptiveParameters();

        // Switch the model into training mode so dropout/batchnorm/etc. layers
        // behave correctly during gradient computation. Without this the
        // mini-batched Optimize path runs every forward pass in inference
        // mode (no dropout, BatchNorm with running stats) while still
        // applying gradient updates — the per-sample Train path already
        // sets training mode at the top of its TrainWithTape call, so the
        // batched Optimize path was the lone exception that produced
        // mode-collapsed Transformer training under BuildAsync. Restored
        // to eval mode in a finally so callers can immediately Predict
        // after Optimize without an extra SetTrainingMode(false) call —
        // mirrors the PyTorch contract that an optimizer.step() pass
        // leaves the model in train mode and the caller flips to eval
        // before validation.
        //
        // SetTrainingMode lives on INeuralNetwork, not IFullModel — for non-NN
        // models (regression, clustering, etc.) there's no training-mode
        // distinction so the gate is skipped.
        //
        // CRITICAL: must follow the LIVE currentSolution, not the original
        // pre-loop instance. UpdateSolution returns WithParameters(...)
        // replacements; if we toggle the pre-loop instance only, the
        // batch-N+1 model is in unknown mode. Sync the flag on every new
        // currentSolution and flip back to eval on the final live
        // instance in the finally block. (PR #1364 review.)
        (currentSolution as AiDotNet.Interfaces.INeuralNetwork<T>)?.SetTrainingMode(true);

        try
        {
            var previousStepData = PrepareAndEvaluateSolution(currentSolution, inputData);

            for (int epoch = 0; epoch < _options.MaxIterations; epoch++)
            {
                // Notify sampler of new epoch (for curriculum/self-paced learning)
                NotifyEpochStart(epoch);

                // Create batcher for the current epoch using DataLoader infrastructure
                var batcher = CreateBatcher(inputData, _options.BatchSize);

                foreach (var (xBatch, yBatch, batchIndices) in batcher.GetBatches())
                {
                    _t++;
                    // Calculate gradient on the batch
                    var gradient = CalculateGradient(currentSolution, xBatch, yBatch);

                    // Update solution using Adam algorithm
                    var newSolution = UpdateSolution(currentSolution, gradient);

                    // Sync training mode onto the new live instance —
                    // WithParameters(...) returns a fresh model that
                    // doesn't inherit the prior instance's training-mode flag.
                    (newSolution as AiDotNet.Interfaces.INeuralNetwork<T>)?.SetTrainingMode(true);

                    currentSolution = newSolution;

                    // Advance the scheduler's per-batch hook so StepPerBatch /
                    // WarmupThenEpoch schedulers actually progress. The
                    // batched Optimize loop previously never called
                    // OnBatchEnd, so any caller wiring an Adam-with-scheduler
                    // optimizer to BuildAsync got a flat learning rate
                    // regardless of configuration. (PR #1364 review.)
                    OnBatchEnd();
                }

                // Evaluate after processing all batches in the epoch
                var currentStepData = EvaluateSolution(currentSolution, inputData);
                UpdateBestSolution(currentStepData, ref bestStepData);
                UpdateAdaptiveParameters(currentStepData, previousStepData);

                // Check early stopping criteria
                if (UpdateIterationHistoryAndCheckEarlyStopping(epoch, bestStepData))
                {
                    return CreateOptimizationResult(bestStepData, inputData);
                }

                // Check convergence against the PREVIOUS epoch, not against
                // bestStepData. UpdateBestSolution above copies currentStepData
                // into bestStepData on the first iteration (because bestStepData
                // starts uninitialised), so |best - current| would always be 0
                // < tolerance and the optimiser would exit after the first epoch
                // — observed as AiModelBuilder.BuildAsync producing uniform
                // (1/V) predictions because only ~3 batched Adam steps ran
                // before Optimize returned. The correct convergence signal is
                // "the fitness stopped changing from one epoch to the next",
                // i.e. |current - previous| < tolerance. Issue #1340.
                //
                // SKIP convergence check on epoch 0 (review #1364 C4nK1):
                // previousStepData at epoch 0 is the pre-training baseline
                // from PrepareAndEvaluateSolution (untrained model evaluation).
                // If the first epoch happens to produce a fitness change
                // smaller than Tolerance (e.g. a warmup scheduler that
                // starts with a very small LR, or a model that's already
                // near a local optimum at init), the optimizer would
                // false-positive-converge before training has actually
                // happened. From epoch 1 onward, previousStepData is the
                // PRIOR EPOCH's post-training fitness, so |current - previous|
                // is a meaningful per-epoch progress signal.
                if (epoch > 0 && NumOps.LessThan(
                    NumOps.Abs(NumOps.Subtract(previousStepData.FitnessScore, currentStepData.FitnessScore)),
                    NumOps.FromDouble(_options.Tolerance)))
                {
                    return CreateOptimizationResult(bestStepData, inputData);
                }

                previousStepData = currentStepData;

                // Per-epoch scheduler tick — same rationale as OnBatchEnd
                // above. Epoch-level schedulers (StepPerEpoch, etc.) need
                // this to advance.
                OnEpochEnd();
            }

            return CreateOptimizationResult(bestStepData, inputData);
        }
        finally
        {
            // Leave the LIVE model (not the original pre-loop instance) in
            // eval mode so the next Predict / evaluation call doesn't
            // accidentally engage dropout / batchnorm-train-stats.
            (currentSolution as AiDotNet.Interfaces.INeuralNetwork<T>)?.SetTrainingMode(false);
        }
    }

    /// <summary>
    /// Updates the adaptive parameters of the optimizer based on the current and previous optimization steps.
    /// </summary>
    /// <param name="currentStepData">Data from the current optimization step.</param>
    /// <param name="previousStepData">Data from the previous optimization step.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method adjusts how the optimizer learns based on its recent performance.
    /// It can change the learning rate and momentum factors to help the optimizer learn more effectively.
    /// </para>
    /// </remarks>
    protected override void UpdateAdaptiveParameters(OptimizationStepData<T, TInput, TOutput> currentStepData, OptimizationStepData<T, TInput, TOutput> previousStepData)
    {
        base.UpdateAdaptiveParameters(currentStepData, previousStepData);

        // Adam-specific adaptive parameter updates
        if (_options.UseAdaptiveLearningRate)
        {
            CurrentLearningRate = MathHelper.Max(NumOps.FromDouble(_options.MinLearningRate),
                MathHelper.Min(NumOps.FromDouble(_options.MaxLearningRate), CurrentLearningRate));
        }

        if (_options.UseAdaptiveBetas)
        {
            _currentBeta1 = MathHelper.Max(NumOps.FromDouble(_options.MinBeta1),
                MathHelper.Min(NumOps.FromDouble(_options.MaxBeta1), _currentBeta1));
            _currentBeta2 = MathHelper.Max(NumOps.FromDouble(_options.MinBeta2),
                MathHelper.Min(NumOps.FromDouble(_options.MaxBeta2), _currentBeta2));
        }
    }

    /// <summary>
    /// Updates the current solution using the Adam update rule. Kept for the
    /// non-NN code path (regression, clustering, classical models where the
    /// solution does NOT implement <see cref="AiDotNet.Interfaces.INeuralNetwork{T}"/>);
    /// the base-class <see cref="GradientBasedOptimizerBase{T,TInput,TOutput}.UpdateSolution"/>
    /// intercepts NN solutions and delegates to <see cref="Step(TapeStepContext{T})"/>
    /// via <see cref="GradientBasedOptimizerBase{T,TInput,TOutput}.SynthesizeTapeStepContext"/>,
    /// so the legacy flat-vector path here only runs for non-NN models — eliminating
    /// the historical two-Adam-implementations split (#1413). All NN training
    /// goes through Step, which has the anomaly guard + gradient clipping safeguards.
    /// </summary>
    /// <param name="currentSolution">The current solution being optimized.</param>
    /// <param name="gradient">The calculated gradient for the current solution.</param>
    /// <returns>A new solution with updated parameters.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method applies the Adam algorithm to adjust the model's parameters.
    /// It uses the current gradient and past information to decide how to change each parameter.
    /// </para>
    /// </remarks>
    // #1413 ARCHITECTURAL CONSOLIDATION: AdamOptimizer's flat-vector
    // UpdateSolution override is REMOVED. NN solutions go through the base
    // class's UpdateSolution which synthesizes a TapeStepContext from the
    // flat gradient and delegates to Step(TapeStepContext) — the SAME code
    // path the per-sample nn.Train bypass uses, with the SAME anomaly
    // guard, gradient clipping, AMSGrad, and float-loop fast path. Non-NN
    // solutions fall through to the base's UpdateParameters dispatch which
    // resolves to AdamOptimizer.UpdateParameters (still present below).
    // This is the elimination of the two-Adam-implementations split that
    // caused #1380.

    /// <summary>
    /// Updates a vector of parameters using the Adam optimization algorithm.
    /// </summary>
    /// <param name="parameters">The current parameter vector to be updated.</param>
    /// <param name="gradient">The gradient vector corresponding to the parameters.</param>
    /// <returns>The updated parameter vector.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method applies the Adam algorithm to a vector of parameters.
    /// It's like adjusting multiple knobs on a machine all at once, where each knob represents a parameter.
    /// The method decides how much to turn each knob based on past adjustments and the current gradient.
    /// </para>
    /// </remarks>
    public override Vector<T> UpdateParameters(Vector<T> parameters, Vector<T> gradient)
    {
        if (_m == null || _v == null || _m.Length != parameters.Length || _m.Length != gradient.Length)
        {
            _m = new Vector<T>(parameters.Length);
            _v = new Vector<T>(parameters.Length);
            _previousM = new Vector<T>(parameters.Length);
            _previousV = new Vector<T>(parameters.Length);
            _t = 0;
        }
        if (_options.UseAMSGrad && (_vMaxVector is null || _vMaxVector.Length != parameters.Length))
        {
            _vMaxVector = new Vector<T>(parameters.Length);
        }

        // Guard against parameter/gradient size mismatch
        if (parameters.Length != gradient.Length)
        {
            throw new ArgumentException(
                $"Parameter vector length ({parameters.Length}) must match gradient vector length ({gradient.Length}).");
        }

        // Save pre-update state for accurate reverse updates
        if (_previousM == null || _previousV == null)
        {
            _previousM = new Vector<T>(parameters.Length);
            _previousV = new Vector<T>(parameters.Length);
        }

        // Copy _m and _v to _previousM and _previousV (vectorized copy)
        _previousM = new Vector<T>(_m);
        _previousV = new Vector<T>(_v);
        _previousT = _t;

        _t++;

        // === Vectorized Adam Update using IEngine ===
        // Phase B: US-GPU-015 - GPU-accelerated gradient updates

        T beta1 = NumOps.FromDouble(_options.Beta1);
        T beta2 = NumOps.FromDouble(_options.Beta2);
        T oneMinusBeta1 = NumOps.FromDouble(1 - _options.Beta1);
        T oneMinusBeta2 = NumOps.FromDouble(1 - _options.Beta2);
        T epsilon = NumOps.FromDouble(_options.Epsilon);
        T biasCorrection1 = NumOps.FromDouble(1 - Math.Pow(_options.Beta1, _t));
        T biasCorrection2 = NumOps.FromDouble(1 - Math.Pow(_options.Beta2, _t));

        // Update biased first moment: m = beta1 * m + (1 - beta1) * gradient
        var mScaled = (Vector<T>)Engine.Multiply(_m, beta1);
        var gradScaled = (Vector<T>)Engine.Multiply(gradient, oneMinusBeta1);
        _m = (Vector<T>)Engine.Add(mScaled, gradScaled);

        // Update biased second moment: v = beta2 * v + (1 - beta2) * gradient^2
        var gradSquared = (Vector<T>)Engine.Multiply(gradient, gradient);
        var vScaled = (Vector<T>)Engine.Multiply(_v, beta2);
        var gradSquaredScaled = (Vector<T>)Engine.Multiply(gradSquared, oneMinusBeta2);
        _v = (Vector<T>)Engine.Add(vScaled, gradSquaredScaled);

        // Compute bias-corrected first moment: mHat = m / (1 - beta1^t)
        var mHat = (Vector<T>)Engine.Divide(_m, biasCorrection1);

        // Compute bias-corrected second moment: vHat = v / (1 - beta2^t)
        var vHat = (Vector<T>)Engine.Divide(_v, biasCorrection2);

        // AMSGrad: track per-coord running max of v̂. The Reddi 2018 fix
        // guarantees the denominator √v̂_max is non-decreasing, which bounds
        // Adam's post-convergence m̂ / √v̂ drift on stochastic-objective
        // models (VGAE reparameterization noise, etc. — see GraphGenerationModel
        // in #1332 cluster 6). For consistency the bias-corrected
        // v̂_t (not raw v_t) is the quantity tracked, mirroring the standard
        // formulation and AdamW's existing AMSGrad path.
        Vector<T> vHatEffective;
        if (_options.UseAMSGrad)
        {
            _vMaxVector = (Vector<T>)Engine.Max(_vMaxVector!, vHat);
            vHatEffective = _vMaxVector;
        }
        else
        {
            vHatEffective = vHat;
        }

        // Compute update: update = mHat / (sqrt(vHatEffective) + epsilon)
        var vHatSqrt = (Vector<T>)Engine.Sqrt(vHatEffective);
        // Create epsilon vector for addition
        var epsilonVec = Vector<T>.CreateDefault(vHatSqrt.Length, epsilon);
        var denominator = (Vector<T>)Engine.Add(vHatSqrt, epsilonVec);
        var update = (Vector<T>)Engine.Divide(mHat, denominator);

        // Apply update: parameters = parameters - learningRate * update
        var scaledUpdate = (Vector<T>)Engine.Multiply(update, CurrentLearningRate);
        var updatedParameters = (Vector<T>)Engine.Subtract(parameters, scaledUpdate);

        return updatedParameters;
    }


    /// <summary>
    /// Updates a matrix of parameters using the Adam optimization algorithm.
    /// </summary>
    /// <param name="parameters">The current parameter matrix to be updated.</param>
    /// <param name="gradient">The gradient matrix corresponding to the parameters.</param>
    /// <returns>The updated parameter matrix.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method is similar to UpdateVector, but it works on a 2D grid of parameters instead of a 1D list.
    /// It's like adjusting a whole panel of knobs, where each knob is positioned in a grid.
    /// </para>
    /// </remarks>
    public override Matrix<T> UpdateParameters(Matrix<T> parameters, Matrix<T> gradient)
    {
        int totalSize = parameters.Rows * parameters.Columns;

        if (_m == null || _v == null || _m.Length != totalSize)
        {
            _m = new Vector<T>(totalSize);
            _v = new Vector<T>(totalSize);
            _t = 0;
        }

        _t++;

        // === Vectorized Adam Update using IEngine ===
        // Phase B: US-GPU-015 - GPU-accelerated gradient updates
        // Flatten matrices to vectors for vectorized operations

        // Flatten matrix to vector
        var paramVec = new Vector<T>(totalSize);
        var gradVec = new Vector<T>(totalSize);
        int idx = 0;
        for (int i = 0; i < parameters.Rows; i++)
        {
            for (int j = 0; j < parameters.Columns; j++)
            {
                paramVec[idx] = parameters[i, j];
                gradVec[idx] = gradient[i, j];
                idx++;
            }
        }

        T beta1 = NumOps.FromDouble(_options.Beta1);
        T beta2 = NumOps.FromDouble(_options.Beta2);
        T oneMinusBeta1 = NumOps.FromDouble(1 - _options.Beta1);
        T oneMinusBeta2 = NumOps.FromDouble(1 - _options.Beta2);
        T epsilon = NumOps.FromDouble(_options.Epsilon);
        T biasCorrection1 = NumOps.FromDouble(1 - Math.Pow(_options.Beta1, _t));
        T biasCorrection2 = NumOps.FromDouble(1 - Math.Pow(_options.Beta2, _t));

        // Update biased first moment: m = beta1 * m + (1 - beta1) * gradient
        var mScaled = (Vector<T>)Engine.Multiply(_m, beta1);
        var gradScaled = (Vector<T>)Engine.Multiply(gradVec, oneMinusBeta1);
        _m = (Vector<T>)Engine.Add(mScaled, gradScaled);

        // Update biased second moment: v = beta2 * v + (1 - beta2) * gradient^2
        var gradSquared = (Vector<T>)Engine.Multiply(gradVec, gradVec);
        var vScaled = (Vector<T>)Engine.Multiply(_v, beta2);
        var gradSquaredScaled = (Vector<T>)Engine.Multiply(gradSquared, oneMinusBeta2);
        _v = (Vector<T>)Engine.Add(vScaled, gradSquaredScaled);

        // Compute bias-corrected moments
        var mHat = (Vector<T>)Engine.Divide(_m, biasCorrection1);
        var vHat = (Vector<T>)Engine.Divide(_v, biasCorrection2);

        // AMSGrad: same v̂_max correction the Vector / UpdateSolution
        // paths apply. Without this branch the Matrix-parameter path
        // silently ran plain Adam even with UseAMSGrad=true. PR #1350 review.
        var vHatForDenominator = vHat;
        if (_options.UseAMSGrad)
        {
            if (_vMaxVector is null || _vMaxVector.Length != vHat.Length)
                _vMaxVector = new Vector<T>(vHat.Length);
            _vMaxVector = (Vector<T>)Engine.Max(_vMaxVector, vHat);
            vHatForDenominator = _vMaxVector;
        }

        // Compute update
        var vHatSqrt = (Vector<T>)Engine.Sqrt(vHatForDenominator);
        var epsilonVec = new Vector<T>(Enumerable.Repeat(epsilon, vHatSqrt.Length));
        var denominator = (Vector<T>)Engine.Add(vHatSqrt, epsilonVec);
        var update = (Vector<T>)Engine.Divide(mHat, denominator);
        var scaledUpdate = (Vector<T>)Engine.Multiply(update, CurrentLearningRate);

        // Apply update
        var updatedVec = (Vector<T>)Engine.Subtract(paramVec, scaledUpdate);

        // Unflatten vector back to matrix
        var updatedMatrix = new Matrix<T>(parameters.Rows, parameters.Columns);
        idx = 0;
        for (int i = 0; i < parameters.Rows; i++)
        {
            for (int j = 0; j < parameters.Columns; j++)
            {
                updatedMatrix[i, j] = updatedVec[idx];
                idx++;
            }
        }

        return updatedMatrix;
    }

    /// <summary>
    /// Reverses an Adam gradient update to recover original parameters.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This override provides accurate reversal for Adam's adaptive update rule:
    /// params_old = params_new + lr * m_hat / (sqrt(v_hat) + epsilon)
    /// </para>
    // Per-parameter Adam state for tape-based training (keyed by tensor reference identity).
    // ConcurrentDictionary so the per-tensor moments survive concurrent
    // TrainWithTape steps once HOGWILD! / DDP-shard trainers land (#1369);
    // until then the TrainWithTape sentinel serializes access. Never call
    // .Count or .IsEmpty on these — bucket-Monitor lock (2026-04-22
    // DeferredArrayMaterializer lesson).
    private readonly ConcurrentDictionary<Tensor<T>, Tensor<T>> _tapeM = new(TensorReferenceComparer<Tensor<T>>.Instance);
    private readonly ConcurrentDictionary<Tensor<T>, Tensor<T>> _tapeV = new(TensorReferenceComparer<Tensor<T>>.Instance);
    /// <summary>
    /// Per-parameter running maximum of v̂_t when AMSGrad is enabled
    /// (Reddi, Kale, Kumar 2018). Used as the denominator's
    /// second-moment estimate instead of v̂_t itself, which prevents
    /// the post-convergence m̂ / √v̂ drift that surfaces in
    /// MoreData_ShouldNotDegrade across NTM / GRU / DBM / etc.
    /// (#1332 cluster 6 + cluster 1.1).
    /// </summary>
    private readonly ConcurrentDictionary<Tensor<T>, Tensor<T>> _tapeVMax = new(TensorReferenceComparer<Tensor<T>>.Instance);
    private int _tapeStep;

    /// <inheritdoc />
    public override void Step(TapeStepContext<T> context)
    {
        PrepareTapeState(context);

        // PyTorch GradScaler-style anomaly guard runs BEFORE advancing the
        // step counter or any other state. Otherwise a skipped step would
        // still bump _tapeStep, which advances bc1/bc2 on the NEXT real step
        // (Adam's bias correction is step-indexed) and silently distorts
        // the optimizer's adaptive scale. With this ordering, a skipped
        // step is a true no-op: no parameters, no moments, no step index.
        //
        // The guard is configurable via AdamOptimizerOptions.AnomalyGuardMode:
        //   Auto    (default for fp32/fp16): scan; rare with bf16/fp64.
        //   Always  : scan regardless of T.
        //   Never   : skip the scan (saves the O(total-grad-elements)
        //             per-Step cost; only safe when upstream NaN/Inf isn't
        //             expected, e.g. fully-deterministic regression tests).
        // Materialize sparse-embedding contributions into context.Gradients BEFORE
        // the anomaly + global-norm clipping pre-scans run. Both scanners walk
        // context.Gradients only — without this, a parameter whose entire
        // gradient lives in the sparse list would slip past the NaN/Inf check
        // and would not contribute to the global norm, so clipping would
        // undercount and bad sparse values could poison m/v on the scatter path.
        // The sparse fast path below skips the dense entry when it scatters (the
        // dense tensor is left in the dict but the sparse continue keeps m/v in
        // sync with it), so this materialization is a one-time cost paid only
        // when AnomalyGuard or clipping is active.
        if (ShouldRunAnomalyGuard() || GradientOptions.MaxGradientNorm > 0.0)
        {
            SparseEmbeddingOptimizerHelpers.MaterializeSparseIntoGradientsDict(context, Engine);
        }

        if (ShouldRunAnomalyGuard() && AnyGradientIsAnomalous(context))
        {
            return;
        }

        _tapeStep++;

        double b1 = _options.Beta1;
        double b2 = _options.Beta2;
        double oneMinusB1 = 1.0 - b1;
        double oneMinusB2 = 1.0 - b2;
        double eps = _options.Epsilon;
        double bc1 = 1.0 - Math.Pow(b1, _tapeStep);
        double bc2 = 1.0 - Math.Pow(b2, _tapeStep);
        double lr = NumOps.ToDouble(CurrentLearningRate);

        // Fast-path detection — for T=double / T=float we can apply the
        // entire Adam update as a single tight loop over the raw underlying
        // arrays, no per-step tensor allocations. The previous engine-op
        // chain allocated 15+ intermediate tensors per parameter group per
        // step; on Hawk-class models that meant ~165 allocations per Adam
        // step and dominated training cost (PerfView profile of Hawk train
        // step at #1224 showed AdamOptimizer.Step at 56% of train wall-time).
        // The slow generic-T path is preserved verbatim below for any T
        // that isn't double or float.
        bool isDouble = typeof(T) == typeof(double);
        bool isFloat = typeof(T) == typeof(float);

        // Pre-cast bias-correction / lr / eps / beta as T once (used by both
        // paths but only constructed here).
        T epsilon = NumOps.FromDouble(eps);
        T biasCorrection1 = NumOps.FromDouble(bc1);
        T biasCorrection2 = NumOps.FromDouble(bc2);
        T beta1 = NumOps.FromDouble(b1);
        T beta2 = NumOps.FromDouble(b2);
        T oneMinusBeta1 = NumOps.FromDouble(oneMinusB1);
        T oneMinusBeta2 = NumOps.FromDouble(oneMinusB2);

        // Global-norm gradient clipping (PyTorch's torch.nn.utils.clip_grad_norm_
        // semantics) before applying the Adam update. Compute the L2 norm
        // across every parameter's gradient, and if the global norm exceeds
        // the configured MaxGradientNorm threshold, scale every gradient by
        // (threshold / global_norm) so the global norm caps at the threshold.
        // Without this, Adam's first-step bias correction (biasC1 ≈ 0.1,
        // biasC2 ≈ 0.001) creates huge updates on randomly-initialised large
        // models — Hawk's 135M-parameter LM diverges from loss=0.43 to 6.97
        // over 10 iterations on default LR=1e-3 without clipping. With it
        // (or with the canonical PyTorch default MaxGradientNorm=1.0) the
        // first-step update is bounded and training converges.
        if (GradientOptions.EnableGradientClipping &&
            GradientOptions.GradientClippingMethod == GradientClippingMethod.ByNorm)
        {
            ApplyGlobalNormGradientClipping(context, GradientOptions.MaxGradientNorm);
        }

        // PyTorch GradScaler-style anomaly guard is already enforced at the
        // top of Step() via ShouldRunAnomalyGuard() + AnyGradientIsAnomalous().
        // That single gate respects AnomalyGuardMode and runs BEFORE
        // _tapeStep++ so a skipped step is a true no-op.

        // GPU-RESIDENT ADAM (Phase 1, env AIDOTNET_GPU_ADAM=1): when on a GPU engine + float, run the Adam
        // update on the GPU (GpuOptimizer.TryAdamStep -> backend.AdamUpdate) so gradients never download to
        // host and weights/moments update in place — eliminating the per-step grad-download + weight-reupload
        // that makes the step host-bound (and uncapturable as a CUDA graph). Moments are allocated GPU-resident
        // (RentPinnedOnGpu) so TryGetGpuBuffer resolves them. Gated OFF by default; falls back to the CPU SIMD
        // path per-parameter whenever any tensor isn't GPU-resident. AMSGrad uses the CPU path (no GPU vMax yet).
        // cudaGraph-safety: the GPU-resident step is only host-read-free when NO
        // host-side gradient scan runs first. The anomaly guard
        // (ShouldRunAnomalyGuard -> AnyGradientIsAnomalous) and global-norm gradient
        // clipping both walk grad.Data.Span on the host, so disable the GPU fast path
        // whenever either is active — otherwise the step is still host-bound and not
        // graph-capturable as advertised. (Those host scans run above this point.)
        bool hostGradientScanActive = ShouldRunAnomalyGuard()
            || (GradientOptions.EnableGradientClipping
                && GradientOptions.GradientClippingMethod == GradientClippingMethod.ByNorm);
        bool gpuAdam = isFloat && !_options.UseAMSGrad
            && !hostGradientScanActive
            && System.Environment.GetEnvironmentVariable("AIDOTNET_GPU_ADAM") == "1"
            && AiDotNet.Tensors.Engines.AiDotNetEngine.Current is AiDotNet.Tensors.Engines.DirectGpuTensorEngine;

        foreach (var param in context.Parameters)
        {
            // Cheap presence check — do NOT materialize sparse→dense yet. When only
            // sparse embedding grads exist, eagerly resolving the "effective" gradient
            // here would ToDense the whole [vocab, dim] table and then the sparse fast
            // path below would scatter + `continue`, throwing the dense tensor away and
            // defeating the entire sparse optimization. Defer materialization until
            // after the sparse path declines (review #1526).
            bool hasDenseGrad = context.Gradients.TryGetValue(param, out var denseGradLookup) && denseGradLookup is not null;
            bool hasSparseGrad = SparseEmbeddingOptimizerHelpers.HasSparseEmbeddingGrad(param);
            if (!hasDenseGrad && !hasSparseGrad)
                continue;

            // Sparse-aware fast path for embedding-table parameters: when the
            // backward came from a SparseEmbeddingGradient (Tensors PR #553),
            // only the gathered rows received non-zero gradients — typically
            // ~16 rows out of a 250 002-vocab table. Touching m/v/θ across all
            // 250 002 rows is the actual bottleneck (~192 M cells of memory
            // traffic per step on LayoutXLM-class models). Scatter the Adam
            // update onto only the indexed rows; if there are no sparse
            // contributions, this is a one-look no-op and the dense path
            // below runs unchanged. Lazy m/v init below the GPU branch still
            // handles the first-step allocation — call this AFTER the m/v
            // resolution so we know they share shape with param.
            // (Re-checked again after m/v init; see the post-init call below.)

            // Lazily initialize per-parameter moment tensors. If the parameter
            // was first seen while a lazy-init layer (e.g.
            // MultiHeadAttentionLayer with IsLazy: true initialization
            // strategy) still had its weights allocated as a placeholder
            // [0, 0] tensor, our cached m / v captured the placeholder shape.
            // Once the layer materializes real weights, the gradient arrives
            // at the real shape — m / v need to be re-allocated to match,
            // otherwise TensorAdd's result has a length larger than m and
            // TensorCopy throws "Destination array was not long enough".
            if (!_tapeM.TryGetValue(param, out var m) || !m._shape.SequenceEqual(param._shape))
            {
                m = gpuAdam ? AiDotNet.Tensors.Helpers.TensorAllocator.RentPinnedOnGpu<T>(param._shape) : new Tensor<T>(param._shape);
                if (gpuAdam) m.AsWritableSpan().Clear();   // Adam moments start at 0
                _tapeM[param] = m;
            }
            if (!_tapeV.TryGetValue(param, out var v) || !v._shape.SequenceEqual(param._shape))
            {
                v = gpuAdam ? AiDotNet.Tensors.Helpers.TensorAllocator.RentPinnedOnGpu<T>(param._shape) : new Tensor<T>(param._shape);
                if (gpuAdam) v.AsWritableSpan().Clear();
                _tapeV[param] = v;
            }
            // AMSGrad running max of v̂. Initialised to a fresh zero tensor on
            // first encounter of `param`; the max-accumulation guarantees the
            // denominator √v̂_max is non-decreasing once gradients have been
            // seen, which is the Reddi 2018 fix for Adam's m̂ / √v̂ drift on
            // sparse-gradient and post-convergence regimes.
            Tensor<T>? vMax = null;
            bool useAmsgrad = _options.UseAMSGrad;
            if (useAmsgrad)
            {
                if (!_tapeVMax.TryGetValue(param, out vMax) || !vMax._shape.SequenceEqual(param._shape))
                {
                    vMax = new Tensor<T>(param._shape);
                    _tapeVMax[param] = vMax;
                }
            }

            // Sparse-embedding scatter path FIRST — before materializing any dense
            // gradient. Tensors#553 ships a sparse representation of the
            // embedding-lookup backward alongside the dense seeding; on a hit we
            // update m/v/θ only on the accessed rows (~16 out of 250 002 for a
            // typical BERT/XLM-R batch) and skip the dense traversal that would read
            // + write all the zero rows. It reads the sparse grads directly (not the
            // dense `grad`), so running it here avoids the full-tensor ToDense for the
            // sparse-only case. Plain Adam here (weightDecay=0); AdamW passes its own
            // configured weight-decay rate. AMSGrad's running max v̂ is intentionally
            // not maintained on the sparse path — vMax is dense and the scatter only
            // touches the indexed rows, which would let the AMSGrad invariant drift.
            // Take the dense path for AMSGrad to keep its monotonic-v̂ guarantee.
            if (!useAmsgrad
                && SparseEmbeddingOptimizerHelpers.TryApplyAdamSparse(
                    param, m, v, lr, b1, b2, bc1, bc2, eps, weightDecay: 0.0))
            {
                continue;
            }

            // Sparse path declined (AMSGrad, non-rank-2 layout, or no sparse grads) —
            // now resolve the dense gradient. For a sparse-only param this is where
            // the ToDense finally happens, and only because a dense update genuinely
            // needs it. Reuse the lookup we already did above.
            if (!SparseEmbeddingOptimizerHelpers.TryGetEffectiveGradient(context, param, Engine, out var grad))
                continue;

            // Reshape gradient to match parameter shape if element counts match
            // (can happen when Reshape adds/removes batch dimensions in forward pass)
            if (!param._shape.SequenceEqual(grad._shape) && param.Length == grad.Length)
            {
                grad = Engine.Reshape(grad, param._shape);
            }

            // GPU-resident Adam update (Phase 1): when param/grad/m/v all resolve to GPU buffers, the kernel
            // updates weights + moments in place on the GPU with no host download. Returns false (-> CPU path)
            // if any tensor isn't GPU-resident this step. weightDecay=0 (plain Adam; AdamW handles its own).
            if (gpuAdam)
            {
                var pf = (Tensor<float>)(object)param;
                var gf = (Tensor<float>)(object)grad;
                var mf = (Tensor<float>)(object)m;
                var vf = (Tensor<float>)(object)v;
                if (AiDotNet.Tensors.Engines.Gpu.GpuOptimizer.TryAdamStep(
                        pf, gf, mf, vf, (float)lr, (float)b1, (float)b2, (float)eps, 0f, _tapeStep))
                    continue;   // weights/moments updated in place on GPU; skip the CPU SIMD path for this param
            }

            int n = param.Length;
            // Buffer-aliased parameter views (NeuralNetworkBase wires every
            // trainable layer's weight tensor as a slice into a shared
            // ParameterBuffer<T> at non-zero _storageOffset) cannot use
            // GetDataArray() — for non-zero-offset tensors that path falls
            // back to ToArray() and returns a COPY, so an in-place
            // mutation on the returned array silently throws away every
            // Adam update and the buffer (the actual single source of
            // truth) never sees the new weights. Symptom on BiomedCLIP /
            // DFNCLIP / any model that goes through GetOrCreateParameterBuffer:
            // train completes with non-zero loss but
            // GradientFlow_ShouldBeNonZeroAndFinite reports "no parameters
            // changed" because the post-Train chunk read returns the same
            // pre-Train values. AsWritableSpan() returns a writable Span
            // sliced at the correct offset into the live storage; mutations
            // through it land on the buffer and are visible to subsequent
            // reads through any view of the same slice.
            if (isDouble)
            {
                // Hybrid fast-path: when the parameter tensor is stored at
                // offset 0 with full storage length (the common case for
                // models that don't go through ParameterBuffer<T> aliasing
                // — every model whose layers own their weight tensors
                // outright), GetLiveBackingArrayOrNull returns the live
                // backing array and we run the original raw-array Adam
                // loop the JIT auto-vectorizes most aggressively. When
                // the tensor is a non-zero-offset view (CLIP-family vision
                // encoders go through GetOrCreateParameterBuffer which
                // hands out per-parameter views as slices into a single
                // contiguous ParameterBuffer<T> at non-zero offsets), the
                // fast array path returns null and we fall back to
                // AsWritableSpan which correctly slices into the buffer
                // at the right offset. Both paths execute the same
                // numerics — only the destination of writes differs.
                double[]? paramArr = (double[]?)(object?)param.GetLiveBackingArrayOrNull();
                double[]? gradArr = (double[]?)(object?)((Tensor<T>)grad).GetLiveBackingArrayOrNull();
                double[]? mArr = (double[]?)(object?)m.GetLiveBackingArrayOrNull();
                double[]? vArr = (double[]?)(object?)v.GetLiveBackingArrayOrNull();
                double[]? vMaxArr = useAmsgrad ? (double[]?)(object?)vMax!.GetLiveBackingArrayOrNull() : null;
                if (paramArr is not null && gradArr is not null && mArr is not null && vArr is not null
                    && (!useAmsgrad || vMaxArr is not null))
                {
                    for (int i = 0; i < n; i++)
                    {
                        double g = gradArr[i];
                        double mNew = b1 * mArr[i] + oneMinusB1 * g;
                        double vNew = b2 * vArr[i] + oneMinusB2 * g * g;
                        mArr[i] = mNew;
                        vArr[i] = vNew;
                        double mHat = mNew / bc1;
                        double vHatEff;
                        if (useAmsgrad)
                        {
                            // AMSGrad: track running max of v̂ across all steps.
                            double vHatNow = vNew / bc2;
                            double vMaxPrev = vMaxArr![i];
                            double vMaxNew = vHatNow > vMaxPrev ? vHatNow : vMaxPrev;
                            vMaxArr[i] = vMaxNew;
                            vHatEff = vMaxNew;
                        }
                        else
                        {
                            vHatEff = vNew / bc2;
                        }
                        paramArr[i] -= lr * mHat / (Math.Sqrt(vHatEff) + eps);
                    }
                }
                else
                {
                    // Buffer-aliased view path. Use ref-T arithmetic via
                    // MemoryMarshal + Unsafe.Add to avoid Span<T>'s
                    // per-iteration bounds check while still slicing into
                    // the real backing storage at the right offset (so
                    // mutations land on the buffer).
                    var paramD = (Tensor<double>)(object)param;
                    var gradD = (Tensor<double>)(object)grad;
                    var mD = (Tensor<double>)(object)m;
                    var vD = (Tensor<double>)(object)v;
                    System.Span<double> paramSpan = paramD.AsWritableSpan();
                    System.ReadOnlySpan<double> gradSpan = gradD.AsSpan();
                    System.Span<double> mSpan = mD.AsWritableSpan();
                    System.Span<double> vSpan = vD.AsWritableSpan();
                    ref double paramR = ref System.Runtime.InteropServices.MemoryMarshal.GetReference(paramSpan);
                    ref double gradR = ref System.Runtime.InteropServices.MemoryMarshal.GetReference(gradSpan);
                    ref double mR = ref System.Runtime.InteropServices.MemoryMarshal.GetReference(mSpan);
                    ref double vR = ref System.Runtime.InteropServices.MemoryMarshal.GetReference(vSpan);
                    ref double vMaxR = ref System.Runtime.CompilerServices.Unsafe.NullRef<double>();
                    if (useAmsgrad)
                    {
                        var vMaxD = (Tensor<double>)(object)vMax!;
                        System.Span<double> vMaxSpan = vMaxD.AsWritableSpan();
                        vMaxR = ref System.Runtime.InteropServices.MemoryMarshal.GetReference(vMaxSpan);
                    }
                    for (int i = 0; i < n; i++)
                    {
                        double g = System.Runtime.CompilerServices.Unsafe.Add(ref gradR, i);
                        double mPrev = System.Runtime.CompilerServices.Unsafe.Add(ref mR, i);
                        double vPrev = System.Runtime.CompilerServices.Unsafe.Add(ref vR, i);
                        double mNew = b1 * mPrev + oneMinusB1 * g;
                        double vNew = b2 * vPrev + oneMinusB2 * g * g;
                        System.Runtime.CompilerServices.Unsafe.Add(ref mR, i) = mNew;
                        System.Runtime.CompilerServices.Unsafe.Add(ref vR, i) = vNew;
                        double mHat = mNew / bc1;
                        double vHatEff;
                        if (useAmsgrad)
                        {
                            double vHatNow = vNew / bc2;
                            double vMaxPrev = System.Runtime.CompilerServices.Unsafe.Add(ref vMaxR, i);
                            double vMaxNew = vHatNow > vMaxPrev ? vHatNow : vMaxPrev;
                            System.Runtime.CompilerServices.Unsafe.Add(ref vMaxR, i) = vMaxNew;
                            vHatEff = vMaxNew;
                        }
                        else
                        {
                            vHatEff = vNew / bc2;
                        }
                        System.Runtime.CompilerServices.Unsafe.Add(ref paramR, i) -=
                            lr * mHat / (Math.Sqrt(vHatEff) + eps);
                    }
                }
            }
            else if (isFloat)
            {
                float[]? paramArr = (float[]?)(object?)param.GetLiveBackingArrayOrNull();
                float[]? gradArr = (float[]?)(object?)((Tensor<T>)grad).GetLiveBackingArrayOrNull();
                float[]? mArr = (float[]?)(object?)m.GetLiveBackingArrayOrNull();
                float[]? vArr = (float[]?)(object?)v.GetLiveBackingArrayOrNull();
                float[]? vMaxArr = useAmsgrad ? (float[]?)(object?)vMax!.GetLiveBackingArrayOrNull() : null;
                float fb1 = (float)b1, fb2 = (float)b2;
                float f1mb1 = (float)oneMinusB1, f1mb2 = (float)oneMinusB2;
                float fbc1 = (float)bc1, fbc2 = (float)bc2;
                float feps = (float)eps, flr = (float)lr;
                if (paramArr is not null && gradArr is not null && mArr is not null && vArr is not null
                    && (!useAmsgrad || vMaxArr is not null))
                {
                    for (int i = 0; i < n; i++)
                    {
                        float g = gradArr[i];
                        float mNew = fb1 * mArr[i] + f1mb1 * g;
                        float vNew = fb2 * vArr[i] + f1mb2 * g * g;
                        mArr[i] = mNew;
                        vArr[i] = vNew;
                        float mHat = mNew / fbc1;
                        float vHatEff;
                        if (useAmsgrad)
                        {
                            float vHatNow = vNew / fbc2;
                            float vMaxPrev = vMaxArr![i];
                            float vMaxNew = vHatNow > vMaxPrev ? vHatNow : vMaxPrev;
                            vMaxArr[i] = vMaxNew;
                            vHatEff = vMaxNew;
                        }
                        else
                        {
                            vHatEff = vNew / fbc2;
                        }
                        paramArr[i] -= flr * mHat / ((float)Math.Sqrt(vHatEff) + feps);
                    }
                }
                else
                {
                    var paramF = (Tensor<float>)(object)param;
                    var gradF = (Tensor<float>)(object)grad;
                    var mF = (Tensor<float>)(object)m;
                    var vF = (Tensor<float>)(object)v;
                    System.Span<float> paramSpan = paramF.AsWritableSpan();
                    System.ReadOnlySpan<float> gradSpan = gradF.AsSpan();
                    System.Span<float> mSpan = mF.AsWritableSpan();
                    System.Span<float> vSpan = vF.AsWritableSpan();
                    ref float paramR = ref System.Runtime.InteropServices.MemoryMarshal.GetReference(paramSpan);
                    ref float gradR = ref System.Runtime.InteropServices.MemoryMarshal.GetReference(gradSpan);
                    ref float mR = ref System.Runtime.InteropServices.MemoryMarshal.GetReference(mSpan);
                    ref float vR = ref System.Runtime.InteropServices.MemoryMarshal.GetReference(vSpan);
                    ref float vMaxR = ref System.Runtime.CompilerServices.Unsafe.NullRef<float>();
                    if (useAmsgrad)
                    {
                        var vMaxF = (Tensor<float>)(object)vMax!;
                        System.Span<float> vMaxSpan = vMaxF.AsWritableSpan();
                        vMaxR = ref System.Runtime.InteropServices.MemoryMarshal.GetReference(vMaxSpan);
                    }
                    for (int i = 0; i < n; i++)
                    {
                        float g = System.Runtime.CompilerServices.Unsafe.Add(ref gradR, i);
                        float mPrev = System.Runtime.CompilerServices.Unsafe.Add(ref mR, i);
                        float vPrev = System.Runtime.CompilerServices.Unsafe.Add(ref vR, i);
                        float mNew = fb1 * mPrev + f1mb1 * g;
                        float vNew = fb2 * vPrev + f1mb2 * g * g;
                        System.Runtime.CompilerServices.Unsafe.Add(ref mR, i) = mNew;
                        System.Runtime.CompilerServices.Unsafe.Add(ref vR, i) = vNew;
                        float mHat = mNew / fbc1;
                        float vHatEff;
                        if (useAmsgrad)
                        {
                            float vHatNow = vNew / fbc2;
                            float vMaxPrev = System.Runtime.CompilerServices.Unsafe.Add(ref vMaxR, i);
                            float vMaxNew = vHatNow > vMaxPrev ? vHatNow : vMaxPrev;
                            System.Runtime.CompilerServices.Unsafe.Add(ref vMaxR, i) = vMaxNew;
                            vHatEff = vMaxNew;
                        }
                        else
                        {
                            vHatEff = vNew / fbc2;
                        }
                        System.Runtime.CompilerServices.Unsafe.Add(ref paramR, i) -=
                            flr * mHat / ((float)Math.Sqrt(vHatEff) + feps);
                    }
                }
            }
            else
            {
                // Generic-T fallback (Half / Decimal / etc.) — use the
                // engine-op chain. Preserves correctness for non-fp T at
                // the cost of the per-step allocations identified above.
                var mScaled = Engine.TensorMultiplyScalar(m, beta1);
                var gradScaled = Engine.TensorMultiplyScalar(grad, oneMinusBeta1);
                var mNew = Engine.TensorAdd(mScaled, gradScaled);
                Engine.TensorCopy(mNew, m);

                var gradSquared = Engine.TensorMultiply(grad, grad);
                var vScaled = Engine.TensorMultiplyScalar(v, beta2);
                var gradSqScaled = Engine.TensorMultiplyScalar(gradSquared, oneMinusBeta2);
                var vNew = Engine.TensorAdd(vScaled, gradSqScaled);
                Engine.TensorCopy(vNew, v);

                var mHat = Engine.TensorDivideScalar(m, biasCorrection1);
                var vHat = Engine.TensorDivideScalar(v, biasCorrection2);
                Tensor<T> vHatEffective;
                if (useAmsgrad)
                {
                    // vMax := max(vMax, vHat) element-wise. Element-wise max
                    // via (a + b + |a - b|) / 2 because IEngine doesn't ship
                    // a generic element-wise Max kernel; for the rare types
                    // that hit this path (Half / Decimal / etc.) the extra
                    // ops are negligible vs the matmul cost the model is
                    // already paying.
                    var diff = Engine.TensorSubtract(vHat, vMax!);
                    var absDiff = Engine.TensorAbs(diff);
                    var sum = Engine.TensorAdd(vHat, vMax!);
                    var maxPlusSum = Engine.TensorAdd(sum, absDiff);
                    var vMaxNew = Engine.TensorMultiplyScalar(maxPlusSum, NumOps.FromDouble(0.5));
                    Engine.TensorCopy(vMaxNew, vMax!);
                    vHatEffective = vMax!;
                }
                else
                {
                    vHatEffective = vHat;
                }
                var vHatSqrt = Engine.TensorSqrt(vHatEffective);
                var denom = Engine.TensorAddScalar(vHatSqrt, epsilon);
                var update = Engine.TensorDivide(mHat, denom);
                var scaledUpdate = Engine.TensorMultiplyScalar(update, CurrentLearningRate);
                Engine.TensorSubtractInPlace(param, scaledUpdate);
            }
        }
    }

    /// <para>
    /// Uses the current moment estimates (_m, _v, _t) to reconstruct the exact
    /// update that was applied, accounting for bias correction and adaptive learning rates.
    /// </para>
    /// <para><b>For Beginners:</b> This accurately undoes an Adam update by accounting
    /// for all of Adam's special features (momentum, adaptive learning rate, bias correction).
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

        // ReverseUpdate reconstructs the forward step from m / v / bias-
        // correction state. Under UseAMSGrad the forward step divides by
        // sqrt(v̂_max) instead of sqrt(v̂), and the running v̂_max from
        // before the step isn't snapshotted on _previousM / _previousV.
        // A naive reverse pass would use the post-update _vMaxVector and
        // produce wrong parameters. Until a snapshot is added (the
        // larger fix), fail fast so callers get a clear error instead
        // of silently incorrect rollback. PR #1350 review.
        if (_options.UseAMSGrad)
        {
            throw new NotSupportedException(
                "ReverseUpdate is not supported when UseAMSGrad is enabled — the AMSGrad " +
                "v̂_max state at the time of the forward step is not snapshotted, so the " +
                "reverse pass cannot reconstruct the original parameters exactly.");
        }

        // Ensure previous moment buffers are initialized
        if (_previousM == null || _previousV == null || _previousM.Length != updatedParameters.Length || _previousT == 0)
        {
            // If moments aren't initialized, fall back to SGD-style reversal
            // This shouldn't happen in normal usage but provides a safe fallback
            return base.ReverseUpdate(updatedParameters, appliedGradients);
        }

        // === Vectorized Reverse Adam Update using IEngine (Phase B: US-GPU-015) ===
        // Recompute the moments that were used during the update
        var beta1Vec = Vector<T>.CreateDefault(_previousM.Length, NumOps.FromDouble(_options.Beta1));
        var oneMinusBeta1Vec = Vector<T>.CreateDefault(_previousM.Length, NumOps.FromDouble(1 - _options.Beta1));
        var beta2Vec = Vector<T>.CreateDefault(_previousV.Length, NumOps.FromDouble(_options.Beta2));
        var oneMinusBeta2Vec = Vector<T>.CreateDefault(_previousV.Length, NumOps.FromDouble(1 - _options.Beta2));

        var mAtUpdateTime = (Vector<T>)Engine.Add(
            (Vector<T>)Engine.Multiply(_previousM, beta1Vec),
            (Vector<T>)Engine.Multiply(appliedGradients, oneMinusBeta1Vec)
        );

        var gradSquared = (Vector<T>)Engine.Multiply(appliedGradients, appliedGradients);
        var vAtUpdateTime = (Vector<T>)Engine.Add(
            (Vector<T>)Engine.Multiply(_previousV, beta2Vec),
            (Vector<T>)Engine.Multiply(gradSquared, oneMinusBeta2Vec)
        );

        // Compute bias-corrected moments
        T biasCorrection1 = NumOps.FromDouble(1 - Math.Pow(_options.Beta1, _previousT + 1));
        T biasCorrection2 = NumOps.FromDouble(1 - Math.Pow(_options.Beta2, _previousT + 1));
        var biasCorrection1Vec = Vector<T>.CreateDefault(mAtUpdateTime.Length, biasCorrection1);
        var biasCorrection2Vec = Vector<T>.CreateDefault(vAtUpdateTime.Length, biasCorrection2);

        var mHat = (Vector<T>)Engine.Divide(mAtUpdateTime, biasCorrection1Vec);
        var vHat = (Vector<T>)Engine.Divide(vAtUpdateTime, biasCorrection2Vec);

        // Compute the update that was applied
        var vHatSqrt = (Vector<T>)Engine.Sqrt(vHat);
        var epsilonVec = Vector<T>.CreateDefault(vHatSqrt.Length, NumOps.FromDouble(_options.Epsilon));
        var denominator = (Vector<T>)Engine.Add(vHatSqrt, epsilonVec);
        var update = (Vector<T>)Engine.Divide(mHat, denominator);
        var currentLrVec = Vector<T>.CreateDefault(update.Length, CurrentLearningRate);
        var scaledUpdate = (Vector<T>)Engine.Multiply(update, currentLrVec);

        // Reverse: params_old = params_new + scaled_update
        return (Vector<T>)Engine.Add(updatedParameters, scaledUpdate);
    }

    /// <summary>
    /// Resets the optimizer's internal state.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is like resetting the optimizer's memory.
    /// It forgets all past adjustments and starts fresh, which can be useful when you want to reuse the optimizer for a new problem.
    /// </para>
    /// </remarks>
    public override void Reset()
    {
        base.Reset();
        _m = Vector<T>.Empty();
        _v = Vector<T>.Empty();
        _t = 0;
        // Also clear the AMSGrad per-coord v̂_max so reuse after Reset
        // doesn't carry a stale upper bound into the next training run.
        _vMaxVector = null;
        // Tape-step moment dictionaries + tape step counter: same
        // rationale — reusing the optimizer with WithParameters-cloned
        // models post-Reset would otherwise leak per-tensor moments
        // from the prior run keyed on stale tensor refs. PR #1350 review.
        _tapeM.Clear();
        _tapeV.Clear();
        _tapeVMax.Clear();
        _tapeStep = 0;
    }

    /// <summary>
    /// Updates the optimizer's options.
    /// </summary>
    /// <param name="options">The new options to be set.</param>
    /// <exception cref="ArgumentException">Thrown when the provided options are not of type AdamOptimizerOptions.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method allows you to change the optimizer's settings mid-way.
    /// It's like adjusting the personal trainer's approach based on new instructions.
    /// </para>
    /// </remarks>
    protected override void UpdateOptions(OptimizationAlgorithmOptions<T, TInput, TOutput> options)
    {
        if (options is AdamOptimizerOptions<T, TInput, TOutput> adamOptions)
        {
            _options = adamOptions;
        }
        else
        {
            throw new ArgumentException("Invalid options type. Expected AdamOptimizerOptions.");
        }
    }

    /// <summary>
    /// Gets the current optimizer options.
    /// </summary>
    /// <returns>The current AdamOptimizerOptions.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method lets you check what settings the optimizer is currently using.
    /// It's like asking your personal trainer about their current training plan for you.
    /// </para>
    /// </remarks>
    public override OptimizationAlgorithmOptions<T, TInput, TOutput> GetOptions()
    {
        return _options;
    }

    /// <summary>
    /// Serializes the optimizer's state into a byte array.
    /// </summary>
    /// <returns>A byte array representing the serialized state of the optimizer.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method saves the optimizer's current state into a compact form.
    /// It's like taking a snapshot of the optimizer's memory and settings, which can be used later to recreate its exact state.
    /// </para>
    /// </remarks>
    public override byte[] Serialize()
    {
        using (MemoryStream ms = new MemoryStream())
        using (BinaryWriter writer = new BinaryWriter(ms))
        {
            // Serialize base class data
            byte[] baseData = base.Serialize();
            writer.Write(baseData.Length);
            writer.Write(baseData);

            // Serialize AdamOptimizerOptions
            string optionsJson = JsonConvert.SerializeObject(_options);
            writer.Write(optionsJson);

            // Serialize Adam-specific data
            writer.Write(_t);
            writer.Write(_m.Length);
            foreach (var value in _m)
            {
                writer.Write(Convert.ToDouble(value));
            }
            writer.Write(_v.Length);
            foreach (var value in _v)
            {
                writer.Write(Convert.ToDouble(value));
            }

            // Serialize the AMSGrad running-max buffer. A length of -1
            // encodes "not yet allocated" (the AMSGrad option is off or
            // the optimizer hasn't seen its first update); any
            // non-negative length is the actual element count followed
            // by that many doubles. Without this, a checkpoint restored
            // on an AMSGrad optimizer would resume with a fresh empty
            // v̂_max and diverge from uninterrupted training.
            // (PR #1350 round-2 review.)
            writer.Write(_vMaxVector?.Length ?? -1);
            if (_vMaxVector is not null)
            {
                foreach (var value in _vMaxVector)
                {
                    writer.Write(Convert.ToDouble(value));
                }
            }

            return ms.ToArray();
        }
    }

    /// <summary>
    /// Deserializes the optimizer's state from a byte array.
    /// </summary>
    /// <param name="data">The byte array containing the serialized optimizer state.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method rebuilds the optimizer's state from a saved snapshot.
    /// It's like restoring the optimizer's memory and settings from a backup, allowing you to continue from where you left off.
    /// </para>
    /// </remarks>
    public override void Deserialize(byte[] data)
    {
        using (MemoryStream ms = new MemoryStream(data))
        using (BinaryReader reader = new BinaryReader(ms))
        {
            // Deserialize base class data
            int baseDataLength = reader.ReadInt32();
            byte[] baseData = reader.ReadBytes(baseDataLength);
            base.Deserialize(baseData);

            // Deserialize AdamOptimizerOptions
            string optionsJson = reader.ReadString();
            _options = JsonConvert.DeserializeObject<AdamOptimizerOptions<T, TInput, TOutput>>(optionsJson)
                ?? throw new InvalidOperationException("Failed to deserialize optimizer options.");

            // Deserialize Adam-specific data
            _t = reader.ReadInt32();
            int mLength = reader.ReadInt32();
            _m = new Vector<T>(mLength);
            for (int i = 0; i < mLength; i++)
            {
                _m[i] = NumOps.FromDouble(reader.ReadDouble());
            }
            int vLength = reader.ReadInt32();
            _v = new Vector<T>(vLength);
            for (int i = 0; i < vLength; i++)
            {
                _v[i] = NumOps.FromDouble(reader.ReadDouble());
            }

            // Restore the AMSGrad running-max buffer if present in the
            // checkpoint. Length -1 indicates "not yet allocated" (the
            // sentinel emitted by Serialize when UseAMSGrad is off or the
            // optimizer hadn't taken its first AMSGrad step yet); any
            // non-negative length is a real vector. Older checkpoints
            // without this trailing field will fail the ReadInt32 here —
            // matching the broader Serialize/Deserialize contract that
            // older checkpoints aren't forward-compatible across schema
            // changes. (PR #1350 round-2 review.)
            int vMaxLength = reader.ReadInt32();
            if (vMaxLength < 0)
            {
                _vMaxVector = null;
            }
            else
            {
                _vMaxVector = new Vector<T>(vMaxLength);
                for (int i = 0; i < vMaxLength; i++)
                {
                    _vMaxVector[i] = NumOps.FromDouble(reader.ReadDouble());
                }
            }

            // Initialize adaptive parameters from deserialized options
            InitializeAdaptiveParameters();
        }
    }

    /// <summary>
    /// Generates a unique key for caching gradients.
    /// </summary>
    /// <param name="model">The symbolic model.</param>
    /// <param name="X">The input matrix.</param>
    /// <param name="y">The target vector.</param>
    /// <returns>A string key for gradient caching.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method creates a unique identifier for a specific optimization scenario.
    /// It's like creating a label for a particular training session, which helps in efficiently storing and retrieving calculated gradients.
    /// </para>
    /// </remarks>
    protected override string GenerateGradientCacheKey(IFullModel<T, TInput, TOutput> model, TInput X, TOutput y)
    {
        var baseKey = base.GenerateGradientCacheKey(model, X, y);
        return $"{baseKey}_Adam_{_options.InitialLearningRate}_{_options.MaxIterations}";
    }

    #region GPU Optimizer Support

    /// <summary>
    /// GPU buffer for first moment estimates (m).
    /// </summary>
    private IGpuBuffer? _gpuM;

    /// <summary>
    /// GPU buffer for second moment estimates (v).
    /// </summary>
    private IGpuBuffer? _gpuV;

    /// <summary>
    /// Gets whether this optimizer supports GPU-accelerated parameter updates.
    /// </summary>
    public override bool SupportsGpuUpdate => true;

    /// <summary>
    /// Initializes Adam optimizer state on the GPU.
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
    /// Updates parameters on the GPU using the Adam kernel.
    /// </summary>
    /// <param name="parameters">GPU buffer containing parameters to update (modified in-place).</param>
    /// <param name="gradients">GPU buffer containing gradients.</param>
    /// <param name="parameterCount">Number of parameters.</param>
    /// <param name="backend">The GPU backend to use for execution.</param>
    public override void UpdateParametersGpu(IGpuBuffer parameters, IGpuBuffer gradients, int parameterCount, IDirectGpuBackend backend)
    {
        if (!_gpuStateInitialized || _gpuM == null || _gpuV == null)
        {
            InitializeGpuState(parameterCount, backend);
        }

        _t++;

        // Call the Adam GPU kernel
        // Note: Adam (unlike AdamW) doesn't use weight decay, so pass 0.0f
        backend.AdamUpdate(
            parameters,
            gradients,
            _gpuM!,
            _gpuV!,
            (float)_options.InitialLearningRate,
            (float)_options.Beta1,
            (float)_options.Beta2,
            (float)_options.Epsilon,
            0.0f, // Adam doesn't use weight decay (use AdamW for that)
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
    /// Applies PyTorch-style global-norm gradient clipping across every
    /// gradient in the tape step's <see cref="TapeStepContext{T}.Gradients"/>
    /// dictionary. Computes the global L2 norm
    /// <c>sqrt(Σ_p ‖grad_p‖²)</c> across all parameter gradients; if that
    /// norm exceeds <paramref name="maxNorm"/>, every gradient is scaled
    /// by <c>maxNorm / globalNorm</c> in place so the post-clip global norm
    /// is exactly <paramref name="maxNorm"/>. Mirrors
    /// <c>torch.nn.utils.clip_grad_norm_(params, max_norm)</c>.
    /// </summary>
    /// <remarks>
    /// Without global-norm clipping, Adam's first-step bias correction
    /// (<c>biasC1 ≈ 0.1</c>, <c>biasC2 ≈ 0.001</c>) creates huge updates on
    /// randomly-initialised large models — Hawk's 135M-parameter LM diverges
    /// from loss 0.43 to 6.97 over 10 iterations on default LR=1e-3
    /// (issue #1275 acceptance criterion 3). With clipping at the canonical
    /// PyTorch transformer-training default <c>MaxGradientNorm=1.0</c>, the
    /// first-step update is bounded and training converges.
    /// </remarks>
    /// <summary>
    /// Returns true iff the anomaly guard's configured mode says to run
    /// the per-step gradient scan. Default <see cref="AdamAnomalyGuardMode.Auto"/>
    /// currently behaves identically to <c>Always</c>; reserved for a future
    /// numeric-type-aware heuristic.
    /// </summary>
    private bool ShouldRunAnomalyGuard()
    {
        return _options.AnomalyGuardMode switch
        {
            AdamAnomalyGuardMode.Never => false,
            AdamAnomalyGuardMode.Always => true,
            AdamAnomalyGuardMode.Auto => true,
            // Unknown values (corrupted config, future enum additions not
            // yet handled here) fail loudly instead of silently enabling
            // the guard. Detecting misconfiguration deterministically beats
            // running with the wrong policy and reporting wrong gradients
            // downstream.
            _ => throw new ArgumentOutOfRangeException(
                nameof(AdamOptimizerOptions<T, TInput, TOutput>.AnomalyGuardMode),
                _options.AnomalyGuardMode,
                $"Unknown AdamAnomalyGuardMode value: {_options.AnomalyGuardMode}. " +
                "Expected one of: Auto, Always, Never."),
        };
    }

    /// <summary>
    /// Scans every gradient tensor for NaN/Inf entries and returns true on
    /// the first sighting. PyTorch GradScaler-style anomaly guard for the
    /// Adam <c>m</c>/<c>v</c> moment accumulators: a single NaN gradient
    /// poisons them permanently, so callers must skip the entire step
    /// (parameters, moments, AND step index) on a positive return.
    /// </summary>
    private bool AnyGradientIsAnomalous(TapeStepContext<T> context)
    {
        foreach (var kvp in context.Gradients)
        {
            var grad = kvp.Value;
            if (grad is null) continue;
            var span = grad.Data.Span;
            for (int i = 0; i < span.Length; i++)
            {
                double v = NumOps.ToDouble(span[i]);
                if (double.IsNaN(v) || double.IsInfinity(v)) return true;
            }
        }
        return false;
    }

    /// <summary>
    /// Flat-vector overload of <see cref="AnyGradientIsAnomalous(TapeStepContext{T})"/>
    /// for the Optimize / UpdateSolution path (#1380 part 2). Iterates the
    /// gradient Vector directly since UpdateSolution doesn't have a
    /// TapeStepContext to walk.
    /// </summary>
    private bool AnyGradientIsAnomalous(Vector<T> gradient)
    {
        for (int i = 0; i < gradient.Length; i++)
        {
            double v = NumOps.ToDouble(gradient[i]);
            if (double.IsNaN(v) || double.IsInfinity(v)) return true;
        }
        return false;
    }

    private static void ApplyGlobalNormGradientClipping(
        TapeStepContext<T> context,
        double maxNorm)
    {
        if (maxNorm <= 0.0) return;

        var numOps = MathHelper.GetNumericOperations<T>();

        // Pass 1: compute global L2 norm. Walk every gradient tensor and
        // accumulate the squared sum across all elements.
        double globalNormSq = 0.0;
        foreach (var kvp in context.Gradients)
        {
            var grad = kvp.Value;
            if (grad is null) continue;
            var span = grad.Data.Span;
            for (int i = 0; i < span.Length; i++)
            {
                double v = numOps.ToDouble(span[i]);
                globalNormSq += v * v;
            }
        }
        double globalNorm = Math.Sqrt(globalNormSq);

        // Below threshold: nothing to do.
        if (globalNorm <= maxNorm || globalNorm == 0.0 || double.IsNaN(globalNorm) || double.IsInfinity(globalNorm))
            return;

        // Pass 2: scale every gradient by (maxNorm / globalNorm) in place.
        double scale = maxNorm / globalNorm;
        foreach (var kvp in context.Gradients)
        {
            var grad = kvp.Value;
            if (grad is null) continue;
            var span = grad.Data.Span;
            for (int i = 0; i < span.Length; i++)
                span[i] = numOps.FromDouble(numOps.ToDouble(span[i]) * scale);
        }
    }

    #endregion
}
