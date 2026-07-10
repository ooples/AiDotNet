using AiDotNet.Helpers;
using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Enums;
using AiDotNet.LossFunctions;
using AiDotNet.Optimizers;
using AiDotNet.Models.Options;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.TimeSeries;

/// <summary>
/// Implements N-HiTS (Neural Hierarchical Interpolation for Time Series) for efficient long-horizon forecasting.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// N-HiTS is an evolution of N-BEATS that addresses limitations in long-horizon forecasting through:
/// </para>
/// <list type="bullet">
/// <item>Multi-rate data sampling via hierarchical interpolation</item>
/// <item>Stack-specific input pooling to capture patterns at different frequencies</item>
/// <item>More efficient parameterization compared to N-BEATS</item>
/// <item>Interpolation-based basis functions for smoother predictions</item>
/// </list>
/// <para>
/// Original paper: Challu et al., "N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting" (AAAI 2023).
/// </para>
/// <para>
/// <b>Production-Ready Features:</b>
/// <list type="bullet">
/// <item>Uses Tensor&lt;T&gt; for GPU-accelerated operations via IEngine</item>
/// <item>Proper backpropagation via automatic differentiation</item>
/// <item>Vectorized operations - no scalar loops in hot paths</item>
/// <item>All parameters are trained (not subsets)</item>
/// </list>
/// </para>
/// <para><b>For Beginners:</b> N-HiTS improves upon N-BEATS by using a "zoom lens" approach to time series.
/// It looks at your data at three different zoom levels:
/// - Zoomed out (low resolution): Captures long-term trends like yearly seasonality
/// - Medium zoom: Captures medium-term patterns like monthly cycles
/// - Zoomed in (high resolution): Captures short-term fluctuations like daily variations
///
/// By combining insights from all three levels, it produces more accurate forecasts,
/// especially for predicting far into the future.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create N-HiTS model with default options for long-horizon forecasting
/// var options = new NHiTSOptions&lt;double&gt;();
/// var model = new NHiTSModel&lt;double&gt;(options);
///
/// // Prepare historical time series data
/// var history = new Vector&lt;double&gt;(new double[] { 112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
///     115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140 });
/// var trainingMatrix = Matrix&lt;double&gt;.Build.Dense(history.Count - 1, 1);
///
/// // Train the model on historical observations
/// model.Train(trainingMatrix, history.SubVector(1, history.Count - 1));
///
/// // Forecast future values using hierarchical interpolation
/// var forecast = model.Predict(trainingMatrix);
/// // Result is available in the returned value
/// </code>
/// </example>
[ModelDomain(ModelDomain.TimeSeries)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.TimeSeriesModel)]
[ModelTask(ModelTask.Forecasting)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting", "https://arxiv.org/abs/2201.12886", Year = 2023, Authors = "Cristian Challu, Kin G. Olivares, Boris N. Oreshkin, Federico Garza, Max Mergenthaler-Canseco, Armin Dubrawski")]
public class NHiTSModel<T> : TimeSeriesModelBase<T>
{
    private readonly NHiTSOptions<T> _options;
    private Vector<T> _trainingSeries = Vector<T>.Empty();
    private readonly List<NHiTSStackTensor<T>> _stacks;
    private readonly Random _random;

    // Normalization statistics computed during training (zero-mean / unit-variance
    // of the training series). Applied to inputs before the network and inverted on
    // the network output so gradient flow stays well-scaled — mirrors NBEATSModel.
    private T _normMean = MathHelper.GetNumericOperations<T>().Zero;
    private T _normStd = MathHelper.GetNumericOperations<T>().One;

    /// <summary>
    /// True when the most recent <c>TrainCore</c> completed via the GPU-resident
    /// fused compiled plan (weights / activations / Adam moments resident on the
    /// device across the whole loop). False when the eager tape path ran instead —
    /// either because <c>CanTrainOnGpu</c> was false, the resident attempt didn't
    /// improve the validation baseline, or the config's pool sizes don't divide the
    /// lookback cleanly (see <see cref="TryTrainGpuResident"/>).
    /// </summary>
    /// <remarks>
    /// Internal diagnostic: the public surface stays limited to the facade
    /// (<c>AiModelBuilder</c>/<c>AiModelResult</c>). Visible to the test and
    /// serving assemblies via <c>InternalsVisibleTo</c>.
    /// </remarks>
    internal bool LastRunUsedGpuResidentPath { get; private set; }

    /// <summary>
    /// Initializes a new instance of the NHiTSModel class.
    /// </summary>
    /// <param name="options">Configuration options for N-HiTS.</param>
    public NHiTSModel(NHiTSOptions<T>? options = null)
        : base(options ?? new NHiTSOptions<T>())
    {
        _options = options ?? new NHiTSOptions<T>();
        Options = _options;
        _stacks = new List<NHiTSStackTensor<T>>();
        _random = RandomHelper.CreateSeededRandom(42);

        ValidateNHiTSOptions();
        InitializeStacks();
    }

    /// <summary>
    /// Validates N-HiTS specific options.
    /// </summary>
    private void ValidateNHiTSOptions()
    {
        if (_options.NumStacks <= 0)
            throw new ArgumentException("Number of stacks must be positive.");

        if (_options.PoolingKernelSizes is null || _options.PoolingKernelSizes.Length != _options.NumStacks)
            throw new ArgumentException($"Pooling kernel sizes length must match number of stacks ({_options.NumStacks}).");

        for (int i = 0; i < _options.PoolingKernelSizes.Length; i++)
        {
            if (_options.PoolingKernelSizes[i] <= 0)
                throw new ArgumentException(
                    $"Pooling kernel size at index {i} must be positive (was {_options.PoolingKernelSizes[i]}); " +
                    "a zero kernel divides by zero and a negative kernel produces an invalid downsampled length.");
        }

        if (_options.LookbackWindow <= 0)
            throw new ArgumentException("Lookback window must be positive.");

        if (_options.ForecastHorizon <= 0)
            throw new ArgumentException("Forecast horizon must be positive.");
    }

    /// <summary>
    /// Initializes all stacks with their respective pooling and interpolation configurations.
    /// </summary>
    private void InitializeStacks()
    {
        _stacks.Clear();

        for (int i = 0; i < _options.NumStacks; i++)
        {
            int poolingSize = _options.PoolingKernelSizes[i];
            // Ceil division so the stack's declared input length matches the number
            // of windows ApplyPoolingTensor actually produces for a LookbackWindow-long
            // series (ceil(L / k)); floor division would leave a size mismatch when L
            // is not divisible by the kernel.
            int downsampledLength = (_options.LookbackWindow + poolingSize - 1) / poolingSize;

            var stack = new NHiTSStackTensor<T>(
                downsampledLength > 0 ? downsampledLength : 1,
                _options.ForecastHorizon,
                _options.HiddenLayerSize,
                _options.NumHiddenLayers,
                _options.NumBlocksPerStack,
                poolingSize,
                seed: 42 + i * 1000
            );

            _stacks.Add(stack);
        }
    }

    /// <summary>
    /// Trains the N-HiTS model with tape-based automatic differentiation and the Adam
    /// optimizer (Challu et al. 2023 use Adam), mirroring the working NBEATSModel path.
    /// </summary>
    /// <remarks>
    /// The previous implementation built an EMPTY gradient dictionary in
    /// <c>ForwardWithGradients</c> (the block backward pass had been stubbed out), so
    /// <c>ApplyGradients</c> updated nothing and the model never learned. This rewrite
    /// re-expresses the forward pass under a <see cref="GradientTape{T}"/> so autodiff
    /// produces the gradients for every stack weight/bias and <c>AdamOptimizer.Step</c>
    /// applies them. Interpreting the label vector <paramref name="y"/> as the univariate
    /// series, each sample supervises the full H-step horizon window (paper §3), and each
    /// stack forecasts from a multi-rate pooled view of the L-step lookback.
    /// </remarks>
    protected override void TrainCore(Matrix<T> x, Vector<T> y)
    {
        // Reject series that cannot produce a single training window BEFORE the
        // mean/variance pass (which divides by y.Length) and the window builder.
        // An empty series would divide by zero; any series shorter than
        // LookbackWindow + ForecastHorizon yields no valid windows and would train
        // silently on nothing (zero parameter updates).
        int requiredLength = checked(_options.LookbackWindow + _options.ForecastHorizon);
        if (y.Length < requiredLength)
        {
            throw new ArgumentException(
                $"Training series must contain at least {requiredLength} values " +
                $"(LookbackWindow {_options.LookbackWindow} + ForecastHorizon {_options.ForecastHorizon}); " +
                $"got {y.Length}.",
                nameof(y));
        }

        // Store training series BEFORE training loop for cancellation safety
        _trainingSeries = new Vector<T>(y.Length);
        for (int i = 0; i < y.Length; i++)
            _trainingSeries[i] = y[i];
        ModelParameters = new Vector<T>(1);
        ModelParameters[0] = NumOps.FromDouble(y.Length);

        // Normalize the series to zero mean / unit variance for stable gradient flow.
        T yMean = NumOps.Zero;
        for (int i = 0; i < y.Length; i++)
            yMean = NumOps.Add(yMean, y[i]);
        yMean = NumOps.Divide(yMean, NumOps.FromDouble(y.Length));

        T yVar = NumOps.Zero;
        for (int i = 0; i < y.Length; i++)
        {
            T diff = NumOps.Subtract(y[i], yMean);
            yVar = NumOps.Add(yVar, NumOps.Multiply(diff, diff));
        }
        yVar = NumOps.Divide(yVar, NumOps.FromDouble(y.Length));
        T yStd = NumOps.Sqrt(yVar);
        if (NumOps.LessThanOrEquals(yStd, NumOps.FromDouble(1e-10)))
            yStd = NumOps.One;

        _normMean = yMean;
        _normStd = yStd;

        var yNorm = new Vector<T>(y.Length);
        for (int i = 0; i < y.Length; i++)
            yNorm[i] = NumOps.Divide(NumOps.Subtract(y[i], yMean), yStd);

        // GPU-RESIDENT fast path (float + DirectGpuTensorEngine + compilation).
        // Same seam NBEATSModel uses via TimeSeriesModelBase.TryFusedResidentStep —
        // forward + backward + Adam captured as a single on-device plan, weights /
        // activations / Adam moments resident across every step. Only in epoch-bounded
        // mode: the resident attempt is validated against the untrained baseline and
        // rejected (with a fresh block reinit) if it didn't help, so in a wall-clock-
        // bounded run a rejected attempt would burn the whole budget and leave nothing
        // for the eager fallback. Epoch budgets don't have that hazard.
        LastRunUsedGpuResidentPath = false;
        if (CanTrainOnGpu && _options.MaxTrainingTimeSeconds <= 0
            && TryTrainGpuResident(yNorm))
        {
            LastRunUsedGpuResidentPath = true;
            return;
        }

        // Adam optimizer (Challu et al. 2023).
        var adamOptions = new AdamOptimizerOptions<T, Matrix<T>, Vector<T>>
        {
            InitialLearningRate = _options.LearningRate
        };
        var optimizer = new AdamOptimizer<T, Matrix<T>, Vector<T>>(null, adamOptions);

        // Collect every trainable weight/bias tensor from all stacks (registered via
        // RegisterTrainableParameter in the stack constructor).
        var allStacks = _stacks.Cast<Interfaces.ILayer<T>>().ToList();
        var trainableParams = Training.TapeTrainingStep<T>.CollectParameters(allStacks, -1);

        var trainingLoss = new MeanSquaredErrorLoss<T>();

        int lookback = _options.LookbackWindow;
        int horizon = _options.ForecastHorizon;
        int numSamples = y.Length;

        bool timeBounded = _options.MaxTrainingTimeSeconds > 0;
        int maxEpochs = timeBounded ? int.MaxValue : _options.Epochs;

        // Best-checkpoint / early-stopping restore. Mini-batch Adam on a small
        // series is stable while descending but, once near the minimum, the noisy
        // per-batch gradient is amplified by Adam's 1/sqrt(v) term and the run can
        // walk away from the optimum in late epochs (full-batch training does not
        // show this). We therefore snapshot the parameters at the end of every
        // epoch whose mean training loss improves on the best seen, and restore the
        // best snapshot after training — so extra epochs can never make the returned
        // model worse. This is standard best-model checkpointing and needs no change
        // to the public options.
        double bestLoss = double.PositiveInfinity;
        List<Vector<T>>? bestSnapshot = null;

        // Valid window positions (idx with a full lookback AND target horizon), used
        // to score each epoch's FROZEN end-of-epoch weights for best-checkpoint
        // selection. Built once — the series doesn't change across epochs.
        var checkpointWindows = new List<int>();
        for (int idx = 0; idx < numSamples; idx++)
            if (idx >= lookback && idx + horizon <= yNorm.Length)
                checkpointWindows.Add(idx);

        for (int epoch = 0; epoch < maxEpochs; epoch++)
        {
            if (timeBounded && TrainingCancellationToken.IsCancellationRequested)
                break;
            TrainingCancellationToken.ThrowIfCancellationRequested();

            var indices = Enumerable.Range(0, numSamples).OrderBy(_ => _random.Next()).ToList();

            int epochSampleCount = 0;

            for (int batchStart = 0; batchStart < numSamples; batchStart += _options.BatchSize)
            {
                if (timeBounded && TrainingCancellationToken.IsCancellationRequested)
                    break;
                TrainingCancellationToken.ThrowIfCancellationRequested();

                int batchEnd = Math.Min(batchStart + _options.BatchSize, numSamples);
                int batchCount = batchEnd - batchStart;

                // Keep only samples with a complete lookback AND target window
                // (idx ∈ [L, N - H]); mirrors the paper's windowed sampling.
                var validIndices = new List<int>(batchCount);
                for (int bi = 0; bi < batchCount; bi++)
                {
                    int idx = indices[batchStart + bi];
                    if (idx < lookback || idx + horizon > yNorm.Length)
                        continue;
                    validIndices.Add(idx);
                }

                if (validIndices.Count == 0)
                    continue;

                int effectiveBatch = validIndices.Count;

                // Target horizon window [B, H] (normalized).
                var targetData = new T[effectiveBatch * horizon];
                for (int bi = 0; bi < effectiveBatch; bi++)
                {
                    int idx = validIndices[bi];
                    for (int h = 0; h < horizon; h++)
                        targetData[bi * horizon + h] = yNorm[idx + h];
                }
                var batchTarget = new Tensor<T>(new[] { effectiveBatch, horizon }, new Vector<T>(targetData));

                using var tape = new GradientTape<T>();

                // Each stack forecasts from its own multi-rate pooled view of the
                // lookback. Pooling has no trainable parameters, so we materialize the
                // pooled inputs eagerly as tape leaves and let ForwardTape carry the
                // gradient back into the stack's MLP weights.
                Tensor<T>? aggregatedForecast = null;
                foreach (var stack in _stacks)
                {
                    int pooledLen = stack.InputLength;
                    var pooledData = new T[effectiveBatch * pooledLen];
                    for (int bi = 0; bi < effectiveBatch; bi++)
                    {
                        int idx = validIndices[bi];
                        var window = new Tensor<T>(new[] { lookback });
                        for (int j = 0; j < lookback; j++)
                            window[j] = yNorm[idx - lookback + j];
                        var pooled = ApplyPoolingTensor(window, stack.PoolingSize);
                        for (int j = 0; j < pooledLen; j++)
                            pooledData[bi * pooledLen + j] = j < pooled.Shape[0] ? pooled[j] : NumOps.Zero;
                    }

                    var pooledInput = new Tensor<T>(new[] { effectiveBatch, pooledLen }, new Vector<T>(pooledData));
                    var stackForecast = stack.ForwardTape(pooledInput); // [B, H]
                    aggregatedForecast = aggregatedForecast is null
                        ? stackForecast
                        : Engine.TensorAdd(aggregatedForecast, stackForecast);
                }

                var batchLoss = trainingLoss.ComputeTapeLoss(aggregatedForecast!, batchTarget);

                var allGrads = tape.ComputeGradients(batchLoss, sources: null);
                var grads = new Dictionary<Tensor<T>, Tensor<T>>(
                    Helpers.TensorReferenceComparer<Tensor<T>>.Instance);
                foreach (var param in trainableParams)
                {
                    if (allGrads.TryGetValue(param, out var grad))
                        grads[param] = grad;
                }

                // Count trained samples this epoch so best-checkpoint selection only
                // fires when training actually ran. The epoch is SCORED separately on
                // the frozen end-of-epoch weights (see ValidationMse below), so the
                // per-batch pre-update loss is no longer accumulated here.
                epochSampleCount += effectiveBatch;

                var context = new TapeStepContext<T>(
                    trainableParams, grads,
                    batchLoss.Length > 0 ? batchLoss[0] : NumOps.Zero);

                optimizer.Step(context);
            }

            // Snapshot the parameters if this epoch's FROZEN end-of-epoch weights are
            // the best so far. Score with ValidationMse (a no-update forward over the
            // window set) so bestLoss measures exactly the weights bestSnapshot
            // captures. Scoring by the epoch's mean PRE-update batch loss instead
            // would measure a mix of intra-epoch weight states, not the snapshot, so
            // a late-diverging epoch (good early batches, bad end weights) could be
            // wrongly selected as best. The extra pass is forward-only (no backprop).
            if (epochSampleCount > 0)
            {
                double epochLoss = ValidationMse(checkpointWindows, yNorm, lookback, horizon);
                if (!double.IsNaN(epochLoss) && !double.IsInfinity(epochLoss) && epochLoss < bestLoss)
                {
                    bestLoss = epochLoss;
                    bestSnapshot = new List<Vector<T>>(_stacks.Count);
                    foreach (var stack in _stacks)
                        bestSnapshot.Add(stack.GetParameters());
                }
            }
        }

        // Restore the best checkpoint so late-epoch divergence cannot degrade the
        // returned model.
        if (bestSnapshot is not null)
        {
            for (int s = 0; s < _stacks.Count; s++)
                _stacks[s].SetParameters(bestSnapshot[s]);
        }
    }

    /// <summary>
    /// Batched, tape-recordable average pooling for the fused-resident forward.
    /// <c>[B, L] → [B, L/kernelSize]</c> via <c>Reshape → ReduceMean(axis=2)</c>.
    /// Requires <c>L % kernelSize == 0</c>; returns null otherwise so the caller
    /// can fall back to the eager path. Kernel=1 is identity (returned as-is).
    /// </summary>
    private Tensor<T>? PoolBatchedTape(Tensor<T> input, int kernelSize)
    {
        if (kernelSize <= 1) return input;
        int B = input.Shape[0];
        int L = input.Shape[1];
        if (L % kernelSize != 0) return null;
        int poolCount = L / kernelSize;
        var reshaped = Engine.Reshape(input, new[] { B, poolCount, kernelSize });
        return Engine.ReduceMean(reshaped, new[] { 2 }, keepDims: false);
    }

    /// <summary>
    /// Runs the full multi-rate stack over a <c>[B, L]</c> batch using on-tape
    /// pooling + per-stack forecast + sum. Returns null when any stack's pooling
    /// size doesn't divide the lookback cleanly (fused path unsupported for that
    /// config; caller falls back to eager).
    /// </summary>
    private Tensor<T>? RunForwardBatched(Tensor<T> input)
    {
        Tensor<T>? aggregated = null;
        foreach (var stack in _stacks)
        {
            var pooled = PoolBatchedTape(input, stack.PoolingSize);
            if (pooled is null) return null;
            var forecast = stack.ForwardTape(pooled);
            aggregated = aggregated is null
                ? forecast
                : Engine.TensorAdd(aggregated, forecast);
        }
        return aggregated;
    }

    /// <summary>
    /// Validation MSE across up to 256 windows for the accept/reject gate. Uses
    /// the current stack weights so it correctly reflects the pre- or post-resident
    /// state depending on when it's called.
    /// </summary>
    private double ValidationMse(List<int> valid, Vector<T> yNorm, int L, int H)
    {
        int m = Math.Min(valid.Count, 256);
        if (m == 0) return double.NaN;
        var inputData = new T[m * L];
        var targetData = new T[m * H];
        for (int bi = 0; bi < m; bi++)
        {
            int idx = valid[bi];
            for (int j = 0; j < L; j++) inputData[bi * L + j] = yNorm[idx - L + j];
            for (int h = 0; h < H; h++) targetData[bi * H + h] = yNorm[idx + h];
        }
        var input = new Tensor<T>(new[] { m, L }, new Vector<T>(inputData));
        var pred = RunForwardBatched(input);
        if (pred is null) return double.NaN;
        double sum = 0.0;
        int n = pred.Length;
        for (int i = 0; i < n; i++)
        {
            double d = NumOps.ToDouble(pred[i]) - NumOps.ToDouble(targetData[i]);
            sum += d * d;
        }
        return sum / n;
    }

    /// <summary>
    /// GPU-resident training via the fused compiled-plan capture path — mirrors
    /// NBEATSModel.TryTrainGpuResident. Returns false when the fused path can't
    /// engage, when the pool sizes don't divide the lookback cleanly, or when the
    /// resident run failed to improve on the untrained baseline (blocks are
    /// re-initialized before returning so the eager fallback starts clean).
    /// </summary>
    private bool TryTrainGpuResident(Vector<T> yNorm)
    {
        int L = _options.LookbackWindow;
        int H = _options.ForecastHorizon;
        int batchSize = _options.BatchSize;

        // Precondition: every stack's pooling divides L cleanly so PoolBatchedTape
        // works. Fall back to eager for non-power-of-two configs.
        foreach (var stack in _stacks)
        {
            if (stack.PoolingSize > 1 && L % stack.PoolingSize != 0)
                return false;
        }

        var valid = new List<int>();
        for (int idx = 0; idx < yNorm.Length; idx++)
            if (idx >= L && idx + H <= yNorm.Length)
                valid.Add(idx);
        if (valid.Count < batchSize) return false;

        var layers = _stacks.Cast<ITrainableLayer<T>>().ToList();
        var trainingLoss = new MeanSquaredErrorLoss<T>();

        Tensor<T> ForwardStack(Tensor<T> input) => RunForwardBatched(input)!;
        Tensor<T> ComputeLoss(Tensor<T> pred, Tensor<T> target) =>
            trainingLoss.ComputeTapeLoss(pred, target);

        double preMse = ValidationMse(valid, yNorm, L, H);

        float lr = (float)_options.LearningRate;
        const float beta1 = 0.9f;
        const float beta2 = 0.999f;
        const float epsilon = 1e-8f;
        const float weightDecay = 0f;

        AiDotNet.Training.CompiledTapeTrainingStep<T>.Invalidate();
        AiDotNet.Training.CompiledTapeTrainingStep<T>.ResetFusedStepCount();

        var random = new Random(42);
        int maxEpochs = _options.Epochs;
        bool fusedEngaged = false;
        bool diverged = false;
        double firstStepLoss = double.NaN;

        for (int epoch = 0; epoch < maxEpochs && !diverged; epoch++)
        {
            TrainingCancellationToken.ThrowIfCancellationRequested();
            var order = valid.OrderBy(_ => random.Next()).ToList();
            int fullBatches = order.Count / batchSize;

            for (int b = 0; b < fullBatches; b++)
            {
                TrainingCancellationToken.ThrowIfCancellationRequested();
                int baseIdx = b * batchSize;
                var inputData = new T[batchSize * L];
                var targetData = new T[batchSize * H];
                for (int bi = 0; bi < batchSize; bi++)
                {
                    int idx = order[baseIdx + bi];
                    for (int j = 0; j < L; j++) inputData[bi * L + j] = yNorm[idx - L + j];
                    for (int h = 0; h < H; h++) targetData[bi * H + h] = yNorm[idx + h];
                }
                var batchInput = new Tensor<T>(new[] { batchSize, L }, new Vector<T>(inputData));
                var batchTarget = new Tensor<T>(new[] { batchSize, H }, new Vector<T>(targetData));

                bool ran = TryFusedResidentStep(
                    layers, batchInput, batchTarget, ForwardStack, ComputeLoss,
                    lr, beta1, beta2, epsilon, weightDecay, out T stepLoss);
                if (!ran)
                {
                    if (!fusedEngaged) return false;
                    continue;
                }
                fusedEngaged = true;
                double stepLossD = NumOps.ToDouble(stepLoss);
                if (double.IsNaN(stepLossD) || double.IsInfinity(stepLossD))
                {
                    diverged = true;
                    break;
                }
                if (double.IsNaN(firstStepLoss)) firstStepLoss = stepLossD;
                else if (stepLossD > 1e3 && stepLossD > firstStepLoss * 1e3)
                {
                    diverged = true;
                    break;
                }
            }
        }

        if (fusedEngaged)
        {
            double postMse = ValidationMse(valid, yNorm, L, H);
            bool improved = !double.IsNaN(postMse) && !double.IsInfinity(postMse)
                            && postMse < preMse * 0.98;
            if (diverged || !improved)
            {
                _stacks.Clear();
                InitializeStacks();
                return false;
            }
        }
        return fusedEngaged;
    }

    public override Vector<T> Predict(Matrix<T> input)
    {
        if (TryPredictFromTimeIndexCalibration(input, _trainingSeries, out var calibratedPredictions))
        {
            return calibratedPredictions;
        }

        int n = input.Rows;
        var predictions = new Vector<T>(n);
        // Forecast every row from its own lookback window (see DeepARModel.Predict: the prior
        // i < _trainingSeries.Length shortcut returned memorized training values for OOS rows).
        for (int i = 0; i < n; i++)
        {
            predictions[i] = PredictSingle(input.GetRow(i));
        }
        return predictions;
    }

    /// <summary>
    /// Applies pooling to downsample the input tensor.
    /// </summary>
    private Tensor<T> ApplyPoolingTensor(Tensor<T> input, int kernelSize)
    {
        if (kernelSize <= 1)
            return input.Clone();

        int inputLength = input.Shape[0];
        int outputLength = (inputLength + kernelSize - 1) / kernelSize;
        var pooled = new Tensor<T>([outputLength]);

        for (int i = 0; i < outputLength; i++)
        {
            int start = i * kernelSize;
            int end = Math.Min(start + kernelSize, inputLength);

            // Average pooling
            T sum = NumOps.Zero;
            for (int j = start; j < end; j++)
            {
                sum = NumOps.Add(sum, input[j]);
            }
            pooled[i] = NumOps.Divide(sum, NumOps.FromDouble(end - start));
        }

        return pooled;
    }

    /// <summary>
    /// Applies linear interpolation to upsample the forecast tensor.
    /// </summary>
    private Tensor<T> ApplyInterpolationTensor(Tensor<T> input, int targetLength)
    {
        int inputLength = input.Shape[0];
        if (inputLength == targetLength)
            return input.Clone();

        var interpolated = new Tensor<T>([targetLength]);

        if (inputLength == 1)
        {
            // Repeat single value
            for (int i = 0; i < targetLength; i++)
            {
                interpolated[i] = input[0];
            }
            return interpolated;
        }

        // Handle single target length - return average of all input values
        if (targetLength == 1)
        {
            T sum = NumOps.Zero;
            for (int i = 0; i < inputLength; i++)
            {
                sum = NumOps.Add(sum, input[i]);
            }
            interpolated[0] = NumOps.Divide(sum, NumOps.FromDouble(inputLength));
            return interpolated;
        }

        double scale = (double)(inputLength - 1) / (targetLength - 1);

        for (int i = 0; i < targetLength; i++)
        {
            double srcIdx = i * scale;
            int idx1 = (int)Math.Floor(srcIdx);
            int idx2 = Math.Min(idx1 + 1, inputLength - 1);
            double weight = srcIdx - idx1;

            T val1 = input[idx1];
            T val2 = input[idx2];
            T interpolatedVal = NumOps.Add(
                NumOps.Multiply(val1, NumOps.FromDouble(1.0 - weight)),
                NumOps.Multiply(val2, NumOps.FromDouble(weight))
            );

            interpolated[i] = interpolatedVal;
        }

        return interpolated;
    }

    public override T PredictSingle(Vector<T> input)
    {
        // If input is shorter than lookback window, construct from training series tail
        if (input.Length < _options.LookbackWindow && _trainingSeries.Length >= _options.LookbackWindow)
        {
            var lookback = new Vector<T>(_options.LookbackWindow);
            int start = _trainingSeries.Length - _options.LookbackWindow;
            for (int j = 0; j < _options.LookbackWindow; j++)
                lookback[j] = _trainingSeries[start + j];
            input = lookback;
        }

        var forecast = ForecastHorizon(input);
        return forecast[0]; // Return first step
    }

    /// <summary>
    /// Generates forecasts for the full horizon using hierarchical processing.
    /// </summary>
    public Vector<T> ForecastHorizon(Vector<T> input)
    {
        // Normalize the input window with the training statistics so inference
        // matches the normalized space the network was trained in.
        var inputTensor = new Tensor<T>([input.Length]);
        for (int i = 0; i < input.Length; i++)
        {
            inputTensor[i] = NumOps.Divide(NumOps.Subtract(input[i], _normMean), _normStd);
        }

        var aggregatedForecast = new Tensor<T>([_options.ForecastHorizon]);

        // Process through each stack
        for (int stackIdx = 0; stackIdx < _stacks.Count; stackIdx++)
        {
            var stack = _stacks[stackIdx];
            var pooledInput = ApplyPoolingTensor(inputTensor, stack.PoolingSize);
            var stackForecast = stack.ForwardInternal(pooledInput);
            var interpolatedForecast = ApplyInterpolationTensor(stackForecast, _options.ForecastHorizon);

            for (int i = 0; i < _options.ForecastHorizon; i++)
            {
                aggregatedForecast[i] = NumOps.Add(aggregatedForecast[i], interpolatedForecast[i]);
            }
        }

        // Denormalize and convert Tensor back to Vector
        var result = new Vector<T>(_options.ForecastHorizon);
        for (int i = 0; i < _options.ForecastHorizon; i++)
        {
            result[i] = NumOps.Add(NumOps.Multiply(aggregatedForecast[i], _normStd), _normMean);
        }

        return result;
    }

    protected override void SerializeCore(BinaryWriter writer)
    {
        writer.Write(_options.NumStacks);
        writer.Write(_options.LookbackWindow);
        writer.Write(_options.ForecastHorizon);

        writer.Write(_stacks.Count);
        foreach (var stack in _stacks)
        {
            stack.Serialize(writer);
        }

        // Normalization statistics learned in TrainCore. Without these a reloaded
        // model denormalizes with the defaults (_normMean=0, _normStd=1), so its
        // forecasts differ from the original trained model. Written as doubles.
        writer.Write(NumOps.ToDouble(_normMean));
        writer.Write(NumOps.ToDouble(_normStd));
    }

    protected override void DeserializeCore(BinaryReader reader)
    {
        _options.NumStacks = reader.ReadInt32();
        _options.LookbackWindow = reader.ReadInt32();
        _options.ForecastHorizon = reader.ReadInt32();

        InitializeStacks();

        int stackCount = reader.ReadInt32();
        for (int s = 0; s < stackCount && s < _stacks.Count; s++)
        {
            _stacks[s].Deserialize(reader);
        }

        // Restore the normalization statistics written by SerializeCore so the
        // reloaded model reproduces the original's forecasts.
        _normMean = NumOps.FromDouble(reader.ReadDouble());
        _normStd = NumOps.FromDouble(reader.ReadDouble());
    }

    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            Name = "N-HiTS",
            Description = "Neural Hierarchical Interpolation for Time Series with multi-rate sampling (Production-Ready)",
            Complexity = ParameterCount,
            FeatureCount = _options.LookbackWindow,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NumStacks", _options.NumStacks },
                { "LookbackWindow", _options.LookbackWindow },
                { "ForecastHorizon", _options.ForecastHorizon },
                { "PoolingKernelSizes", _options.PoolingKernelSizes! },
                { "ProductionReady", true }
            }
        };
    }

    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateInstance()
    {
        return new NHiTSModel<T>(new NHiTSOptions<T>(_options));
    }

    public override long ParameterCount
    {
        get
        {
            int total = 0;
            foreach (var stack in _stacks)
                total += (int)stack.ParameterCount;
            return total;
        }
    }

    public override IFullModel<T, Matrix<T>, Vector<T>> Clone()
    {
        var clone = new NHiTSModel<T>(_options);
        clone._stacks.Clear();
        clone._stacks.AddRange(_stacks);
        if (_trainingSeries.Length > 0)
            clone._trainingSeries = new Vector<T>(_trainingSeries);
        if (ModelParameters is not null && ModelParameters.Length > 0)
            clone.ModelParameters = new Vector<T>(ModelParameters);
        clone._normMean = _normMean;
        clone._normStd = _normStd;
        return clone;
    }

    public override IFullModel<T, Matrix<T>, Vector<T>> DeepCopy() => Clone();
}

/// <summary>
/// Represents a single stack in the N-HiTS architecture using Tensor operations.
/// </summary>
internal class NHiTSStackTensor<T> : NeuralNetworks.Layers.LayerBase<T>
{
    private readonly int _inputLength;
    private readonly int _outputLength;
    private readonly int _hiddenSize;
    private readonly int _numLayers;
    private readonly Random _random;

    // Tensor-based weights and biases (registered as trainable parameters; updated
    // by the Adam optimizer from tape-computed gradients).
    private readonly List<Tensor<T>> _weights;
    private readonly List<Tensor<T>> _biases;

    public int PoolingSize { get; }

    /// <summary>
    /// The pooled input length this stack's MLP expects (number of pooling windows
    /// over the lookback). Used by the model to shape the pooled batch tensor.
    /// </summary>
    public int InputLength => _inputLength;

    public override long ParameterCount
    {
        get
        {
            int count = 0;
            foreach (var w in _weights)
                count += w.Length;
            foreach (var b in _biases)
                count += b.Length;
            return count;
        }
    }

    public override bool SupportsTraining => true;
    public override void ResetState() { _lastForwardInput = null; }
    public override void UpdateParameters(T learningRate) { /* tape-based optimizer updates registered params */ }

    public override Vector<T> GetParameters()
    {
        var allParams = new List<T>();
        foreach (var w in _weights)
            for (int i = 0; i < w.Length; i++) allParams.Add(w[i]);
        foreach (var b in _biases)
            for (int i = 0; i < b.Length; i++) allParams.Add(b[i]);
        return new Vector<T>(allParams.ToArray());
    }

    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length != ParameterCount)
        {
            throw new ArgumentException(
                $"Expected {ParameterCount} parameters, but got {parameters.Length}.",
                nameof(parameters));
        }

        int idx = 0;
        foreach (var w in _weights)
        {
            for (int i = 0; i < w.Length; i++)
                w[i] = parameters[idx++];
        }
        foreach (var b in _biases)
        {
            for (int i = 0; i < b.Length; i++)
                b[i] = parameters[idx++];
        }
    }

    /// <summary>
    /// Persists the constructor's parameters so DeserializationHelper can
    /// reconstruct the layer with paper-faithful dimensions instead of the
    /// 16 / 4 / 64 / 1 / 1 / 2 fallback defaults. <c>numBlocks</c> and
    /// <c>seed</c> are intentionally NOT persisted: numBlocks is a vestigial
    /// ctor parameter that doesn't influence internal state in this
    /// implementation, and seed is consumed at construction time to seed
    /// <c>_random</c> — the random state has advanced past the original
    /// seed by the time GetMetadata runs, so persisting it would mislead
    /// callers into thinking the same seed reproduces the same weights
    /// (it doesn't, post-training).
    /// </summary>
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["InputLength"] = _inputLength.ToString();
        metadata["OutputLength"] = _outputLength.ToString();
        metadata["HiddenSize"] = _hiddenSize.ToString();
        metadata["NumLayers"] = _numLayers.ToString();
        metadata["PoolingSize"] = PoolingSize.ToString();
        return metadata;
    }

    public NHiTSStackTensor(int inputLength, int outputLength, int hiddenSize, int numLayers, int numBlocks, int poolingSize, int seed = 42)
        : base(new[] { inputLength }, new[] { outputLength })
    {
        _inputLength = inputLength;
        _outputLength = outputLength;
        _hiddenSize = hiddenSize;
        _numLayers = numLayers;
        PoolingSize = poolingSize;
        _random = RandomHelper.CreateSeededRandom(seed);

        _weights = new List<Tensor<T>>();
        _biases = new List<Tensor<T>>();

        InitializeWeights();

        // Register every weight/bias so TapeTrainingStep.CollectParameters picks
        // them up and the Adam optimizer updates them from tape-computed gradients.
        foreach (var w in _weights)
            RegisterTrainableParameter(w, PersistentTensorRole.Weights);
        foreach (var b in _biases)
            RegisterTrainableParameter(b, PersistentTensorRole.Biases);
    }

    private void InitializeWeights()
    {
        // Input layer: [hiddenSize, inputLength]
        double stddev = Math.Sqrt(2.0 / (_inputLength + _hiddenSize));
        _weights.Add(CreateRandomTensor([_hiddenSize, _inputLength], stddev));
        _biases.Add(new Tensor<T>([_hiddenSize]));

        // Hidden layers: [hiddenSize, hiddenSize]
        for (int i = 1; i < _numLayers; i++)
        {
            stddev = Math.Sqrt(2.0 / (_hiddenSize + _hiddenSize));
            _weights.Add(CreateRandomTensor([_hiddenSize, _hiddenSize], stddev));
            _biases.Add(new Tensor<T>([_hiddenSize]));
        }

        // Output layer: [outputLength, hiddenSize]
        stddev = Math.Sqrt(2.0 / (_hiddenSize + _outputLength));
        _weights.Add(CreateRandomTensor([_outputLength, _hiddenSize], stddev));
        _biases.Add(new Tensor<T>([_outputLength]));
    }

    private Tensor<T> CreateRandomTensor(int[] shape, double stddev)
    {
        var tensor = new Tensor<T>(shape);
        int total = tensor.Length;
        for (int i = 0; i < total; i++)
        {
            tensor[i] = NumOps.FromDouble((_random.NextDouble() * 2 - 1) * stddev);
        }
        return tensor;
    }

    private Tensor<T>? _lastForwardInput;

    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastForwardInput = input;
        return ForwardInternal(input);
    }

    public Tensor<T> ForwardInternal(Tensor<T> input)
    {
        // Inference forward. Uses the EXACT same Engine tensor ops as ForwardTape
        // so a trained model's inference output matches what training optimized —
        // a prior hand-rolled scalar matmul here disagreed with the tape path,
        // producing garbage predictions from correctly-trained weights. Running
        // outside a GradientTape, these Engine ops execute eagerly (and stay
        // GPU-dispatchable).
        var x = input;

        // Ensure input matches expected size
        if (x.Shape[0] != _inputLength)
        {
            var resized = new Tensor<T>([_inputLength]);
            for (int i = 0; i < _inputLength; i++)
            {
                int srcIdx = (i * x.Shape[0]) / _inputLength;
                resized[i] = x[Math.Min(srcIdx, x.Shape[0] - 1)];
            }
            x = resized;
        }

        // Column vector [inputLength, 1] so weight[out, in] @ col = [out, 1].
        var col = Engine.Reshape(x, new[] { _inputLength, 1 });

        for (int layer = 0; layer < _weights.Count; layer++)
        {
            var weight = _weights[layer];
            var linear = Engine.TensorMatMul(weight, col);                 // [out, 1]
            var biasCol = Engine.Reshape(_biases[layer], new[] { weight.Shape[0], 1 });
            linear = Engine.TensorBroadcastAdd(linear, biasCol);
            col = layer < _weights.Count - 1 ? Engine.ReLU(linear) : linear;
        }

        return Engine.Reshape(col, new[] { _outputLength });
    }

    /// <summary>
    /// Tape-tracked forward pass over a batched, already-pooled input <c>[B, inputLength]</c>,
    /// returning the stack forecast <c>[B, outputLength]</c>. Uses <c>Engine.Tensor*</c>
    /// ops so <see cref="GradientTape{T}"/> can differentiate the loss with respect to every
    /// registered weight and bias. This is the training-time counterpart of the eager
    /// <see cref="ForwardInternal"/> used at inference — both read the same weight tensors, so
    /// Adam updates applied to the registered tensors are visible to inference immediately.
    /// </summary>
    public Tensor<T> ForwardTape(Tensor<T> input)
    {
        // [B, in] -> [in, B] so weight[out, in] @ x[in, B] = [out, B].
        var x = Engine.TensorPermute(input, new[] { 1, 0 });

        for (int layer = 0; layer < _weights.Count; layer++)
        {
            var weight = _weights[layer];               // [outSize, inSize]
            var linear = Engine.TensorMatMul(weight, x); // [outSize, B]
            var biasCol = Engine.Reshape(_biases[layer], new[] { weight.Shape[0], 1 });
            linear = Engine.TensorBroadcastAdd(linear, biasCol);

            // ReLU on every layer except the linear output head.
            x = layer < _weights.Count - 1 ? Engine.ReLU(linear) : linear;
        }

        // [outputLength, B] -> [B, outputLength]
        return Engine.TensorPermute(x, new[] { 1, 0 });
    }

    public override void Serialize(BinaryWriter writer)
    {
        writer.Write(_inputLength);
        writer.Write(_outputLength);
        writer.Write(_hiddenSize);
        writer.Write(_numLayers);
        writer.Write(PoolingSize);

        writer.Write(_weights.Count);
        foreach (var weight in _weights)
        {
            writer.Write(weight.Shape.Length);
            foreach (var dim in weight._shape)
                writer.Write(dim);
            for (int i = 0; i < weight.Length; i++)
                writer.Write(Convert.ToDouble(weight[i]));
        }

        writer.Write(_biases.Count);
        foreach (var bias in _biases)
        {
            writer.Write(bias.Shape.Length);
            foreach (var dim in bias._shape)
                writer.Write(dim);
            for (int i = 0; i < bias.Length; i++)
                writer.Write(Convert.ToDouble(bias[i]));
        }
    }

    public override void Deserialize(BinaryReader reader)
    {
        // Skip reading dimensions as they should match constructor
        reader.ReadInt32(); // inputLength
        reader.ReadInt32(); // outputLength
        reader.ReadInt32(); // hiddenSize
        reader.ReadInt32(); // numLayers
        reader.ReadInt32(); // poolingSize

        int weightCount = reader.ReadInt32();
        // Consume ALL serialized tensors to keep stream aligned, even if counts differ
        for (int w = 0; w < weightCount; w++)
        {
            int rank = reader.ReadInt32();
            var shape = new int[rank];
            for (int d = 0; d < rank; d++)
                shape[d] = reader.ReadInt32();

            int total = shape.Aggregate(1, (a, b) => a * b);
            for (int i = 0; i < total; i++)
            {
                double v = reader.ReadDouble();
                if (w < _weights.Count && i < _weights[w].Length)
                    _weights[w][i] = NumOps.FromDouble(v);
            }
        }

        int biasCount = reader.ReadInt32();
        // Consume ALL serialized tensors to keep stream aligned, even if counts differ
        for (int b = 0; b < biasCount; b++)
        {
            int rank = reader.ReadInt32();
            var shape = new int[rank];
            for (int d = 0; d < rank; d++)
                shape[d] = reader.ReadInt32();

            int total = shape.Aggregate(1, (a, b) => a * b);
            for (int i = 0; i < total; i++)
            {
                double v = reader.ReadDouble();
                if (b < _biases.Count && i < _biases[b].Length)
                    _biases[b][i] = NumOps.FromDouble(v);
            }
        }
    }
}
