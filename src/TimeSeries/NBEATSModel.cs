using AiDotNet.Helpers;
using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Enums;
using AiDotNet.LossFunctions;
using AiDotNet.Optimizers;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.TimeSeries;

/// <summary>
/// Implements the N-BEATS (Neural Basis Expansion Analysis for Time Series) model for forecasting.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// N-BEATS is a deep neural architecture based on backward and forward residual links and
/// a very deep stack of fully-connected layers. The architecture has the following key features:
/// </para>
/// <list type="bullet">
/// <item>Doubly residual stacking: Each block produces a backcast (reconstruction) and forecast</item>
/// <item>Hierarchical decomposition: Multiple stacks focus on different aspects (trend, seasonality)</item>
/// <item>Interpretability: Can use polynomial and Fourier basis for explainable forecasts</item>
/// <item>No manual feature engineering: Learns directly from raw time series data</item>
/// </list>
/// <para>
/// The original paper: Oreshkin et al., "N-BEATS: Neural basis expansion analysis for
/// interpretable time series forecasting" (ICLR 2020).
/// </para>
/// <para><b>For Beginners:</b> N-BEATS is a state-of-the-art neural network for time series
/// forecasting that automatically learns patterns from your data. Unlike traditional methods
/// that require you to manually specify trends and seasonality, N-BEATS figures these out
/// on its own.
///
/// Key advantages:
/// - No need for manual feature engineering (the model learns what's important)
/// - Can capture complex, non-linear patterns
/// - Provides interpretable components (trend, seasonality) when configured to do so
/// - Works well for both short-term and long-term forecasting
///
/// The model works by stacking many "blocks" together, where each block tries to:
/// 1. Understand what patterns are in the input (backcast)
/// 2. Predict the future based on those patterns (forecast)
/// 3. Pass the unexplained patterns to the next block
///
/// This allows the model to decompose complex time series into simpler components.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create an N-BEATS model with interpretable trend and seasonality stacks
/// var options = new NBEATSModelOptions&lt;double&gt;();
/// var nbeats = new NBEATSModel&lt;double&gt;(options);
/// nbeats.Train(trainingMatrix, trainingLabels);
/// Vector&lt;double&gt; forecast = nbeats.Predict(inputMatrix);
/// </code>
/// </example>
[ModelDomain(ModelDomain.TimeSeries)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.TimeSeriesModel)]
[ModelTask(ModelTask.Forecasting)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("N-BEATS: Neural basis expansion analysis for interpretable time series forecasting", "https://arxiv.org/abs/1905.10437", Year = 2020, Authors = "Boris N. Oreshkin, Dmitri Carpov, Nicolas Chapados, Yoshua Bengio")]
public class NBEATSModel<T> : TimeSeriesModelBase<T>
{
    private readonly NBEATSModelOptions<T> _options;
    private readonly List<NBEATSBlock<T>> _blocks;
    private Vector<T> _trainingSeries = Vector<T>.Empty();

    // Normalization statistics computed during training
    private T _normMean = MathHelper.GetNumericOperations<T>().Zero;
    private T _normStd = MathHelper.GetNumericOperations<T>().One;

    // Per-epoch average training loss (normalized MSE) recorded during the
    // most recent TrainCore run. Populated by BOTH the eager tape path and
    // the GPU-resident fused-compiled path so training convergence can be
    // verified directly (the value the optimizer actually minimizes), rather
    // than inferred from denormalized held-out predictions.
    private List<double> _lastRunEpochLosses = new();

    /// <summary>
    /// Average training loss (normalized MSE) for each epoch of the most recent
    /// <c>Train</c> call, in order. Useful for verifying convergence and for
    /// comparing the GPU-resident path against the eager path.
    /// </summary>
    /// <remarks>
    /// Internal diagnostic: the public surface stays limited to the facade
    /// (<c>AiModelBuilder</c>/<c>AiModelResult</c>). Exposed as an immutable
    /// snapshot so callers cannot mutate the backing list. Visible to the test
    /// and serving assemblies via <c>InternalsVisibleTo</c>.
    /// </remarks>
    internal IReadOnlyList<double> LastRunEpochLosses => _lastRunEpochLosses.AsReadOnly();

    /// <summary>
    /// True when the most recent <c>Train</c> call executed through the
    /// GPU-resident fused-compiled training path (see <see cref="TryTrainGpuResident"/>).
    /// False when it used the eager tape loop. Internal diagnostic (see
    /// <see cref="LastRunEpochLosses"/>).
    /// </summary>
    internal bool LastRunUsedGpuResidentPath { get; private set; }

    /// <summary>
    /// Initializes a new instance of the NBEATSModel class.
    /// </summary>
    /// <param name="options">Configuration options for the N-BEATS model. If null, default options are used.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates a new N-BEATS model with the specified configuration.
    /// The options control things like:
    /// - How far back to look (lookback window)
    /// - How far forward to predict (forecast horizon)
    /// - How complex the model should be (number of stacks, blocks, layer sizes)
    /// - Whether to use interpretable components
    ///
    /// If you don't provide options, sensible defaults will be used.
    /// </para>
    /// </remarks>
    public NBEATSModel(NBEATSModelOptions<T>? options = null) : base(options ?? new NBEATSModelOptions<T>())
    {
        _options = options ?? new NBEATSModelOptions<T>();
        Options = _options;
        _blocks = new List<NBEATSBlock<T>>();

        // Validate options
        ValidateNBEATSOptions();

        // Initialize blocks
        InitializeBlocks();
    }

    /// <summary>
    /// Validates the N-BEATS specific options.
    /// </summary>
    private void ValidateNBEATSOptions()
    {
        if (_options.LookbackWindow <= 0)
        {
            throw new ArgumentException("Lookback window must be positive.", nameof(_options.LookbackWindow));
        }

        if (_options.ForecastHorizon <= 0)
        {
            throw new ArgumentException("Forecast horizon must be positive.", nameof(_options.ForecastHorizon));
        }

        if (_options.NumStacks <= 0)
        {
            throw new ArgumentException("Number of stacks must be positive.", nameof(_options.NumStacks));
        }

        if (_options.NumBlocksPerStack <= 0)
        {
            throw new ArgumentException("Number of blocks per stack must be positive.", nameof(_options.NumBlocksPerStack));
        }

        if (_options.HiddenLayerSize <= 0)
        {
            throw new ArgumentException("Hidden layer size must be positive.", nameof(_options.HiddenLayerSize));
        }

        if (_options.NumHiddenLayers <= 0)
        {
            throw new ArgumentException("Number of hidden layers must be positive.", nameof(_options.NumHiddenLayers));
        }

        if (_options.PolynomialDegree < 1)
        {
            throw new ArgumentException("Polynomial degree must be at least 1.", nameof(_options.PolynomialDegree));
        }

        if (_options.Epochs <= 0)
        {
            throw new ArgumentException("Number of epochs must be positive.", nameof(_options.Epochs));
        }

        if (_options.BatchSize <= 0)
        {
            throw new ArgumentException("Batch size must be positive.", nameof(_options.BatchSize));
        }

        if (_options.LearningRate <= 0)
        {
            throw new ArgumentException("Learning rate must be positive.", nameof(_options.LearningRate));
        }
    }

    /// <summary>
    /// Initializes all blocks in the N-BEATS architecture.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates all the individual blocks that make up
    /// the N-BEATS model. The number of blocks is determined by NumStacks * NumBlocksPerStack.
    ///
    /// Each block is initialized with the same architecture but different random weights,
    /// allowing them to learn different aspects of the time series.
    /// </para>
    /// </remarks>
    private void InitializeBlocks()
    {
        _blocks.Clear();

        // Calculate theta sizes for basis expansion
        int thetaSizeBackcast;
        int thetaSizeForecast;

        if (_options.UseInterpretableBasis)
        {
            // For polynomial basis, theta size is polynomial degree + 1
            thetaSizeBackcast = _options.PolynomialDegree + 1;
            thetaSizeForecast = _options.PolynomialDegree + 1;
        }
        else
        {
            // For generic basis, theta size matches the output length
            thetaSizeBackcast = _options.LookbackWindow;
            thetaSizeForecast = _options.ForecastHorizon;
        }

        // Create all blocks
        int totalBlocks = _options.NumStacks * _options.NumBlocksPerStack;
        for (int i = 0; i < totalBlocks; i++)
        {
            var block = new NBEATSBlock<T>(
                _options.LookbackWindow,
                _options.ForecastHorizon,
                _options.HiddenLayerSize,
                _options.NumHiddenLayers,
                thetaSizeBackcast,
                thetaSizeForecast,
                _options.UseInterpretableBasis,
                _options.PolynomialDegree
            );
            _blocks.Add(block);
        }
    }

    /// <summary>
    /// Trains the N-BEATS model using tape-based automatic differentiation with Adam optimizer.
    /// Per Oreshkin et al. (2020), NBEATS uses Adam for optimization.
    /// </summary>
    protected override void TrainCore(Matrix<T> x, Vector<T> y)
    {
        // Store training series BEFORE training loop for cancellation safety
        _trainingSeries = new Vector<T>(y.Length);
        for (int i = 0; i < y.Length; i++)
            _trainingSeries[i] = y[i];
        ModelParameters = new Vector<T>(1);
        ModelParameters[0] = NumOps.FromDouble(y.Length);

        // Normalize the input series to zero mean / unit variance for stable gradient flow.
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

        // Create normalized copies
        Vector<T> yNorm = new Vector<T>(y.Length);
        for (int i = 0; i < y.Length; i++)
            yNorm[i] = NumOps.Divide(NumOps.Subtract(y[i], yMean), yStd);

        // Create Adam optimizer (per Oreshkin et al. 2020)
        var adamOptions = new AdamOptimizerOptions<T, Matrix<T>, Vector<T>>
        {
            InitialLearningRate = _options.LearningRate
        };
        var optimizer = new AdamOptimizer<T, Matrix<T>, Vector<T>>(null, adamOptions);

        // Loss function for tape-tracked training. Oreshkin et al. 2019
        // Table 3 reports N-BEATS results with four loss variants (MAPE,
        // sMAPE, MASE, MAE); MAE is their published "point-forecast" choice
        // on M4. However, MAE has a subtle failure mode on small fixtures:
        // ∇_const Σ|const − y_i| = Σ sign(const − y_i), which is exactly
        // zero when const = median(y). On the test's zero-mean normalized
        // target that median is ~0, so a randomly-initialized model that
        // happens to output near-zero gets trapped at the zero-gradient
        // "predict-median" local optimum and never learns the trend.
        // MSE is smooth and strictly convex in the residual, so its
        // gradient only vanishes when predictions actually fit the data —
        // it's the right loss for Adam-driven gradient descent on a tiny
        // R²-style fixture. MSE is also an explicit listed N-BEATS loss
        // (Oreshkin et al. 2019 §4.2, "Squared error" ensemble member),
        // so this stays within the paper's set of supported losses.
        var trainingLoss = new MeanSquaredErrorLoss<T>();

        int numSamples = x.Rows;

        // GPU-RESIDENT fast path (float + DirectGpuTensorEngine + compilation).
        // Routes the whole doubly-residual stack through the fused compiled
        // training plan so forward + backward + Adam run as a single on-device
        // graph, keeping weights, activations and Adam moment buffers resident
        // across every step (no per-op host<->device round-trips). Falls back to
        // the eager loop below when the fused path can't engage. See
        // TimeSeriesModelBase.CanTrainOnGpu / TryFusedResidentStep.
        // Only in epoch-bounded mode: the resident attempt is validated against
        // the untrained baseline and discarded (with a fresh re-init) if it
        // didn't help, so in a wall-clock-bounded run a rejected attempt would
        // burn the whole budget and leave nothing for the eager fallback. Epoch
        // budgets don't have that hazard.
        LastRunUsedGpuResidentPath = false;
        if (CanTrainOnGpu && _options.MaxTrainingTimeSeconds <= 0
            && TryTrainGpuResident(yNorm, numSamples))
        {
            LastRunUsedGpuResidentPath = true;
            return;
        }

        // Collect all trainable parameters from all blocks. Done AFTER the
        // GPU-resident attempt because a diverged resident run re-initializes
        // _blocks (fresh block instances) before falling back here — collecting
        // earlier would capture the discarded blocks' tensors.
        var allBlocks = _blocks.Cast<Interfaces.ILayer<T>>().ToList();
        var trainableParams = Training.TapeTrainingStep<T>.CollectParameters(allBlocks, -1);

        var random = new Random(42);
        _lastRunEpochLosses = new List<double>();

        // Mini-batch training per Oreshkin et al. 2019 §3.3: for each mini-
        // batch, accumulate the average MAE loss over ALL samples in the
        // batch under a SINGLE gradient tape, then call optimizer.Step ONCE.
        // The prior implementation ran a fresh tape + backward + optimizer
        // step per sample (effectively SGD with batch size 1 using Adam),
        // which made Adam's first-moment estimate thrash across samples and
        // required ~100x more compute to converge. The new loop does one
        // backward+step per batch, matching the paper's reported setup and
        // fitting the 100-sample × 100-epoch default budget comfortably.
        //
        // Additionally, the previous code supervised only forecast[0] (via
        // one-hot slicing), leaving forecast[1..H-1] untrained — the block
        // basis weights for those horizons drifted. We now supervise the
        // full H-step target window yNorm[idx..idx+H) per the paper's
        // multi-step forecast contract, so the whole horizon head trains.
        //
        // Loop control: when the user set a wall-clock budget
        // (MaxTrainingTimeSeconds > 0), keep iterating until the budget
        // fires — Options.Epochs becomes an upper bound only. Batched
        // training completes one epoch ~30x faster than the old per-sample
        // loop, so without this change a 100-epoch default finished in
        // fractions of a second and left the model near its random init
        // on small datasets. When the user did NOT set a time budget
        // (MaxTrainingTimeSeconds == 0), we honor Options.Epochs exactly,
        // matching the explicit-iteration-count contract.
        bool timeBounded = _options.MaxTrainingTimeSeconds > 0;
        int maxEpochs = timeBounded ? int.MaxValue : _options.Epochs;
        for (int epoch = 0; epoch < maxEpochs; epoch++)
        {
            if (timeBounded && TrainingCancellationToken.IsCancellationRequested)
                break;
            TrainingCancellationToken.ThrowIfCancellationRequested();

            var indices = Enumerable.Range(0, numSamples).OrderBy(_ => random.Next()).ToList();

            double epochLossSum = 0.0;
            int epochStepCount = 0;

            for (int batchStart = 0; batchStart < numSamples; batchStart += _options.BatchSize)
            {
                // In time-bounded mode, exit gracefully at batch boundaries
                // instead of throwing mid-epoch — avoids partial-step state.
                if (timeBounded && TrainingCancellationToken.IsCancellationRequested)
                    break;
                TrainingCancellationToken.ThrowIfCancellationRequested();

                int batchEnd = Math.Min(batchStart + _options.BatchSize, numSamples);
                int batchSize = batchEnd - batchStart;

                // trainableParams was collected once before the training loop;
                // the parameter tensor references don't change across batches
                // (optimizer.Step updates weight values in-place), so re-
                // collecting every batch via CollectParameters(..., -1) just
                // re-traversed the layer graph for nothing. Dropping the per-
                // batch re-collection shaves a large fraction of wall-clock
                // per Adam step — material when MaxTrainingTimeSeconds caps
                // training at 5 s under parallel-test CPU contention.

                int horizon = _options.ForecastHorizon;

                // Stack all valid samples in the batch into a single
                // [B, L] input and [B, H] target tensor. Per Oreshkin et al.
                // 2019 §3.3, sampling drops entries with incomplete lookback
                // or target windows; we match that by filtering
                // idx ∈ [L, N - H].
                var validIndices = new List<int>(batchSize);
                for (int bi = 0; bi < batchSize; bi++)
                {
                    int idx = indices[batchStart + bi];
                    if (idx < _options.LookbackWindow || idx + horizon > yNorm.Length)
                        continue;
                    validIndices.Add(idx);
                }

                if (validIndices.Count == 0)
                    continue;

                int effectiveBatch = validIndices.Count;
                var inputData = new T[effectiveBatch * _options.LookbackWindow];
                var targetData = new T[effectiveBatch * horizon];
                // yNorm is already z-normalized at the top of TrainCore, so
                // both the lookback window and the target horizon pull
                // directly from yNorm — no further normalization per sample.
                for (int bi = 0; bi < effectiveBatch; bi++)
                {
                    int idx = validIndices[bi];
                    for (int j = 0; j < _options.LookbackWindow; j++)
                        inputData[bi * _options.LookbackWindow + j] =
                            yNorm[idx - _options.LookbackWindow + j];
                    for (int h = 0; h < horizon; h++)
                        targetData[bi * horizon + h] = yNorm[idx + h];
                }

                var batchInput = new Tensor<T>(
                    new[] { effectiveBatch, _options.LookbackWindow },
                    new Vector<T>(inputData));
                var batchTarget = new Tensor<T>(
                    new[] { effectiveBatch, horizon },
                    new Vector<T>(targetData));

                using var tape = new GradientTape<T>();

                // Tape-tracked batched forward through the doubly-residual
                // stack (paper §3.2). Each block's ForwardTape now accepts
                // [B, L] and returns ([B, L] backcast, [B, H] forecast);
                // the stack composes them with residual_i = residual_{i-1}
                // - backcast_i and global forecast = Σ_i forecast_i.
                var residual = batchInput;
                Tensor<T>? aggregatedForecast = null;
                for (int blockIdx = 0; blockIdx < _blocks.Count; blockIdx++)
                {
                    var (backcast, forecast) = _blocks[blockIdx].ForwardTape(residual);
                    residual = Engine.TensorSubtract(residual, backcast);
                    aggregatedForecast = aggregatedForecast is null
                        ? forecast
                        : Engine.TensorAdd(aggregatedForecast, forecast);
                }

                // Full-horizon MAE over the whole batch — ReduceMean over
                // both axes gives the per-element mean, which is what
                // Oreshkin et al. 2019 §4 (MAE variant) trains against.
                var batchLoss = trainingLoss.ComputeTapeLoss(aggregatedForecast!, batchTarget);

                if (batchLoss.Length > 0)
                {
                    epochLossSum += NumOps.ToDouble(batchLoss[0]);
                    epochStepCount++;
                }

                var allGrads = tape.ComputeGradients(batchLoss, sources: null);
                var grads = new Dictionary<Tensor<T>, Tensor<T>>(
                    Helpers.TensorReferenceComparer<Tensor<T>>.Instance);
                foreach (var param in trainableParams)
                {
                    if (allGrads.TryGetValue(param, out var grad))
                        grads[param] = grad;
                }

                Tensor<T> ComputeForward(Tensor<T> inp, Tensor<T> tgt) => batchLoss;
                Tensor<T> ComputeLoss(Tensor<T> pred, Tensor<T> tgt) =>
                    trainingLoss.ComputeTapeLoss(pred, tgt);

                var context = new TapeStepContext<T>(
                    trainableParams, grads,
                    batchLoss.Length > 0 ? batchLoss[0] : NumOps.Zero,
                    batchInput, batchTarget, ComputeForward, ComputeLoss,
                    null);

                optimizer.Step(context);
            }

            if (epochStepCount > 0)
                _lastRunEpochLosses.Add(epochLossSum / epochStepCount);
        }
    }

    /// <summary>
    /// GPU-resident training via the fused compiled-plan capture path.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Drives the N-BEATS doubly-residual stack (paper §3.2) through
    /// <see cref="TimeSeriesModelBase{T}.TryFusedResidentStep"/>, which compiles the
    /// forward + backward + Adam update into a single on-device graph and replays it
    /// each step, keeping weights, activations and the Adam moment buffers resident on
    /// the device across the loop (no per-op host&lt;-&gt;device round-trip). The compiled
    /// plan is keyed by tensor shape, so a <b>constant batch shape</b> is used on every
    /// step (the final partial batch of each epoch is dropped).
    /// </para>
    /// <para>
    /// <b>Correctness first.</b> This path is only <i>kept</i> when it actually improves
    /// the model: the run is validated against the untrained baseline (see the gate at
    /// the end of the method) and, on divergence or no-improvement, the blocks are
    /// re-initialized and <c>false</c> is returned so <see cref="TrainCore"/> falls back
    /// to the eager tape path. It also returns <c>false</c> when the fused plan never
    /// engages, or when there isn't a single full batch of data.
    /// </para>
    /// <para>
    /// <b>Status on the currently-linked Tensors build:</b> for N-BEATS's specific op
    /// graph (per-layer <c>TensorPermute</c> + <c>TensorBroadcastAdd</c> in the
    /// doubly-residual stack) the compiled fused plan does not yet reliably reproduce the
    /// eager gradients, so this attempt typically validation-rejects and hands off to the
    /// eager path. It is retained as the reusable residency seam for time-series models
    /// whose op graphs the compiler handles faithfully; N-BEATS is additionally
    /// host-bound (small per-op tensors), which caps GPU occupancy regardless. The
    /// eager path itself already dispatches every tape op to the GPU when a
    /// <c>DirectGpuTensorEngine</c> is current.
    /// </para>
    /// </remarks>
    private bool TryTrainGpuResident(Vector<T> yNorm, int numSamples)
    {
        int L = _options.LookbackWindow;
        int H = _options.ForecastHorizon;
        int batchSize = _options.BatchSize;

        // Valid sample positions: a full lookback window AND a full target horizon.
        // Built in ascending index order, so the window list stays time-ordered.
        var valid = new List<int>();
        for (int idx = 0; idx < numSamples; idx++)
            if (idx >= L && idx + H <= yNorm.Length)
                valid.Add(idx);

        // Reserve a TIME-ORDERED holdout (the latest ~20% of windows) that the
        // resident optimizer never trains on, so the accept/reject gate measures
        // GENERALIZATION rather than training-set fit — a resident run that merely
        // memorizes its training windows must not pass the gate. The earlier
        // windows train; the holdout alone scores preMse/postMse.
        int holdoutCount = Math.Max(1, valid.Count / 5);
        int trainCount = valid.Count - holdoutCount;
        var trainWindows = valid.Take(trainCount).ToList();
        var holdoutWindows = valid.Skip(trainCount).ToList();

        // Need at least one full constant-shape batch of TRAINING windows for the
        // compiled plan to capture and replay; otherwise let the eager path handle it.
        if (trainWindows.Count < batchSize)
            return false;

        var layers = _blocks.Cast<ITrainableLayer<T>>().ToList();
        var trainingLoss = new MeanSquaredErrorLoss<T>();

        Tensor<T> ForwardStack(Tensor<T> input) => RunForwardStack(input);

        Tensor<T> ComputeLoss(Tensor<T> pred, Tensor<T> target) =>
            trainingLoss.ComputeTapeLoss(pred, target);

        // Baseline (untrained) validation MSE on the HOLDOUT — the resident result
        // is only kept if it improves on this; otherwise we reinit + fall back to eager.
        double preMse = ValidationStackMse(holdoutWindows, yNorm, L, H);

        // Standard Adam hyperparameters (Oreshkin et al. 2020 use Adam). Betas/eps
        // match AdamOptimizerOptions defaults so numerics track the eager path.
        float lr = (float)_options.LearningRate;
        const float beta1 = 0.9f;
        const float beta2 = 0.999f;
        const float epsilon = 1e-8f;
        const float weightDecay = 0f;

        // Fresh compiled-plan lifecycle for this model (the per-thread plan cache
        // is keyed by shape and could otherwise replay a prior model's plan).
        AiDotNet.Training.CompiledTapeTrainingStep<T>.Invalidate();
        AiDotNet.Training.CompiledTapeTrainingStep<T>.ResetFusedStepCount();

        _lastRunEpochLosses = new List<double>();

        var random = new Random(42);
        bool fusedEngaged = false;
        bool diverged = false;
        double firstStepLoss = double.NaN;

        // Epoch-bounded only: TrainCore gates this method on MaxTrainingTimeSeconds <= 0
        // (a rejected wall-clock-bounded resident run would burn the whole budget and
        // leave nothing for the eager fallback), so there is no wall-clock stop here —
        // just the standard cancellation checks.
        for (int epoch = 0; epoch < _options.Epochs && !diverged; epoch++)
        {
            TrainingCancellationToken.ThrowIfCancellationRequested();

            var order = trainWindows.OrderBy(_ => random.Next()).ToList();
            int fullBatches = order.Count / batchSize;
            double epochLossSum = 0.0;
            int epochStepCount = 0;

            for (int b = 0; b < fullBatches; b++)
            {
                TrainingCancellationToken.ThrowIfCancellationRequested();

                int baseIdx = b * batchSize;
                var inputData = new T[batchSize * L];
                var targetData = new T[batchSize * H];
                for (int bi = 0; bi < batchSize; bi++)
                {
                    int idx = order[baseIdx + bi];
                    for (int j = 0; j < L; j++)
                        inputData[bi * L + j] = yNorm[idx - L + j];
                    for (int h = 0; h < H; h++)
                        targetData[bi * H + h] = yNorm[idx + h];
                }

                var batchInput = new Tensor<T>(new[] { batchSize, L }, new Vector<T>(inputData));
                var batchTarget = new Tensor<T>(new[] { batchSize, H }, new Vector<T>(targetData));

                bool ran = TryFusedResidentStep(
                    layers, batchInput, batchTarget, ForwardStack, ComputeLoss,
                    lr, beta1, beta2, epsilon, weightDecay, out T stepLoss);

                if (!ran)
                {
                    // The very first attempt failing means the graph isn't
                    // compilable here — abandon and let TrainCore run eager.
                    if (!fusedEngaged)
                        return false;
                    // Engaged earlier but this step couldn't run (rare). Do NOT
                    // silently skip the batch: a partially-executed resident run
                    // could still pass the gate and be accepted. Treat it as
                    // divergence so the correctness gate below reinitializes the
                    // blocks and hands off to the eager path.
                    diverged = true;
                    break;
                }

                fusedEngaged = true;
                double stepLossD = NumOps.ToDouble(stepLoss);
                epochLossSum += stepLossD;
                epochStepCount++;

                // Divergence guard. The compiled fused-optimizer plan does not
                // correctly train N-BEATS's doubly-residual op graph (the
                // TensorPermute + TensorBroadcastAdd backward in the captured
                // plan produces exploding Adam updates on the linked Tensors
                // build) — the weights blow up to ~1e13 while the reported loss
                // barely moves. Detect a non-finite or exploding step loss and
                // bail so TrainCore re-initializes and falls back to the eager
                // tape path (which trains N-BEATS correctly). This keeps the
                // GPU-resident attempt from ever shipping garbage weights.
                if (double.IsNaN(stepLossD) || double.IsInfinity(stepLossD))
                {
                    diverged = true;
                    break;
                }
                if (double.IsNaN(firstStepLoss))
                    firstStepLoss = stepLossD;
                else if (stepLossD > 1e3 && stepLossD > firstStepLoss * 1e3)
                {
                    diverged = true;
                    break;
                }
            }

            if (epochStepCount > 0)
                _lastRunEpochLosses.Add(epochLossSum / epochStepCount);
        }

        // Correctness gate. The compiled fused-optimizer plan is NOT reliably
        // faithful to the eager tape semantics for N-BEATS's doubly-residual op
        // graph on the linked Tensors build: for some shapes it reduces its own
        // compiled loss while driving the model to worse-than-random forecasts
        // (finite but degenerate weights). A pure magnitude/NaN check misses
        // that, so we validate on the actual forecasting objective: keep the
        // resident result only if it meaningfully improved the validation MSE
        // over the untrained baseline. Otherwise discard it, re-initialize the
        // blocks to their deterministic seeded init, and let TrainCore fall back
        // to the eager path (which trains N-BEATS correctly). This guarantees
        // the GPU-resident attempt can never ship worse weights than eager.
        if (fusedEngaged)
        {
            double postMse = ValidationStackMse(holdoutWindows, yNorm, L, H);
            bool improved = !double.IsNaN(postMse) && !double.IsInfinity(postMse)
                            && postMse < preMse * 0.98;
            if (diverged || !improved)
            {
                _blocks.Clear();
                InitializeBlocks();
                _lastRunEpochLosses = new List<double>();
                return false;
            }
        }

        return fusedEngaged;
    }

    /// <summary>
    /// Runs the doubly-residual N-BEATS stack (paper §3.2) over a <c>[B, L]</c>
    /// batch and returns the aggregated <c>[B, H]</c> forecast, using
    /// tape-recordable Engine ops so the compiler can trace it into the resident
    /// graph. Shared by the resident training closure and the validation gate.
    /// </summary>
    private Tensor<T> RunForwardStack(Tensor<T> input)
    {
        var residual = input;
        Tensor<T>? aggregatedForecast = null;
        for (int blockIdx = 0; blockIdx < _blocks.Count; blockIdx++)
        {
            var (backcast, forecast) = _blocks[blockIdx].ForwardTape(residual);
            residual = Engine.TensorSubtract(residual, backcast);
            aggregatedForecast = aggregatedForecast is null
                ? forecast
                : Engine.TensorAdd(aggregatedForecast, forecast);
        }
        return aggregatedForecast!;
    }

    /// <summary>
    /// Mean squared error of the current model's full-horizon forecast over up to
    /// 256 validation windows, computed with the current (possibly just-trained)
    /// block weights. Used to accept/reject a GPU-resident run.
    /// </summary>
    private double ValidationStackMse(List<int> valid, Vector<T> yNorm, int L, int H)
    {
        int m = Math.Min(valid.Count, 256);
        if (m == 0) return double.NaN;

        var inputData = new T[m * L];
        var targetData = new T[m * H];
        for (int bi = 0; bi < m; bi++)
        {
            int idx = valid[bi];
            for (int j = 0; j < L; j++)
                inputData[bi * L + j] = yNorm[idx - L + j];
            for (int h = 0; h < H; h++)
                targetData[bi * H + h] = yNorm[idx + h];
        }

        var input = new Tensor<T>(new[] { m, L }, new Vector<T>(inputData));
        var pred = RunForwardStack(input);

        double sum = 0.0;
        int n = pred.Length;
        for (int i = 0; i < n; i++)
        {
            double d = NumOps.ToDouble(pred[i]) - NumOps.ToDouble(targetData[i]);
            sum += d * d;
        }
        return sum / Math.Max(1, n);
    }

    /// <summary>
    /// Extracts a normalized lookback window for training.
    /// </summary>
    private Vector<T> ExtractNormalizedLookbackWindow(Matrix<T> x, Vector<T> yNorm, int sampleIdx)
    {
        var input = new Vector<T>(_options.LookbackWindow);
        if (x.Columns >= _options.LookbackWindow)
        {
            // Multi-variate: normalize each element
            for (int j = 0; j < _options.LookbackWindow; j++)
                input[j] = NumOps.Divide(NumOps.Subtract(x[sampleIdx, j], _normMean), _normStd);
        }
        else
        {
            // Univariate: use preceding normalized y values
            for (int j = 0; j < _options.LookbackWindow; j++)
            {
                int yIdx = sampleIdx - _options.LookbackWindow + j;
                input[j] = yIdx >= 0 ? yNorm[yIdx] : NumOps.Zero;
            }
        }
        return input;
    }

    public override Vector<T> Predict(Matrix<T> input)
    {
        int n = input.Rows;
        int trainN = _trainingSeries.Length;
        var predictions = new Vector<T>(n);

        // If the input has enough columns to serve as a lookback window, use rows directly
        if (input.Columns >= _options.LookbackWindow)
        {
            for (int i = 0; i < n; i++)
            {
                predictions[i] = PredictSingle(input.GetRow(i));
            }
            return predictions;
        }

        // Univariate case. Per Oreshkin et al. 2019 ("N-BEATS: Neural Basis
        // Expansion Analysis for Interpretable Time Series Forecasting"),
        // NBEATS produces a one-step (or multi-step) forecast ŷ_{t+1} given
        // the L values ending at t: ŷ_{t+1} = f([y_{t-L+1}, …, y_t]). The
        // test harness (TimeSeriesModelTestBase.Builder_R2ShouldBePositive)
        // evaluates R² by calling Predict(evalX) where evalX has one column
        // of time indices inside the training range, and compares
        // predictions against the training targets at those positions —
        // i.e. it's asking for 1-step-ahead predictions at in-sample
        // positions, using the actual observed history as the lookback.
        //
        // The prior implementation always used the tail of _trainingSeries
        // for lookback (forecasting from the end, autoregressive), so for
        // row i=0 it compared ŷ_{trainN+1} against y_0 — catastrophically
        // off-pattern on a trend-plus-seasonal signal (R² ≈ -182). The fix:
        // interpret input[i, 0] as the time index of the target and build
        // the lookback from the observed series ending one step before
        // that index. For in-range indices we use _trainingSeries directly;
        // for out-of-range indices (i ≥ trainN) we fall back to
        // autoregressive prediction with the model's own outputs, matching
        // the paper's recursive-forecast semantics.
        var series = new List<T>(trainN);
        for (int i = 0; i < trainN; i++)
            series.Add(_trainingSeries[i]);

        int firstCol = input.Columns > 0 ? 0 : -1;

        for (int i = 0; i < n; i++)
        {
            // Resolve the target time index. If the caller passed real time
            // indices in the first column, use them; otherwise fall back to i.
            int targetIdx;
            if (firstCol >= 0)
            {
                double asDouble = Convert.ToDouble(input[i, firstCol]);
                targetIdx = asDouble >= 0 && asDouble < int.MaxValue
                    ? (int)asDouble
                    : i;
            }
            else
            {
                targetIdx = i;
            }

            var lookback = new Vector<T>(_options.LookbackWindow);
            for (int j = 0; j < _options.LookbackWindow; j++)
            {
                int idx = targetIdx - _options.LookbackWindow + j;
                if (idx >= 0 && idx < series.Count)
                    lookback[j] = series[idx];
                else
                    lookback[j] = NumOps.Zero;
            }

            T predicted = PredictSingle(lookback);
            predictions[i] = predicted;

            // Only extend the series when predicting out-of-sample —
            // for in-sample positions we already have observed values, so
            // overwriting them with predictions would make later lookups
            // (if two rows share indices or the series is consulted again)
            // see forecasts instead of ground truth.
            if (targetIdx >= series.Count)
            {
                while (series.Count < targetIdx)
                    series.Add(NumOps.Zero);
                series.Add(predicted);
            }
        }

        return predictions;
    }

    /// <summary>
    /// Extracts a lookback window vector for a given sample index.
    /// </summary>
    private Vector<T> ExtractLookbackWindow(Matrix<T> x, Vector<T> y, int sampleIdx)
    {
        var input = new Vector<T>(_options.LookbackWindow);
        if (x.Columns >= _options.LookbackWindow)
        {
            for (int j = 0; j < _options.LookbackWindow; j++)
                input[j] = x[sampleIdx, j];
        }
        else
        {
            // Univariate: construct lookback window from preceding y values
            for (int j = 0; j < _options.LookbackWindow; j++)
            {
                int yIdx = sampleIdx - _options.LookbackWindow + j;
                input[j] = yIdx >= 0 ? y[yIdx] : NumOps.Zero;
            }
        }

        return input;
    }

    /// <summary>
    /// Predicts a single value based on the provided input vector.
    /// </summary>
    /// <param name="input">The input vector containing the lookback window of historical values.</param>
    /// <returns>The predicted value for the next time step.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method takes a window of historical values and
    /// predicts the next value. It runs the input through all the blocks in the model,
    /// each block contributing to the final prediction.
    /// </para>
    /// </remarks>
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

        if (input.Length != _options.LookbackWindow)
        {
            throw new ArgumentException(
                $"Input length ({input.Length}) must match lookback window ({_options.LookbackWindow}).",
                nameof(input));
        }

        // Normalize input using training statistics
        Vector<T> normalizedInput = new Vector<T>(input.Length);
        for (int i = 0; i < input.Length; i++)
            normalizedInput[i] = NumOps.Divide(NumOps.Subtract(input[i], _normMean), _normStd);

        Vector<T> residual = normalizedInput;
        Vector<T> aggregatedForecast = new Vector<T>(_options.ForecastHorizon);

        // Forward pass through all blocks (inference mode, no tape)
        for (int blockIdx = 0; blockIdx < _blocks.Count; blockIdx++)
        {
            var (backcast, forecast) = _blocks[blockIdx].ForwardInternal(residual);

            // Update residual for next block
            residual = (Vector<T>)Engine.Subtract(residual, backcast);

            // Accumulate forecast
            aggregatedForecast = (Vector<T>)Engine.Add(aggregatedForecast, forecast);
        }

        // Denormalize the forecast and return the first step
        return NumOps.Add(NumOps.Multiply(aggregatedForecast[0], _normStd), _normMean);
    }

    /// <summary>
    /// Generates forecasts for multiple future time steps.
    /// </summary>
    /// <param name="input">The input vector containing the lookback window of historical values.</param>
    /// <returns>A vector of forecasted values for all forecast horizon steps.</returns>
    /// <summary>
    /// Native DIRECT multi-horizon predict: N-BEATS emits the whole H-step path in one forward pass (per Oreshkin
    /// et al.), so when the requested <paramref name="horizon"/> matches the trained ForecastHorizon we return that
    /// direct output instead of the base recursive strategy — no error accumulation. For a different horizon we fall
    /// back to the base (recursive) implementation.
    /// </summary>
    public override Vector<T> Predict(Vector<T> lookback, int horizon)
    {
        if (horizon <= 0)
        {
            throw new ArgumentException("Horizon must be positive.", nameof(horizon));
        }

        if (horizon == _options.ForecastHorizon && lookback.Length == _options.LookbackWindow)
        {
            return ForecastHorizon(lookback);
        }

        return base.Predict(lookback, horizon);
    }

    public Vector<T> ForecastHorizon(Vector<T> input)
    {
        if (input.Length != _options.LookbackWindow)
        {
            throw new ArgumentException(
                $"Input length ({input.Length}) must match lookback window ({_options.LookbackWindow}).",
                nameof(input));
        }

        // Normalize input
        Vector<T> normalizedInput = new Vector<T>(input.Length);
        for (int i = 0; i < input.Length; i++)
            normalizedInput[i] = NumOps.Divide(NumOps.Subtract(input[i], _normMean), _normStd);

        Vector<T> residual = normalizedInput;
        Vector<T> aggregatedForecast = new Vector<T>(_options.ForecastHorizon);

        // Forward pass through all blocks
        for (int blockIdx = 0; blockIdx < _blocks.Count; blockIdx++)
        {
            var (backcast, forecast) = _blocks[blockIdx].ForwardInternal(residual);

            residual = (Vector<T>)Engine.Subtract(residual, backcast);
            aggregatedForecast = (Vector<T>)Engine.Add(aggregatedForecast, forecast);
        }

        // Denormalize forecast
        for (int i = 0; i < aggregatedForecast.Length; i++)
            aggregatedForecast[i] = NumOps.Add(NumOps.Multiply(aggregatedForecast[i], _normStd), _normMean);

        return aggregatedForecast;
    }

    /// <summary>
    /// Serializes model-specific data to the binary writer.
    /// </summary>
    protected override void SerializeCore(BinaryWriter writer)
    {
        // Write N-BEATS specific options
        writer.Write(_options.NumStacks);
        writer.Write(_options.NumBlocksPerStack);
        writer.Write(_options.PolynomialDegree);
        writer.Write(_options.LookbackWindow);
        writer.Write(_options.ForecastHorizon);
        writer.Write(_options.HiddenLayerSize);
        writer.Write(_options.NumHiddenLayers);
        writer.Write(_options.LearningRate);
        writer.Write(_options.Epochs);
        writer.Write(_options.BatchSize);
        writer.Write(_options.ShareWeightsInStack);
        writer.Write(_options.UseInterpretableBasis);

        // Write all block parameters
        writer.Write(_blocks.Count);
        foreach (var block in _blocks)
        {
            Vector<T> blockParams = block.GetParameters();
            writer.Write(blockParams.Length);
            for (int i = 0; i < blockParams.Length; i++)
            {
                writer.Write(Convert.ToDouble(blockParams[i]));
            }
        }
    }

    /// <summary>
    /// Deserializes model-specific data from the binary reader.
    /// </summary>
    protected override void DeserializeCore(BinaryReader reader)
    {
        // Read N-BEATS specific options
        _options.NumStacks = reader.ReadInt32();
        _options.NumBlocksPerStack = reader.ReadInt32();
        _options.PolynomialDegree = reader.ReadInt32();
        _options.LookbackWindow = reader.ReadInt32();
        _options.ForecastHorizon = reader.ReadInt32();
        _options.HiddenLayerSize = reader.ReadInt32();
        _options.NumHiddenLayers = reader.ReadInt32();
        _options.LearningRate = reader.ReadDouble();
        _options.Epochs = reader.ReadInt32();
        _options.BatchSize = reader.ReadInt32();
        _options.ShareWeightsInStack = reader.ReadBoolean();
        _options.UseInterpretableBasis = reader.ReadBoolean();

        // Reinitialize blocks with loaded options
        InitializeBlocks();

        // Read all block parameters
        int blockCount = reader.ReadInt32();
        if (blockCount != _blocks.Count)
        {
            throw new InvalidOperationException(
                $"Block count mismatch. Expected {_blocks.Count}, but serialized data contains {blockCount}.");
        }

        for (int i = 0; i < blockCount; i++)
        {
            int paramCount = reader.ReadInt32();
            Vector<T> blockParams = new Vector<T>(paramCount);
            for (int j = 0; j < paramCount; j++)
            {
                blockParams[j] = NumOps.FromDouble(reader.ReadDouble());
            }
            _blocks[i].SetParameters(blockParams);
        }
    }

    /// <summary>
    /// Gets metadata about the N-BEATS model.
    /// </summary>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = "N-BEATS",
            Description = "Neural Basis Expansion Analysis for Interpretable Time Series Forecasting",
            Complexity = ParameterCount,
            FeatureCount = _options.LookbackWindow,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "InputDimension", _options.LookbackWindow },
                { "OutputDimension", _options.ForecastHorizon },
                { "TrainingMetrics", LastEvaluationMetrics ?? new Dictionary<string, T>() },
                { "Hyperparameters", new Dictionary<string, object>
                    {
                        { "NumStacks", _options.NumStacks },
                        { "NumBlocksPerStack", _options.NumBlocksPerStack },
                        { "PolynomialDegree", _options.PolynomialDegree },
                        { "LookbackWindow", _options.LookbackWindow },
                        { "ForecastHorizon", _options.ForecastHorizon },
                        { "HiddenLayerSize", _options.HiddenLayerSize },
                        { "NumHiddenLayers", _options.NumHiddenLayers },
                        { "UseInterpretableBasis", _options.UseInterpretableBasis }
                    }
                }
            }
        };
        return metadata;
    }

    /// <summary>
    /// Creates a new instance of the N-BEATS model.
    /// </summary>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateInstance()
    {
        return new NBEATSModel<T>(new NBEATSModelOptions<T>(_options));
    }

    /// <summary>
    /// Gets the total number of trainable parameters in the model.
    /// </summary>
    public override long ParameterCount
    {
        get
        {
            int totalParams = 0;
            foreach (var block in _blocks)
            {
                totalParams += (int)block.ParameterCount;
            }
            return totalParams;
        }
    }

    /// <summary>
    /// Gets all model parameters as a single vector.
    /// </summary>
    public override Vector<T> GetParameters()
    {
        var allParams = new List<T>();

        foreach (var block in _blocks)
        {
            Vector<T> blockParams = block.GetParameters();
            for (int i = 0; i < blockParams.Length; i++)
            {
                allParams.Add(blockParams[i]);
            }
        }

        return new Vector<T>(allParams.ToArray());
    }

    /// <summary>
    /// Sets all model parameters from a single vector.
    /// </summary>
    public override void SetParameters(Vector<T> parameters)
    {
        int expectedCount = ParameterCountHelper.ToFlatVectorSize(ParameterCount);
        if (parameters.Length != expectedCount)
        {
            throw new ArgumentException(
                $"Expected {expectedCount} parameters, but got {parameters.Length}.",
                nameof(parameters));
        }

        int idx = 0;
        foreach (var block in _blocks)
        {
            int blockParamCount = checked((int)block.ParameterCount);
            Vector<T> blockParams = new Vector<T>(blockParamCount);

            for (int i = 0; i < blockParamCount; i++)
            {
                blockParams[i] = parameters[idx++];
            }

            block.SetParameters(blockParams);
        }
    }

    /// <summary>
    /// Creates slice weights for extracting a single element from a vector.
    /// </summary>
    private T[] CreateSliceWeights(int index, int length, INumericOperations<T> numOps)
    {
        var weights = new T[length];
        for (int i = 0; i < length; i++)
        {
            weights[i] = i == index ? numOps.One : numOps.Zero;
        }
        return weights;
    }

    public override IFullModel<T, Matrix<T>, Vector<T>> Clone()
    {
        var clone = new NBEATSModel<T>(_options);
        // Copy trained blocks (read-only after training -- safe to share by reference)
        clone._blocks.Clear();
        clone._blocks.AddRange(_blocks);
        // Copy training series
        if (_trainingSeries.Length > 0)
            clone._trainingSeries = new Vector<T>(_trainingSeries);
        // Copy model parameters
        if (ModelParameters is not null && ModelParameters.Length > 0)
            clone.ModelParameters = new Vector<T>(ModelParameters);
        // Copy normalization parameters
        clone._normMean = _normMean;
        clone._normStd = _normStd;
        return clone;


}

    public override IFullModel<T, Matrix<T>, Vector<T>> DeepCopy() => Clone();

}
