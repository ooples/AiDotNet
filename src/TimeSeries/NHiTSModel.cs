using AiDotNet.LearningRateSchedulers;
using AiDotNet.Helpers;
using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Enums;
using AiDotNet.LossFunctions;
using AiDotNet.Optimizers;
using AiDotNet.Models.Options;
using AiDotNet.Models.Parameters;
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
///
/// // Prepare historical time series data
/// var history = new Vector&lt;double&gt;(new double[] { 112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
///     115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140 });
/// var trainingMatrix = new Matrix&lt;double&gt;(history.Length - 1, 1);
///
/// // Train the model on historical observations
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new NHiTSModel&lt;double&gt;(options))
///     .Build(trainingMatrix, history.SubVector(1, history.Length - 1));
///
/// // Forecast future values using hierarchical interpolation
/// var forecast = result.Predict(trainingMatrix);
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
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 1e-3, ReferenceBatchSize = 256,
                Source = "Challu et al. 2023, Sec. 4: trained with the ADAM optimizer and MAE loss at a "
                        + "batch size of 256 and an initial learning rate of 1e-3, halved three times "
                        + "across the training procedure. No schedule is declared because the paper "
                        + "gives neither the interval nor the points at which the halving occurs.")]
public partial class NHiTSModel<T> : TimeSeriesModelBase<T>, ISupportsLossFunction<T>
{
    /// <inheritdoc />
    /// <remarks>
    /// N-HiTS is a point forecaster: its head emits a single value per horizon step, so any
    /// pointwise loss is meaningful. Defaults to mean squared error when none is configured.
    /// </remarks>
    public void SetLossFunction(ILossFunction<T> lossFunction) => ApplyLossFunction(lossFunction);

    private readonly NHiTSOptions<T> _options;
    [Buffer(Availability = ParameterAvailability.Fit)]
    private Vector<T> _trainingSeries = Vector<T>.Empty();
    private readonly List<NHiTSStackTensor<T>> _stacks;
    private readonly Random _random;

    // Normalization statistics computed during training (zero-mean / unit-variance
    // of the training series). Applied to inputs before the network and inverted on
    // the network output so gradient flow stays well-scaled — mirrors NBEATSModel.
    private T _normMean = MathHelper.GetNumericOperations<T>().Zero;
    private T _normStd = MathHelper.GetNumericOperations<T>().One;

    /// <summary>
    /// Initializes a new instance of the NHiTSModel class.
    /// </summary>
    /// <param name="options">Configuration options for N-HiTS.</param>
    public NHiTSModel(NHiTSOptions<T>? options = null)
        : base(options ??= new NHiTSOptions<T>())
    {
        _options = options;
        Options = _options;
        _stacks = new List<NHiTSStackTensor<T>>();
        _random = RandomHelper.CreateSeededRandom(SeedOr(42));

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
                seed: SeedOr(42) + i * 1000
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

        // Adam optimizer (Challu et al. 2023).
        var adamOptions = new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
        {
            InitialLearningRate = _options.LearningRate
        };
        var optimizer = new AdamOptimizer<T, Tensor<T>, Tensor<T>>(null, adamOptions);

        // Collect every trainable weight/bias tensor from all stacks (registered via
        // RegisterTrainableParameter in the stack constructor).
        var allStacks = _stacks.Cast<Interfaces.ILayer<T>>().ToList();
        var trainableParams = Training.TapeTrainingStep<T>.CollectParameters(allStacks, -1);
        var trainableLayers = _stacks.Cast<ITrainableLayer<T>>().ToList();

        var trainingLoss = TrainingLoss;

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

            // Only windows with a complete lookback AND target (idx ∈ [L, N - H]) train, mirroring the paper's
            // windowed sampling. They are filtered once (checkpointWindows), before shuffling, so every batch but
            // the last has exactly BatchSize samples: each distinct batch shape is a separate compiled plan.
            var order = checkpointWindows.OrderBy(_ => _random.Next()).ToList();

            int epochSampleCount = 0;

            for (int batchStart = 0; batchStart < order.Count; batchStart += _options.BatchSize)
            {
                if (timeBounded && TrainingCancellationToken.IsCancellationRequested)
                    break;
                TrainingCancellationToken.ThrowIfCancellationRequested();

                int effectiveBatch = Math.Min(_options.BatchSize, order.Count - batchStart);
                var inputData = new T[effectiveBatch * lookback];
                var targetData = new T[effectiveBatch * horizon];
                for (int bi = 0; bi < effectiveBatch; bi++)
                {
                    int idx = order[batchStart + bi];
                    for (int j = 0; j < lookback; j++)
                        inputData[bi * lookback + j] = yNorm[idx - lookback + j];
                    for (int h = 0; h < horizon; h++)
                        targetData[bi * horizon + h] = yNorm[idx + h];
                }
                var batchInput = new Tensor<T>(new[] { effectiveBatch, lookback }, new Vector<T>(inputData));
                var batchTarget = new Tensor<T>(new[] { effectiveBatch, horizon }, new Vector<T>(targetData));

                // Multi-rate pooling, every stack's forecast and their sum (RunForwardBatched), the loss, the
                // backward and the Adam update: one fused compiled plan when it applies (CPU or GPU), the eager
                // tape otherwise (TrainTapeBatch). The epoch is SCORED separately on the frozen end-of-epoch
                // weights (see ValidationMse), so the per-batch loss is not accumulated here.
                TrainTapeBatch(trainableLayers, batchInput, batchTarget, RunStacksForTraining, trainingLoss.ComputeTapeLoss, optimizer);
                epochSampleCount += effectiveBatch;
            }

            // Snapshot the parameters if this epoch's FROZEN end-of-epoch weights are
            // the best so far. Score with ValidationMse (a no-update forward over the
            // window set) so bestLoss measures exactly the weights bestSnapshot
            // captures. Scoring by the epoch's mean PRE-update batch loss instead
            // would measure a mix of intra-epoch weight states, not the snapshot, so
            // a late-diverging epoch (good early batches, bad end weights) could be
            // wrongly selected as best. The extra pass is forward-only (no backprop).
            double epochLoss = epochSampleCount > 0
                ? ValidationMse(checkpointWindows, yNorm, lookback, horizon)
                : double.PositiveInfinity;
            if (!double.IsNaN(epochLoss) && !double.IsInfinity(epochLoss) && epochLoss < bestLoss)
            {
                bestLoss = epochLoss;
                bestSnapshot = new List<Vector<T>>(_stacks.Count);
                foreach (var stack in _stacks)
                    bestSnapshot.Add(stack.GetParameters());
            }

            // Surface the epoch to facade callbacks / patience-based early stopping (a diverged epoch reports
            // a non-finite loss, which counts as non-improving). Break on a callback/early-stopping veto.
            if (!ReportEpoch(epoch, timeBounded ? 0 : _options.Epochs, NumOps.FromDouble(epochLoss)))
            {
                break;
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
    /// Batched, tape-recordable average pooling: <c>[B, L] → [B, ceil(L / kernelSize)]</c>, the same windows
    /// <see cref="ApplyPoolingTensor"/> produces per sample (the last one shorter when the kernel does not divide
    /// the lookback). Kernel 1 is identity.
    /// </summary>
    private Tensor<T> PoolBatchedTape(Tensor<T> input, int kernelSize)
    {
        if (kernelSize <= 1) return input;
        int B = input.Shape[0];
        int L = input.Shape[1];
        int fullWindows = L / kernelSize;
        int tail = L - fullWindows * kernelSize;

        Tensor<T>? full = null;
        if (fullWindows > 0)
        {
            var body = tail == 0 ? input : Engine.TensorNarrow(input, 1, 0, fullWindows * kernelSize);
            full = Engine.ReduceMean(Engine.Reshape(body, new[] { B, fullWindows, kernelSize }), new[] { 2 }, keepDims: false);
        }
        if (tail == 0 && full is not null)
            return full;

        // The partial last window averages only the elements it has.
        var last = Engine.ReduceMean(Engine.TensorNarrow(input, 1, fullWindows * kernelSize, tail), new[] { 1 }, keepDims: true);
        return full is null ? last : Engine.TensorConcatenate(new[] { full, last }, axis: 1);
    }

    /// <summary>
    /// Runs the full multi-rate stack over a <c>[B, L]</c> batch using on-tape pooling, each stack's forecast and
    /// their sum, returning <c>[B, H]</c>. Null only when the model has no stacks.
    /// </summary>
    internal Tensor<T>? RunForwardBatched(Tensor<T> input)
    {
        Tensor<T>? aggregated = null;
        foreach (var stack in _stacks)
        {
            var pooled = PoolBatchedTape(input, stack.PoolingSize);
            // Summed column-major ([H, B]) and transposed once below: see ForwardTapeColumns.
            var forecast = stack.ForwardTapeColumns(pooled);
            aggregated = aggregated is null
                ? forecast
                : Engine.TensorAdd(aggregated, forecast);
        }
        return aggregated is null ? null : Engine.TensorPermute(aggregated, new[] { 1, 0 });
    }

    private Tensor<T> RunStacksForTraining(Tensor<T> input)
        => RunForwardBatched(input) ?? throw new InvalidOperationException("N-HiTS has no stacks to run; the model was not initialized.");

    /// <summary>
    /// Validation MSE across up to 256 windows, scoring each epoch on its frozen end-of-epoch weights. Uses
    /// the current stack weights.
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
        //
        // Rows that reach the stacks as a full lookback window (a row of exactly LookbackWindow values, or a shorter
        // row that PredictSingle replaces with the training-series tail) run as one batched forward per chunk instead
        // of one per row: AiModelBuilder predicts the whole dataset after fitting, so per-row inference was a large
        // share of a production fit. Same values as PredictSingle (NHiTSBatchedPredictTests): the batched pooling is
        // the same tail-mean average pool, and each stack already emits ForecastHorizon values, so the per-row
        // interpolation is the identity. Any other row length keeps the per-row path.
        int lookback = _options.LookbackWindow;
        bool tailFill = _trainingSeries.Length >= lookback;
        var batched = new List<int>(n);
        for (int i = 0; i < n; i++)
        {
            if (input.Columns == lookback || (input.Columns < lookback && tailFill))
                batched.Add(i);
            else
                predictions[i] = PredictSingle(input.GetRow(i));
        }

        for (int c0 = 0; c0 < batched.Count; c0 += PredictChunkRows)
        {
            int rows = Math.Min(PredictChunkRows, batched.Count - c0);
            // A nested arena per chunk: its scratch is recycled when the chunk ends without touching any arena the
            // caller has open (whose tensors may still be live).
            using var chunkArena = AiDotNet.Tensors.Helpers.TensorArena.Create();
            var data = new T[rows * lookback];
            for (int r = 0; r < rows; r++)
            {
                int row = batched[c0 + r];
                for (int t = 0; t < lookback; t++)
                {
                    T v = input.Columns == lookback ? input[row, t] : _trainingSeries[_trainingSeries.Length - lookback + t];
                    data[r * lookback + t] = NumOps.Divide(NumOps.Subtract(v, _normMean), _normStd);
                }
            }
            var outNorm = RunForwardBatched(new Tensor<T>(new[] { rows, lookback }, new Vector<T>(data)))
                ?? throw new InvalidOperationException("N-HiTS has no stacks to run; the model was not initialized.");
            var outSpan = outNorm.IsContiguous ? outNorm.AsSpan() : outNorm.Contiguous().AsSpan();   // [rows, horizon]
            int horizon = _options.ForecastHorizon;
            for (int r = 0; r < rows; r++)
                predictions[batched[c0 + r]] = NumOps.Add(NumOps.Multiply(outSpan[r * horizon], _normStd), _normMean);
        }
        return predictions;
    }

    // Rows per batched predict forward; bounds activation memory ([rows, HiddenLayerSize] per stack layer).
    private const int PredictChunkRows = 1024;

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
}

/// <summary>
/// Represents a single stack in the N-HiTS architecture using Tensor operations.
/// </summary>
// Rank 1 only, and that is enforced rather than assumed: ForwardInternal reshapes its argument to
// `[_inputLength, 1]`, which only succeeds when the whole tensor holds exactly _inputLength values.
// The axis is Time on both sides - in is the (already pooled) lookback window, out is the forecast
// horizon - which is also what the constructor declares:
// `base(new[] { inputLength }, new[] { outputLength })`.
[TensorLayout(TensorAxis.Time, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Time, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
internal partial class NHiTSStackTensor<T> : NeuralNetworks.Layers.LayerBase<T>, IShapeContract
{
    /// <inheritdoc />
    /// <remarks>
    /// <para>
    /// HAND-WRITTEN, and Fixed rather than Same for a reason worth stating: this stack does not
    /// merely resize its input, it REFUSES to be resized BY it. ForwardInternal's last statement is
    /// <c>Engine.Reshape(col, new[] { _outputLength })</c>, and the loop before it walks the MLP's
    /// own weight list, so the horizon comes from the stack's configuration alone.
    /// </para>
    /// <para>
    /// The guard at the top of ForwardInternal makes that emphatic: an input whose length is not
    /// <c>_inputLength</c> is RESAMPLED onto _inputLength ("Ensure input matches expected size")
    /// rather than rejected. So no input length - not even a wrong one - reaches the output.
    /// </para>
    /// </remarks>
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
    {
        if (inputRank != 1 || _outputLength <= 0) return null;

        return new[]
        {
            new OutputAxisContract(TensorAxis.Time, AxisRelation.Fixed(_outputLength)),
        };
    }

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

    public override bool SupportsTraining => true;
    public override void ResetState() { _lastForwardInput = null; }
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

    /// <summary>Construction state: kept so the generated clone factory can rebuild this layer.</summary>
    private readonly int _numBlocks;

    public NHiTSStackTensor(int inputLength, int outputLength, int hiddenSize, int numLayers, int numBlocks, int poolingSize, int seed = 42)
        : base(new[] { inputLength }, new[] { outputLength })
    {
        _numBlocks = numBlocks;
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

    [Scratch]
    private Tensor<T>? _lastForwardInput;

    protected override Tensor<T> ForwardTraced(Tensor<T> input)
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
            linear = Engine.TensorAdd(linear, biasCol);
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
    /// <remarks>
    /// The result is a permuted view. A caller that sums several stacks' forecasts should use
    /// <see cref="ForwardTapeColumns"/> and transpose the sum once: adding permuted views of
    /// device-resident results gave a wrong sum on the DirectGpu engine (#1804,
    /// AiDotNet.Tensors#1090), and costs a strided permute per stack either way.
    /// </remarks>
    public Tensor<T> ForwardTape(Tensor<T> input)
        => Engine.TensorPermute(ForwardTapeColumns(input), new[] { 1, 0 });

    /// <summary>
    /// Tape-tracked forward pass over a batched, already-pooled input <c>[B, inputLength]</c>,
    /// returning the stack forecast column-major, <c>[outputLength, B]</c>: the layout the stack
    /// computes in (weight <c>[out, in]</c> @ x <c>[in, B]</c>), so forecasts from several stacks
    /// sum as dense tensors.
    /// </summary>
    internal Tensor<T> ForwardTapeColumns(Tensor<T> input)
    {
        // [B, in] -> [in, B] so weight[out, in] @ x[in, B] = [out, B].
        var x = Engine.TensorPermute(input, new[] { 1, 0 });

        for (int layer = 0; layer < _weights.Count; layer++)
        {
            var weight = _weights[layer];               // [outSize, inSize]
            var linear = Engine.TensorMatMul(weight, x); // [outSize, B]
            var biasCol = Engine.Reshape(_biases[layer], new[] { weight.Shape[0], 1 });
            linear = Engine.TensorAdd(linear, biasCol);

            // ReLU on every layer except the linear output head.
            x = layer < _weights.Count - 1 ? Engine.ReLU(linear) : linear;
        }

        return x; // [outputLength, B]
    }
}
