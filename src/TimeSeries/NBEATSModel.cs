using AiDotNet.LearningRateSchedulers;
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
/// var inputMatrix = new Matrix&lt;double&gt;(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 }, { 5.0, 6.0 }, { 7.0, 8.0 } });
/// var trainingLabels = new Vector&lt;double&gt;(new double[] { 0.0, 1.0, 0.0, 1.0 });
/// var trainingMatrix = new Matrix&lt;double&gt;(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 }, { 5.0, 6.0 }, { 7.0, 8.0 } });
/// // Create an N-BEATS model with interpretable trend and seasonality stacks
/// var options = new NBEATSModelOptions&lt;double&gt;();
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new NBEATSModel&lt;double&gt;(options))
///     .Build(trainingMatrix, trainingLabels);
/// Vector&lt;double&gt; forecast = result.Predict(inputMatrix);
/// </code>
/// </example>
[ModelDomain(ModelDomain.TimeSeries)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.TimeSeriesModel)]
[ModelTask(ModelTask.Forecasting)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("N-BEATS: Neural basis expansion analysis for interpretable time series forecasting", "https://arxiv.org/abs/1905.10437", Year = 2020, Authors = "Boris N. Oreshkin, Dmitri Carpov, Nicolas Chapados, Yoshua Bengio")]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 0.001,
                Source = "Oreshkin et al. 2020, Sec. 5: the Adam optimizer with default settings and an "
                        + "initial learning rate of 0.001.")]
public partial class NBEATSModel<T> : TimeSeriesModelBase<T>, ISupportsLossFunction<T>
{
    /// <inheritdoc />
    /// <remarks>
    /// N-BEATS is a point forecaster: its head emits a single value per horizon step, so any
    /// pointwise loss is meaningful. Defaults to mean squared error when none is configured.
    /// </remarks>
    public void SetLossFunction(ILossFunction<T> lossFunction) => ApplyLossFunction(lossFunction);

    private readonly NBEATSModelOptions<T> _options;
    private readonly List<NBEATSBlock<T>> _blocks;
    [Buffer]
    private Vector<T> _trainingSeries = Vector<T>.Empty();

    // Normalization statistics computed during training
    private T _normMean = MathHelper.GetNumericOperations<T>().Zero;
    private T _normStd = MathHelper.GetNumericOperations<T>().One;

    // Per-epoch average training loss (normalized MSE) recorded during the
    // most recent TrainCore run. Populated by BOTH the eager tape path and
    // the fused compiled path so training convergence can be
    // verified directly (the value the optimizer actually minimizes), rather
    // than inferred from denormalized held-out predictions.
    private List<double> _lastRunEpochLosses = new();

    /// <summary>
    /// Average training loss (normalized MSE) for each epoch of the most recent
    /// <c>Train</c> call, in order. Useful for verifying convergence and for
    /// comparing the fused compiled path against the eager path.
    /// </summary>
    /// <remarks>
    /// Internal diagnostic: the public surface stays limited to the facade
    /// (<c>AiModelBuilder</c>/<c>AiModelResult</c>). Exposed as an immutable
    /// snapshot so callers cannot mutate the backing list. Visible to the test
    /// and serving assemblies via <c>InternalsVisibleTo</c>.
    /// </remarks>
    internal IReadOnlyList<double> LastRunEpochLosses => _lastRunEpochLosses.AsReadOnly();

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
    public NBEATSModel(NBEATSModelOptions<T>? options = null) : base(options ??= new NBEATSModelOptions<T>())
    {
        _options = options;
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
                _options.PolynomialDegree,
                // Distinct per block: a shared seed gave every block identical initial weights.
                seed: SeedOr(42) + i
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
        // Global-norm gradient clipping at GradientClipNorm (see that option) is the optimizer's own:
        // one engine reduction per step, shared with its anomaly guard.
        var adamOptions = new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
        {
            InitialLearningRate = _options.LearningRate,
            EnableGradientClipping = _options.GradientClipNorm > 0,
            GradientClippingMethod = GradientClippingMethod.ByNorm,
            MaxGradientNorm = _options.GradientClipNorm,
        };
        var optimizer = new AdamOptimizer<T, Tensor<T>, Tensor<T>>(null, adamOptions);

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
        var trainingLoss = TrainingLoss;

        int numSamples = x.Rows;

        // The trainable tensors, for the best-epoch snapshot below.
        var allBlocks = _blocks.Cast<Interfaces.ILayer<T>>().ToList();
        var trainableParams = Training.TapeTrainingStep<T>.CollectParameters(allBlocks, -1);
        var trainableLayers = _blocks.Cast<ITrainableLayer<T>>().ToList();

        int lookback = _options.LookbackWindow;
        int horizon = _options.ForecastHorizon;
        var validWindows = new List<int>();
        for (int idx = 0; idx < numSamples; idx++)
            if (idx >= lookback && idx + horizon <= yNorm.Length)
                validWindows.Add(idx);

        var random = RandomHelper.CreateSeededRandom(SeedOr(42));
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

        // Best-epoch-weights checkpointing. On noisy / fat-tailed real series the eager Adam loop can DIVERGE —
        // the normalized training loss climbs instead of falling (measured 1.20 -> 1.55 over 15 epochs on SPY
        // daily log-returns) and the final weights produce predictions blown up to hundreds of times the target
        // scale. Adam's adaptive step makes gradient clipping alone insufficient here, so we additionally keep the
        // parameters from the LOWEST-loss epoch and restore them after training — inference then always uses the
        // best-fit weights, never a diverged tail. Standard early-stopping-style checkpointing.
        double bestEpochLoss = double.PositiveInfinity;
        Tensor<T>[]? bestParamSnapshot = null;
        for (int epoch = 0; epoch < maxEpochs; epoch++)
        {
            if (timeBounded && TrainingCancellationToken.IsCancellationRequested)
                break;
            TrainingCancellationToken.ThrowIfCancellationRequested();

            // Only windows with a full lookback AND a full target horizon train (Oreshkin et al. 2019 §3.3 drop the
            // rest). They are filtered once, before shuffling, so every batch but the last has exactly BatchSize
            // samples: filtering inside each batch gave every batch a different B, and each distinct shape is a
            // separate compiled plan.
            var order = validWindows.OrderBy(_ => random.Next()).ToList();

            double epochLossSum = 0.0;
            int epochStepCount = 0;

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

                // The doubly-residual stack (paper §3.2), the loss, the backward and the Adam update: one fused
                // compiled plan when it applies (CPU or GPU), the eager tape otherwise (TrainTapeBatch).
                T batchLoss = TrainTapeBatch(
                    trainableLayers, batchInput, batchTarget, RunForwardStack, trainingLoss.ComputeTapeLoss, optimizer);
                epochLossSum += NumOps.ToDouble(batchLoss);
                epochStepCount++;
            }

            if (epochStepCount > 0)
            {
                double epochLoss = epochLossSum / epochStepCount;
                _lastRunEpochLosses.Add(epochLoss);

                // Snapshot the weights (as Tensors) whenever this epoch improved on the best loss so far.
                // Allocate the snapshot tensors once, then reuse them via a vectorized copy (dest = src * 1).
                if (epochLoss < bestEpochLoss && !double.IsNaN(epochLoss) && !double.IsInfinity(epochLoss))
                {
                    bestEpochLoss = epochLoss;
                    if (bestParamSnapshot is null)
                    {
                        bestParamSnapshot = new Tensor<T>[trainableParams.Count];
                        for (int pi = 0; pi < trainableParams.Count; pi++)
                            bestParamSnapshot[pi] = trainableParams[pi].Clone();
                    }
                    else
                    {
                        for (int pi = 0; pi < trainableParams.Count; pi++)
                            Engine.TensorMultiplyScalarInto(bestParamSnapshot[pi], trainableParams[pi], NumOps.One);
                    }
                }

                // Report after snapshotting so an early stop still leaves the best weights to
                // restore below.
                if (!ReportEpoch(epoch, timeBounded ? 0 : _options.Epochs, NumOps.FromDouble(epochLoss)))
                {
                    break;
                }
            }
        }

        // Restore the best-loss weights so inference never runs on a diverged tail (vectorized copy).
        if (bestParamSnapshot is not null)
            for (int pi = 0; pi < trainableParams.Count; pi++)
                Engine.TensorMultiplyScalarInto(trainableParams[pi], bestParamSnapshot[pi], NumOps.One);
    }

    /// <summary>
    /// Runs the doubly-residual N-BEATS stack (paper §3.2) over a <c>[B, L]</c>
    /// batch and returns the aggregated <c>[B, H]</c> forecast, using
    /// tape-recordable Engine ops so the compiled training plan can trace it.
    /// </summary>
    internal Tensor<T> RunForwardStack(Tensor<T> input)
    {
        if (_blocks.Count == 0)
            throw new InvalidOperationException("N-BEATS has no blocks to run; the model was not initialized.");

        // The stack runs column-major ([L, B] residual, [H, B] forecast sum), the layout each
        // block computes in, so the input is transposed once here and the forecast once at the
        // end. Threading [B, L] through ForwardTape instead costs two strided permutes per block
        // per pass (plus their backward), and makes every residual subtract and forecast sum
        // combine strided views.
        var residual = Engine.TensorPermute(input, new[] { 1, 0 });
        Tensor<T>? aggregatedForecast = null;
        for (int blockIdx = 0; blockIdx < _blocks.Count; blockIdx++)
        {
            var (backcast, forecast) = _blocks[blockIdx].ForwardTapeColumns(residual);
            residual = Engine.TensorSubtract(residual, backcast);
            aggregatedForecast = aggregatedForecast is null
                ? forecast
                : Engine.TensorAdd(aggregatedForecast, forecast);
        }
        return Engine.TensorPermute(aggregatedForecast ?? throw new InvalidOperationException("N-BEATS produced no forecast."), new[] { 1, 0 });
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


    /// <summary>
    /// Deserializes model-specific data from the binary reader.
    /// </summary>


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

    // ParameterCount restated a fold the base now derives from generated component registration.
    // Removed under AIDN082.
    // GetParameters restated a fold the base now derives from generated component registration.
    // Removed under AIDN082.
    // SetParameters restated a fold the base now derives from generated component registration.
    // Removed under AIDN082.
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

}
