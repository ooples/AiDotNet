using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Enums;
using AiDotNet.LossFunctions;
using AiDotNet.Optimizers;
using AiDotNet.Models.Options;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Validation;

namespace AiDotNet.TimeSeries;

/// <summary>
/// Implements DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// DeepAR is a probabilistic forecasting model that produces full probability distributions
/// rather than point estimates. Key features include:
/// </para>
/// <list type="bullet">
/// <item>Autoregressive RNN architecture (LSTM-based)</item>
/// <item>Probabilistic forecasts with a Gaussian, negative-binomial or Student-t likelihood</item>
/// <item>Quantile predictions via Monte-Carlo sampling of the predictive path</item>
/// <item>Effective for cold-start scenarios</item>
/// </list>
/// <para>
/// Original paper: Salinas et al., "DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks" (2020).
/// </para>
/// <para>
/// The network is the paper's (<see cref="DeepARNetwork{T}"/>, shared with the Finance DeepAR and checked against a
/// PyTorch transcription): an LSTM over the previous value divided by the series scale ν and the covariates, whose top
/// state parametrizes the likelihood. Training maximizes the log-likelihood over whole windows of conditioning and
/// prediction range (§3.2); forecasts are medians and quantiles of ancestral sample paths (§2). The paper's age
/// covariate is added when <see cref="DeepAROptions{T}.UseAgeFeature"/> is set; its time-of-day and calendar
/// covariates need timestamps, which the time-series interface does not carry, so pass them as covariates.
/// </para>
/// <para><b>For Beginners:</b> DeepAR is like a weather forecaster that doesn't just say
/// "it will be 70 degrees tomorrow" but rather "there's a 50% chance it'll be between 65-75 degrees."
/// It uses an LSTM (a recurrent neural network good at remembering patterns over time) and predicts
/// a probability distribution (a mean and a spread) for each future step.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var contextWindow = new Matrix&lt;double&gt;(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 }, { 5.0, 6.0 }, { 7.0, 8.0 } });
/// var trainingLabels = new Vector&lt;double&gt;(new double[] { 0.0, 1.0, 0.0, 1.0 });
/// var trainingMatrix = new Matrix&lt;double&gt;(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 }, { 5.0, 6.0 }, { 7.0, 8.0 } });
/// var options = new DeepAROptions&lt;double&gt;
/// {
///     LookbackWindow = 30, ForecastHorizon = 7,
///     HiddenSize = 40, NumLayers = 2
/// };
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new DeepARModel&lt;double&gt;(options))
///     .Build(trainingMatrix, trainingLabels);
/// Vector&lt;double&gt; forecast = result.Predict(contextWindow);
/// </code>
/// </example>
[ModelDomain(ModelDomain.TimeSeries)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.RecurrentNetwork)]
[ModelTask(ModelTask.Forecasting)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks", "https://arxiv.org/abs/1704.04110", Year = 2020, Authors = "David Salinas, Valentin Flunkert, Jan Gasthaus, Tim Januschowski")]
public partial class DeepARModel<T> : TimeSeriesModelBase<T>
{
    private readonly DeepAROptions<T> _options;
    private readonly DeepARLikelihood _likelihood;
    private readonly List<LayerBase<T>> _layers = new();
    private readonly DeepARNetwork<T> _network;
    private readonly Random _random;
    private readonly List<double> _epochLosses = new();
    [Buffer]
    private Vector<T> _trainingSeries = Vector<T>.Empty();

    // Standardization of the covariates (§3.4, "we standardize all covariates") from the training rows; the age
    // covariate's from the training indices.
    private double[] _covariateMean = Array.Empty<double>();
    private double[] _covariateStd = Array.Empty<double>();
    private double _ageMean;
    private double _ageStd = 1.0;

    /// <summary>
    /// Initializes a new instance of the DeepARModel class.
    /// </summary>
    /// <param name="options">Configuration options for DeepAR.</param>
    public DeepARModel(DeepAROptions<T>? options = null)
        : base(options ??= new DeepAROptions<T>())
    {
        _options = options;
        Options = _options;
        if (_options.LookbackWindow <= 0) throw new ArgumentException("Lookback window must be positive.");
        if (_options.ForecastHorizon <= 0) throw new ArgumentException("Forecast horizon must be positive.");
        if (_options.HiddenSize <= 0) throw new ArgumentException("Hidden size must be positive.");
        if (_options.NumLayers <= 0) throw new ArgumentException("Number of layers must be positive.");
        if (_options.NumSamples <= 0) throw new ArgumentException("Number of samples must be positive.");
        if (_options.CovariateSize < 0) throw new ArgumentException("CovariateSize must not be negative.");
        if (_options.Epochs <= 0 || _options.BatchSize <= 0 || _options.LearningRate <= 0)
            throw new ArgumentException("Epochs, BatchSize and LearningRate must be positive.");
        _likelihood = _options.ResolveLikelihood();
        int seed = _options.Seed ?? 42;
        _random = RandomHelper.CreateSeededRandom(seed + 1);
        _network = new DeepARNetwork<T>(Engine, _layers, new DeepARConfiguration(
            Covariates, _options.HiddenSize, _options.NumLayers, _options.DropoutRate, _likelihood, _options.ScaleHandling));
        _network.InitializeLikeReference(RandomHelper.CreateSeededRandom(seed));
    }

    // The network's covariates: the age feature, then the caller's.
    private int Covariates => (_options.UseAgeFeature ? 1 : 0) + _options.CovariateSize;

    /// <inheritdoc />
    protected override void RegisterComponents()
    {
        for (int i = 0; i < _layers.Count; i++)
            RegisterParameterComponent($"deepar.layer{i}", _layers[i]);
    }

    private static Random InferenceRandom() => DeepARNetwork<T>.ForecastRandom();

    // The network covariates of series index `index` (the age, then the caller's row, standardized).
    private void FillCovariates(double[,,] target, int row, int step, int index, Func<int, double>? userCovariate)
    {
        int k = 0;
        if (_options.UseAgeFeature) target[row, step, k++] = (index - _ageMean) / _ageStd;
        for (int c = 0; c < _options.CovariateSize; c++)
            target[row, step, k++] = (userCovariate!(c) - _covariateMean[c]) / _covariateStd[c];
    }

    /// <summary>
    /// Trains DeepAR as the paper does (§3.2): every window of LookbackWindow conditioning steps and ForecastHorizon
    /// prediction steps is a training instance, the negative log-likelihood of every one of its steps is minimized
    /// (teacher forcing, z₀ being the value before the window or zero at the series start), and each window's scale ν
    /// comes from its conditioning range. Adam over shuffled batches.
    /// </summary>
    protected override void TrainCore(Matrix<T> x, Vector<T> y)
    {
        int conditioning = _options.LookbackWindow, horizon = _options.ForecastHorizon, steps = conditioning + horizon;
        int n = y.Length, windows = n - steps + 1;
        if (windows <= 0)
            throw new ArgumentException($"Not enough data for one training window: need {steps} points, got {n}.", nameof(y));
        int cov = _options.CovariateSize;
        if (cov > 0 && (x is null || x.Rows != n || x.Columns < cov))
            throw new ArgumentException(
                $"DeepAR covariate contract violated: with CovariateSize={cov}, x must be [{n}, >={cov}] (one covariate row per series index), " +
                $"but got [{x?.Rows ?? 0}, {x?.Columns ?? 0}].");
        _trainingSeries = new Vector<T>(n);
        for (int i = 0; i < n; i++) _trainingSeries[i] = y[i];
        ModelParameters = new Vector<T>(1);
        ModelParameters[0] = NumOps.FromDouble(n);
        var values = y.Select(v => NumOps.ToDouble(v)).ToArray();

        _ageMean = (n - 1) / 2.0;
        _ageStd = Math.Max(Math.Sqrt((n * (double)n - 1) / 12.0), 1e-10);
        _covariateMean = new double[cov];
        _covariateStd = new double[cov];
        for (int c = 0; c < cov; c++)
        {
            double mean = 0, variance = 0;
            for (int i = 0; i < n; i++) mean += NumOps.ToDouble(x![i, c]);
            mean /= n;
            for (int i = 0; i < n; i++) variance += Math.Pow(NumOps.ToDouble(x![i, c]) - mean, 2);
            _covariateMean[c] = mean;
            _covariateStd[c] = Math.Max(Math.Sqrt(variance / n), 1e-10);
        }

        var optimizer = new AdamOptimizer<T, Tensor<T>, Tensor<T>>(null, new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
        {
            InitialLearningRate = _options.LearningRate,
            UseAdaptiveLearningRate = false,
            UseAdaptiveBetas = false,
            EnableGradientClipping = false,
        });
        var parameters = Training.TapeTrainingStep<T>.CollectParameters(_layers);
        bool timeBounded = _options.MaxTrainingTimeSeconds > 0;
        int epochs = timeBounded ? int.MaxValue : _options.Epochs;
        _epochLosses.Clear();
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            if (timeBounded && TrainingCancellationToken.IsCancellationRequested) break;
            TrainingCancellationToken.ThrowIfCancellationRequested();
            var order = Enumerable.Range(0, windows).OrderBy(_ => _random.Next()).ToArray();
            double lossSum = 0;
            int batches = 0;
            for (int start = 0; start < windows; start += _options.BatchSize)
            {
                if (timeBounded && TrainingCancellationToken.IsCancellationRequested) break;
                TrainingCancellationToken.ThrowIfCancellationRequested();
                int batch = Math.Min(_options.BatchSize, windows - start);
                // One step's data [B, T + 1, 1 + C]: channel 0 is the window z₀ … z_T, the others the covariates of
                // z₁ … z_T (row 0 unused). Packed so the shared step rebinds all of it on every compiled replay.
                var covariates = Covariates > 0 ? new double[1, steps, Covariates] : null;
                var data = new Tensor<T>(new[] { batch, steps + 1, 1 + Covariates });
                for (int b = 0; b < batch; b++)
                {
                    int s = order[start + b];
                    data[b, 0, 0] = NumOps.FromDouble(s > 0 ? values[s - 1] : 0.0);
                    for (int t = 0; t < steps; t++)
                    {
                        data[b, t + 1, 0] = NumOps.FromDouble(values[s + t]);
                        if (covariates is not null)
                        {
                            int index = s + t;
                            FillCovariates(covariates, 0, t, index, c => NumOps.ToDouble(x![index, c]));
                            for (int k = 0; k < Covariates; k++) data[b, t + 1, 1 + k] = NumOps.FromDouble(covariates[0, t, k]);
                        }
                    }
                }
                // The window's likelihood, the backward and the Adam update: one fused compiled plan when it applies
                // (CPU or GPU), the eager tape otherwise.
                T loss = TrainTapeBatch(Array.Empty<ITrainableLayer<T>>(), data, data,
                    step => WindowLoss(step, conditioning), static (value, _) => value, optimizer, extraParameters: parameters);
                lossSum += NumOps.ToDouble(loss);
                batches++;
            }
            if (batches == 0) continue;
            double epochLoss = lossSum / batches;
            _epochLosses.Add(epochLoss);
            if (!ReportEpoch(epoch, timeBounded ? 0 : _options.Epochs, NumOps.FromDouble(epochLoss))) break;
        }
    }

    // The mean negative log-likelihood of packed windows [B, T + 1, 1 + C] (see TrainCore).
    private Tensor<T> WindowLoss(Tensor<T> data, int conditioning)
    {
        int batch = data.Shape[0], length = data.Shape[1];
        var window = Engine.Reshape(Engine.TensorSlice(data, new[] { 0, 0, 0 }, new[] { batch, length, 1 }), new[] { batch, length });
        var covariates = Covariates > 0 ? Engine.TensorSlice(data, new[] { 0, 1, 1 }, new[] { batch, length - 1, Covariates }) : null;
        return _network.Loss(window, covariates, conditioning, training: true, _random);
    }

    // ------------------------------------------------------------------ forecasting

    // The history [1, t₀ + 1] of the last LookbackWindow values (z₀ the value before them, or 0) and their covariates.
    private (double[,] History, double[,,]? Covariates) History(IReadOnlyList<double> series, int endIndex, Func<int, int, double>? userCovariate)
    {
        int length = Math.Min(_options.LookbackWindow, series.Count);
        if (length == 0) throw new ArgumentException("DeepAR needs at least one observed value to forecast from.");
        int first = series.Count - length;
        var history = new double[1, length + 1];
        history[0, 0] = first > 0 ? series[first - 1] : 0.0;
        for (int t = 0; t < length; t++) history[0, t + 1] = series[first + t];
        double[,,]? covariates = null;
        if (Covariates > 0)
        {
            covariates = new double[1, length, Covariates];
            for (int t = 0; t < length; t++)
            {
                int position = first + t;
                FillCovariates(covariates, 0, t, endIndex - series.Count + position, userCovariate is null ? null : c => userCovariate(position, c));
            }
        }
        return (history, covariates);
    }

    private double[,,]? Future(int startIndex, int horizon, Func<int, int, double>? userCovariate)
    {
        if (Covariates == 0) return null;
        var future = new double[1, horizon, Covariates];
        for (int h = 0; h < horizon; h++)
        {
            int step = h;
            FillCovariates(future, 0, h, startIndex + h, userCovariate is null ? null : c => userCovariate(step, c));
        }
        return future;
    }

    private double[][] Paths(IReadOnlyList<double> series, int endIndex, Func<int, int, double>? historyCovariate,
        Func<int, int, double>? futureCovariate, int horizon)
    {
        var (history, covariates) = History(series, endIndex, historyCovariate);
        return _network.Sample(history, covariates, Future(endIndex, horizon, futureCovariate), horizon, _options.NumSamples, InferenceRandom())[0];
    }

    private static double Quantile(double[] sorted, double q)
    {
        int index = Math.Max(0, Math.Min((int)Math.Floor(q * sorted.Length), sorted.Length - 1));
        return sorted[index];
    }

    private Dictionary<double, Vector<T>> Quantiles(double[][] paths, double[] quantiles)
    {
        int horizon = paths[0].Length;
        var result = new Dictionary<double, Vector<T>>();
        foreach (var q in quantiles)
        {
            var forecast = new Vector<T>(horizon);
            for (int h = 0; h < horizon; h++)
            {
                var step = paths.Select(p => p[h]).OrderBy(v => v).ToArray();
                forecast[h] = NumOps.FromDouble(Quantile(step, q));
            }
            result[q] = forecast;
        }
        return result;
    }

    // The median of the next value's predictive distribution after the series.
    private T NextMedian(IReadOnlyList<double> series, int endIndex, Func<int, int, double>? historyCovariate, Func<int, int, double>? futureCovariate)
    {
        var paths = Paths(series, endIndex, historyCovariate, futureCovariate, 1);
        var step = paths.Select(p => p[0]).OrderBy(v => v).ToArray();
        return GuardPrediction(NumOps.FromDouble(Quantile(step, 0.5)));
    }

    /// <summary>
    /// Predicts the series: with as many rows as the training series, the median one-step forecast of each observation
    /// from the values before it; with one column (time steps), the medians of the next input.Rows steps after the
    /// training series, from NumSamples ancestral sample paths; otherwise each row is a history window and its
    /// prediction is <see cref="PredictSingle"/> of that row, all rows sampled in one batched pass.
    /// </summary>
    /// <remarks>The first observation has no history; its in-sample prediction is the median one-step forecast from an
    /// empty conditioning range (z₀ = 0, ν = 1). Out of sample, each row draws its NumSamples paths from its own
    /// generator, seeded as <see cref="PredictSingle"/> seeds one, so the batch returns exactly the per-row
    /// medians.</remarks>
    public override Vector<T> Predict(Matrix<T> input)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (_trainingSeries.Length == 0) throw new InvalidOperationException("The model must be trained before predicting.");
        if (_options.CovariateSize > 0)
            throw new InvalidOperationException(
                "This DeepAR model was trained with covariates (CovariateSize > 0). Use ForecastWithCovariates(...) or PredictNextWithCovariates(...).");
        int n = input.Rows;
        var series = _trainingSeries.Select(v => NumOps.ToDouble(v)).ToArray();
        var predictions = new Vector<T>(n);
        if (n == series.Length)
        {
            for (int i = 0; i < n; i++)
                predictions[i] = i == 0
                    ? FirstStepMedian()
                    : NextMedian(new ArraySegment<double>(series, 0, i), i, null, null);
            return predictions;
        }
        if (input.Columns == 1)
        {
            var paths = Paths(series, series.Length, null, null, n);
            for (int h = 0; h < n; h++)
                predictions[h] = GuardPrediction(NumOps.FromDouble(Quantile(paths.Select(p => p[h]).OrderBy(v => v).ToArray(), 0.5)));
            return predictions;
        }
        return NextMedians(input);
    }

    // The median next value after each row of `rows` (a history window per row), sampled for all rows at once: the
    // batched form of PredictSingle.
    private Vector<T> NextMedians(Matrix<T> rows)
    {
        int count = rows.Rows, width = rows.Columns;
        var predictions = new Vector<T>(count);
        if (count == 0) return predictions;
        int length = Math.Min(_options.LookbackWindow, width);
        if (length == 0) throw new ArgumentException("DeepAR needs at least one observed value to forecast from.");
        int first = width - length, end = Math.Max(_trainingSeries.Length, width);
        var history = new double[count, length + 1];
        var covariates = Covariates > 0 ? new double[count, length, Covariates] : null;
        var future = Covariates > 0 ? new double[count, 1, Covariates] : null;
        var randoms = new Random[count];
        for (int r = 0; r < count; r++)
        {
            history[r, 0] = first > 0 ? NumOps.ToDouble(rows[r, first - 1]) : 0.0;
            for (int t = 0; t < length; t++)
            {
                history[r, t + 1] = NumOps.ToDouble(rows[r, first + t]);
                if (covariates is not null) FillCovariates(covariates, r, t, end - width + first + t, null);
            }
            if (future is not null) FillCovariates(future, r, 0, end, null);
            randoms[r] = InferenceRandom();
        }
        var paths = _network.Sample(history, covariates, future, 1, _options.NumSamples, InferenceRandom(), rowRandoms: randoms);
        for (int r = 0; r < count; r++)
            predictions[r] = GuardPrediction(NumOps.FromDouble(Quantile(paths[r].Select(p => p[0]).OrderBy(v => v).ToArray(), 0.5)));
        return predictions;
    }

    private T FirstStepMedian()
    {
        var history = new double[1, 2];
        var covariates = Covariates > 0 ? new double[1, 1, Covariates] : null;
        if (covariates is not null) FillCovariates(covariates, 0, 0, 0, null);
        var paths = _network.Sample(history, covariates, Future(1, 1, null), 1, _options.NumSamples, InferenceRandom())[0];
        return GuardPrediction(NumOps.FromDouble(Quantile(paths.Select(p => p[0]).OrderBy(v => v).ToArray(), 0.5)));
    }

    /// <summary>
    /// The median forecast of the value after the given history (its last LookbackWindow values), taken to end where
    /// the training series ends.
    /// </summary>
    public override T PredictSingle(Vector<T> input)
    {
        if (_options.CovariateSize > 0)
            throw new InvalidOperationException("This DeepAR model has covariates; use PredictNextWithCovariates(...).");
        var series = input.Select(v => NumOps.ToDouble(v)).ToArray();
        return NextMedian(series, Math.Max(_trainingSeries.Length, series.Length), null, null);
    }

    /// <summary>
    /// The requested quantiles of the ForecastHorizon steps after the history, from NumSamples ancestral sample paths
    /// (the history is taken to end where the training series ends, which places the age feature).
    /// </summary>
    public Dictionary<double, Vector<T>> ForecastWithQuantiles(Vector<T> history, double[] quantiles)
    {
        Guard.NotNull(history);
        if (_options.CovariateSize > 0)
            throw new InvalidOperationException("This DeepAR model has covariates; use ForecastWithCovariates(...).");
        var series = history.Select(v => NumOps.ToDouble(v)).ToArray();
        int end = Math.Max(_trainingSeries.Length, series.Length);
        return Quantiles(Paths(series, end, null, null, _options.ForecastHorizon), quantiles);
    }

    /// <summary>
    /// The requested quantiles of the ForecastHorizon steps after the history, given the covariates of the history
    /// <c>[L, CovariateSize]</c> and of the forecast steps <c>[≥ ForecastHorizon, CovariateSize]</c>.
    /// </summary>
    public Dictionary<double, Vector<T>> ForecastWithCovariates(
        Vector<T> history, Matrix<T> historyCovariates, Matrix<T> futureCovariates, double[] quantiles)
    {
        int cov = _options.CovariateSize;
        if (cov <= 0)
            throw new InvalidOperationException("ForecastWithCovariates requires CovariateSize > 0; use ForecastWithQuantiles for the univariate model.");
        Guard.NotNull(history);
        Guard.NotNull(historyCovariates);
        Guard.NotNull(futureCovariates);
        int length = history.Length, horizon = _options.ForecastHorizon;
        if (historyCovariates.Rows != length || historyCovariates.Columns < cov)
            throw new ArgumentException($"historyCovariates must be [{length}, >={cov}]; got [{historyCovariates.Rows}, {historyCovariates.Columns}].");
        if (futureCovariates.Rows < horizon || futureCovariates.Columns < cov)
            throw new ArgumentException($"futureCovariates must be [>={horizon}, >={cov}]; got [{futureCovariates.Rows}, {futureCovariates.Columns}].");
        var series = history.Select(v => NumOps.ToDouble(v)).ToArray();
        int end = Math.Max(_trainingSeries.Length, series.Length);
        return Quantiles(Paths(series, end, (t, c) => NumOps.ToDouble(historyCovariates[t, c]),
            (h, c) => NumOps.ToDouble(futureCovariates[h, c]), horizon), quantiles);
    }

    /// <summary>
    /// The median forecast of the value after the history window, given its covariates <c>[L, CovariateSize]</c>; the
    /// next step's covariates are taken to equal the last row's.
    /// </summary>
    public T PredictNextWithCovariates(Vector<T> historyWindow, Matrix<T> historyCovariates)
    {
        int cov = _options.CovariateSize;
        if (cov <= 0)
            throw new InvalidOperationException("PredictNextWithCovariates requires CovariateSize > 0; use PredictSingle for the univariate model.");
        Guard.NotNull(historyWindow);
        Guard.NotNull(historyCovariates);
        int length = historyWindow.Length;
        if (historyCovariates.Rows != length || historyCovariates.Columns < cov)
            throw new ArgumentException($"historyCovariates must be [{length}, >={cov}]; got [{historyCovariates.Rows}, {historyCovariates.Columns}].");
        var series = historyWindow.Select(v => NumOps.ToDouble(v)).ToArray();
        int end = Math.Max(_trainingSeries.Length, series.Length);
        return NextMedian(series, end, (t, c) => NumOps.ToDouble(historyCovariates[t, c]),
            (_, c) => NumOps.ToDouble(historyCovariates[length - 1, c]));
    }

    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            Name = "DeepAR",
            Description = $"Probabilistic forecasting with autoregressive recurrent networks ({_likelihood} likelihood)",
            Complexity = ParameterCount,
            FeatureCount = _options.LookbackWindow,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "HiddenSize", _options.HiddenSize },
                { "NumLayers", _options.NumLayers },
                { "LikelihoodType", _likelihood.ToString() },
                { "ScaleHandling", _options.ScaleHandling.ToString() },
                { "ForecastHorizon", _options.ForecastHorizon },
            }
        };
    }

    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateInstance() => new DeepARModel<T>(new DeepAROptions<T>(_options));

    internal IReadOnlyList<double> TrainingLossHistory => _epochLosses.AsReadOnly();

    /// <inheritdoc />
    /// <remarks>The network's parameters are saved by the base through the registered layers; this adds the training
    /// series and the covariate and age standardization.</remarks>
    protected override void SerializeCore(BinaryWriter writer)
    {
        writer.Write(_trainingSeries.Length);
        for (int i = 0; i < _trainingSeries.Length; i++) writer.Write(NumOps.ToDouble(_trainingSeries[i]));
        writer.Write(_covariateMean.Length);
        for (int c = 0; c < _covariateMean.Length; c++)
        {
            writer.Write(_covariateMean[c]);
            writer.Write(_covariateStd[c]);
        }
        writer.Write(_ageMean);
        writer.Write(_ageStd);
    }

    /// <inheritdoc />
    protected override void DeserializeCore(BinaryReader reader)
    {
        int length = reader.ReadInt32();
        _trainingSeries = new Vector<T>(length);
        for (int i = 0; i < length; i++) _trainingSeries[i] = NumOps.FromDouble(reader.ReadDouble());
        int cov = reader.ReadInt32();
        _covariateMean = new double[cov];
        _covariateStd = new double[cov];
        for (int c = 0; c < cov; c++)
        {
            _covariateMean[c] = reader.ReadDouble();
            _covariateStd[c] = reader.ReadDouble();
        }
        _ageMean = reader.ReadDouble();
        _ageStd = reader.ReadDouble();
    }
}
