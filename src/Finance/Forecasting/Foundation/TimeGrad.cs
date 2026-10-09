using AiDotNet.LearningRateSchedulers;
using System.IO;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Finance.Interfaces;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Helpers;
using Microsoft.ML.OnnxRuntime;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

using AiDotNet.Finance.Base;
namespace AiDotNet.Finance.Forecasting.Foundation;

/// <summary>
/// TimeGrad — Autoregressive Denoising Diffusion Model for Time Series Forecasting.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// TimeGrad combines an autoregressive RNN with a conditional diffusion process for
/// probabilistic multi-step forecasting. It generates multiple forecast samples to
/// provide well-calibrated uncertainty estimates.
/// </para>
/// <para><b>For Beginners:</b> TimeGrad predicts time series step by step, where at each step
/// it uses a diffusion process to generate the next value. Think of it as a storyteller who
/// writes one sentence at a time, but for each sentence uses a careful drafting process to get
/// it right. By generating many possible futures, TimeGrad provides not just a single forecast
/// but a range of scenarios with probabilities, helping you understand how confident the
/// prediction is.</para>
/// <para>
/// <b>Reference:</b> Rasul et al., "Autoregressive Denoising Diffusion Models for Multivariate Probabilistic Time Series Forecasting", ICML 2021.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a TimeGrad autoregressive diffusion model for probabilistic forecasting
/// // Combines RNN with conditional diffusion for step-by-step forecast generation
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.Regression,
///     inputHeight: 512, inputWidth: 1, inputDepth: 1, outputSize: 24);
///
/// // Training mode with RNN encoder and denoising diffusion decoder
/// var model = new TimeGrad&lt;double&gt;(architecture);
///
/// // ONNX inference mode with pre-trained model
/// var onnxModel = new TimeGrad&lt;double&gt;(architecture, "timegrad.onnx");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Finance)]
[ModelDomain(ModelDomain.TimeSeries)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Diffusion)]
[ModelCategory(ModelCategory.RecurrentNetwork)]
[ModelTask(ModelTask.Forecasting)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Autoregressive Denoising Diffusion Models for Multivariate Probabilistic Time Series Forecasting", "https://arxiv.org/abs/2101.12072", Year = 2021, Authors = "Kashif Rasul, Calvin Seward, Ingmar Schuster, Roland Vollgraf")]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 1e-3, ReferenceBatchSize = 64,
                Source = "Rasul et al. 2021, Sec. 4: Adam with a learning rate of 1e-3 and batches of size 64.")]
public partial class TimeGrad<T> : TimeSeriesFoundationModelBase<T>
{
    #region Fields

    private readonly bool _useNativeMode;
    // The trainable graph; its layers are this model's Layers, bound by position before every forward.
    private TimeGradNetwork<T>? _network;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ILossFunction<T> _lossFunction;
    private readonly TimeGradOptions<T> _options;

    private int _contextLength;
    private int _forecastHorizon;
    private int _numDiffusionSteps;
    private int _numSamples;
    private int? _seed;

    // Seeds the training draws when no Seed is configured: fixed per instance, so the draw index alone
    // decides each step's (k, eps) and a gradient check re-evaluates the same objective.
    private readonly int _unseededDrawBase = RandomHelper.CreateSecureRandom().Next();

    private double[] _betas = Array.Empty<double>();
    private double[] _alphas = Array.Empty<double>();
    private double[] _alphasCumprod = Array.Empty<double>();
    private double[] _sqrtAlphasCumprod = Array.Empty<double>();
    private double[] _sqrtOneMinusAlphasCumprod = Array.Empty<double>();
    private double[] _posteriorVariance = Array.Empty<double>();

    // The univariate series is one target dimension (D = 1 in the paper's notation).
    private const int TargetDimension = 1;

    #endregion

    #region Properties

    /// <inheritdoc/>
    public override int SequenceLength => _contextLength;
    /// <inheritdoc/>
    public override int PredictionHorizon => _forecastHorizon;
    /// <inheritdoc/>
    public override int NumFeatures => 1;
    /// <inheritdoc/>
    public override int PatchSize => 1;
    /// <inheritdoc/>
    public override int Stride => 1;
    /// <inheritdoc/>
    public override bool IsChannelIndependent => false;
    /// <inheritdoc/>
    public override bool UseNativeMode => _useNativeMode;
    /// <inheritdoc/>
    public override FoundationModelSize ModelSize => FoundationModelSize.Small;
    /// <inheritdoc/>
    public override int MaxContextLength => _contextLength;
    /// <inheritdoc/>
    public override int MaxPredictionHorizon => _forecastHorizon;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a TimeGrad model using a pretrained ONNX model.
    /// </summary>
    public TimeGrad(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        TimeGradOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"ONNX model not found: {onnxModelPath}");

        options ??= new TimeGradOptions<T>();
        _options = options;
        Options = _options;

        _useNativeMode = false;
        OnnxModelPath = onnxModelPath;
        OnnxSession = new InferenceSession(onnxModelPath);

        _optimizer = optimizer
    ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this)
    ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();

        CopyOptionsToFields(options);
    }

    /// <summary>
    /// Creates a TimeGrad model in native mode for training.
    /// </summary>
    public TimeGrad(
        NeuralNetworkArchitecture<T> architecture,
        TimeGradOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        options ??= new TimeGradOptions<T>();
        _options = options;
        Options = _options;

        _useNativeMode = true;
        OnnxSession = null;
        OnnxModelPath = null;

        _optimizer = optimizer
    ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this)
    ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();

        CopyOptionsToFields(options);
        InitializeLayers();
    }

    private void CopyOptionsToFields(TimeGradOptions<T> options)
    {
        if (options.ContextLength <= 0) throw new ArgumentOutOfRangeException(nameof(options), "ContextLength must be positive.");
        if (options.ForecastHorizon <= 0) throw new ArgumentOutOfRangeException(nameof(options), "ForecastHorizon must be positive.");
        if (options.NumDiffusionSteps <= 0) throw new ArgumentOutOfRangeException(nameof(options), "NumDiffusionSteps must be positive.");
        _contextLength = options.ContextLength;
        _forecastHorizon = options.ForecastHorizon;
        _numDiffusionSteps = options.NumDiffusionSteps;
        _numSamples = Math.Max(1, options.NumSamples);
        _seed = options.Seed;
        ComputeNoiseSchedule(options);
    }

    /// <summary>
    /// The variance schedule beta_1..beta_N (linear from 1e-4 to 0.1 over N = 100 in the paper) and the quantities
    /// the forward process and the sampler read from it, including the posterior variance
    /// beta~_k = (1 - alphaBar_{k-1}) / (1 - alphaBar_k) beta_k used by the reverse step.
    /// </summary>
    private void ComputeNoiseSchedule(TimeGradOptions<T> options)
    {
        int n = _numDiffusionSteps;
        double start = options.BetaStart, end = options.BetaEnd;
        if (double.IsNaN(start) || double.IsNaN(end) || start <= 0 || end <= 0 || end >= 1 || start > end)
            throw new ArgumentOutOfRangeException(nameof(options), "BetaStart and BetaEnd must satisfy 0 < BetaStart <= BetaEnd < 1.");

        _betas = new double[n];
        for (int k = 0; k < n; k++)
        {
            double fraction = n > 1 ? (double)k / (n - 1) : 0.0;
            _betas[k] = options.BetaSchedule switch
            {
                AiDotNet.Enums.BetaSchedule.Linear => start + (end - start) * fraction,
                AiDotNet.Enums.BetaSchedule.ScaledLinear => Math.Pow(Math.Sqrt(start) + (Math.Sqrt(end) - Math.Sqrt(start)) * fraction, 2),
                AiDotNet.Enums.BetaSchedule.SquaredCosine => SquaredCosineBeta(k, n),
                _ => throw new ArgumentOutOfRangeException(nameof(options), $"Unknown beta schedule {options.BetaSchedule}.")
            };
        }

        _alphas = new double[n];
        _alphasCumprod = new double[n];
        _sqrtAlphasCumprod = new double[n];
        _sqrtOneMinusAlphasCumprod = new double[n];
        _posteriorVariance = new double[n];
        double cumulative = 1.0;
        for (int k = 0; k < n; k++)
        {
            _alphas[k] = 1.0 - _betas[k];
            double previous = cumulative;
            cumulative *= _alphas[k];
            _alphasCumprod[k] = cumulative;
            _sqrtAlphasCumprod[k] = Math.Sqrt(cumulative);
            _sqrtOneMinusAlphasCumprod[k] = Math.Sqrt(1.0 - cumulative);
            _posteriorVariance[k] = k == 0 ? 0.0 : (1.0 - previous) / (1.0 - cumulative) * _betas[k];
        }
    }

    // Nichol & Dhariwal 2021, s = 0.008, clipped at 0.999.
    private static double SquaredCosineBeta(int k, int n)
    {
        static double AlphaBar(double t) => Math.Pow(Math.Cos((t + 0.008) / 1.008 * Math.PI / 2), 2);
        return Math.Min(1.0 - AlphaBar((k + 1.0) / n) / AlphaBar((double)k / n), 0.999);
    }

    #endregion

    #region Initialization

    /// <summary>
    /// Publishes TimeGrad's graph through Layers. Caller-supplied layers are bound to the same roles by position;
    /// the forward is not a sequential chain, so a list that does not match the layout is refused.
    /// </summary>
    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        var network = new TimeGradNetwork<T>(
            _options.NumRnnLayers, _options.HiddenDimension, _options.DropoutRate, TargetDimension,
            _options.ResidualLayers, _options.ResidualChannels, _options.DilationCycleLength,
            _options.TimeEmbeddingDim, _options.DenoisingNetworkDim);
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            network.BindTo(Architecture.Layers);
            Layers.AddRange(Architecture.Layers);
        }
        else
        {
            Layers.AddRange(network.Layers);
        }

        _network = network;
    }

    private bool _lazyShapesProbed;

    // True while the probe runs: toggling training mode builds the parameter layout, which calls back here.
    private bool _lazyShapesProbing;

    /// <inheritdoc/>
    /// <remarks>
    /// TimeGrad's graph is not a sequential chain (an RNN feeding a conditioned denoiser over a different axis), so
    /// the base walk would size the LSTM from the context length. Resolve through the real training forward with
    /// one zero row instead.
    /// </remarks>
    protected override void ResolveLazyLayerShapes()
    {
        if (!_useNativeMode || _lazyShapesProbed || _lazyShapesProbing) return;
        _lazyShapesProbing = true;
        bool wasTraining = IsTrainingMode;
        try
        {
            if (wasTraining) SetTrainingMode(false);
            using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
            _ = ForwardForTraining(new Tensor<T>(new[] { 1, PackedWidth }));
            _lazyShapesProbed = true;
        }
        finally
        {
            try
            {
                if (wasTraining) SetTrainingMode(true);
            }
            finally
            {
                _lazyShapesProbing = false;
            }
        }
    }

    /// <inheritdoc/>
    /// <remarks>
    /// TimeGrad's layers are not a sequential chain, so the family's fold over Layers would push the series into the
    /// denoiser's convolutions. One training forward on a target-less pair runs every layer exactly once, in the
    /// order the model uses it, and that is what is recorded.
    /// </remarks>
    public override Dictionary<string, Tensor<T>> GetNamedLayerActivations(Tensor<T> input)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        var activations = new Dictionary<string, Tensor<T>>();
        if (!_useNativeMode) return activations;

        var pair = PrepareTrainingPair(input, null, 0);
        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        using var trace = new AiDotNet.NeuralNetworks.Graph.LayerForwardObserver<T>();
        _ = ForwardForTraining(pair.Input);
        foreach (var (layer, _, output) in trace.Calls)
        {
            if (output is null) continue;
            int index = Layers.IndexOf(layer);
            if (index < 0) continue;
            activations[$"Layer_{index}_{layer.GetType().Name}"] = output.Clone();
        }

        return activations;
    }

    private TimeGradNetwork<T> BoundNetwork()
    {
        var network = _network ?? throw new InvalidOperationException("TimeGrad has no native network in ONNX mode.");
        network.BindTo(Layers);
        return network;
    }

    #endregion

    #region NeuralNetworkBase Overrides

    /// <inheritdoc/>
    public override bool SupportsTraining => _useNativeMode;

    /// <inheritdoc/>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        return _useNativeMode ? ForwardNative(input) : ForecastOnnx(input);
    }

    // Width of a training row: the teacher-forced sequence, then x_k and k per future step.
    private int TeacherForcedLength => _contextLength + _forecastHorizon - 1;
    private int PackedWidth => TeacherForcedLength + 2 * _forecastHorizon;

    /// <summary>
    /// Draws the diffusion training pair (Rasul et al. 2021, Algorithm 1). For every future step t of every series,
    /// a diffusion step k ~ U{1..N} and noise eps ~ N(0, I) give x_k = sqrt(alphaBar_k) x_t + sqrt(1 - alphaBar_k) eps.
    /// The packed row holds the mean-scaled series the RNN reads under teacher forcing (the context, then the true
    /// future up to t - 1), every x_k, and every k; the target is eps.
    /// </summary>
    protected override (Tensor<T> Input, Tensor<T>? Target) PrepareTrainingPair(Tensor<T> input, Tensor<T>? target, long draw)
    {
        if (!_useNativeMode) return (input, target);
        var context = ContextRows(input, out int batch);
        var future = new double[batch, _forecastHorizon];
        if (target is not null)
        {
            if (target.Length != batch * _forecastHorizon)
                throw new ArgumentException(
                    $"TimeGrad's target holds {target.Length} values; [{batch}, {_forecastHorizon}] needs {batch * _forecastHorizon}.",
                    nameof(target));
            for (int b = 0; b < batch; b++)
                for (int t = 0; t < _forecastHorizon; t++)
                    future[b, t] = NumOps.ToDouble(target[b * _forecastHorizon + t]);
        }

        var random = RandomHelper.CreateSeededRandom(DrawSeed(draw));
        var packed = new Tensor<T>(new[] { batch, PackedWidth });
        var noise = new Tensor<T>(new[] { batch, _forecastHorizon });
        for (int b = 0; b < batch; b++)
        {
            double scale = MeanScale(context, b);
            int row = b * PackedWidth;
            for (int t = 0; t < _contextLength; t++)
                packed[row + t] = NumOps.FromDouble(context[b, t] / scale);
            for (int t = 0; t < _forecastHorizon - 1; t++)
                packed[row + _contextLength + t] = NumOps.FromDouble(future[b, t] / scale);
            for (int t = 0; t < _forecastHorizon; t++)
            {
                int k = random.Next(_numDiffusionSteps);
                double eps = StandardNormal(random);
                double noisy = _sqrtAlphasCumprod[k] * future[b, t] / scale + _sqrtOneMinusAlphasCumprod[k] * eps;
                packed[row + TeacherForcedLength + t] = NumOps.FromDouble(noisy);
                packed[row + TeacherForcedLength + _forecastHorizon + t] = NumOps.FromDouble(k);
                noise[b * _forecastHorizon + t] = NumOps.FromDouble(eps);
            }
        }

        return (packed, noise);
    }

    /// <summary>
    /// epsilon_theta(x_k, h_{t-1}, k) for every future step of a pair from <see cref="PrepareTrainingPair"/>, as
    /// <c>[B, ForecastHorizon]</c>: the RNN reads the teacher-forced sequence and its state before each future step
    /// conditions the denoiser. Every operation is recorded, so a compiled replay recomputes it from the replayed row.
    /// </summary>
    public override Tensor<T> ForwardForTraining(Tensor<T> input)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Training is only supported in native mode.");
        if (input.Rank != 2 || input.Shape[1] != PackedWidth)
            throw new ArgumentException(
                $"TimeGrad trains on rows prepared by {nameof(PrepareTrainingPair)}: [B, {PackedWidth}] " +
                $"(sequence {TeacherForcedLength}, then x_k and k for {_forecastHorizon} steps); got [{string.Join(", ", input.Shape.ToArray())}].",
                nameof(input));

        var network = BoundNetwork();
        int batch = input.Shape[0];
        int rows = batch * _forecastHorizon;
        var sequence = Engine.Reshape(Engine.TensorNarrow(input, 1, 0, TeacherForcedLength), new[] { batch, TeacherForcedLength, TargetDimension });
        var hidden = network.EncodeHistory(sequence);
        int hiddenSize = hidden.Shape[2];
        // The state after reading step t - 1 (sequence position contextLength - 1 + t) conditions step t.
        var condition = Engine.Reshape(
            Engine.TensorNarrow(hidden, 1, _contextLength - 1, _forecastHorizon), new[] { rows, hiddenSize });
        var noisy = Engine.Reshape(Engine.TensorNarrow(input, 1, TeacherForcedLength, _forecastHorizon), new[] { rows, TargetDimension });
        var steps = Engine.Reshape(Engine.TensorNarrow(input, 1, TeacherForcedLength + _forecastHorizon, _forecastHorizon), new[] { rows, 1 });
        var predicted = network.PredictNoise(Engine, noisy, condition, network.StepEmbedding(Engine, steps));
        return Engine.Reshape(predicted, new[] { batch, _forecastHorizon });
    }
    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NetworkType", "TimeGrad" },
                { "ContextLength", _contextLength },
                { "ForecastHorizon", _forecastHorizon },
                { "HiddenDimension", _options.HiddenDimension },
                { "NumRnnLayers", _options.NumRnnLayers },
                { "NumDiffusionSteps", _numDiffusionSteps },
                { "ResidualLayers", _options.ResidualLayers },
                { "ResidualChannels", _options.ResidualChannels },
                { "NumSamples", _numSamples },
                { "UseNativeMode", _useNativeMode }
            },
            ModelDataProvider = () => _useNativeMode ? this.Serialize() : Array.Empty<byte>()
        };
    }

    #endregion

    #region IForecastingModel Implementation

    /// <inheritdoc/>
    public override Tensor<T> Forecast(Tensor<T> historicalData, double[]? quantiles = null)
    {
        if (quantiles is not null && quantiles.Length > 0)
        {
            if (!_useNativeMode)
                throw new NotSupportedException("Quantile forecasts sample the native model; an ONNX TimeGrad returns its point forecast only.");
            return ForecastQuantiles(historicalData, quantiles);
        }

        return _useNativeMode ? ForwardNative(historicalData) : ForecastOnnx(historicalData);
    }

    /// <inheritdoc/>
    public override Tensor<T> AutoregressiveForecast(Tensor<T> input, int steps)
    {
        // TimeGrad is inherently autoregressive (step-by-step diffusion)
        var predictions = new List<Tensor<T>>();
        var currentInput = input;
        int stepsRemaining = steps;

        while (stepsRemaining > 0)
        {
            var prediction = Forecast(currentInput, null);
            predictions.Add(prediction);
            int stepsUsed = Math.Min(_forecastHorizon, stepsRemaining);
            stepsRemaining -= stepsUsed;

            if (stepsRemaining > 0)
                currentInput = ShiftInputWithPredictions(currentInput, prediction, stepsUsed);
        }

        return ConcatenatePredictions(predictions, steps);
    }

    /// <inheritdoc/>
    public override Dictionary<string, T> Evaluate(Tensor<T> predictions, Tensor<T> actuals)
    {
        var metrics = new Dictionary<string, T>();
        T mse = NumOps.Zero;
        T mae = NumOps.Zero;
        int count = 0;

        for (int i = 0; i < predictions.Length && i < actuals.Length; i++)
        {
            var diff = NumOps.Subtract(predictions[i], actuals[i]);
            mse = NumOps.Add(mse, NumOps.Multiply(diff, diff));
            mae = NumOps.Add(mae, NumOps.Abs(diff));
            count++;
        }

        if (count > 0)
        {
            mse = NumOps.Divide(mse, NumOps.FromDouble(count));
            mae = NumOps.Divide(mae, NumOps.FromDouble(count));
        }

        metrics["MSE"] = mse;
        metrics["MAE"] = mae;
        metrics["RMSE"] = NumOps.Sqrt(mse);
        return metrics;
    }

    /// <inheritdoc/>
    public override Tensor<T> ApplyInstanceNormalization(Tensor<T> input)
        // RevIN forward (Kim et al. 2022), delegated to the shared tape-tracked helper. The previous
        // hand-rolled version accumulated mean/variance with scalar NumOps arithmetic and wrote the
        // output through result.Data.Span[...], which the autodiff tape cannot observe: the normalised
        // tensor came back as a LEAF, so no gradient could flow through the normalisation. RevIN is a
        // differentiable layer in the paper, not a preprocessing step.
        => NormalizeInstanceOnTape(input, DefaultRevInEpsilon, out _, out _);

    /// <inheritdoc/>
    public override Dictionary<string, T> GetFinancialMetrics()
    {
        T lastLoss = LastLoss is not null ? LastLoss : NumOps.Zero;
        return new Dictionary<string, T>
        {
            ["ContextLength"] = NumOps.FromDouble(_contextLength),
            ["ForecastHorizon"] = NumOps.FromDouble(_forecastHorizon),
            ["HiddenDimension"] = NumOps.FromDouble(_options.HiddenDimension),
            ["NumDiffusionSteps"] = NumOps.FromDouble(_numDiffusionSteps),
            ["LastLoss"] = lastLoss
        };
    }

    #endregion

    #region Forward/Backward Pass

    /// <summary>The point forecast: the mean of <see cref="TimeGradOptions{T}.NumSamples"/> sampled paths.</summary>
    private Tensor<T> ForwardNative(Tensor<T> input)
    {
        var paths = SampleForecasts(input, out int batch);
        var mean = new Tensor<T>(new[] { batch, _forecastHorizon });
        for (int b = 0; b < batch; b++)
            for (int t = 0; t < _forecastHorizon; t++)
            {
                double sum = 0;
                for (int s = 0; s < _numSamples; s++) sum += paths[s, b, t];
                mean[b * _forecastHorizon + t] = NumOps.FromDouble(sum / _numSamples);
            }

        return input.Rank == 1 ? Engine.Reshape(mean, new[] { _forecastHorizon }) : mean;
    }

    /// <summary>Quantiles of the sampled paths, <c>[batch, ForecastHorizon, quantiles]</c>.</summary>
    private Tensor<T> ForecastQuantiles(Tensor<T> input, double[] quantiles)
    {
        foreach (double q in quantiles)
            if (double.IsNaN(q) || q < 0 || q > 1)
                throw new ArgumentOutOfRangeException(nameof(quantiles), $"Quantile {q} is outside [0, 1].");
        var paths = SampleForecasts(input, out int batch);
        var result = new Tensor<T>(new[] { batch, _forecastHorizon, quantiles.Length });
        var values = new double[_numSamples];
        for (int b = 0; b < batch; b++)
            for (int t = 0; t < _forecastHorizon; t++)
            {
                for (int s = 0; s < _numSamples; s++) values[s] = paths[s, b, t];
                Array.Sort(values);
                for (int q = 0; q < quantiles.Length; q++)
                {
                    double position = quantiles[q] * (_numSamples - 1);
                    int lower = (int)Math.Floor(position);
                    int upper = Math.Min(lower + 1, _numSamples - 1);
                    double value = values[lower] + (position - lower) * (values[upper] - values[lower]);
                    result[(b * _forecastHorizon + t) * quantiles.Length + q] = NumOps.FromDouble(value);
                }
            }

        return result;
    }

    /// <summary>
    /// Samples forecast paths (Rasul et al. 2021, Algorithm 2), <c>[samples, batch, ForecastHorizon]</c> in the
    /// series' own scale. Autoregressive over the horizon: the RNN reads the context plus the values sampled so far,
    /// and each step runs the reverse chain x_{k-1} = (x_k - beta_k / sqrt(1 - alphaBar_k) eps_theta) / sqrt(alpha_k)
    /// + sqrt(beta~_k) z from x_N ~ N(0, I). Paths of every series run as one batch. Seeded from Seed, so a seeded
    /// model forecasts reproducibly.
    /// </summary>
    private double[,,] SampleForecasts(Tensor<T> input, out int batch)
    {
        var network = BoundNetwork();
        var context = ContextRows(input, out batch);
        int paths = batch * _numSamples;
        var scales = new double[batch];
        for (int b = 0; b < batch; b++) scales[b] = MeanScale(context, b);

        int fullLength = _contextLength + _forecastHorizon;
        var values = new double[paths, fullLength];
        for (int p = 0; p < paths; p++)
        {
            int b = p % batch;
            for (int t = 0; t < _contextLength; t++) values[p, t] = context[b, t] / scales[b];
        }

        var random = _seed.HasValue ? RandomHelper.CreateSeededRandom(_seed.Value) : RandomHelper.CreateSecureRandom();
        bool wasTraining = IsTrainingMode;
        if (wasTraining) SetTrainingMode(false);
        try
        {
            using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
            var x = new double[paths];
            var hiddenStates = new Tensor<T>?[network.RecurrentLayerCount];
            var cellStates = new Tensor<T>?[network.RecurrentLayerCount];
            var contextSteps = new Tensor<T>(new[] { paths, _contextLength, TargetDimension });
            for (int p = 0; p < paths; p++)
                for (int i = 0; i < _contextLength; i++)
                    contextSteps[p * _contextLength + i] = NumOps.FromDouble(values[p, i]);
            // The RNN reads the context once; each sampled value then advances it by one step.
            var condition = network.AdvanceHistory(Engine, contextSteps, hiddenStates, cellStates);
            for (int t = 0; t < _forecastHorizon; t++)
            {
                int length = _contextLength + t;
                if (t > 0)
                {
                    var previous = new Tensor<T>(new[] { paths, 1, TargetDimension });
                    for (int p = 0; p < paths; p++) previous[p] = NumOps.FromDouble(values[p, length - 1]);
                    condition = network.AdvanceHistory(Engine, previous, hiddenStates, cellStates);
                }

                for (int p = 0; p < paths; p++) x[p] = StandardNormal(random);
                for (int k = _numDiffusionSteps - 1; k >= 0; k--)
                {
                    var noisy = new Tensor<T>(new[] { paths, TargetDimension });
                    var steps = new Tensor<T>(new[] { paths, 1 });
                    T stepValue = NumOps.FromDouble(k);
                    for (int p = 0; p < paths; p++)
                    {
                        noisy[p] = NumOps.FromDouble(x[p]);
                        steps[p] = stepValue;
                    }

                    var eps = network.PredictNoise(Engine, noisy, condition, network.StepEmbedding(Engine, steps));
                    double noiseCoefficient = _betas[k] / _sqrtOneMinusAlphasCumprod[k];
                    double inverseSqrtAlpha = 1.0 / Math.Sqrt(_alphas[k]);
                    double sigma = Math.Sqrt(_posteriorVariance[k]);
                    for (int p = 0; p < paths; p++)
                    {
                        double mean = inverseSqrtAlpha * (x[p] - noiseCoefficient * NumOps.ToDouble(eps[p]));
                        x[p] = k > 0 ? mean + sigma * StandardNormal(random) : mean;
                    }
                }

                for (int p = 0; p < paths; p++) values[p, length] = x[p];
            }
        }
        finally
        {
            if (wasTraining) SetTrainingMode(true);
        }

        var result = new double[_numSamples, batch, _forecastHorizon];
        for (int p = 0; p < paths; p++)
        {
            int s = p / batch, b = p % batch;
            for (int t = 0; t < _forecastHorizon; t++)
                result[s, b, t] = values[p, _contextLength + t] * scales[b];
        }

        return result;
    }

    /// <summary>
    /// The context as rows <c>[batch, ContextLength]</c> from <c>[ContextLength]</c>, <c>[B, ContextLength]</c> or
    /// <c>[B, ContextLength, 1]</c>; a longer history keeps its most recent ContextLength steps.
    /// </summary>
    private double[,] ContextRows(Tensor<T> input, out int batch)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        int length;
        switch (input.Rank)
        {
            case 1: batch = 1; length = input.Shape[0]; break;
            case 2: batch = input.Shape[0]; length = input.Shape[1]; break;
            case 3 when input.Shape[2] == TargetDimension: batch = input.Shape[0]; length = input.Shape[1]; break;
            default:
                throw new ArgumentException(
                    $"TimeGrad takes a univariate history [T], [B, T] or [B, T, 1]; got [{string.Join(", ", input.Shape.ToArray())}].",
                    nameof(input));
        }

        if (length < _contextLength)
            throw new ArgumentException($"TimeGrad needs at least {_contextLength} past steps; got {length}.", nameof(input));
        var rows = new double[batch, _contextLength];
        int offset = length - _contextLength;
        for (int b = 0; b < batch; b++)
            for (int t = 0; t < _contextLength; t++)
                rows[b, t] = NumOps.ToDouble(input[b * length + offset + t]);
        return rows;
    }

    /// <summary>
    /// The reference implementation's mean scaler: each series is divided by the mean absolute value of its context
    /// (1 when that is zero), so one network serves series of any magnitude.
    /// </summary>
    private double MeanScale(double[,] context, int row)
    {
        double sum = 0;
        for (int t = 0; t < _contextLength; t++) sum += Math.Abs(context[row, t]);
        double scale = sum / _contextLength;
        return scale > 1e-10 && !double.IsInfinity(scale) ? scale : 1.0;
    }

    private int DrawSeed(long draw)
    {
        unchecked
        {
            ulong mixed = (ulong)(uint)(_seed ?? _unseededDrawBase) * 0x9E3779B97F4A7C15UL ^ (ulong)draw * 0xBF58476D1CE4E5B9UL;
            mixed ^= mixed >> 31;
            mixed *= 0x94D049BB133111EBUL;
            mixed ^= mixed >> 29;
            return (int)(mixed & 0x7FFFFFFF);
        }
    }

    private static double StandardNormal(Random random)
    {
        double u1 = 1.0 - random.NextDouble();
        double u2 = 1.0 - random.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }

    protected override Tensor<T> ForecastOnnx(Tensor<T> input)
    {
        if (OnnxSession == null)
            throw new InvalidOperationException("ONNX session is not initialized.");

        int batchSize = input.Rank > 1 ? input.Shape[0] : 1;
        int seqLen = input.Rank > 1 ? input.Shape[1] : input.Length;
        int features = input.Rank > 2 ? input.Shape[2] : 1;

        var inputData = new float[batchSize * seqLen * features];
        for (int i = 0; i < input.Length && i < inputData.Length; i++)
            inputData[i] = (float)NumOps.ToDouble(input[i]);

        var inputTensor = new OnnxTensors.DenseTensor<float>(
            inputData, new[] { batchSize, seqLen, features });

        string inputName = OnnxSession.InputMetadata.Keys.FirstOrDefault() ?? "input";
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(inputName, inputTensor)
        };

        using var results = OnnxSession.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();

        var outputShape = outputTensor.Dimensions.ToArray();
        var output = new Tensor<T>(outputShape);

        int totalElements = 1;
        foreach (var dim in outputShape) totalElements *= dim;

        for (int i = 0; i < totalElements && i < output.Length; i++)
            output.Data.Span[i] = NumOps.FromDouble(outputTensor.GetValue(i));

        return output;
    }

    #endregion
}
