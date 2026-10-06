using AiDotNet.LearningRateSchedulers;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Finance.Interfaces;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Helpers;
using Microsoft.ML.OnnxRuntime;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

using AiDotNet.Finance.Base;
namespace AiDotNet.Finance.Probabilistic;

/// <summary>
/// DiffusionTS (Interpretable Diffusion for Time Series) for probabilistic forecasting with seasonal-trend decomposition.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para>
/// DiffusionTS is an interpretable diffusion model that uses seasonal-trend decomposition
/// to generate forecasts with clear interpretable components.
/// </para>
/// <para><b>For Beginners:</b> DiffusionTS makes diffusion models more interpretable
/// by decomposing time series into understandable components:
///
/// <b>The Key Insight:</b>
/// Time series often have clear structure (trends, seasonality) that gets lost in
/// "black box" models. DiffusionTS preserves this structure by generating each
/// component separately and combining them.
///
/// <b>How DiffusionTS Works:</b>
/// 1. <b>Window:</b> The history and the horizon form one window, standardised per series
/// 2. <b>Denoiser:</b> A transformer predicts the clean window from a noisy one and the noise level
/// 3. <b>Decomposition:</b> Its decoder writes each prediction as a trend (a low-order polynomial)
///    plus seasonality (the strongest Fourier components) plus a residual
/// 4. <b>Forecasting:</b> Starting from noise, each denoising step keeps the observed history
///    in place, so the horizon is generated to continue it
///
/// <b>Training:</b> The model learns to recover the clean window, with an extra loss on its
/// Fourier coefficients so that periodic structure is learned explicitly.
///
/// <b>Key Benefits:</b>
/// - Interpretable decomposition of forecasts
/// - Can enforce structural constraints (smooth trends, periodic seasons)
/// - Better uncertainty quantification per component
/// - Enables "what-if" analysis by modifying components
/// </para>
/// <para>
/// <b>Reference:</b> Yuan and Qiao, "Diffusion-TS: Interpretable Diffusion for General Time Series Generation", 2024.
/// https://arxiv.org/abs/2403.01742
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a Diffusion-TS model for interpretable time series generation
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.Regression,
///     inputHeight: 100, inputWidth: 1, inputDepth: 1, outputSize: 24);
/// var model = new DiffusionTS&lt;double&gt;(architecture);
///
/// // Or load a pre-trained ONNX model for diffusion time series generation
/// var onnxModel = new DiffusionTS&lt;double&gt;(architecture, "diffusionts.onnx");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Finance)]
[ModelDomain(ModelDomain.TimeSeries)]
[ModelCategory(ModelCategory.Diffusion)]
[ModelTask(ModelTask.Forecasting)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.VeryHigh)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Diffusion-TS: Interpretable Diffusion for General Time Series Generation", "https://arxiv.org/abs/2403.01742", Year = 2024, Authors = "Xinyu Yuan, Yan Qiao")]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 0.0008, Beta1 = 0.9, Beta2 = 0.96,
                WarmupSteps = 500, Schedule = LearningRateSchedulerType.LinearWarmup,
                Source = "Yuan and Qiao 2024, Sec. 4: the network is optimized with Adam at betas 0.9 "
                        + "and 0.96, under a linearly decaying learning rate that starts at 0.0008 after "
                        + "500 iterations of warmup. The cosine schedule named alongside is the "
                        + "diffusion noise schedule, not the learning rate.")]
public partial class DiffusionTS<T> : ForecastingModelBase<T>
{
    #region Fields

    private readonly bool _useNativeMode;
    // The denoiser; its layers are this model's Layers, bound by position before every forward.
    private DiffusionTSNetwork<T>? _network;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ILossFunction<T> _lossFunction;
    private readonly DiffusionTSOptions<T> _options;

    private readonly int _sequenceLength;
    private readonly int _forecastHorizon;
    private readonly int _numFeatures;
    private readonly int _numDiffusionSteps;
    private readonly int _numSamples;
    private readonly int? _seed;

    // Seeds the training draws when no Seed is configured, fixed per instance (see PrepareTrainingPair).
    private readonly int _unseededDrawBase = RandomHelper.CreateSecureRandom().Next();

    private double[] _betas = Array.Empty<double>();
    private double[] _alphasCumprod = Array.Empty<double>();
    private double[] _sqrtAlphasCumprod = Array.Empty<double>();
    private double[] _sqrtOneMinusAlphasCumprod = Array.Empty<double>();
    private double[] _posteriorMeanClean = Array.Empty<double>();
    private double[] _posteriorMeanNoisy = Array.Empty<double>();
    private double[] _posteriorVariance = Array.Empty<double>();
    private double[] _lossWeight = Array.Empty<double>();

    private bool _lazyShapesProbed;
    // True while the probe runs: toggling training mode builds the parameter layout, which calls back into the probe.
    private bool _lazyShapesProbing;

    #endregion

    #region IForecastingModel Properties

    /// <inheritdoc/>
    public override int SequenceLength => _sequenceLength;

    /// <inheritdoc/>
    public override int PredictionHorizon => _forecastHorizon;

    /// <inheritdoc/>
    public override int NumFeatures => _numFeatures;

    /// <inheritdoc/>
    public override int PatchSize => 1;

    /// <inheritdoc/>
    public override int Stride => 1;

    /// <inheritdoc/>
    public override bool IsChannelIndependent => false;

    /// <inheritdoc/>
    public override bool UseNativeMode => _useNativeMode;

    /// <summary>Gets the number of future steps forecast.</summary>
    public int ForecastHorizon => _forecastHorizon;

    /// <inheritdoc/>
    public override bool SupportsTraining => _useNativeMode;

    /// <summary>Gets the number of diffusion steps T.</summary>
    public int NumDiffusionSteps => _numDiffusionSteps;

    /// <summary>Gets how many generated windows a forecast averages.</summary>
    public int NumSamples => _numSamples;

    // The model generates context and horizon as one window.
    private int Window => _sequenceLength + _forecastHorizon;
    private int WindowValues => Window * _numFeatures;

    #endregion

    #region Constructors

    /// <summary>Creates a DiffusionTS model that runs a pretrained ONNX graph.</summary>
    public DiffusionTS(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        DiffusionTSOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? DefaultLoss(options), 1.0)
    {
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentNullException(nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"ONNX model not found: {onnxModelPath}");

        _useNativeMode = false;
        OnnxModelPath = onnxModelPath;
        OnnxSession = new InferenceSession(onnxModelPath);
        _options = options ?? new DiffusionTSOptions<T>();
        Options = _options;
        _lossFunction = lossFunction ?? DefaultLoss(_options);
        _optimizer = optimizer
            ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this)
            ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        (_sequenceLength, _forecastHorizon, _numFeatures, _numDiffusionSteps, _numSamples, _seed) = ReadSizes(_options);
        ComputeNoiseSchedule();
    }

    /// <summary>Creates a native DiffusionTS model that can be trained.</summary>
    public DiffusionTS(
        NeuralNetworkArchitecture<T> architecture,
        DiffusionTSOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? DefaultLoss(options), 1.0)
    {
        _useNativeMode = true;
        _options = options ?? new DiffusionTSOptions<T>();
        Options = _options;
        _lossFunction = lossFunction ?? DefaultLoss(_options);
        _optimizer = optimizer
            ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this)
            ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        (_sequenceLength, _forecastHorizon, _numFeatures, _numDiffusionSteps, _numSamples, _seed) = ReadSizes(_options);
        ComputeNoiseSchedule();
        InitializeLayers();
    }

    // The reference trains with L1 unless configured for L2; PrepareTrainingPair scales both sides to match.
    private static ILossFunction<T> DefaultLoss(DiffusionTSOptions<T>? options)
        => (options?.ReconstructionLoss ?? DiffusionReconstructionLoss.L1) == DiffusionReconstructionLoss.L2
            ? new MeanSquaredErrorLoss<T>()
            : new MeanAbsoluteErrorLoss<T>();

    private static (int, int, int, int, int, int?) ReadSizes(DiffusionTSOptions<T> options)
    {
        if (options.SequenceLength <= 0) throw new ArgumentOutOfRangeException(nameof(options), "SequenceLength must be positive.");
        if (options.ForecastHorizon <= 0) throw new ArgumentOutOfRangeException(nameof(options), "ForecastHorizon must be positive.");
        if (options.NumFeatures <= 0) throw new ArgumentOutOfRangeException(nameof(options), "NumFeatures must be positive.");
        if (options.NumDiffusionSteps <= 0) throw new ArgumentOutOfRangeException(nameof(options), "NumDiffusionSteps must be positive.");
        return (options.SequenceLength, options.ForecastHorizon, options.NumFeatures, options.NumDiffusionSteps,
            Math.Max(1, options.NumSamples), options.Seed);
    }

    #endregion

    #region Diffusion Schedule

    /// <summary>
    /// The variance schedule (cosine in the paper) and what the forward process, the x_0-parameterised reverse step
    /// and the reference's per-step loss reweighting read from it.
    /// </summary>
    private void ComputeNoiseSchedule()
    {
        int n = _numDiffusionSteps;
        double start = _options.BetaStart, end = _options.BetaEnd;
        _betas = new double[n];
        for (int k = 0; k < n; k++)
        {
            double fraction = n > 1 ? (double)k / (n - 1) : 0.0;
            _betas[k] = _options.BetaSchedule switch
            {
                BetaSchedule.Linear => start + (end - start) * fraction,
                BetaSchedule.ScaledLinear => Math.Pow(Math.Sqrt(start) + (Math.Sqrt(end) - Math.Sqrt(start)) * fraction, 2),
                BetaSchedule.SquaredCosine => Math.Min(1.0 - CosineAlphaBar((k + 1.0) / n) / CosineAlphaBar((double)k / n), 0.999),
                _ => throw new ArgumentOutOfRangeException(nameof(_options), $"Unknown beta schedule {_options.BetaSchedule}.")
            };
            if (double.IsNaN(_betas[k]) || _betas[k] <= 0 || _betas[k] >= 1)
                throw new ArgumentOutOfRangeException(nameof(_options), "Every beta must lie in (0, 1).");
        }

        _alphasCumprod = new double[n];
        _sqrtAlphasCumprod = new double[n];
        _sqrtOneMinusAlphasCumprod = new double[n];
        _posteriorMeanClean = new double[n];
        _posteriorMeanNoisy = new double[n];
        _posteriorVariance = new double[n];
        _lossWeight = new double[n];
        double cumulative = 1.0;
        for (int k = 0; k < n; k++)
        {
            double previous = cumulative;
            double alpha = 1.0 - _betas[k];
            cumulative *= alpha;
            _alphasCumprod[k] = cumulative;
            _sqrtAlphasCumprod[k] = Math.Sqrt(cumulative);
            _sqrtOneMinusAlphasCumprod[k] = Math.Sqrt(1.0 - cumulative);
            // q(x_{k-1} | x_k, x_0): mean = c0 x_0 + ck x_k, variance beta~_k (Ho et al. 2020, eq. 7).
            _posteriorMeanClean[k] = _betas[k] * Math.Sqrt(previous) / (1.0 - cumulative);
            _posteriorMeanNoisy[k] = (1.0 - previous) * Math.Sqrt(alpha) / (1.0 - cumulative);
            _posteriorVariance[k] = _betas[k] * (1.0 - previous) / (1.0 - cumulative);
            // The reference's loss_weight: sqrt(alpha_k) sqrt(1 - alphaBar_k) / beta_k / 100.
            _lossWeight[k] = Math.Sqrt(alpha) * Math.Sqrt(1.0 - cumulative) / _betas[k] / 100.0;
        }
    }

    // Nichol & Dhariwal 2021, s = 0.008.
    private static double CosineAlphaBar(double t) => Math.Pow(Math.Cos((t + 0.008) / 1.008 * Math.PI / 2), 2);

    #endregion

    #region Initialization

    /// <summary>
    /// Publishes the denoiser through Layers. Caller-supplied layers are bound to the same roles by position; the
    /// forward is not a sequential chain, so a list that does not match the layout is refused.
    /// </summary>
    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        var network = new DiffusionTSNetwork<T>(
            _numFeatures, Window, _options.HiddenDimension, _options.NumHeads, _options.NumEncoderLayers,
            _options.NumDecoderLayers, _options.MlpHiddenTimes, _options.DropoutRate, _options.FourierTopKFactor);
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

    private DiffusionTSNetwork<T> BoundNetwork()
    {
        var network = _network ?? throw new InvalidOperationException("DiffusionTS has no native network in ONNX mode.");
        network.BindTo(Layers);
        return network;
    }

    /// <inheritdoc/>
    /// <remarks>The denoiser is not a sequential chain, so shapes resolve through one training forward on a zero row.</remarks>
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

    #endregion

    #region Training

    // A training row: x_k over the window, then the step k, then the loss scale for that row.
    private int PackedWidth => WindowValues + 2;

    // The prediction and target are compared as [x_0 | Re F(x_0) | Im F(x_0)], each scaled per row.
    private int ObjectiveWidth => (_options.UseFourierLoss ? 3 : 1) * WindowValues;

    private double FourierWeight => _options.FourierLossWeight ?? Math.Sqrt(Window) / 5.0;

    // What both sides of the Fourier terms are multiplied by so the loss weights them by FourierWeight: the weight itself
    // under L1, whose distance is degree-1 homogeneous, and its square root under L2, which squares the factor.
    private double FourierScale => _options.ReconstructionLoss == DiffusionReconstructionLoss.L2
        ? Math.Sqrt(FourierWeight)
        : FourierWeight;
    /// <summary>
    /// Draws the Diffusion-TS training pair. The window is the context followed by the target horizon, normalised per
    /// series by the context's mean and spread. A step k ~ U{1..T} and noise eps ~ N(0, I) give
    /// x_k = sqrt(alphaBar_k) x_0 + sqrt(1 - alphaBar_k) eps; the network predicts x_0. The objective is the reference's:
    /// the distance between predicted and true x_0, plus, with the Fourier loss on, sqrt(L) / 5 times the distance
    /// between their Fourier coefficients (norm "forward"), each reweighted by the step's loss weight. The pair carries
    /// those as [x_0 | w_F Re F x_0 | w_F Im F x_0] times a per-row scale, so the configured L1 (or L2) loss over the
    /// row reproduces the reference sum.
    /// </summary>
    protected override (Tensor<T> Input, Tensor<T>? Target) PrepareTrainingPair(Tensor<T> input, Tensor<T>? target, long draw)
    {
        if (!_useNativeMode) return (input, target);
        var context = ContextRows(input, out int batch);
        var future = new double[batch, _forecastHorizon, _numFeatures];
        if (target is not null)
        {
            if (target.Length != batch * _forecastHorizon * _numFeatures)
                throw new ArgumentException(
                    $"DiffusionTS's target holds {target.Length} values; [{batch}, {_forecastHorizon}, {_numFeatures}] needs " +
                    $"{batch * _forecastHorizon * _numFeatures}.", nameof(target));
            for (int b = 0; b < batch; b++)
                for (int t = 0; t < _forecastHorizon; t++)
                    for (int f = 0; f < _numFeatures; f++)
                        future[b, t, f] = NumOps.ToDouble(target[(b * _forecastHorizon + t) * _numFeatures + f]);
        }

        var random = RandomHelper.CreateSeededRandom(DrawSeed(draw));
        var packed = new Tensor<T>(new[] { batch, PackedWidth });
        var objective = new Tensor<T>(new[] { batch, ObjectiveWidth });
        var clean = new double[Window * _numFeatures];
        bool l2 = _options.ReconstructionLoss == DiffusionReconstructionLoss.L2;
        int terms = _options.UseFourierLoss ? 3 : 1;
        for (int b = 0; b < batch; b++)
        {
            var (mean, spread) = Statistics(context, b);
            for (int t = 0; t < Window; t++)
                for (int f = 0; f < _numFeatures; f++)
                {
                    double raw = t < _sequenceLength ? context[b, t, f] : future[b, t - _sequenceLength, f];
                    clean[t * _numFeatures + f] = (raw - mean[f]) / spread[f];
                }

            int k = random.Next(_numDiffusionSteps);
            int row = b * PackedWidth;
            for (int i = 0; i < WindowValues; i++)
                packed[row + i] = NumOps.FromDouble(
                    _sqrtAlphasCumprod[k] * clean[i] + _sqrtOneMinusAlphasCumprod[k] * StandardNormal(random));
            packed[row + WindowValues] = NumOps.FromDouble(k);
            // Both sides are multiplied by this scale, so the mean loss over the row is the reference's weighted mean.
            double scale = l2 ? Math.Sqrt(terms * _lossWeight[k]) : terms * _lossWeight[k];
            packed[row + WindowValues + 1] = NumOps.FromDouble(scale);

            var targetRow = ObjectiveOf(clean, scale);
            for (int i = 0; i < targetRow.Length; i++) objective[b * ObjectiveWidth + i] = NumOps.FromDouble(targetRow[i]);
        }

        return (packed, objective);
    }

    /// <summary>
    /// x_0-hat for a pair from <see cref="PrepareTrainingPair"/>, laid out as the objective row
    /// [x_0-hat | w_F Re F x_0-hat | w_F Im F x_0-hat] times the row's scale. Every operation is recorded, so a compiled
    /// replay recomputes it from the replayed row.
    /// </summary>
    public override Tensor<T> ForwardForTraining(Tensor<T> input)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Training is only supported in native mode.");
        if (input.Rank != 2 || input.Shape[1] != PackedWidth)
            throw new ArgumentException(
                $"DiffusionTS trains on rows prepared by {nameof(PrepareTrainingPair)}: [B, {PackedWidth}] (the noisy window, " +
                $"the step, the loss scale); got [{string.Join(", ", input.Shape.ToArray())}].", nameof(input));

        var network = BoundNetwork();
        int batch = input.Shape[0];
        var noisy = Engine.Reshape(Engine.TensorNarrow(input, 1, 0, WindowValues), new[] { batch, Window, _numFeatures });
        var steps = Engine.TensorNarrow(input, 1, WindowValues, 1);
        var scale = Engine.TensorNarrow(input, 1, WindowValues + 1, 1);
        var predicted = network.PredictCleanWindow(Engine, noisy, steps);

        var flat = Engine.Reshape(predicted, new[] { batch, WindowValues });
        var parts = new List<Tensor<T>> { flat };
        if (_options.UseFourierLoss)
        {
            // F along time for every feature: [B, L, F] -> [B * F, L] rows against the L x L "forward"-normalised basis.
            var series = Engine.Reshape(Engine.TensorPermute(predicted, new[] { 0, 2, 1 }), new[] { batch * _numFeatures, Window });
            var (cosine, sine) = FourierBasis();
            T weight = NumOps.FromDouble(FourierScale);
            parts.Add(Engine.TensorMultiplyScalar(Engine.Reshape(Engine.TensorMatMul(series, cosine), new[] { batch, WindowValues }), weight));
            parts.Add(Engine.TensorMultiplyScalar(Engine.Reshape(Engine.TensorMatMul(series, sine), new[] { batch, WindowValues }), weight));
        }

        var row = parts.Count == 1 ? flat : Engine.TensorConcatenate(parts.ToArray(), axis: 1);
        return Engine.TensorMultiply(row, Engine.TensorBroadcastTo(scale, new[] { batch, ObjectiveWidth }));
    }

    // The objective row of a clean window, in the same layout ForwardForTraining produces.
    private double[] ObjectiveOf(double[] clean, double scale)
    {
        var row = new double[ObjectiveWidth];
        for (int i = 0; i < WindowValues; i++) row[i] = scale * clean[i];
        if (!_options.UseFourierLoss) return row;
        double weight = FourierScale;
        for (int f = 0; f < _numFeatures; f++)
            for (int k = 0; k < Window; k++)
            {
                double real = 0, imaginary = 0;
                for (int t = 0; t < Window; t++)
                {
                    double angle = 2.0 * Math.PI * k * t / Window;
                    double value = clean[t * _numFeatures + f];
                    real += value * Math.Cos(angle);
                    imaginary -= value * Math.Sin(angle);
                }

                row[WindowValues + f * Window + k] = scale * weight * real / Window;
                row[2 * WindowValues + f * Window + k] = scale * weight * imaginary / Window;
            }

        return row;
    }

    private (Tensor<T> Cosine, Tensor<T> Sine) FourierBasis()
    {
        var cosine = new Tensor<T>(new[] { Window, Window });
        var sine = new Tensor<T>(new[] { Window, Window });
        for (int t = 0; t < Window; t++)
            for (int k = 0; k < Window; k++)
            {
                double angle = 2.0 * Math.PI * k * t / Window;
                cosine[t * Window + k] = NumOps.FromDouble(Math.Cos(angle) / Window);
                sine[t * Window + k] = NumOps.FromDouble(-Math.Sin(angle) / Window);
            }

        return (cosine, sine);
    }

    #endregion

    #region Forecasting

    /// <inheritdoc/>
    protected override Tensor<T> PredictCore(Tensor<T> input)
        => _useNativeMode ? ForecastNative(input) : ForecastOnnx(input);

    /// <inheritdoc/>
    public override Tensor<T> Forecast(Tensor<T> historicalData, double[]? quantiles = null)
    {
        if (quantiles is not null && quantiles.Length > 0)
        {
            if (!_useNativeMode)
                throw new NotSupportedException("Quantile forecasts sample the native model; an ONNX DiffusionTS returns its point forecast only.");
            return ForecastQuantiles(historicalData, quantiles);
        }

        return _useNativeMode ? ForecastNative(historicalData) : ForecastOnnx(historicalData);
    }

    /// <summary>The forecast with a central prediction interval at <paramref name="confidenceLevel"/>.</summary>
    public (Tensor<T> Forecast, Tensor<T> Lower, Tensor<T> Upper) ForecastWithIntervals(Tensor<T> input, double confidenceLevel = 0.95)
    {
        if (double.IsNaN(confidenceLevel) || confidenceLevel <= 0 || confidenceLevel >= 1)
            throw new ArgumentOutOfRangeException(nameof(confidenceLevel));
        double tail = (1.0 - confidenceLevel) / 2.0;
        // One set of generated windows serves the mean and both bounds; each set costs NumDiffusionSteps denoiser passes.
        var windows = SampleWindows(input, out int batch);
        var bounds = QuantilesOf(windows, batch, new[] { tail, 1.0 - tail }, PointShape(input));
        var lower = new Tensor<T>(PointShape(input));
        var upper = new Tensor<T>(PointShape(input));
        for (int i = 0; i < lower.Length; i++)
        {
            lower[i] = bounds[2 * i];
            upper[i] = bounds[2 * i + 1];
        }

        return (MeanOf(windows, batch, input), lower, upper);
    }

    /// <inheritdoc/>
    public override Tensor<T> AutoregressiveForecast(Tensor<T> input, int steps)
    {
        if (steps <= 0) throw new ArgumentOutOfRangeException(nameof(steps));
        var predictions = new List<Tensor<T>>();
        var current = input;
        int remaining = steps;
        while (remaining > 0)
        {
            var prediction = Forecast(current, null);
            predictions.Add(prediction);
            int used = Math.Min(_forecastHorizon, remaining);
            remaining -= used;
            if (remaining > 0) current = ShiftInputWithPredictions(current, prediction, used);
        }

        return ConcatenatePredictions(predictions, steps);
    }

    /// <summary>The point forecast: the mean horizon of <see cref="NumSamples"/> generated windows.</summary>
    private Tensor<T> ForecastNative(Tensor<T> input)
    {
        var windows = SampleWindows(input, out int batch);
        return MeanOf(windows, batch, input);
    }

    private Tensor<T> MeanOf(double[,,] windows, int batch, Tensor<T> input)
    {
        var mean = new Tensor<T>(new[] { batch, _forecastHorizon, _numFeatures });
        int values = _forecastHorizon * _numFeatures;
        for (int b = 0; b < batch; b++)
            for (int i = 0; i < values; i++)
            {
                double sum = 0;
                for (int s = 0; s < _numSamples; s++) sum += windows[s, b, i];
                mean[b * values + i] = NumOps.FromDouble(sum / _numSamples);
            }

        return IsUnbatched(input) ? Engine.Reshape(mean, new[] { _forecastHorizon, _numFeatures }) : mean;
    }

    // Quantiles over the generated windows, as an extra last axis of the point forecast's shape.
    private Tensor<T> ForecastQuantiles(Tensor<T> input, double[] quantiles)
    {
        foreach (double q in quantiles)
            if (double.IsNaN(q) || q < 0 || q > 1)
                throw new ArgumentOutOfRangeException(nameof(quantiles), $"Quantile {q} is outside [0, 1].");
        var windows = SampleWindows(input, out int batch);
        return QuantilesOf(windows, batch, quantiles, PointShape(input));
    }

    private Tensor<T> QuantilesOf(double[,,] windows, int batch, double[] quantiles, int[] pointShape)
    {
        int values = _forecastHorizon * _numFeatures;
        var shape = pointShape.Concat(new[] { quantiles.Length }).ToArray();
        var result = new Tensor<T>(shape);
        var sorted = new double[_numSamples];
        for (int b = 0; b < batch; b++)
            for (int i = 0; i < values; i++)
            {
                for (int s = 0; s < _numSamples; s++) sorted[s] = windows[s, b, i];
                Array.Sort(sorted);
                for (int q = 0; q < quantiles.Length; q++)
                {
                    double position = quantiles[q] * (_numSamples - 1);
                    int lower = (int)Math.Floor(position);
                    int upper = Math.Min(lower + 1, _numSamples - 1);
                    result[(b * values + i) * quantiles.Length + q] =
                        NumOps.FromDouble(sorted[lower] + (position - lower) * (sorted[upper] - sorted[lower]));
                }
            }

        return result;
    }

    /// <summary>
    /// Generates windows conditioned on the context, <c>[samples, batch, horizon x features]</c> in the series' own
    /// scale. Every reverse step predicts x_0, samples x_{k-1} from the posterior q(x_{k-1} | x_k, x_0), and then puts
    /// the observed context back at noise level k - 1, so the generated horizon is always denoised against the history; the
    /// predicted x_0 is clamped to <see cref="DiffusionTSOptions{T}.DenoisedClip"/> first, as the reference clamps it
    /// (the replacement step of the reference's conditional sampler). The reference can additionally take gradient steps
    /// on x_k toward the observations; that guidance is not applied here.
    /// </summary>
    private double[,,] SampleWindows(Tensor<T> input, out int batch)
    {
        var network = BoundNetwork();
        var context = ContextRows(input, out batch);
        int paths = batch * _numSamples;
        var means = new double[batch][];
        var spreads = new double[batch][];
        for (int b = 0; b < batch; b++) (means[b], spreads[b]) = Statistics(context, b);

        var random = _seed.HasValue ? RandomHelper.CreateSeededRandom(_seed.Value) : RandomHelper.CreateSecureRandom();
        var x = new double[paths, Window, _numFeatures];
        for (int p = 0; p < paths; p++)
            for (int t = 0; t < Window; t++)
                for (int f = 0; f < _numFeatures; f++)
                    x[p, t, f] = StandardNormal(random);
        ImposeContext(x, context, means, spreads, batch, _numDiffusionSteps - 1, random);

        bool wasTraining = IsTrainingMode;
        if (wasTraining) SetTrainingMode(false);
        try
        {
            using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
            var noisy = new Tensor<T>(new[] { paths, Window, _numFeatures });
            var steps = new Tensor<T>(new[] { paths, 1 });
            for (int k = _numDiffusionSteps - 1; k >= 0; k--)
            {
                for (int p = 0; p < paths; p++)
                {
                    steps[p] = NumOps.FromDouble(k);
                    for (int t = 0; t < Window; t++)
                        for (int f = 0; f < _numFeatures; f++)
                            noisy[(p * Window + t) * _numFeatures + f] = NumOps.FromDouble(x[p, t, f]);
                }

                var clean = network.PredictCleanWindow(Engine, noisy, steps);
                double sigma = Math.Sqrt(_posteriorVariance[k]);
                double bound = _options.DenoisedClip ?? double.PositiveInfinity;
                for (int p = 0; p < paths; p++)
                    for (int t = 0; t < Window; t++)
                        for (int f = 0; f < _numFeatures; f++)
                        {
                            double predicted = Math.Max(-bound, Math.Min(bound, NumOps.ToDouble(clean[(p * Window + t) * _numFeatures + f])));
                            x[p, t, f] = k > 0
                                ? _posteriorMeanClean[k] * predicted + _posteriorMeanNoisy[k] * x[p, t, f] + sigma * StandardNormal(random)
                                : predicted;
                        }

                ImposeContext(x, context, means, spreads, batch, k - 1, random);
            }
        }
        finally
        {
            if (wasTraining) SetTrainingMode(true);
        }

        var result = new double[_numSamples, batch, _forecastHorizon * _numFeatures];
        for (int p = 0; p < paths; p++)
        {
            int s = p / batch, b = p % batch;
            for (int t = 0; t < _forecastHorizon; t++)
                for (int f = 0; f < _numFeatures; f++)
                    result[s, b, t * _numFeatures + f] = x[p, _sequenceLength + t, f] * spreads[b][f] + means[b][f];
        }

        return result;
    }

    // Writes the normalised context into the context positions at noise level `level` (exactly, when level < 0).
    private void ImposeContext(double[,,] x, double[,,] context, double[][] means, double[][] spreads, int batch, int level, Random random)
    {
        int paths = x.GetLength(0);
        for (int p = 0; p < paths; p++)
        {
            int b = p % batch;
            for (int t = 0; t < _sequenceLength; t++)
                for (int f = 0; f < _numFeatures; f++)
                {
                    double clean = (context[b, t, f] - means[b][f]) / spreads[b][f];
                    x[p, t, f] = level < 0
                        ? clean
                        : _sqrtAlphasCumprod[level] * clean + _sqrtOneMinusAlphasCumprod[level] * StandardNormal(random);
                }
        }
    }

    /// <summary>
    /// The context as <c>[batch, SequenceLength, features]</c> from <c>[B, T, F]</c>, an unbatched <c>[T, F]</c>, or for a
    /// univariate model <c>[B, T]</c> or <c>[T]</c>; a longer history keeps its most recent SequenceLength steps.
    /// </summary>
    private double[,,] ContextRows(Tensor<T> input, out int batch)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        int length;
        switch (input.Rank)
        {
            case 3 when input.Shape[2] == _numFeatures: batch = input.Shape[0]; length = input.Shape[1]; break;
            case 2 when IsUnbatched(input): batch = 1; length = input.Shape[0]; break;
            case 2 when _numFeatures == 1: batch = input.Shape[0]; length = input.Shape[1]; break;
            case 1 when _numFeatures == 1: batch = 1; length = input.Shape[0]; break;
            default:
                throw new ArgumentException(
                    $"DiffusionTS takes [B, T, {_numFeatures}] or [T, {_numFeatures}]" +
                    (_numFeatures == 1 ? ", [B, T] or [T]" : string.Empty) +
                    $"; got [{string.Join(", ", input.Shape.ToArray())}].", nameof(input));
        }

        if (length < _sequenceLength)
            throw new ArgumentException($"DiffusionTS needs at least {_sequenceLength} past steps; got {length}.", nameof(input));
        var rows = new double[batch, _sequenceLength, _numFeatures];
        int offset = length - _sequenceLength;
        for (int b = 0; b < batch; b++)
            for (int t = 0; t < _sequenceLength; t++)
                for (int f = 0; f < _numFeatures; f++)
                    rows[b, t, f] = NumOps.ToDouble(input[(b * length + offset + t) * _numFeatures + f]);
        return rows;
    }

    // An unbatched [T, F] window: rank 2 whose last axis is the feature axis (for one feature, a [T, 1] column).
    private bool IsUnbatched(Tensor<T> input)
        => input.Rank == 1 || (input.Rank == 2 && input.Shape[1] == _numFeatures && (_numFeatures > 1 || input.Shape[0] >= _sequenceLength));

    private int[] PointShape(Tensor<T> input)
    {
        ContextRows(input, out int batch);
        return IsUnbatched(input) ? new[] { _forecastHorizon, _numFeatures } : new[] { batch, _forecastHorizon, _numFeatures };
    }

    /// <summary>
    /// Each series' mean and spread over its context. Windows are standardised by them, so the denoiser sees series of
    /// any level and scale on one footing; a flat feature keeps a spread of 1.
    /// </summary>
    private (double[] Mean, double[] Spread) Statistics(double[,,] context, int row)
    {
        var mean = new double[_numFeatures];
        var spread = new double[_numFeatures];
        for (int f = 0; f < _numFeatures; f++)
        {
            double sum = 0;
            for (int t = 0; t < _sequenceLength; t++) sum += context[row, t, f];
            mean[f] = sum / _sequenceLength;
            double squares = 0;
            for (int t = 0; t < _sequenceLength; t++) squares += Math.Pow(context[row, t, f] - mean[f], 2);
            double deviation = Math.Sqrt(squares / _sequenceLength);
            spread[f] = deviation > 1e-5 && !double.IsInfinity(deviation) ? deviation : 1.0;
        }

        return (mean, spread);
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

    #endregion

    #region Model Reporting

    /// <inheritdoc/>
    /// <remarks>
    /// The layers are not a sequential chain, so the family's fold over Layers does not describe the model. One training
    /// forward on a target-less pair runs every layer in the order the model uses it, and that is what is recorded.
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

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NetworkType", "DiffusionTS" },
                { "SequenceLength", _sequenceLength },
                { "ForecastHorizon", _forecastHorizon },
                { "NumFeatures", _numFeatures },
                { "HiddenDimension", _options.HiddenDimension },
                { "NumEncoderLayers", _options.NumEncoderLayers },
                { "NumDecoderLayers", _options.NumDecoderLayers },
                { "NumDiffusionSteps", _numDiffusionSteps },
                { "NumSamples", _numSamples },
                { "UseNativeMode", _useNativeMode }
            },
            ModelDataProvider = () => _useNativeMode ? this.Serialize() : Array.Empty<byte>()
        };
    }

    /// <inheritdoc/>
    public override Dictionary<string, T> Evaluate(Tensor<T> predictions, Tensor<T> actuals)
    {
        var metrics = new Dictionary<string, T>();
        T mse = NumOps.Zero;
        T mae = NumOps.Zero;
        int count = Math.Min(predictions.Length, actuals.Length);
        for (int i = 0; i < count; i++)
        {
            T diff = NumOps.Subtract(predictions[i], actuals[i]);
            mse = NumOps.Add(mse, NumOps.Multiply(diff, diff));
            mae = NumOps.Add(mae, NumOps.Abs(diff));
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
    /// <remarks>Windows are standardised per series inside the model (see <c>Statistics</c>), so the input is left as given.</remarks>
    public override Tensor<T> ApplyInstanceNormalization(Tensor<T> input) => input;

    /// <inheritdoc/>
    public override Dictionary<string, T> GetFinancialMetrics()
    {
        T lastLoss = LastLoss is not null ? LastLoss : NumOps.Zero;
        return new Dictionary<string, T>
        {
            ["LastLoss"] = lastLoss,
            ["SequenceLength"] = NumOps.FromDouble(_sequenceLength),
            ["ForecastHorizon"] = NumOps.FromDouble(_forecastHorizon),
            ["NumDiffusionSteps"] = NumOps.FromDouble(_numDiffusionSteps),
            ["NumSamples"] = NumOps.FromDouble(_numSamples)
        };
    }

    #endregion

    #region ONNX

    /// <inheritdoc/>
    protected override Tensor<T> ForecastOnnx(Tensor<T> input)
    {
        if (OnnxSession is null)
            throw new InvalidOperationException("ONNX session not initialized.");
        var context = ContextRows(input, out int batch);
        var data = new float[batch * _sequenceLength * _numFeatures];
        for (int b = 0; b < batch; b++)
            for (int t = 0; t < _sequenceLength; t++)
                for (int f = 0; f < _numFeatures; f++)
                    data[(b * _sequenceLength + t) * _numFeatures + f] = (float)context[b, t, f];

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(
                OnnxSession.InputMetadata.Keys.FirstOrDefault() ?? "input",
                new OnnxTensors.DenseTensor<float>(data, new[] { batch, _sequenceLength, _numFeatures }))
        };
        using var results = OnnxSession.Run(inputs);
        var output = results.First().AsTensor<float>();
        var result = new Tensor<T>(output.Dimensions.ToArray());
        for (int i = 0; i < result.Length; i++) result[i] = NumOps.FromDouble(output.GetValue(i));
        return result;
    }

    /// <inheritdoc/>
    protected override void Dispose(bool disposing)
    {
        if (disposing) OnnxSession?.Dispose();
        base.Dispose(disposing);
    }

    #endregion
}
