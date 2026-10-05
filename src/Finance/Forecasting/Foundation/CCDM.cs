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
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using Microsoft.ML.OnnxRuntime;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

using AiDotNet.Finance.Base;
namespace AiDotNet.Finance.Forecasting.Foundation;

/// <summary>
/// CCDM — Channel-aware Contrastive Conditional Diffusion for multivariate probabilistic time series
/// forecasting (Li, Chen and Xiong, 2024).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// A DDPM forecaster (paper Section 3, following the authors' reference implementation at
/// github.com/LSY-Cython/CCDM). The denoiser treats every variable as a token: the past window and
/// the noisy future of each variable are embedded independently by channel-independent dense
/// modules, concatenated, mixed across variables by channel-wise diffusion transformers whose norms
/// the diffusion step modulates (adaLN-Zero), and decoded back to the horizon as a noise estimate.
/// Training minimizes the noise-prediction error (Equation 1), optionally plus the denoising-based
/// InfoNCE term (Equation 6); forecasting runs ancestral DDPM sampling for NumSamples paths and
/// reports their per-position median, with quantiles taken from the same paths.
/// </para>
/// <para>
/// Windows are normalized per variable by the context's mean and standard deviation; the target is
/// normalized with the same statistics for training and the sampled forecast is mapped back.
/// </para>
/// <para><b>For Beginners:</b> CCDM generates future values the way image generators create
/// pictures: it starts from random noise and removes it step by step, guided by the history. Every
/// run starts from different noise, so it draws many possible futures; the median is the forecast
/// and the spread tells you how uncertain it is. With several series it lets them inform each
/// other through attention across the series.</para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.Regression,
///     inputSize: 168, outputSize: 24);
///
/// var model = new CCDM&lt;double&gt;(architecture);
/// var forecast = model.Forecast(history);                                  // median of 100 paths
/// var bands = model.Forecast(history, quantiles: new[] { 0.1, 0.5, 0.9 }); // [1, horizon, 3]
/// </code>
/// </example>
[ModelDomain(ModelDomain.Finance)]
[ModelDomain(ModelDomain.TimeSeries)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Diffusion)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.Forecasting)]
[ModelComplexity(ModelComplexity.High)]
[ResearchPaper("Channel-aware Contrastive Conditional Diffusion for Multivariate Probabilistic Time Series Forecasting", "https://arxiv.org/abs/2410.02168")]
    [ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 1e-3, WeightDecay = 1e-6, ReferenceBatchSize = 64,
                Source = "Li, Chen and Xiong 2024 reference implementation (github.com/LSY-Cython/CCDM): "
                        + "optim.Adam(lr=init_lr, weight_decay=1e-6) with init_lr = 1e-3 and a training "
                        + "batch size of 64 in run_I48_O96.py.")]
public partial class CCDM<T> : TimeSeriesFoundationModelBase<T>, ITrainingObjectiveProvider<T>
{
    #region Fields

    private readonly bool _useNativeMode;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ILossFunction<T> _lossFunction;
    private readonly CCDMOptions<T> _options;

    public override ModelOptions GetOptions() => _options;

    private int _contextLength;
    private int _forecastHorizon;
    private int _numFeatures;
    private int _hiddenDimension;
    private int _numLayers;
    private int _numHeads;
    private int _embeddingLayers;
    private double _mlpRatio;
    private int _diffusionSteps;
    private int _numSamples;
    private int _trainingBatchSize;
    private double _dropout;
    private double _attentionDropout;
    private BetaSchedule _betaSchedule;
    private double _betaStart;
    private double _betaEnd;
    private double _contrastiveWeight;
    private double _contrastiveTemperature;
    private int _numNegatives;

    // DDPM schedule (reference diffusion.py): betas, alphas, alpha-bar and alpha-bar of the
    // previous step, which the posterior variance needs.
    [Buffer]
    private Vector<T> _betas = Vector<T>.Empty();
    [Buffer]
    private Vector<T> _alphas = Vector<T>.Empty();
    [Buffer]
    private Vector<T> _alphasCumprod = Vector<T>.Empty();
    [Buffer]
    private Vector<T> _alphasCumprodPrev = Vector<T>.Empty();
    [Buffer]
    private Vector<T> _sqrtAlphasCumprod = Vector<T>.Empty();
    [Buffer]
    private Vector<T> _sqrtOneMinusAlphasCumprod = Vector<T>.Empty();

    // Constant gamma = 1 / beta = 0 for the non-affine adaLN norms (elementwise_affine=False):
    // rebuilt on demand, never trained or saved.
    [Scratch]
    private Tensor<T>? _plainNormGamma;
    [Scratch]
    private Tensor<T>? _plainNormBeta;

    /// <summary>Frequencies of the sinusoidal step embedding (reference StepEmbedding freq_dim).</summary>
    private const int StepFrequencyDim = 256;

    /// <summary>Epsilon of the adaLN norms (reference nn.LayerNorm(eps=1e-6)).</summary>
    private const double PlainNormEpsilon = 1e-6;

    /// <summary>Patch length of the "variation" negatives (reference negative_sampling patch_size).</summary>
    private const int NegativePatchSize = 8;

    /// <summary>Seed of the negatives in the deterministic objective (paper arXiv id 2410.02168).</summary>
    private const int ObjectiveNegativeSeed = 241002168;

    private const int LayersPerMlpResidual = 5;
    private const int LayersPerDiTBlock = 9;

    #endregion

    #region Properties

    public override int SequenceLength => _contextLength;
    public override int PredictionHorizon => _forecastHorizon;
    public override int NumFeatures => _numFeatures;
    public override int PatchSize => 1;
    public override int Stride => 1;
    public override bool IsChannelIndependent => _numFeatures == 1;
    public override bool UseNativeMode => _useNativeMode;
    public override FoundationModelSize ModelSize => FoundationModelSize.Small;
    public override int MaxContextLength => _contextLength;
    public override int MaxPredictionHorizon => _forecastHorizon;

    #endregion

    #region Constructors

    public CCDM(NeuralNetworkArchitecture<T> architecture, string onnxModelPath,
        CCDMOptions<T>? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null, ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        if (string.IsNullOrWhiteSpace(onnxModelPath)) throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath)) throw new FileNotFoundException($"ONNX model not found: {onnxModelPath}");
        options ??= new CCDMOptions<T>(); _options = options; Options = _options;
        _useNativeMode = false; OnnxModelPath = onnxModelPath; OnnxSession = new InferenceSession(onnxModelPath);
        _optimizer = optimizer
            ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this)
            ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this, new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>> { InitialLearningRate = options.LearningRate }); _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        CopyOptionsToFields(options);
    }

    public CCDM(NeuralNetworkArchitecture<T> architecture,
        CCDMOptions<T>? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null, ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        options ??= new CCDMOptions<T>(); _options = options; Options = _options;
        _useNativeMode = true; OnnxSession = null; OnnxModelPath = null;
        _optimizer = optimizer
            ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this)
            ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this, new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>> { InitialLearningRate = options.LearningRate }); _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        CopyOptionsToFields(options); InitializeLayers();
    }

    private void CopyOptionsToFields(CCDMOptions<T> options)
    {
        if (options.NumFeatures < 1) throw new ArgumentOutOfRangeException(nameof(options), "NumFeatures must be at least 1.");
        if (options.NumSamples < 1) throw new ArgumentOutOfRangeException(nameof(options), "NumSamples must be at least 1.");
        if (options.ContrastiveWeight < 0) throw new ArgumentOutOfRangeException(nameof(options), "ContrastiveWeight cannot be negative.");
        if (options.ContrastiveWeight > 0 && options.NumNegatives < 2)
            throw new ArgumentOutOfRangeException(nameof(options), "The contrastive loss needs NumNegatives >= 2 (half are scaled up, half down).");
        if (options.ContrastiveWeight > 0 && options.ContrastiveTemperature <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), "ContrastiveTemperature must be positive.");

        _contextLength = options.ContextLength; _forecastHorizon = options.ForecastHorizon;
        _numFeatures = options.NumFeatures;
        _hiddenDimension = options.HiddenDimension; _numLayers = options.NumLayers;
        _numHeads = options.NumHeads; _embeddingLayers = options.EmbeddingLayers;
        _mlpRatio = options.MlpRatio; _diffusionSteps = options.DiffusionSteps;
        _dropout = options.DropoutRate; _attentionDropout = options.AttentionDropoutRate;
        _betaSchedule = options.BetaSchedule; _betaStart = options.BetaStart; _betaEnd = options.BetaEnd;
        _numSamples = options.NumSamples; _trainingBatchSize = options.TrainingBatchSize;
        _contrastiveWeight = options.ContrastiveWeight;
        _contrastiveTemperature = options.ContrastiveTemperature;
        _numNegatives = options.NumNegatives;
        ComputeNoiseSchedule();
    }

    /// <summary>
    /// The reference DDPM schedule (diffusion.py): linear, "quad" (= scaled linear) or cosine betas,
    /// then alphas, their running product alpha-bar and alpha-bar of the previous step.
    /// </summary>
    private void ComputeNoiseSchedule()
    {
        if (_diffusionSteps <= 0)
            throw new ArgumentOutOfRangeException(nameof(_diffusionSteps), "DiffusionSteps must be positive.");

        int k = _diffusionSteps;
        var betas = new double[k];
        switch (_betaSchedule)
        {
            case BetaSchedule.Linear:
                for (int i = 0; i < k; i++)
                    betas[i] = _betaStart + (_betaEnd - _betaStart) * i / Math.Max(1, k - 1);
                break;
            case BetaSchedule.ScaledLinear:
            {
                // torch.linspace(beta_start**0.5, beta_end**0.5, K)**2 - the reference "quad".
                double a = Math.Sqrt(_betaStart), b = Math.Sqrt(_betaEnd);
                for (int i = 0; i < k; i++)
                {
                    double root = a + (b - a) * i / Math.Max(1, k - 1);
                    betas[i] = root * root;
                }
                break;
            }
            case BetaSchedule.SquaredCosine:
            {
                // cosine_beta_schedule(n_steps, s=0.008): alpha-bar from a squared cosine, betas
                // from consecutive ratios, clipped to [0, 0.999].
                const double s = 0.008;
                double Bar(double x) => Math.Pow(Math.Cos((x / k + s) / (1 + s) * Math.PI * 0.5), 2);
                double bar0 = Bar(0);
                for (int i = 0; i < k; i++)
                {
                    double beta = 1 - (Bar(i + 1) / bar0) / (Bar(i) / bar0);
                    betas[i] = Math.Min(0.999, Math.Max(0.0, beta));
                }
                break;
            }
            default:
                throw new NotSupportedException($"CCDM does not support the {_betaSchedule} beta schedule.");
        }

        _betas = new Vector<T>(k);
        _alphas = new Vector<T>(k);
        _alphasCumprod = new Vector<T>(k);
        _alphasCumprodPrev = new Vector<T>(k);
        _sqrtAlphasCumprod = new Vector<T>(k);
        _sqrtOneMinusAlphasCumprod = new Vector<T>(k);
        double bar = 1.0;
        for (int i = 0; i < k; i++)
        {
            _alphasCumprodPrev[i] = NumOps.FromDouble(bar);
            double alpha = 1.0 - betas[i];
            bar *= alpha;
            _betas[i] = NumOps.FromDouble(betas[i]);
            _alphas[i] = NumOps.FromDouble(alpha);
            _alphasCumprod[i] = NumOps.FromDouble(bar);
            _sqrtAlphasCumprod[i] = NumOps.FromDouble(Math.Sqrt(bar));
            _sqrtOneMinusAlphasCumprod[i] = NumOps.FromDouble(Math.Sqrt(1.0 - bar));
        }
    }

    private T SampleStandardNormal(Random rand)
    {
        double u1 = 1.0 - rand.NextDouble();
        double u2 = 1.0 - rand.NextDouble();
        return NumOps.FromDouble(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
    }

    #endregion

    #region Initialization

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
            Layers.AddRange(Architecture.Layers);
        else
            Layers.AddRange(LayerHelper<T>.CreateDefaultCCDMLayers(
                Architecture, _contextLength, _forecastHorizon, _hiddenDimension, _numLayers, _numHeads,
                _dropout, _embeddingLayers, _mlpRatio, _attentionDropout));
        _ = Parts();
    }

    /// <summary>
    /// A view of the flat layer list as the denoiser's parts, in the factory's order.
    /// </summary>
    /// <remarks>
    /// Built from <c>Layers</c> on every forward rather than cached: Clone and Deserialize replace the
    /// entries of <c>Layers</c> (and dispose the ones they replace), so references captured at
    /// construction would point at disposed layers.
    /// </remarks>
    private DenoiserParts Parts()
    {
        int expected = 2 * _embeddingLayers * LayersPerMlpResidual + 2 + _numLayers * LayersPerDiTBlock
                       + 2 + (_embeddingLayers - 1) * LayersPerMlpResidual + 1;
        if (Layers.Count != expected)
        {
            throw new ArgumentException(
                $"CCDM expects the {expected} layers LayerHelper<T>.CreateDefaultCCDMLayers produces for " +
                $"EmbeddingLayers = {_embeddingLayers} and NumLayers = {_numLayers}; the architecture supplied " +
                $"{Layers.Count}. Custom layers must follow the same order.");
        }

        int idx = 0;
        MlpResidualBlock NextMlp() => new(Layers[idx++], Layers[idx++], Layers[idx++], Layers[idx++], Layers[idx++]);

        var past = new List<MlpResidualBlock>();
        var future = new List<MlpResidualBlock>();
        for (int i = 0; i < _embeddingLayers; i++) past.Add(NextMlp());
        for (int i = 0; i < _embeddingLayers; i++) future.Add(NextMlp());
        var stepProjection = Layers[idx++];
        var stepOutput = Layers[idx++];
        var blocks = new List<DiTBlock>();
        for (int b = 0; b < _numLayers; b++)
        {
            blocks.Add(new DiTBlock(
                Layers[idx++], Layers[idx++], Layers[idx++], Layers[idx++], Layers[idx++],
                Layers[idx++], Layers[idx++], Layers[idx++], Layers[idx++]));
        }
        var decoderActivation = Layers[idx++];
        var decoderModulation = Layers[idx++];
        var decoderEmbedding = new List<MlpResidualBlock>();
        for (int i = 0; i < _embeddingLayers - 1; i++) decoderEmbedding.Add(NextMlp());
        var decoderOutput = Layers[idx++];
        return new DenoiserParts(past, future, stepProjection, stepOutput, blocks,
            decoderActivation, decoderModulation, decoderEmbedding, decoderOutput);
    }

    #endregion

    #region NeuralNetworkBase Overrides

    public override bool SupportsTraining => _useNativeMode;
    protected override Tensor<T> PredictCore(Tensor<T> input) => _useNativeMode ? ForwardNative(input, quantiles: null) : ForecastOnnx(input);

    /// <summary>
    /// One training step of paper Equation 6 (Equation 1 alone when ContrastiveWeight is 0).
    /// </summary>
    /// <remarks>
    /// <para>
    /// Per window: normalize context and target by the context's per-variable statistics, draw
    /// TrainingBatchSize diffusion steps antithetically (k and K-1-k in pairs, the reference's
    /// "uniform" step distribution), draw epsilon, form y_k = sqrt(alphaBar_k) y_0 +
    /// sqrt(1 - alphaBar_k) epsilon and regress the denoiser onto epsilon through the configured
    /// loss (mean squared error by default, as the reference).
    /// </para>
    /// <para>
    /// The draws are deliberately NOT seeded from <c>Options.Seed</c>. That seed makes inference
    /// reproducible; reusing it here would hand every call the same steps and the same noise, so the
    /// model would be fitted at a fixed set of noise levels.
    /// </para>
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Training is only supported in native mode.");
        if (expectedOutput.Length <= 0) return;

        // Every training entry point on the base puts the layers in training mode for the duration
        // of the step; without it the dropout layers stay in inference mode.
        SetTrainingMode(true);
        try
        {
            var trainableParams = Training.TapeTrainingStep<T>.CollectParameters(Layers).ToArray();
            var windows = SplitWindows(input);
            var targets = SplitTargets(expectedOutput, windows.Count);
            var rand = RandomHelper.CreateSecureRandom();

            var draws = new List<DenoisingDraw>(windows.Count);
            for (int w = 0; w < windows.Count; w++)
                draws.Add(DrawDenoisingBatch(windows[w], targets[w], rand, antithetic: true));

            // One seed for this step's contrastive negatives, shared by the gradient pass and the
            // optimizer's recompute closure below so both see the same negatives.
            int negativeSeed = rand.Next();

            using var tape = new GradientTape<T>();
            var lossTensor = TrainingLoss(draws, RandomHelper.CreateSeededRandom(negativeSeed));
            var grads = ComputeAndPublishParameterGradients(tape, lossTensor, trainableParams);

            T lossValue = lossTensor.Length > 0 ? lossTensor[0] : NumOps.Zero;
            LastLoss = lossValue;

            // Pinned to this step's draws: a line-searching optimizer that re-drew them would be
            // comparing losses at different noise levels.
            Tensor<T> ComputeForward(Tensor<T> _, Tensor<T> __) => DenoiseDraws(draws);
            Tensor<T> RecomputeLoss(Tensor<T> _, Tensor<T> __) =>
                TrainingLoss(draws, RandomHelper.CreateSeededRandom(negativeSeed));

            var context = new TapeStepContext<T>(
                trainableParams, grads, lossValue,
                input, expectedOutput, ComputeForward, RecomputeLoss);

            MarkTrainMutationStarted();
            _optimizer.Step(context);
            InvalidateWeightCachesAfterSuccessfulWeightUpdate();
            StepSchedulerIfSupported(_optimizer);
        }
        finally
        {
            SetTrainingMode(false);
        }
    }

    /// <summary>
    /// Tape-aware forward over the denoiser at the last diffusion step, with fixed noise.
    /// </summary>
    /// <remarks>
    /// Exists so callers that probe the training graph without a target - gradient reachability,
    /// parameter-movement probes - drive the same denoiser that <see cref="Train"/> and the sampler
    /// drive. The noise comes from <c>Options.Seed</c> so two calls on unchanged weights agree.
    /// </remarks>
    public override Tensor<T> ForwardForTraining(Tensor<T> input)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Training is only supported in native mode.");

        var windows = SplitWindows(input);
        var rand = _options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(_options.Seed.Value)
            : RandomHelper.CreateSecureRandom();
        var outputs = new List<Tensor<T>>(windows.Count);
        foreach (var window in windows)
        {
            var yk = new Tensor<T>(new[] { 1, _forecastHorizon, _numFeatures });
            var span = yk.Data.Span;
            for (int i = 0; i < span.Length; i++) span[i] = SampleStandardNormal(rand);
            outputs.Add(Denoise(window.Tokens, yk, new[] { _diffusionSteps - 1 }, rows: 1));
        }
        return ShapeLikeForecast(input, outputs, rowsPerWindow: 1);
    }

    public override ModelMetadata<T> GetModelMetadata() => new()
    {
        AdditionalInfo = new Dictionary<string, object>
        {
            { "NetworkType", "CCDM" }, { "ContextLength", _contextLength }, { "ForecastHorizon", _forecastHorizon },
            { "NumFeatures", _numFeatures }, { "HiddenDimension", _hiddenDimension }, { "NumLayers", _numLayers },
            { "DiffusionSteps", _diffusionSteps }, { "BetaSchedule", _betaSchedule.ToString() },
            { "NumSamples", _numSamples }, { "ContrastiveWeight", _contrastiveWeight }, { "UseNativeMode", _useNativeMode }
        },
        ModelDataProvider = () => _useNativeMode ? this.Serialize() : Array.Empty<byte>()
    };

    #endregion

    #region IForecastingModel Implementation

    /// <summary>
    /// Forecasts the horizon. Without quantiles this is the per-position median of NumSamples paths;
    /// with quantiles it is the requested empirical quantiles of the same paths, shaped
    /// [batch, horizon, quantiles] (or [batch, horizon, variables, quantiles] with several variables).
    /// </summary>
    public override Tensor<T> Forecast(Tensor<T> historicalData, double[]? quantiles = null)
    {
        if (quantiles is not null)
        {
            for (int i = 0; i < quantiles.Length; i++)
            {
                if (double.IsNaN(quantiles[i]) || quantiles[i] < 0 || quantiles[i] > 1)
                    throw new ArgumentOutOfRangeException(nameof(quantiles), $"Quantile at index {i} is {quantiles[i]}, but must be within [0, 1].");
            }
        }
        if (!_useNativeMode)
        {
            if (quantiles is not null && quantiles.Length > 0)
                throw new NotSupportedException("Quantile forecasts need the native sampler; the ONNX graph returns point forecasts only.");
            return ForecastOnnx(historicalData);
        }
        return ForwardNative(historicalData, quantiles is { Length: > 0 } ? quantiles : null);
    }

    public override Tensor<T> AutoregressiveForecast(Tensor<T> input, int steps) { var predictions = new List<Tensor<T>>(); var currentInput = input; int stepsRemaining = steps; while (stepsRemaining > 0) { var prediction = Forecast(currentInput, null); predictions.Add(prediction); int stepsUsed = Math.Min(_forecastHorizon, stepsRemaining); stepsRemaining -= stepsUsed; if (stepsRemaining > 0) currentInput = ShiftInputWithPredictions(currentInput, prediction, stepsUsed); } return ConcatenatePredictions(predictions, steps); }

    public override Dictionary<string, T> Evaluate(Tensor<T> predictions, Tensor<T> actuals) { T mse = NumOps.Zero; T mae = NumOps.Zero; int count = 0; for (int i = 0; i < predictions.Length && i < actuals.Length; i++) { var diff = NumOps.Subtract(predictions[i], actuals[i]); mse = NumOps.Add(mse, NumOps.Multiply(diff, diff)); mae = NumOps.Add(mae, NumOps.Abs(diff)); count++; } if (count > 0) { mse = NumOps.Divide(mse, NumOps.FromDouble(count)); mae = NumOps.Divide(mae, NumOps.FromDouble(count)); } return new Dictionary<string, T> { ["MSE"] = mse, ["MAE"] = mae, ["RMSE"] = NumOps.Sqrt(mse) }; }

    /// <summary>
    /// Per-variable window normalization (the reference instance_normalization): each variable of
    /// the context is centred and scaled by its own mean and standard deviation over time.
    /// </summary>
    public override Tensor<T> ApplyInstanceNormalization(Tensor<T> input)
        => NormalizePerFeatureOnTape(
            _numFeatures == 1 ? Engine.Reshape(input, new[] { input.Length, 1 }) : input,
            DefaultRevInEpsilon, out _, out _);

    public override Dictionary<string, T> GetFinancialMetrics() { T lastLoss = LastLoss is not null ? LastLoss : NumOps.Zero; return new Dictionary<string, T> { ["ContextLength"] = NumOps.FromDouble(_contextLength), ["ForecastHorizon"] = NumOps.FromDouble(_forecastHorizon), ["LastLoss"] = lastLoss }; }

    #endregion

    #region Windows and shapes

    /// <summary>One normalized context window: tokens [1, D, L] and the statistics to undo it.</summary>
    private sealed class Window
    {
        public Window(Tensor<T> tokens, double[] mean, double[] std) { Tokens = tokens; Mean = mean; Std = std; }
        public Tensor<T> Tokens { get; }
        public double[] Mean { get; }
        public double[] Std { get; }
    }

    /// <summary>
    /// Reads the context as one or more windows. Layouts: [L] or [B, L] with one variable,
    /// [L, D] or [B, L, D] with several (a trailing variable axis of one is also accepted).
    /// </summary>
    private List<Window> SplitWindows(Tensor<T> input)
    {
        int perWindow = _contextLength * _numFeatures;
        if (input.Length == 0 || input.Length % perWindow != 0)
        {
            throw new ArgumentException(
                $"CCDM expects context windows of {_contextLength} steps x {_numFeatures} variable(s) " +
                $"({perWindow} values each); got shape [{string.Join(",", input.Shape.ToArray())}].",
                nameof(input));
        }

        int windows = input.Length / perWindow;
        var values = input.ToArray();
        var result = new List<Window>(windows);
        for (int w = 0; w < windows; w++)
        {
            var mean = new double[_numFeatures];
            var std = new double[_numFeatures];
            var tokens = new Tensor<T>(new[] { 1, _numFeatures, _contextLength });
            var span = tokens.Data.Span;
            for (int f = 0; f < _numFeatures; f++)
            {
                double sum = 0, sumSq = 0;
                for (int t = 0; t < _contextLength; t++)
                {
                    double v = NumOps.ToDouble(values[w * perWindow + t * _numFeatures + f]);
                    sum += v; sumSq += v * v;
                }
                double m = sum / _contextLength;
                double variance = Math.Max(0.0, sumSq / _contextLength - m * m);
                double s = Math.Sqrt(variance + 1e-5);
                mean[f] = m; std[f] = s;
                for (int t = 0; t < _contextLength; t++)
                {
                    double v = NumOps.ToDouble(values[w * perWindow + t * _numFeatures + f]);
                    span[f * _contextLength + t] = NumOps.FromDouble((v - m) / s);
                }
            }
            result.Add(new Window(tokens, mean, std));
        }
        return result;
    }

    /// <summary>Reads the targets as one [H, D] row-major array per window.</summary>
    private List<double[]> SplitTargets(Tensor<T> target, int windows)
    {
        int perWindow = _forecastHorizon * _numFeatures;
        var values = target.ToArray();
        var result = new List<double[]>(windows);
        for (int w = 0; w < windows; w++)
        {
            // A target shorter than the horizon leaves the tail at the window's mean (zero once
            // normalized) rather than changing a width the embedding layers have already baked.
            var row = new double[perWindow];
            for (int i = 0; i < perWindow; i++)
            {
                int src = w * perWindow + i;
                row[i] = src < values.Length ? NumOps.ToDouble(values[src]) : double.NaN;
            }
            result.Add(row);
        }
        return result;
    }

    /// <summary>
    /// Assembles per-window results ([rows, H, D] each) in the caller's layout: [H] or [B, H] with
    /// one variable, [H, D] or [B, H, D] with several.
    /// </summary>
    private Tensor<T> ShapeLikeForecast(Tensor<T> input, List<Tensor<T>> perWindow, int rowsPerWindow)
    {
        var joined = perWindow.Count == 1 ? perWindow[0] : Engine.TensorConcatenate(perWindow.ToArray(), axis: 0);
        int batch = perWindow.Count * rowsPerWindow;
        bool batched = input.Rank >= (_numFeatures == 1 ? 2 : 3) || batch > 1;
        if (_numFeatures == 1)
            return Engine.Reshape(joined, batched ? new[] { batch, _forecastHorizon } : new[] { _forecastHorizon });
        return Engine.Reshape(joined, batched ? new[] { batch, _forecastHorizon, _numFeatures } : new[] { _forecastHorizon, _numFeatures });
    }

    #endregion

    #region Denoiser

    private sealed class DenoiserParts
    {
        public DenoiserParts(
            List<MlpResidualBlock> pastEmbedding, List<MlpResidualBlock> futureEmbedding,
            ILayer<T> stepProjection, ILayer<T> stepOutput, List<DiTBlock> blocks,
            ILayer<T> decoderActivation, ILayer<T> decoderModulation,
            List<MlpResidualBlock> decoderEmbedding, ILayer<T> decoderOutput)
        {
            PastEmbedding = pastEmbedding; FutureEmbedding = futureEmbedding;
            StepProjection = stepProjection; StepOutput = stepOutput; Blocks = blocks;
            DecoderActivation = decoderActivation; DecoderModulation = decoderModulation;
            DecoderEmbedding = decoderEmbedding; DecoderOutput = decoderOutput;
        }
        public List<MlpResidualBlock> PastEmbedding { get; }
        public List<MlpResidualBlock> FutureEmbedding { get; }
        public ILayer<T> StepProjection { get; }
        public ILayer<T> StepOutput { get; }
        public List<DiTBlock> Blocks { get; }
        public ILayer<T> DecoderActivation { get; }
        public ILayer<T> DecoderModulation { get; }
        public List<MlpResidualBlock> DecoderEmbedding { get; }
        public ILayer<T> DecoderOutput { get; }
    }

    private sealed class MlpResidualBlock
    {
        public MlpResidualBlock(ILayer<T> first, ILayer<T> second, ILayer<T> dropout, ILayer<T> residual, ILayer<T> norm)
        { First = first; Second = second; Dropout = dropout; Residual = residual; Norm = norm; }
        public ILayer<T> First { get; }
        public ILayer<T> Second { get; }
        public ILayer<T> Dropout { get; }
        public ILayer<T> Residual { get; }
        public ILayer<T> Norm { get; }
    }

    private sealed class DiTBlock
    {
        public DiTBlock(ILayer<T> activation, ILayer<T> modulation, ILayer<T> query, ILayer<T> key, ILayer<T> value,
            ILayer<T> output, ILayer<T> attentionDropout, ILayer<T> mlpIn, ILayer<T> mlpOut)
        {
            Activation = activation; Modulation = modulation; Query = query; Key = key; Value = value;
            Output = output; AttentionDropout = attentionDropout; MlpIn = mlpIn; MlpOut = mlpOut;
        }
        public ILayer<T> Activation { get; }
        public ILayer<T> Modulation { get; }
        public ILayer<T> Query { get; }
        public ILayer<T> Key { get; }
        public ILayer<T> Value { get; }
        public ILayer<T> Output { get; }
        public ILayer<T> AttentionDropout { get; }
        public ILayer<T> MlpIn { get; }
        public ILayer<T> MlpOut { get; }
    }

    /// <summary>
    /// The reference Denoiser.forward: epsilon-hat for noisy futures <paramref name="yk"/>
    /// [rows, H, D] given the context tokens [1, D, L] at the given step(s) (one per row, or one
    /// shared by every row).
    /// </summary>
    private Tensor<T> Denoise(Tensor<T> contextTokens, Tensor<T> yk, IReadOnlyList<int> steps, int rows)
    {
        var parts = Parts();
        var context = rows == 1 ? contextTokens : Engine.TensorTile(contextTokens, new[] { rows, 1, 1 });
        var past = ApplyMlpStack(parts.PastEmbedding, context);                         // [R, D, e]
        var future = ApplyMlpStack(parts.FutureEmbedding, Engine.TensorPermute(yk, new[] { 0, 2, 1 })); // [R, D, e]
        var h = Engine.TensorConcatenate(new[] { past, future }, axis: 2);              // [R, D, 2e]

        var c = StepEmbedding(parts, steps, rows);                                      // [R, e]
        foreach (var block in parts.Blocks) h = DiTBlockForward(h, c, block, rows);

        var decoded = Decode(parts, h, c, rows);                                        // [R, D, H]
        return Engine.TensorPermute(decoded, new[] { 0, 2, 1 });                         // [R, H, D]
    }

    private Tensor<T> ApplyMlpStack(List<MlpResidualBlock> stack, Tensor<T> x)
    {
        foreach (var block in stack)
        {
            // LayerNorm(Dropout(Linear(ReLU(Linear(x)))) + Linear(x)) - the reference MLPResidual.
            var embedded = block.Dropout.Forward(block.Second.Forward(block.First.Forward(x)));
            var residual = block.Residual.Forward(x);
            x = block.Norm.Forward(Engine.TensorAdd(embedded, residual));
        }
        return x;
    }

    /// <summary>The reference StepEmbedding: 256 sinusoidal frequencies -> Linear + SiLU -> Linear.</summary>
    private Tensor<T> StepEmbedding(DenoiserParts parts, IReadOnlyList<int> steps, int rows)
    {
        var frequencies = new Tensor<T>(new[] { rows, StepFrequencyDim });
        var span = frequencies.Data.Span;
        for (int r = 0; r < rows; r++)
        {
            int step = steps.Count == 1 ? steps[0] : steps[r];
            WriteDiffusionTimestepEmbedding(span.Slice(r * StepFrequencyDim, StepFrequencyDim), step);
        }
        return parts.StepOutput.Forward(parts.StepProjection.Forward(frequencies));
    }

    /// <summary>LayerNorm without learned affine (reference elementwise_affine=False, eps 1e-6).</summary>
    private Tensor<T> PlainNorm(Tensor<T> x)
    {
        int width = x.Shape[x.Rank - 1];
        var gamma = _plainNormGamma;
        var beta = _plainNormBeta;
        if (gamma is null || beta is null || gamma.Length != width)
        {
            gamma = new Tensor<T>(new[] { width });
            gamma.Fill(NumOps.One);
            beta = new Tensor<T>(new[] { width });
            _plainNormGamma = gamma;
            _plainNormBeta = beta;
        }
        return Engine.LayerNorm(x, gamma, beta, PlainNormEpsilon, out _, out _);
    }

    /// <summary>modulate(x, shift, scale) = x * (1 + scale) + shift, broadcast over the variables.</summary>
    private Tensor<T> Modulate(Tensor<T> x, Tensor<T> shift, Tensor<T> scale)
        => Engine.TensorAdd(Engine.TensorMultiply(x, Engine.TensorAddScalar(scale, NumOps.One)), shift);

    /// <summary>
    /// The reference DiTBlock with adaLN-Zero: the step embedding yields shift/scale/gate for the
    /// attention and the MLP branch; attention runs across the variable tokens.
    /// </summary>
    private Tensor<T> DiTBlockForward(Tensor<T> h, Tensor<T> c, DiTBlock block, int rows)
    {
        int d = 2 * _hiddenDimension;
        var modulation = Engine.Reshape(block.Modulation.Forward(block.Activation.Forward(c)), new[] { rows, 6, 1, d });
        var shiftAttention = Engine.TensorSliceAxis(modulation, axis: 1, index: 0);
        var scaleAttention = Engine.TensorSliceAxis(modulation, axis: 1, index: 1);
        var gateAttention = Engine.TensorSliceAxis(modulation, axis: 1, index: 2);
        var shiftMlp = Engine.TensorSliceAxis(modulation, axis: 1, index: 3);
        var scaleMlp = Engine.TensorSliceAxis(modulation, axis: 1, index: 4);
        var gateMlp = Engine.TensorSliceAxis(modulation, axis: 1, index: 5);

        var attended = ChannelAttention(Modulate(PlainNorm(h), shiftAttention, scaleAttention), block, rows);
        h = Engine.TensorAdd(h, Engine.TensorMultiply(attended, gateAttention));

        var mlp = block.MlpOut.Forward(block.MlpIn.Forward(Modulate(PlainNorm(h), shiftMlp, scaleMlp)));
        return Engine.TensorAdd(h, Engine.TensorMultiply(mlp, gateMlp));
    }

    /// <summary>Scaled dot-product multi-head attention over the variable axis (reference FullAttention).</summary>
    private Tensor<T> ChannelAttention(Tensor<T> x, DiTBlock block, int rows)
    {
        int d = 2 * _hiddenDimension;
        int tokens = _numFeatures;
        int headDim = d / _numHeads;

        Tensor<T> Heads(Tensor<T> t) => Engine.Reshape(
            Engine.TensorPermute(Engine.Reshape(t, new[] { rows, tokens, _numHeads, headDim }), new[] { 0, 2, 1, 3 }),
            new[] { rows * _numHeads, tokens, headDim });

        var q = Heads(block.Query.Forward(x));
        var k = Heads(block.Key.Forward(x));
        var v = Heads(block.Value.Forward(x));

        var scores = Engine.TensorMultiplyScalar(
            Engine.TensorBatchMatMul(q, Engine.TensorPermute(k, new[] { 0, 2, 1 })),
            NumOps.FromDouble(1.0 / Math.Sqrt(headDim)));
        var weights = block.AttentionDropout.Forward(Engine.Softmax(scores, axis: -1));
        var aggregated = Engine.TensorBatchMatMul(weights, v);                           // [R*heads, D, hd]
        var merged = Engine.Reshape(
            Engine.TensorPermute(Engine.Reshape(aggregated, new[] { rows, _numHeads, tokens, headDim }), new[] { 0, 2, 1, 3 }),
            new[] { rows, tokens, d });
        return block.Output.Forward(merged);
    }

    /// <summary>The reference Decoder: adaLN-modulated norm, n-1 MLPResidual blocks, projection to H.</summary>
    private Tensor<T> Decode(DenoiserParts parts, Tensor<T> h, Tensor<T> c, int rows)
    {
        int d = 2 * _hiddenDimension;
        var modulation = Engine.Reshape(parts.DecoderModulation.Forward(parts.DecoderActivation.Forward(c)), new[] { rows, 2, 1, d });
        var shift = Engine.TensorSliceAxis(modulation, axis: 1, index: 0);
        var scale = Engine.TensorSliceAxis(modulation, axis: 1, index: 1);
        var x = ApplyMlpStack(parts.DecoderEmbedding, Modulate(PlainNorm(h), shift, scale));
        return parts.DecoderOutput.Forward(x);
    }

    #endregion

    #region Training objective

    /// <summary>One window's training rows: noised futures, the noise, the steps and the clean target.</summary>
    private sealed class DenoisingDraw
    {
        public DenoisingDraw(Window window, Tensor<T> noisy, Tensor<T> noise, int[] steps, double[] cleanNormalized)
        { Window = window; Noisy = noisy; Noise = noise; Steps = steps; CleanNormalized = cleanNormalized; }
        public Window Window { get; }
        public Tensor<T> Noisy { get; }
        public Tensor<T> Noise { get; }
        public int[] Steps { get; }
        public double[] CleanNormalized { get; }
    }

    /// <summary>
    /// Normalizes the target with the window's statistics and draws TrainingBatchSize (step, noise)
    /// rows. Antithetic steps pair k with K-1-k (the reference "uniform" step distribution).
    /// </summary>
    private DenoisingDraw DrawDenoisingBatch(Window window, double[] target, Random rand, bool antithetic)
    {
        int rows = Math.Max(1, _trainingBatchSize);
        int perRow = _forecastHorizon * _numFeatures;
        var clean = NormalizeTarget(window, target);

        var steps = new int[rows];
        for (int r = 0; r < rows; r++)
        {
            steps[r] = antithetic && (r % 2) == 1
                ? _diffusionSteps - 1 - steps[r - 1]
                : rand.Next(_diffusionSteps);
        }

        var noise = new Tensor<T>(new[] { rows, _forecastHorizon, _numFeatures });
        var noisy = new Tensor<T>(new[] { rows, _forecastHorizon, _numFeatures });
        var noiseSpan = noise.Data.Span;
        var noisySpan = noisy.Data.Span;
        for (int r = 0; r < rows; r++)
        {
            double a = NumOps.ToDouble(_sqrtAlphasCumprod[steps[r]]);
            double b = NumOps.ToDouble(_sqrtOneMinusAlphasCumprod[steps[r]]);
            for (int i = 0; i < perRow; i++)
            {
                T epsilon = SampleStandardNormal(rand);
                noiseSpan[r * perRow + i] = epsilon;
                noisySpan[r * perRow + i] = NumOps.FromDouble(a * clean[i] + b * NumOps.ToDouble(epsilon));
            }
        }
        return new DenoisingDraw(window, noisy, noise, steps, clean);
    }

    /// <summary>y_0 normalized by the context statistics of its own variable; missing tail = 0.</summary>
    private double[] NormalizeTarget(Window window, double[] target)
    {
        var clean = new double[target.Length];
        for (int i = 0; i < target.Length; i++)
        {
            int f = i % _numFeatures;
            clean[i] = double.IsNaN(target[i]) ? 0.0 : (target[i] - window.Mean[f]) / window.Std[f];
        }
        return clean;
    }

    private Tensor<T> DenoiseDraws(List<DenoisingDraw> draws)
    {
        var predictions = new Tensor<T>[draws.Count];
        for (int w = 0; w < draws.Count; w++)
            predictions[w] = Denoise(draws[w].Window.Tokens, draws[w].Noisy, draws[w].Steps, draws[w].Steps.Length);
        return predictions.Length == 1 ? predictions[0] : Engine.TensorConcatenate(predictions, axis: 0);
    }

    /// <summary>
    /// Equation 6: the noise-prediction loss over every drawn row, plus lambda times the InfoNCE
    /// term when ContrastiveWeight is positive.
    /// </summary>
    private Tensor<T> TrainingLoss(List<DenoisingDraw> draws, Random negativeRandom)
    {
        var predicted = DenoiseDraws(draws);
        var noise = draws.Count == 1 ? draws[0].Noise : Engine.TensorConcatenate(draws.Select(x => x.Noise).ToArray(), axis: 0);
        var loss = TapeLoss(predicted, noise);
        if (_contrastiveWeight <= 0) return loss;

        var contrast = ContrastiveLoss(draws, predicted, noise, negativeRandom);
        return Engine.TensorAdd(loss, Engine.TensorMultiplyScalar(contrast, NumOps.FromDouble(_contrastiveWeight)));
    }

    private Tensor<T> TapeLoss(Tensor<T> predicted, Tensor<T> target)
    {
        if (_lossFunction is LossFunctionBase<T> tapeLoss) return tapeLoss.ComputeTapeLoss(predicted, target);
        throw new InvalidOperationException(
            $"CCDM trains through the autodiff tape, which needs a LossFunctionBase<T>; {_lossFunction.GetType().Name} is not one.");
    }

    /// <summary>
    /// The denoising-based temporal contrastive loss (paper Equation 4, reference cal_train_loss with
    /// loss_type "similarity"): the cosine similarity between predicted and true noise scores the
    /// real future against NumNegatives patch-shuffled and NumNegatives rescaled ones, each noised
    /// with fresh noise at the same step, and InfoNCE with temperature tau picks out the real one.
    /// </summary>
    private Tensor<T> ContrastiveLoss(List<DenoisingDraw> draws, Tensor<T> predicted, Tensor<T> noise, Random rand)
    {
        int perRow = _forecastHorizon * _numFeatures;
        var positive = CosineSimilarity(predicted, noise, perRow);                      // [R, 1]
        var logits = new List<Tensor<T>> { positive };

        foreach (var mode in new[] { false, true })
        {
            var negatives = new Tensor<T>[draws.Count];
            for (int w = 0; w < draws.Count; w++)
            {
                var draw = draws[w];
                int rows = draw.Steps.Length;
                int n = _numNegatives;
                var negNoisy = new Tensor<T>(new[] { rows * n, _forecastHorizon, _numFeatures });
                var negNoise = new Tensor<T>(new[] { rows * n, _forecastHorizon, _numFeatures });
                var negSteps = new int[rows * n];
                var noisySpan = negNoisy.Data.Span;
                var noiseSpan = negNoise.Data.Span;
                for (int j = 0; j < n; j++)
                {
                    var augmented = mode ? ScaledNegative(draw.CleanNormalized, j, n, rand) : ShuffledNegative(draw.CleanNormalized, rand);
                    for (int r = 0; r < rows; r++)
                    {
                        int row = r * n + j;
                        negSteps[row] = draw.Steps[r];
                        double a = NumOps.ToDouble(_sqrtAlphasCumprod[draw.Steps[r]]);
                        double b = NumOps.ToDouble(_sqrtOneMinusAlphasCumprod[draw.Steps[r]]);
                        for (int i = 0; i < perRow; i++)
                        {
                            T epsilon = SampleStandardNormal(rand);
                            noiseSpan[row * perRow + i] = epsilon;
                            noisySpan[row * perRow + i] = NumOps.FromDouble(a * augmented[i] + b * NumOps.ToDouble(epsilon));
                        }
                    }
                }
                var negPredicted = Denoise(draw.Window.Tokens, negNoisy, negSteps, rows * n);
                negatives[w] = Engine.Reshape(CosineSimilarity(negPredicted, negNoise, perRow), new[] { rows, n });
            }
            logits.Add(negatives.Length == 1 ? negatives[0] : Engine.TensorConcatenate(negatives, axis: 0));
        }

        var scaled = Engine.TensorMultiplyScalar(Engine.TensorConcatenate(logits.ToArray(), axis: 1), NumOps.FromDouble(1.0 / _contrastiveTemperature));
        var probabilities = Engine.Softmax(scaled, axis: -1);
        int total = probabilities.Shape[0];
        var positiveProbability = Engine.Reshape(Engine.TensorSliceAxis(probabilities, axis: 1, index: 0), new[] { total, 1 });
        var nll = Engine.TensorNegate(Engine.TensorLog(Engine.TensorAddScalar(positiveProbability, NumOps.FromDouble(1e-12))));
        return Engine.ReduceMean(nll, new[] { 0, 1 }, keepDims: false);
    }

    /// <summary>Cosine similarity of each row of two [rows, H, D] tensors, as [rows, 1].</summary>
    private Tensor<T> CosineSimilarity(Tensor<T> a, Tensor<T> b, int perRow)
    {
        int rows = a.Length / perRow;
        var fa = Engine.Reshape(a, new[] { rows, perRow });
        var fb = Engine.Reshape(b, new[] { rows, perRow });
        var dot = Engine.ReduceSum(Engine.TensorMultiply(fa, fb), new[] { 1 }, keepDims: true);
        var na = Engine.TensorSqrt(Engine.ReduceSum(Engine.TensorMultiply(fa, fa), new[] { 1 }, keepDims: true));
        var nb = Engine.TensorSqrt(Engine.ReduceSum(Engine.TensorMultiply(fb, fb), new[] { 1 }, keepDims: true));
        var denominator = Engine.TensorAddScalar(Engine.TensorMultiply(na, nb), NumOps.FromDouble(1e-8));
        return Engine.TensorDivide(dot, denominator);
    }

    /// <summary>
    /// The reference "variation" negative: the future cut into patches of 8 steps whose order is
    /// shuffled (every variable together). A horizon shorter than two patches shuffles single steps.
    /// </summary>
    private double[] ShuffledNegative(double[] clean, Random rand)
    {
        int patch = _forecastHorizon >= 2 * NegativePatchSize ? NegativePatchSize : 1;
        int patches = _forecastHorizon / patch;
        var order = Enumerable.Range(0, patches).ToArray();
        for (int i = patches - 1; i > 0; i--) { int j = rand.Next(i + 1); (order[i], order[j]) = (order[j], order[i]); }
        var result = (double[])clean.Clone();
        for (int p = 0; p < patches; p++)
            for (int t = 0; t < patch; t++)
                for (int f = 0; f < _numFeatures; f++)
                    result[(p * patch + t) * _numFeatures + f] = clean[(order[p] * patch + t) * _numFeatures + f];
        return result;
    }

    /// <summary>
    /// The reference "scaling" negative: the future rescaled per variable by a factor from
    /// [1.5, 2.0] (first half of the negatives) or [0, 0.5] (second half).
    /// </summary>
    private double[] ScaledNegative(double[] clean, int index, int count, Random rand)
    {
        bool up = index < count / 2;
        var factors = new double[_numFeatures];
        for (int f = 0; f < _numFeatures; f++)
            factors[f] = up ? 1.5 + 0.5 * rand.NextDouble() : 0.5 * rand.NextDouble();
        var result = new double[clean.Length];
        for (int i = 0; i < clean.Length; i++) result[i] = clean[i] * factors[i % _numFeatures];
        return result;
    }

    #endregion

    #region Sampling

    /// <summary>
    /// Ancestral DDPM sampling (reference DDPM.sampling / p_sample) for NumSamples paths per window,
    /// then the per-position median (or the requested quantiles), mapped back to the data scale.
    /// </summary>
    /// <remarks>
    /// The posterior is the reference q_posterior for the noise parameterization:
    /// mean = (y_k - beta_k / sqrt(1 - alphaBar_k) * epsilon-hat) / sqrt(alpha_k) and variance
    /// (1 - alphaBar_{k-1}) beta_k / (1 - alphaBar_k), which is zero at k = 0 so the path ends
    /// denoised. The noise restarts at <c>Options.Seed</c> so repeated forecasts agree; the paths
    /// within one forecast still differ, so their spread is a real estimate.
    /// </remarks>
    private Tensor<T> ForwardNative(Tensor<T> input, double[]? quantiles)
    {
        var windows = SplitWindows(input);
        int samples = Math.Max(1, _numSamples);
        int perRow = _forecastHorizon * _numFeatures;
        var rand = _options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(_options.Seed.Value)
            : RandomHelper.CreateSecureRandom();

        var results = new List<Tensor<T>>(windows.Count);
        foreach (var window in windows)
        {
            var yk = new Tensor<T>(new[] { samples, _forecastHorizon, _numFeatures });
            var span = yk.Data.Span;
            for (int i = 0; i < span.Length; i++) span[i] = SampleStandardNormal(rand);

            for (int k = _diffusionSteps - 1; k >= 0; k--)
            {
                var epsilon = Denoise(window.Tokens, yk, new[] { k }, samples).ToArray();
                double beta = NumOps.ToDouble(_betas[k]);
                double alpha = NumOps.ToDouble(_alphas[k]);
                double bar = NumOps.ToDouble(_alphasCumprod[k]);
                double barPrev = NumOps.ToDouble(_alphasCumprodPrev[k]);
                double coefficient = beta / Math.Sqrt(Math.Max(1e-20, 1.0 - bar));
                double invSqrtAlpha = 1.0 / Math.Sqrt(alpha);
                double sigma = Math.Sqrt(Math.Max(0.0, (1.0 - barPrev) * beta / Math.Max(1e-20, 1.0 - bar)));

                var next = new Tensor<T>(new[] { samples, _forecastHorizon, _numFeatures });
                var nextSpan = next.Data.Span;
                var current = yk.Data.Span;
                for (int i = 0; i < nextSpan.Length; i++)
                {
                    double mean = (NumOps.ToDouble(current[i]) - coefficient * NumOps.ToDouble(epsilon[i])) * invSqrtAlpha;
                    double z = NumOps.ToDouble(SampleStandardNormal(rand));
                    nextSpan[i] = NumOps.FromDouble(mean + sigma * z);
                }
                yk = next;
            }

            results.Add(Summarize(yk, window, samples, perRow, quantiles));
        }

        if (quantiles is null) return ShapeLikeForecast(input, results, rowsPerWindow: 1);

        var joined = results.Count == 1 ? results[0] : Engine.TensorConcatenate(results.ToArray(), axis: 0);
        return _numFeatures == 1
            ? Engine.Reshape(joined, new[] { results.Count, _forecastHorizon, quantiles.Length })
            : Engine.Reshape(joined, new[] { results.Count, _forecastHorizon, _numFeatures, quantiles.Length });
    }

    /// <summary>
    /// Per-position median ([1, H, D]) or quantiles ([1, H, D, Q]) of the sample paths, mapped back to
    /// the data scale with the window's statistics.
    /// </summary>
    private Tensor<T> Summarize(Tensor<T> paths, Window window, int samples, int perRow, double[]? quantiles)
    {
        var values = paths.ToArray();
        var levels = quantiles ?? new[] { 0.5 };
        var result = new Tensor<T>(quantiles is null
            ? new[] { 1, _forecastHorizon, _numFeatures }
            : new[] { 1, _forecastHorizon, _numFeatures, levels.Length });
        var span = result.Data.Span;
        var column = new double[samples];
        for (int i = 0; i < perRow; i++)
        {
            int f = i % _numFeatures;
            for (int s = 0; s < samples; s++)
                column[s] = NumOps.ToDouble(values[s * perRow + i]) * window.Std[f] + window.Mean[f];
            Array.Sort(column);
            for (int q = 0; q < levels.Length; q++)
                span[i * levels.Length + q] = NumOps.FromDouble(EmpiricalQuantile(column, levels[q]));
        }
        return result;
    }

    /// <summary>Linearly interpolated empirical quantile of a sorted sample (the median for 0.5).</summary>
    private static double EmpiricalQuantile(double[] sorted, double level)
    {
        if (sorted.Length == 1) return sorted[0];
        double position = level * (sorted.Length - 1);
        int lower = (int)Math.Floor(position);
        int upper = Math.Min(sorted.Length - 1, lower + 1);
        double fraction = position - lower;
        return sorted[lower] + (sorted[upper] - sorted[lower]) * fraction;
    }

    protected override Tensor<T> ForecastOnnx(Tensor<T> input) { if (OnnxSession == null) throw new InvalidOperationException("ONNX session is not initialized."); int batchSize = input.Rank > 1 ? input.Shape[0] : 1; int seqLen = input.Rank > 1 ? input.Shape[1] : input.Length; int features = input.Rank > 2 ? input.Shape[2] : 1; var inputData = new float[batchSize * seqLen * features]; for (int i = 0; i < input.Length && i < inputData.Length; i++) inputData[i] = (float)NumOps.ToDouble(input[i]); var inputTensor = new OnnxTensors.DenseTensor<float>(inputData, new[] { batchSize, seqLen, features }); string inputName = OnnxSession.InputMetadata.Keys.FirstOrDefault() ?? "input"; var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(inputName, inputTensor) }; using var results = OnnxSession.Run(inputs); var outputTensor = results.First().AsTensor<float>(); var outputShape = outputTensor.Dimensions.ToArray(); var output = new Tensor<T>(outputShape); int totalElements = 1; foreach (var dim in outputShape) totalElements *= dim; for (int i = 0; i < totalElements && i < output.Length; i++) output.Data.Span[i] = NumOps.FromDouble(outputTensor.GetValue(i)); return output; }

    #endregion

    #region ITrainingObjectiveProvider

    /// <summary>
    /// The learner is denoising diffusion, not supervised regression of the forecast onto the target:
    /// <see cref="Train"/> minimizes Equation 6 and <see cref="Predict"/> is an ancestral sampler run
    /// on top of it.
    /// </summary>
    TrainingObjectiveKind ITrainingObjectiveProvider<T>.TrainingObjectiveKind =>
        TrainingObjectiveKind.DiffusionDenoising;

    /// <summary>The supplied forecast target IS the y_0 the denoiser learns to recover.</summary>
    Tensor<T> ITrainingObjectiveProvider<T>.ResolveTrainingTarget(Tensor<T> input, Tensor<T> proposedTarget)
        => proposedTarget;

    /// <summary>
    /// Equation 6 over a FIXED quadrature: an even grid of diffusion steps with seeded noise (and
    /// seeded negatives when the contrastive term is on), so two evaluations of an unchanged model
    /// agree and a before/after comparison is a statement about the parameters.
    /// </summary>
    T ITrainingObjectiveProvider<T>.EvaluateTrainingObjective(Tensor<T> input, Tensor<T> target)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("The training objective is only defined in native mode.");

        var windows = SplitWindows(input);
        var targets = SplitTargets(target, windows.Count);
        var draws = new List<DenoisingDraw>(windows.Count);
        int perRow = _forecastHorizon * _numFeatures;
        for (int w = 0; w < windows.Count; w++)
        {
            var clean = NormalizeTarget(windows[w], targets[w]);
            var cleanTensor = new Tensor<T>(new[] { perRow });
            for (int i = 0; i < perRow; i++) cleanTensor[i] = NumOps.FromDouble(clean[i]);
            var (noisy, noise, steps) = BuildDeterministicDenoisingBatch(
                cleanTensor, perRow, _diffusionSteps, _sqrtAlphasCumprod, _sqrtOneMinusAlphasCumprod);
            draws.Add(new DenoisingDraw(
                windows[w],
                Engine.Reshape(noisy, new[] { steps.Length, _forecastHorizon, _numFeatures }),
                Engine.Reshape(noise, new[] { steps.Length, _forecastHorizon, _numFeatures }),
                steps, clean));
        }

        var loss = TrainingLoss(draws, RandomHelper.CreateSeededRandom(ObjectiveNegativeSeed));
        return loss.Length > 0 ? loss[0] : NumOps.Zero;
    }

    #endregion
}
