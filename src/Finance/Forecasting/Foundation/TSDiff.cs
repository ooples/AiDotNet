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
using AiDotNet.Validation;
using Microsoft.ML.OnnxRuntime;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

using AiDotNet.Finance.Base;
namespace AiDotNet.Finance.Forecasting.Foundation;

/// <summary>
/// TSDiff — Self-Guiding Diffusion Models for Probabilistic Time Series Forecasting.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// TSDiff uses unconditional denoising diffusion as a self-supervised pretraining objective
/// with self-guided refinement for high-quality probabilistic forecasting.
/// </para>
/// <para><b>For Beginners:</b> TSDiff generates probabilistic forecasts using a three-step
/// process: predict, refine, and synthesize. It first learns general time series patterns
/// through diffusion (gradually adding and removing noise), then refines predictions using
/// the model's own internal guidance. This self-guided approach produces high-quality
/// forecasts with well-calibrated uncertainty estimates.</para>
/// <para>
/// <b>Reference:</b> Kollovieh et al., "Predict, Refine, Synthesize: Self-Guiding Diffusion Models for Probabilistic Time Series Forecasting", NeurIPS 2023.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a TSDiff self-guiding diffusion model for probabilistic forecasting
/// // Three-step process: predict, refine, and synthesize with self-guided refinement
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.Regression,
///     inputHeight: 512, inputWidth: 1, inputDepth: 1, outputSize: 24);
///
/// // Training mode with unconditional diffusion and self-guided refinement
/// var model = new TSDiff&lt;double&gt;(architecture);
///
/// // ONNX inference mode with pre-trained model
/// var onnxModel = new TSDiff&lt;double&gt;(architecture, "tsdiff.onnx");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Finance)]
[ModelDomain(ModelDomain.TimeSeries)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Diffusion)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.Forecasting)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Predict, Refine, Synthesize: Self-Guiding Diffusion Models for Probabilistic Time Series Forecasting", "https://arxiv.org/abs/2307.11494", Year = 2023, Authors = "Marcel Kollovieh, Abdul Fatir Ansari, Michael Bohlke-Schneider, Jasper Zschiegner, Hao Wang, Yuyang Wang")]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 1e-3, ReferenceBatchSize = 64,
                Source = "Kollovieh et al. 2023, experimental setup: Adam for 1,000 epochs with a learning rate of 1e-3, each epoch 128 batches of 64 sequences.")]
public partial class TSDiff<T> : TimeSeriesFoundationModelBase<T>, ITrainingObjectiveProvider<T>
{
    #region Fields

    private readonly bool _useNativeMode;
    private ILayer<T>? _inputProjection;
    private readonly List<ILayer<T>> _residualLayers = [];
    private ILayer<T>? _outputProjection;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ILossFunction<T> _lossFunction;
    private readonly TSDiffOptions<T> _options;

    public override ModelOptions GetOptions() => _options;

    private int _sequenceLength;
    private int _forecastHorizon;
    private int _hiddenDimension;
    private int _numResidualBlocks;
    private int _numDiffusionSteps;
    private int _numAttentionHeads;
    private double _dropout;
    private double _betaStart;
    private double _betaEnd;
    private double _guidanceScale;
    private double _unconditionalProbability;
    private int _trainingBatchSize;
    private int _numSamples;

    // DDPM noise schedule (precomputed as generic vectors)
    [Buffer]
    private Vector<T> _betas = Vector<T>.Empty();
    [Buffer]
    private Vector<T> _alphas = Vector<T>.Empty();
    [Buffer]
    private Vector<T> _alphasCumprod = Vector<T>.Empty();
    [Buffer]
    private Vector<T> _sqrtAlphasCumprod = Vector<T>.Empty();
    [Buffer]
    private Vector<T> _sqrtOneMinusAlphasCumprod = Vector<T>.Empty();

    #endregion

    #region Properties

    public override int SequenceLength => _sequenceLength;
    public override int PredictionHorizon => _forecastHorizon;
    public override int NumFeatures => 1;
    public override int PatchSize => 1;
    public override int Stride => 1;
    public override bool IsChannelIndependent => true;
    public override bool UseNativeMode => _useNativeMode;
    public override FoundationModelSize ModelSize => FoundationModelSize.Small;
    public override int MaxContextLength => _sequenceLength;
    public override int MaxPredictionHorizon => _forecastHorizon;

    #endregion

    #region Constructors

    public TSDiff(NeuralNetworkArchitecture<T> architecture, string onnxModelPath,
        TSDiffOptions<T>? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null, ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        if (string.IsNullOrWhiteSpace(onnxModelPath)) throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath)) throw new FileNotFoundException($"ONNX model not found: {onnxModelPath}");
        options ??= new TSDiffOptions<T>(); _options = options; Options = _options;
        _useNativeMode = false; OnnxModelPath = onnxModelPath; OnnxSession = new InferenceSession(onnxModelPath);
        _optimizer = optimizer
            ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this)
            ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this, new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>> { InitialLearningRate = options.LearningRate }); _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        CopyOptionsToFields(options);
    }

    public TSDiff(NeuralNetworkArchitecture<T> architecture,
        TSDiffOptions<T>? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null, ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        options ??= new TSDiffOptions<T>(); _options = options; Options = _options;
        _useNativeMode = true; OnnxSession = null; OnnxModelPath = null;
        _optimizer = optimizer
            ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this)
            ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this, new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>> { InitialLearningRate = options.LearningRate }); _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        CopyOptionsToFields(options); InitializeLayers();
    }

    private void CopyOptionsToFields(TSDiffOptions<T> options)
    {
        Guard.Positive(options.SequenceLength, nameof(options.SequenceLength));
        Guard.Positive(options.ForecastHorizon, nameof(options.ForecastHorizon));
        Guard.Positive(options.HiddenDimension, nameof(options.HiddenDimension));
        Guard.Positive(options.NumResidualBlocks, nameof(options.NumResidualBlocks));
        Guard.Positive(options.NumDiffusionSteps, nameof(options.NumDiffusionSteps));
        Guard.Positive(options.NumAttentionHeads, nameof(options.NumAttentionHeads));

        if (options.BetaStart <= 0 || options.BetaEnd <= 0 || options.BetaEnd <= options.BetaStart)
            throw new ArgumentOutOfRangeException(nameof(options), "BetaStart and BetaEnd must be positive, and BetaEnd must be greater than BetaStart.");

        _sequenceLength = options.SequenceLength; _forecastHorizon = options.ForecastHorizon;
        _hiddenDimension = options.HiddenDimension; _numResidualBlocks = options.NumResidualBlocks;
        _numDiffusionSteps = options.NumDiffusionSteps; _numAttentionHeads = options.NumAttentionHeads;
        _dropout = options.DropoutRate; _betaStart = options.BetaStart;
        _betaEnd = options.BetaEnd; _guidanceScale = options.GuidanceScale;
        _unconditionalProbability = options.UnconditionalProbability;
        _trainingBatchSize = options.TrainingBatchSize;
        _numSamples = options.NumSamples;
        ComputeNoiseSchedule();
    }

    private void ComputeNoiseSchedule()
    {
        _betas = new Vector<T>(_numDiffusionSteps);
        _alphas = new Vector<T>(_numDiffusionSteps);
        _alphasCumprod = new Vector<T>(_numDiffusionSteps);
        _sqrtAlphasCumprod = new Vector<T>(_numDiffusionSteps);
        _sqrtOneMinusAlphasCumprod = new Vector<T>(_numDiffusionSteps);
        T one = NumOps.One;
        T betaStartT = NumOps.FromDouble(_betaStart);
        T betaRangeT = NumOps.FromDouble(_betaEnd - _betaStart);
        T maxDenom = NumOps.FromDouble(Math.Max(1, _numDiffusionSteps - 1));
        for (int t = 0; t < _numDiffusionSteps; t++)
        {
            _betas[t] = NumOps.Add(betaStartT, NumOps.Divide(NumOps.Multiply(betaRangeT, NumOps.FromDouble(t)), maxDenom));
            _alphas[t] = NumOps.Subtract(one, _betas[t]);
        }
        _alphasCumprod[0] = _alphas[0];
        for (int t = 1; t < _numDiffusionSteps; t++)
            _alphasCumprod[t] = NumOps.Multiply(_alphasCumprod[t - 1], _alphas[t]);
        for (int t = 0; t < _numDiffusionSteps; t++)
        {
            _sqrtAlphasCumprod[t] = NumOps.Sqrt(_alphasCumprod[t]);
            _sqrtOneMinusAlphasCumprod[t] = NumOps.Sqrt(NumOps.Subtract(one, _alphasCumprod[t]));
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
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0) { Layers.AddRange(Architecture.Layers); ExtractLayerReferences(); }
        else if (_useNativeMode) { Layers.AddRange(LayerHelper<T>.CreateDefaultTSDiffLayers(Architecture, _sequenceLength, _forecastHorizon, _hiddenDimension, _numResidualBlocks, _numAttentionHeads, _dropout)); ExtractLayerReferences(); }
    }

    private void ExtractLayerReferences()
    {
        int idx = 0;
        if (idx < Layers.Count) _inputProjection = Layers[idx++];
        _residualLayers.Clear();
        while (idx < Layers.Count - 1) _residualLayers.Add(Layers[idx++]);
        if (idx < Layers.Count) _outputProjection = Layers[idx++];
    }

    #endregion

    #region NeuralNetworkBase Overrides

    public override bool SupportsTraining => _useNativeMode;
    protected override Tensor<T> PredictCore(Tensor<T> input) => _useNativeMode ? ForwardNative(input) : ForecastOnnx(input);

    /// <summary>
    /// One epsilon-prediction training step over the same denoiser graph the sampler runs.
    /// </summary>
    /// <remarks>
    /// What this replaces ran the whole Layers stack as a deterministic context-to-forecast
    /// regression head, while <see cref="ForwardNative"/> read the very same layer objects as a
    /// denoiser over an additively composed [x_t + condition + timestep] row. Nothing inference
    /// did had ever been trained. The two paths also pack different widths, and the input
    /// projection is lazily sized, so whichever path ran first baked the layer widths.
    ///
    /// Training now follows Ho et al. 2020 Algorithm 1: draw t uniformly, draw epsilon, form
    /// x_t = sqrt(alphaBar_t) * y + sqrt(1 - alphaBar_t) * epsilon, and regress the denoiser
    /// output onto epsilon through the configured <see cref="ILossFunction{T}"/>.
    ///
    /// With probability <c>UnconditionalProbability</c> the conditioning is dropped for the step.
    /// That is Ho and Salimans, "Classifier-Free Diffusion Guidance" (2022) Section 3: a single
    /// network is trained jointly on the conditional and unconditional objectives so the guidance
    /// blend in <see cref="ForwardNative"/> has a trained unconditional estimate to blend in. It
    /// is applied whatever the guidance scale, because GuidanceScale is an inference knob and
    /// raising it after training must not find a branch the model has never seen.
    ///
    /// The draw is deliberately NOT seeded from <c>Options.Seed</c>. That seed makes inference
    /// reproducible; reusing it here would hand every call the same t and the same epsilon, so
    /// the model would be fitted at one single noise level out of NumDiffusionSteps.
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Training is only supported in native mode.");

        if (expectedOutput.Length <= 0) return;

        // Without this the denoiser's DropoutLayers stay in inference mode and the configured
        // DropoutRate is silently inert for the whole of training.
        SetTrainingMode(true);
        try
        {
            var trainableParams = Training.TapeTrainingStep<T>.CollectParameters(Layers).ToArray();

            var conditioned = ApplyInstanceNormalization(input);
            var condHidden = conditioned.Rank == 2
                ? conditioned
                : Engine.Reshape(conditioned, new[] { 1, conditioned.Length });

            // The denoiser is sized for exactly one horizon, so the target is read into a row of
            // that width whatever the caller's framing. A shorter target leaves the tail at zero
            // rather than changing the packed width, which the lazily-sized input projection
            // would bake and then reject on the next call.
            int outputLen = _forecastHorizon;
            var rand = RandomHelper.CreateSecureRandom();

            // One row per (timestep, noise) draw. Ho et al. 2020 Algorithm 1 draws a single t per
            // EXAMPLE and averages the step over a minibatch (128 in their Section 4); a caller
            // here hands us one example, so the minibatch is taken over the noise process instead.
            // Without it the gradient is a one-sample estimate of an expectation over
            // NumDiffusionSteps noise levels, so consecutive steps are dominated by which t came
            // up rather than by what the model learned - and the reported loss is too, which is
            // what made the training probes read as noise. The conditioning drop is decided per
            // row for the same reason: an all-or-nothing decision per step would train the two
            // branches in alternating bursts.
            int batch = Math.Max(1, _trainingBatchSize);
            var timesteps = new int[batch];
            var dropConditioning = new bool[batch];
            var epsilonTrue = new Tensor<T>(new[] { batch, outputLen });
            var xt = new Tensor<T>(new[] { batch, outputLen });
            for (int b = 0; b < batch; b++)
            {
                int tb = rand.Next(_numDiffusionSteps);
                timesteps[b] = tb;
                dropConditioning[b] = rand.NextDouble() < _unconditionalProbability;
                T sqrtAlphaBar = _sqrtAlphasCumprod[tb];
                T sqrtOneMinus = _sqrtOneMinusAlphasCumprod[tb];
                for (int i = 0; i < outputLen; i++)
                {
                    T noise = SampleStandardNormal(rand);
                    T y = i < expectedOutput.Length ? expectedOutput[i] : NumOps.Zero;
                    int flat = b * outputLen + i;
                    epsilonTrue.Data.Span[flat] = noise;
                    xt.Data.Span[flat] = NumOps.Add(
                        NumOps.Multiply(sqrtAlphaBar, y),
                        NumOps.Multiply(sqrtOneMinus, noise));
                }
            }
            using var tape = new GradientTape<T>();
            var epsilonPred = DenoiserForward(xt, condHidden, timesteps, dropConditioning);
            var lossTensor = _lossFunction.ComputeTapeLoss(epsilonPred, epsilonTrue);

            // Publish through the base rather than calling tape.ComputeGradients directly.
            // GetParameterGradients() answers from the published surface, and with nothing
            // published it falls back to the per-layer accessors, which fabricate an exact zero
            // for every parameter - indistinguishable from a severed tape.
            var grads = ComputeAndPublishParameterGradients(tape, lossTensor, trainableParams);

            T lossValue = lossTensor.Length > 0 ? lossTensor[0] : NumOps.Zero;
            LastLoss = lossValue;

            // All three closures are pinned to this step's (t, epsilon, conditioning). A
            // line-searching optimizer that re-drew them would be comparing losses from two
            // different noise levels and reading the difference as progress.
            Tensor<T> ComputeForward(Tensor<T> _, Tensor<T> __) =>
                DenoiserForward(xt, condHidden, timesteps, dropConditioning);
            Tensor<T> RecomputeLoss(Tensor<T> pred, Tensor<T> __) =>
                _lossFunction.ComputeTapeLoss(pred, epsilonTrue);

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
    /// Tape-aware forward over the denoiser graph, at the noisiest step, from pure noise. This is
    /// the function <see cref="Train"/> fits and <see cref="ForwardNative"/> iterates, so a caller
    /// that records a tape over it is recording the graph inference actually uses.
    /// </summary>
    public override Tensor<T> ForwardForTraining(Tensor<T> input)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Training is only supported in native mode.");

        var conditioned = ApplyInstanceNormalization(input);
        var condHidden = conditioned.Rank == 2
            ? conditioned
            : Engine.Reshape(conditioned, new[] { 1, conditioned.Length });

        var rand = _options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(_options.Seed.Value)
            : RandomHelper.CreateSecureRandom();
        var xt = new Tensor<T>(new[] { 1, _forecastHorizon });
        for (int i = 0; i < _forecastHorizon; i++)
            xt.Data.Span[i] = SampleStandardNormal(rand);

        return DenoiserForward(xt, condHidden, _numDiffusionSteps - 1);
    }

    /// <summary>
    /// The denoiser: one epsilon estimate for <paramref name="xt"/> at diffusion step
    /// <paramref name="t"/>, optionally conditioned on the encoded context.
    /// </summary>
    /// <remarks>
    /// Ho et al. 2020 Section 3.2 add the timestep to the residual stream through an embedding
    /// rather than concatenating it, so x_t, the conditioning and the timestep are summed at the
    /// hidden width. The sum is built with Engine ops because the conditioning arrives through
    /// ApplyInstanceNormalization and the whole composition has to stay on the tape. The
    /// unconditional estimate the guidance blend needs comes from the batched overload's drop
    /// mask, which zeroes the conditioning slots without changing the packed row width.
    /// </remarks>
    private Tensor<T> DenoiserForward(Tensor<T> xt, Tensor<T>? condHidden, int t)
        => DenoiserForward(xt, condHidden, new[] { t }, dropConditioning: null);

    /// <summary>
    /// The batched denoiser: one row per (x_t, timestep) pair. A <paramref name="timesteps"/> of
    /// length one means every row shares that step, which is what the sampler needs; one entry per
    /// row is what a training step needs, since each draw sits at its own noise level. A row whose
    /// <paramref name="dropConditioning"/> entry is set gets the unconditional estimate.
    /// </summary>
    private Tensor<T> DenoiserForward(
        Tensor<T> xt, Tensor<T>? condHidden, IReadOnlyList<int> timesteps, IReadOnlyList<bool>? dropConditioning)
    {
        // One row per sample path. The additive composition is built by a direct span fill
        // rather than traced engine arithmetic because none of its three parts carries a
        // parameter dependency: x_t is either prior noise or a noised target, the conditioning
        // is a normalized copy of the input, and the timestep encoding is a constant. Only the
        // layer chain below needs the tape.
        int rows = xt.Rank == 2 ? xt.Shape[0] : 1;
        int xtCols = rows > 0 ? xt.Length / rows : 0;
        var condFlat = condHidden is null
            ? null
            : (condHidden.Rank == 1 ? condHidden : Engine.Reshape(condHidden, new[] { condHidden.Length }));
        int condLen = condFlat is null ? 0 : Math.Min(condFlat.Length, _hiddenDimension);

        // x_t, the conditioning and the timestep occupy SEPARATE slots of the packed row, and
        // _inputProjection maps the whole row to hidden width. Tashiro et al. 2021 (CSDI,
        // Section 3.3) feed the conditional observations as their own channel, and Kollovieh et
        // al. 2023 condition TSDiff the same way; only the timestep is added to the residual
        // stream (Ho et al. 2020 Section 3.2). Summing x_t INTO the conditioning slots, as this
        // did, leaves the denoiser unable to tell the noised target from its context - it saw one
        // blurred vector, and more training moved the sampler further from the target.
        int rowLen = xtCols + condLen + DiffusionTimestepEmbeddingDim;
        var packed = new Tensor<T>(new[] { rows, rowLen });
        var din = packed.Data.Span;
        for (int r = 0; r < rows; r++)
        {
            int stepR = timesteps.Count == 1 ? timesteps[0] : timesteps[r];
            bool dropR = dropConditioning is not null
                && (dropConditioning.Count == 1 ? dropConditioning[0] : dropConditioning[r]);
            int baseIdx = r * rowLen;
            for (int i = 0; i < xtCols; i++)
            {
                int flat = r * xtCols + i;
                din[baseIdx + i] = flat < xt.Length ? xt[flat] : NumOps.Zero;
            }

            // The unconditional branch keeps the same row WIDTH with the conditioning zeroed, so
            // classifier-free guidance blends two estimates from one lazily-sized projection
            // rather than baking a second input width on the first unconditional call.
            if (!dropR && condFlat is not null)
                for (int i = 0; i < condLen; i++) din[baseIdx + xtCols + i] = condFlat[i];

            WriteDiffusionTimestepEmbedding(
                din.Slice(baseIdx + xtCols + condLen, DiffusionTimestepEmbeddingDim), stepR);
        }

        var eps = packed;

        if (_inputProjection is not null) eps = _inputProjection.Forward(eps);
        foreach (var layer in _residualLayers) eps = layer.Forward(eps);
        if (_outputProjection is not null) eps = _outputProjection.Forward(eps);
        return eps;
    }

    // UpdateParameters was an empty override, silently dropping every restore. The base
    // distributes the vector over the declared enumeration.
    public override ModelMetadata<T> GetModelMetadata() => new()
    {
        AdditionalInfo = new Dictionary<string, object> { { "NetworkType", "TSDiff" }, { "SequenceLength", _sequenceLength }, { "ForecastHorizon", _forecastHorizon }, { "HiddenDimension", _hiddenDimension }, { "NumDiffusionSteps", _numDiffusionSteps }, { "GuidanceScale", _guidanceScale }, { "NumSamples", _numSamples }, { "UseNativeMode", _useNativeMode } },
        ModelDataProvider = () => _useNativeMode ? this.Serialize() : Array.Empty<byte>()
    };




    #endregion

    #region IForecastingModel Implementation

    public override Tensor<T> Forecast(Tensor<T> historicalData, double[]? quantiles = null) { if (quantiles is not null && quantiles.Length > 0) throw new NotSupportedException("TSDiff does not support quantile forecasting. Pass null for point forecasts."); return _useNativeMode ? ForwardNative(historicalData) : ForecastOnnx(historicalData); }
    public override Tensor<T> AutoregressiveForecast(Tensor<T> input, int steps) { var predictions = new List<Tensor<T>>(); var currentInput = input; int stepsRemaining = steps; while (stepsRemaining > 0) { var prediction = Forecast(currentInput, null); predictions.Add(prediction); int stepsUsed = Math.Min(_forecastHorizon, stepsRemaining); stepsRemaining -= stepsUsed; if (stepsRemaining > 0) currentInput = ShiftInputWithPredictions(currentInput, prediction, stepsUsed); } return ConcatenatePredictions(predictions, steps); }

    public override Dictionary<string, T> Evaluate(Tensor<T> predictions, Tensor<T> actuals) { T mse = NumOps.Zero; T mae = NumOps.Zero; int count = 0; for (int i = 0; i < predictions.Length && i < actuals.Length; i++) { var diff = NumOps.Subtract(predictions[i], actuals[i]); mse = NumOps.Add(mse, NumOps.Multiply(diff, diff)); mae = NumOps.Add(mae, NumOps.Abs(diff)); count++; } if (count > 0) { mse = NumOps.Divide(mse, NumOps.FromDouble(count)); mae = NumOps.Divide(mae, NumOps.FromDouble(count)); } return new Dictionary<string, T> { ["MSE"] = mse, ["MAE"] = mae, ["RMSE"] = NumOps.Sqrt(mse) }; }

    public override Tensor<T> ApplyInstanceNormalization(Tensor<T> input)
        // RevIN forward (Kim et al. 2022), delegated to the shared tape-tracked helper. The previous
        // hand-rolled version accumulated mean/variance with scalar NumOps arithmetic and wrote the
        // output through result.Data.Span[...], which the autodiff tape cannot observe: the normalised
        // tensor came back as a LEAF, so no gradient could flow through the normalisation. RevIN is a
        // differentiable layer in the paper, not a preprocessing step.
        => NormalizeInstanceOnTape(input, DefaultRevInEpsilon, out _, out _);

    public override Dictionary<string, T> GetFinancialMetrics() { T lastLoss = LastLoss is not null ? LastLoss : NumOps.Zero; return new Dictionary<string, T> { ["SequenceLength"] = NumOps.FromDouble(_sequenceLength), ["ForecastHorizon"] = NumOps.FromDouble(_forecastHorizon), ["GuidanceScale"] = NumOps.FromDouble(_guidanceScale), ["LastLoss"] = lastLoss }; }

    #endregion

    #region Forward/Backward Pass

    /// <summary>
    /// DDPM reverse process with self-guided diffusion refinement.
    /// TSDiff uses unconditional denoising as a pretraining objective, then refines
    /// predictions at inference by blending unconditional and conditioned noise estimates.
    /// </summary>
    private Tensor<T> ForwardNative(Tensor<T> input)
    {
        var conditioned = ApplyInstanceNormalization(input);
        bool addedBatchDim = false;
        if (conditioned.Rank == 1) { conditioned = conditioned.Reshape(new[] { 1, conditioned.Length }); addedBatchDim = true; }

        // Raw conditioning context. As in CSDI (Tashiro 2021) / the layer-helper
        // layout, _inputProjection projects the WHOLE packed per-step denoiser
        // input to hidden width — so the conditioning is packed RAW here, not
        // pre-projected (pre-projecting consumed _inputProjection on the
        // conditioning shape and let the raw packed input fall through to the
        // residual stack whose BatchNorm channels are hiddenDimension).
        var condHidden = conditioned.Rank == 2
            ? conditioned
            : Engine.Reshape(conditioned, new[] { 1, conditioned.Length });

        int outputLen = _forecastHorizon;
        // A diffusion forecaster is generative: one path carries the full spread of the
        // predictive distribution rather than its centre, so NumSamples paths are drawn as rows
        // and the per-position median is returned. That is the estimate Tashiro et al. (CSDI,
        // NeurIPS 2021) report and what the sibling CSDI implementation in this library does.
        // NumSamples was already declared on the options and simply never read here.
        int samples = Math.Max(1, _numSamples);

        // Restart the noise stream at the configured seed so Predict called twice on the same
        // input returns the same answer. Seeding here does not collapse the sample set: the
        // NumSamples paths draw successive values from the one stream and so still differ from
        // one another. With Seed null the draw is secure and deliberately not reproducible.
        var rand = _options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(_options.Seed.Value)
            : RandomHelper.CreateSecureRandom();

        // Start from pure Gaussian noise
        var xt = new Tensor<T>(new[] { samples, outputLen });
        for (int i = 0; i < samples * outputLen; i++)
            xt.Data.Span[i] = SampleStandardNormal(rand);

        // Iterative DDPM reverse process: t = T-1 ... 0
        for (int t = _numDiffusionSteps - 1; t >= 0; t--)
        {
            var epsCond = DenoiserForward(xt, condHidden, t);

            // Self-guided diffusion: compute unconditional estimate (no conditioning)
            // eps_guided = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            // When guidance_scale > 1, this amplifies the conditioned signal
            Tensor<T> epsGuided;
            if (Math.Abs(_guidanceScale - 1.0) > 1e-6)
            {
                // Unconditional estimate: the same denoiser with the conditioning dropped,
                // which is the branch Train's conditioning dropout keeps trained.
                // Drop the conditioning by MASK, not by passing null: the packed row keeps its
                // width so both estimates go through the one lazily-sized input projection.
                var epsUncond = DenoiserForward(xt, condHidden, new[] { t }, new[] { true });

                // Classifier-free guidance blend (Ho and Salimans 2022):
                //   eps_guided = eps_uncond + scale * (eps_cond - eps_uncond)
                // Built out of Engine SIMD ops so the per-element work
                // vectorises at paper scale (outputLen x hiddenDim >= 1024).
                T guidanceT = NumOps.FromDouble(_guidanceScale);
                var delta = Engine.TensorSubtract(epsCond, epsUncond);
                var weighted = Engine.TensorMultiplyScalar(delta, guidanceT);
                epsGuided = Engine.TensorAdd(epsUncond, weighted);
            }
            else
            {
                epsGuided = epsCond;
            }

            // DDPM reverse step: x_{t-1} = (x_t - beta_t/sqrt(1-alpha_bar_t) * eps) / sqrt(alpha_t) + sigma_t * z
            T alphaT = _alphas[t];
            T betaT = _betas[t];
            T eps10 = NumOps.FromDouble(1e-10);
            T sqrtOneMinusAlphaBarT = NumOps.Sqrt(NumOps.Subtract(NumOps.One, _alphasCumprod[t]));
            T noiseCoeffT = NumOps.Divide(betaT, NumOps.Add(sqrtOneMinusAlphaBarT, eps10));
            T sqrtAlphaT = NumOps.Sqrt(alphaT);
            T sigmaT = t > 0 ? NumOps.Sqrt(betaT) : NumOps.Zero;

            int epsCols = samples > 0 ? epsGuided.Length / samples : 0;
            for (int s = 0; s < samples; s++)
            {
                for (int i = 0; i < outputLen; i++)
                {
                    int flat = s * outputLen + i;
                    if (flat >= xt.Length) break;
                    int epsIdx = s * epsCols + i;
                    T epsVal = i < epsCols && epsIdx < epsGuided.Length ? epsGuided[epsIdx] : NumOps.Zero;
                    T meanT = NumOps.Divide(NumOps.Subtract(xt[flat], NumOps.Multiply(noiseCoeffT, epsVal)), NumOps.Add(sqrtAlphaT, eps10));
                    T z = t > 0 ? SampleStandardNormal(rand) : NumOps.Zero;
                    xt.Data.Span[flat] = NumOps.Add(meanT, NumOps.Multiply(sigmaT, z));
                }
            }
        }

        var median = MedianAcrossSamples(xt, samples, outputLen);
        if (addedBatchDim) return median;
        return Engine.Reshape(median, new[] { 1, outputLen });
    }

    /// <summary>
    /// Per-position median over the sample axis - the deterministic estimate a probabilistic
    /// forecaster reports. The same sample set is what a prediction interval would come from.
    /// </summary>
    private Tensor<T> MedianAcrossSamples(Tensor<T> paths, int samples, int outputLen)
    {
        var result = new Tensor<T>(new[] { outputLen });
        var column = new T[samples];
        for (int i = 0; i < outputLen; i++)
        {
            for (int s = 0; s < samples; s++) column[s] = paths[s * outputLen + i];
            Array.Sort(column, (a, b) => NumOps.LessThan(a, b) ? -1 : NumOps.LessThan(b, a) ? 1 : 0);
            result[i] = (samples % 2) == 1
                ? column[samples / 2]
                : NumOps.Divide(NumOps.Add(column[samples / 2 - 1], column[samples / 2]), NumOps.FromDouble(2.0));
        }

        return result;
    }

    protected override Tensor<T> ForecastOnnx(Tensor<T> input) { if (OnnxSession == null) throw new InvalidOperationException("ONNX session is not initialized."); int batchSize = input.Rank > 1 ? input.Shape[0] : 1; int seqLen = input.Rank > 1 ? input.Shape[1] : input.Length; int features = input.Rank > 2 ? input.Shape[2] : 1; var inputData = new float[batchSize * seqLen * features]; for (int i = 0; i < input.Length && i < inputData.Length; i++) inputData[i] = (float)NumOps.ToDouble(input[i]); var inputTensor = new OnnxTensors.DenseTensor<float>(inputData, new[] { batchSize, seqLen, features }); string inputName = OnnxSession.InputMetadata.Keys.FirstOrDefault() ?? "input"; var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(inputName, inputTensor) }; using var results = OnnxSession.Run(inputs); var outputTensor = results.First().AsTensor<float>(); var outputShape = outputTensor.Dimensions.ToArray(); var output = new Tensor<T>(outputShape); int totalElements = 1; foreach (var dim in outputShape) totalElements *= dim; for (int i = 0; i < totalElements && i < output.Length; i++) output.Data.Span[i] = NumOps.FromDouble(outputTensor.GetValue(i)); return output; }


    #region ITrainingObjectiveProvider

    /// <summary>
    /// The learner is denoising diffusion, not supervised regression of the forecast onto the
    /// target: <see cref="Train"/> minimizes the noise-prediction error of Ho et al. 2020
    /// Algorithm 1, and the forecast is an ancestral sampler run on top of it.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Declaring the objective is what lets a loss-trajectory probe measure the quantity training
    /// actually descends. Judging the model on the sampler instead measures something the
    /// optimizer never sees: the reverse chain divides by sqrt(alpha_bar_T) overall, about 16x on
    /// the T=100, beta_T=0.1 schedule Kollovieh et al. 2023 specify, so it magnifies whatever bias
    /// a partially trained noise predictor still has.
    /// </para>
    /// </remarks>
    TrainingObjectiveKind ITrainingObjectiveProvider<T>.TrainingObjectiveKind =>
        TrainingObjectiveKind.DiffusionDenoising;

    /// <summary>The supplied forecast target IS the x_0 the denoiser learns to recover.</summary>
    Tensor<T> ITrainingObjectiveProvider<T>.ResolveTrainingTarget(Tensor<T> input, Tensor<T> proposedTarget)
        => proposedTarget;

    /// <summary>
    /// Evaluates L_simple over a fixed (timestep, noise) quadrature, through the configured loss
    /// function so a caller who overrides it is scored on what the model is optimizing. Every row
    /// keeps its conditioning: the unconditional branch is a training-time dropout, not part of
    /// the objective being measured.
    /// </summary>
    T ITrainingObjectiveProvider<T>.EvaluateTrainingObjective(Tensor<T> input, Tensor<T> target)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("The training objective is only defined in native mode.");

        var conditioned = ApplyInstanceNormalization(input);
        var condHidden = conditioned.Rank == 2
            ? conditioned
            : Engine.Reshape(conditioned, new[] { 1, conditioned.Length });

        var (noisy, noise, timesteps) = BuildDeterministicDenoisingBatch(
            target, _forecastHorizon, _numDiffusionSteps, _sqrtAlphasCumprod, _sqrtOneMinusAlphasCumprod);

        var predicted = DenoiserForward(noisy, condHidden, timesteps, dropConditioning: null);
        return _lossFunction.ComputeLoss(predicted, noise);
    }

    #endregion
    #endregion
}
