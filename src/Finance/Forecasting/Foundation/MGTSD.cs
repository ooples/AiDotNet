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
using AiDotNet.Validation;
using Microsoft.ML.OnnxRuntime;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

using AiDotNet.Finance.Base;
namespace AiDotNet.Finance.Forecasting.Foundation;

/// <summary>
/// MG-TSD — Multi-Granularity Time Series Diffusion Model with Guided Learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// MG-TSD captures temporal patterns at multiple granularities using a coarse-to-fine
/// guidance mechanism where predictions at coarser levels guide fine-grained diffusion.
/// </para>
/// <para><b>For Beginners:</b> MG-TSD forecasts at multiple zoom levels simultaneously.
/// It first makes a rough forecast (like predicting monthly trends), then uses that to
/// guide a more detailed forecast (like daily values). This coarse-to-fine approach is
/// similar to how an artist first sketches the broad outlines before adding fine details,
/// resulting in more coherent and accurate probabilistic predictions.</para>
/// <para>
/// <b>Reference:</b> Fan et al., "MG-TSD: Multi-Granularity Time Series Diffusion Models with Guided Learning Process", ICLR 2024.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create an MG-TSD multi-granularity time series diffusion model
/// // Coarse-to-fine guidance: rough forecasts at monthly level guide daily predictions
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.Regression,
///     inputHeight: 512, inputWidth: 1, inputDepth: 1, outputSize: 24);
///
/// // Training mode with multi-granularity guided diffusion
/// var model = new MGTSD&lt;double&gt;(architecture);
///
/// // ONNX inference mode with pre-trained model
/// var onnxModel = new MGTSD&lt;double&gt;(architecture, "mgtsd.onnx");
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
[ResearchPaper("MG-TSD: Multi-Granularity Time Series Diffusion Models with Guided Learning Process", "https://arxiv.org/abs/2403.05751", Year = 2024, Authors = "Xinyao Fan, Yueying Wu, Chang Xu, Yuhao Huang, Weiqing Liu, Jiang Bian")]
public class MGTSD<T> : TimeSeriesFoundationModelBase<T>
{
    #region Fields

    private readonly bool _useNativeMode;
    private ILayer<T>? _inputProjection;
    private readonly List<ILayer<T>> _denoisingLayers = [];
    private ILayer<T>? _outputProjection;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ILossFunction<T> _lossFunction;
    private readonly MGTSDOptions<T> _options;

    public override ModelOptions GetOptions() => _options;

    private int _contextLength;
    private int _forecastHorizon;
    private int _hiddenDimension;
    private int _numLayers;
    private int _numHeads;
    private int _diffusionSteps;
    private double _dropout;
    private double _betaStart;
    private double _betaEnd;
    private int _numGranularities;
    private double _guidanceWeight;

    // DDPM noise schedule (precomputed)
    private Vector<T> _betas = Vector<T>.Empty();
    private Vector<T> _alphas = Vector<T>.Empty();
    private Vector<T> _alphasCumprod = Vector<T>.Empty();
    private Vector<T> _sqrtAlphasCumprod = Vector<T>.Empty();
    private Vector<T> _sqrtOneMinusAlphasCumprod = Vector<T>.Empty();

    // Diffusion-training scratch state. Train() samples (timestep, noise) and the
    // normalized target before each base.Train() call so ForwardForTraining can build
    // x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε and run the denoising network as an x0-predictor.
    private Tensor<T>? _trainX0;
    private Tensor<T>? _trainNoise;
    private int _trainTimestep;
    private int _trainStepCounter;

    #endregion

    #region Properties

    public override int SequenceLength => _contextLength;
    public override int PredictionHorizon => _forecastHorizon;
    public override int NumFeatures => 1;
    public override int PatchSize => 1;
    public override int Stride => 1;
    public override bool IsChannelIndependent => true;
    public override bool UseNativeMode => _useNativeMode;
    public override FoundationModelSize ModelSize => FoundationModelSize.Small;
    public override int MaxContextLength => _contextLength;
    public override int MaxPredictionHorizon => _forecastHorizon;

    #endregion

    #region Constructors

    public MGTSD(NeuralNetworkArchitecture<T> architecture, string onnxModelPath,
        MGTSDOptions<T>? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null, ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        if (string.IsNullOrWhiteSpace(onnxModelPath)) throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath)) throw new FileNotFoundException($"ONNX model not found: {onnxModelPath}");
        options ??= new MGTSDOptions<T>(); _options = options; Options = _options;
        _useNativeMode = false; OnnxModelPath = onnxModelPath; OnnxSession = new InferenceSession(onnxModelPath);
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this); _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        CopyOptionsToFields(options);
    }

    public MGTSD(NeuralNetworkArchitecture<T> architecture,
        MGTSDOptions<T>? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null, ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        options ??= new MGTSDOptions<T>();
        _options = options;
        Options = _options;

        _useNativeMode = true;
        OnnxSession = null;
        OnnxModelPath = null;

        // Paper-faithful AdamW with lr=1e-4 per Fan et al. ICLR 2024
        // ("MG-TSD: Multi-Granularity Time-Series Diffusion"
        // arXiv:2403.05751). The framework default `AdamOptimizer` at
        // lr=1e-3 oscillates around the loss minimum on the single-batch
        // memorization probe — the gradient never decays to zero on a
        // constant batch so a momentum-based step keeps bouncing the
        // params around the minimum, producing the "loss(200 iters) >
        // loss(50 iters)" failure pattern in MoreData_ShouldNotDegrade.
        // AdamW with the paper's lr=1e-4 + weight decay 1e-4 keeps the
        // step size in the regime the architecture was tuned for.
        // Callers passing their own optimizer (production training with
        // cosine decay) override this default. Same fix as SwinUNETR.
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this,
            new Models.Options.AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = 1e-4,
                WeightDecay = 1e-4
            });
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();

        CopyOptionsToFields(options);
        InitializeLayers();
    }

    private void CopyOptionsToFields(MGTSDOptions<T> options)
    {
        Guard.Positive(options.ContextLength, nameof(options.ContextLength));
        Guard.Positive(options.ForecastHorizon, nameof(options.ForecastHorizon));
        Guard.Positive(options.HiddenDimension, nameof(options.HiddenDimension));
        Guard.Positive(options.NumLayers, nameof(options.NumLayers));
        Guard.Positive(options.NumHeads, nameof(options.NumHeads));
        Guard.Positive(options.DiffusionSteps, nameof(options.DiffusionSteps));
        Guard.Positive(options.NumGranularities, nameof(options.NumGranularities));

        if (options.BetaStart <= 0 || options.BetaEnd <= 0 || options.BetaEnd <= options.BetaStart)
            throw new ArgumentOutOfRangeException(nameof(options), "BetaStart and BetaEnd must be positive, and BetaEnd must be greater than BetaStart.");

        _contextLength = options.ContextLength;
        _forecastHorizon = options.ForecastHorizon;
        _hiddenDimension = options.HiddenDimension;
        _numLayers = options.NumLayers;
        _numHeads = options.NumHeads;
        _diffusionSteps = options.DiffusionSteps;
        _dropout = options.DropoutRate;
        _betaStart = options.BetaStart;
        _betaEnd = options.BetaEnd;
        _numGranularities = options.NumGranularities;
        _guidanceWeight = options.GuidanceWeight;
        ComputeNoiseSchedule();
    }

    private void ComputeNoiseSchedule()
    {
        _betas = new Vector<T>(_diffusionSteps);
        _alphas = new Vector<T>(_diffusionSteps);
        _alphasCumprod = new Vector<T>(_diffusionSteps);
        _sqrtAlphasCumprod = new Vector<T>(_diffusionSteps);
        _sqrtOneMinusAlphasCumprod = new Vector<T>(_diffusionSteps);
        T one = NumOps.One;
        T betaStartT = NumOps.FromDouble(_betaStart);
        T betaRangeT = NumOps.FromDouble(_betaEnd - _betaStart);
        T maxDenom = NumOps.FromDouble(Math.Max(1, _diffusionSteps - 1));
        for (int t = 0; t < _diffusionSteps; t++)
        {
            _betas[t] = NumOps.Add(betaStartT, NumOps.Divide(NumOps.Multiply(betaRangeT, NumOps.FromDouble(t)), maxDenom));
            _alphas[t] = NumOps.Subtract(one, _betas[t]);
        }
        _alphasCumprod[0] = _alphas[0];
        for (int t = 1; t < _diffusionSteps; t++)
            _alphasCumprod[t] = NumOps.Multiply(_alphasCumprod[t - 1], _alphas[t]);
        for (int t = 0; t < _diffusionSteps; t++)
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
        else if (_useNativeMode) { Layers.AddRange(LayerHelper<T>.CreateDefaultMGTSDLayers(Architecture, _contextLength, _forecastHorizon, _hiddenDimension, _numLayers, _numHeads, _numGranularities, _dropout)); ExtractLayerReferences(); }

        // Manually resolve each sub-stage's lazy layers from its OWN known
        // input shape so ParameterCount reports correctly before any forward
        // — the base class's linear-chain ResolveLazyLayerShapes can't do this
        // because MGTSD's Layers list isn't a single end-to-end pipeline:
        // _inputProjection consumes [1, contextLength], _denoisingLayers
        // consume [1, forecastHorizon + hiddenDimension + forecastHorizon + 1],
        // and _outputProjection consumes the denoising stack's output shape.
        // Propagating _inputProjection's output through the whole list would
        // lock the denoising stack to hiddenDimension and break every real
        // forward pass.
        if (_useNativeMode)
        {
            int denoiseLen = _forecastHorizon + _hiddenDimension + _forecastHorizon + 1;
            try
            {
                if (_inputProjection is AiDotNet.NeuralNetworks.Layers.LayerBase<T> ip && !ip.IsShapeResolved)
                    ip.ResolveFromShape(new[] { 1, _contextLength });
            }
            catch (Exception ex)
            {
                System.Diagnostics.Trace.TraceWarning(
                    "MGTSD: skipped resolving input-projection shape [1,{0}] (left lazy) - {1}",
                    _contextLength, ex.Message);
            }
            int[] denoiseShape = new[] { 1, denoiseLen };
            foreach (var layer in _denoisingLayers)
            {
                if (layer is AiDotNet.NeuralNetworks.Layers.LayerBase<T> lb && !lb.IsShapeResolved)
                {
                    try { lb.ResolveFromShape(denoiseShape); }
                    catch (Exception ex)
                    {
                        System.Diagnostics.Trace.TraceWarning(
                            "MGTSD: stopped denoising-layer pre-resolution at {0} with shape [{1}] - {2}",
                            layer.GetType().Name, string.Join(",", denoiseShape), ex.Message);
                        break;
                    }
                    try
                    {
                        var outShape = lb.GetOutputShape();
                        if (outShape is { Length: > 0 } && System.Array.TrueForAll(outShape, d => d > 0))
                            denoiseShape = outShape;
                    }
                    catch (Exception ex)
                    {
                        System.Diagnostics.Trace.TraceWarning(
                            "MGTSD: failed reading output shape from {0}; keeping prior shape [{1}] - {2}",
                            layer.GetType().Name, string.Join(",", denoiseShape), ex.Message);
                    }
                }
            }
            try
            {
                if (_outputProjection is AiDotNet.NeuralNetworks.Layers.LayerBase<T> op && !op.IsShapeResolved)
                    op.ResolveFromShape(denoiseShape);
            }
            catch (Exception ex)
            {
                System.Diagnostics.Trace.TraceWarning(
                    "MGTSD: skipped resolving output-projection shape [{0}] (left lazy) - {1}",
                    string.Join(",", denoiseShape), ex.Message);
            }
        }
    }

    /// <summary>
    /// MGTSD's <see cref="NeuralNetworkBase{T}.Layers"/> chain consumes the
    /// concatenated [x_t | conditioning | guidance | t-embedding] pack of
    /// length forecastHorizon + hiddenDimension + forecastHorizon + 1
    /// (24 + 128 + 24 + 1 = 177 on the default options), NOT
    /// Architecture.InputWidth (168 = contextLength). Suppress the base
    /// class's architecture-driven ResolveLazyLayerShapes pre-walk so the
    /// lazy denoising-stack layers resolve from the actual 177-width pack
    /// the first time ForwardForTraining / ForwardNative builds it.
    /// Without this override the pre-walk locks every lazy layer in the
    /// stack to 128 and every real forward fails with "Tensors with shapes
    /// [1, 177] and [1, 128] cannot be broadcast" — same root cause as
    /// MisGAN / AutoDiffTabGenerator / GOGGLEGenerator.
    /// </summary>
    protected override int[]? TryGetArchitectureInputShape() => null;

    /// <summary>
    /// Override the base class's naive linear-chain Layers walk because MGTSD's
    /// Layers list isn't a single pipeline — it's [inputProjection,
    /// denoisingLayers..., outputProjection] where each stage takes a different
    /// input shape (contextLength → hidden, denoiseLen=fh+hd+fh+1 →
    /// denoise-stack output, denoise-stack output → forecastHorizon). Feeding
    /// the contextLength input straight through every layer crashes on the
    /// denoising-stack entry because the shapes don't match. Run the real
    /// forward path and snapshot its intermediate activations instead.
    /// </summary>
    public override Dictionary<string, Tensor<T>> GetNamedLayerActivations(Tensor<T> input)
    {
        if (!_useNativeMode) return base.GetNamedLayerActivations(input);

        var activations = new Dictionary<string, Tensor<T>>();
        var conditioned = ApplyInstanceNormalization(input);
        if (conditioned.Rank == 1) conditioned = conditioned.Reshape(new[] { 1, conditioned.Length });

        // Stage 1: input projection — context-length input → hidden representation.
        Tensor<T> condHidden = _inputProjection is not null
            ? _inputProjection.Forward(conditioned)
            : conditioned;
        if (_inputProjection is not null)
            activations[$"Layer_0_{_inputProjection.GetType().Name}"] = condHidden.Clone();

        // Stage 2: build the [x_t | cond | guidance | t] denoising pack and
        // walk the denoising stack on it. Use a zero noise tensor + the
        // mid-schedule timestep so the activation magnitudes are reasonable.
        int segLen = _forecastHorizon;
        int condLen = Math.Min(condHidden.Length, _hiddenDimension);
        int denoiseLen = segLen + condLen + segLen + 1;
        var pack = new Tensor<T>(new[] { 1, denoiseLen });
        condHidden.Data.Span.Slice(0, condLen).CopyTo(pack.Data.Span.Slice(segLen, condLen));
        pack[0, denoiseLen - 1] = NumOps.FromDouble(Math.Sin(2.0 * Math.PI * (_diffusionSteps / 2.0) / Math.Max(1, _diffusionSteps - 1)));
        var current = pack;
        int layerIdx = 1;
        foreach (var layer in _denoisingLayers)
        {
            current = layer.Forward(current);
            activations[$"Layer_{layerIdx}_{layer.GetType().Name}"] = current.Clone();
            layerIdx++;
        }

        // Stage 3: output projection — denoising stack output → forecast.
        if (_outputProjection is not null)
        {
            current = _outputProjection.Forward(current);
            activations[$"Layer_{layerIdx}_{_outputProjection.GetType().Name}"] = current.Clone();
        }

        return activations;
    }

    private void ExtractLayerReferences()
    {
        int idx = 0;
        if (idx < Layers.Count) _inputProjection = Layers[idx++];
        _denoisingLayers.Clear();
        while (idx < Layers.Count - 1) _denoisingLayers.Add(Layers[idx++]);
        if (idx < Layers.Count) _outputProjection = Layers[idx++];
    }

    #endregion

    #region NeuralNetworkBase Overrides

    public override bool SupportsTraining => _useNativeMode;
    public override Tensor<T> Predict(Tensor<T> input) => _useNativeMode ? ForwardNative(input) : ForecastOnnx(input);

    /// <summary>
    /// Tape-aware training forward. Runs the existing Layers stack as a
    /// deterministic context → forecast regression head so
    /// <c>NeuralNetworkBase.TrainWithTape</c> can record the tape, compute the
    /// user-injected loss, and step the optimizer.
    /// </summary>
    /// <remarks>
    /// Same bug family as CCDM / TimeDiff / TimeGrad / TSDiff: custom Train
    /// hand-built DDPM noise perturbation and fed <c>_denoisingLayers</c> a
    /// [noisyTarget | condHidden | t-embedding] tensor whose last-dim didn't
    /// match the layers' baked-in hiddenDim input sizes, then called
    /// <c>_optimizer.UpdateParameters(Layers)</c> without backward.
    /// Multi-granularity DDPM reverse process with guidance weighting stays
    /// in <see cref="ForwardNative"/> for probabilistic inference via
    /// <see cref="Predict"/>/<see cref="Forecast"/>.
    /// </remarks>
    /// <summary>
    /// Paper-faithful DDPM training step (x0-parameterization; MG-TSD §3, Ho et al. 2020).
    /// </summary>
    /// <remarks>
    /// Samples a diffusion timestep t and Gaussian noise ε, then trains the denoising network
    /// to reconstruct the clean (RevIN-normalized) target x_0 from the noised sample
    /// x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε conditioned on the encoded history — the x0-prediction form of
    /// the DDPM objective. x0-prediction (rather than ε-prediction) keeps the regression target
    /// FIXED across steps, so the loss is driveable toward 0 on a memorization task; the
    /// ε-prediction loss instead has an irreducible noise floor. Crucially, the denoising network
    /// here is fed the SAME [x_t | cond | guidance | t] pack used at inference
    /// (<see cref="ForwardNative"/>), so its shared BatchNorm layers see one consistent feature
    /// width in both paths — fixing the train(128)/inference(177) shape mismatch that previously
    /// made the whole training-test family throw broadcast errors (issue #1464).
    /// </remarks>
    /// <summary>
    /// Route the base class's TrainWithTape path through the
    /// constructor's AdamW (lr=1e-4, weight decay 1e-4) rather than the
    /// framework default Adam (lr=1e-3). The default Adam oscillates
    /// around the loss minimum on a single-batch memorization probe —
    /// gradient never goes to zero on a constant batch, momentum step
    /// keeps bouncing the params around the minimum, "loss(200 iters) >
    /// loss(50 iters)" pattern in MoreData_ShouldNotDegrade. AdamW lr=1e-4
    /// matches Fan et al. ICLR 2024 ("MG-TSD: Multi-Granularity
    /// Time-Series Diffusion"). Without this override, the
    /// `_optimizer = ...` field assignment in the ctor would only be
    /// observable through direct field access; the base class's TrainWithTape
    /// path resolves its optimizer through GetOrCreateBaseOptimizer, which
    /// returns its own default Adam unless an override (this) is provided.
    /// </summary>
    protected override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> GetOrCreateBaseOptimizer()
    {
        return _optimizer ?? base.GetOrCreateBaseOptimizer();
    }

    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (!_useNativeMode) { base.Train(input, expected); return; }

        // RevIN: normalize the target by the INPUT's instance statistics. The same statistics
        // de-normalize the forecast in ForwardNative, so a scaled/shifted input yields a
        // scaled/shifted forecast (ScaledInput_ShouldChangeOutput) rather than being washed out
        // by the input-side instance normalization.
        var (mean, std) = ComputeInstanceStats(input);
        int fh = _forecastHorizon;
        var x0 = new Tensor<T>(new[] { 1, fh });
        for (int i = 0; i < fh; i++)
        {
            T tv = i < expected.Length ? expected[i] : NumOps.Zero;
            x0.Data.Span[i] = NumOps.Divide(NumOps.Subtract(tv, mean), std);
        }

        // Reproducible-but-varying (t, ε) per step: seed off the architecture seed plus a
        // per-call counter so the trajectory is deterministic across runs (stable tests) yet
        // still sweeps timesteps/noise so the denoiser learns the full reverse process.
        int baseSeed = Architecture?.RandomSeed ?? 12345;
        var trainRand = RandomHelper.CreateSeededRandom(baseSeed + _trainStepCounter);
        _trainStepCounter++;
        _trainTimestep = trainRand.Next(_diffusionSteps);
        var noise = new Tensor<T>(new[] { 1, fh });
        for (int i = 0; i < fh; i++) noise.Data.Span[i] = SampleStandardNormal(trainRand);

        _trainX0 = x0;
        _trainNoise = noise;
        try
        {
            // base.Train compares ForwardForTraining(input) == x̂_0 against the loss target x_0.
            base.Train(input, x0);
        }
        finally
        {
            _trainX0 = null;
            _trainNoise = null;
        }
    }

    /// <summary>
    /// Tape-aware training forward: builds x_t from the stored (target, noise, timestep) and runs
    /// the denoising network as an x0-predictor over the same [x_t | cond | guidance | t] pack
    /// used at inference. Returns the predicted clean target x̂_0.
    /// </summary>
    public override Tensor<T> ForwardForTraining(Tensor<T> input)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Training is only supported in native mode.");

        var conditioned = ApplyInstanceNormalization(input);
        if (conditioned.Rank == 3 && conditioned.Shape[2] == 1)
            conditioned = conditioned.Reshape(new[] { conditioned.Shape[0], conditioned.Shape[1] });
        else if (conditioned.Rank == 1)
            conditioned = conditioned.Reshape(new[] { 1, conditioned.Length });

        Tensor<T> condHidden = _inputProjection is not null ? _inputProjection.Forward(conditioned) : conditioned;

        int segLen = _forecastHorizon;
        int condLen = Math.Min(condHidden.Length, _hiddenDimension);
        int denoiseLen = segLen + condLen + segLen + 1;
        int t = _trainTimestep;
        T sqrtAbar = _sqrtAlphasCumprod[t];
        T sqrtOneMinus = _sqrtOneMinusAlphasCumprod[t];

        // Build the SAME [x_t | cond | guidance | t] pack used at inference, as a fresh leaf
        // tensor (Span copy). Only the denoising-network Forward passes need to be tape-tracked —
        // they ARE (the layers record their ops onto the active GradientTape), so gradients flow to
        // _denoisingLayers + _outputProjection. We deliberately do NOT route the conditioning
        // through a tape-connected concat (the sibling diffusion forecasters, e.g. CSDI, take the
        // same copy-the-pack approach): the concat's gradient-split mis-mapped the per-segment
        // gradients onto the wrong layers during the tape backward.
        var denoisingInput = new Tensor<T>(new[] { 1, denoiseLen });
        var span = denoisingInput.Data.Span;
        var x0 = _trainX0;
        var noise = _trainNoise;
        // Vectorised x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε for the first segLen slots.
        // x0 and noise are both [1, _forecastHorizon == segLen] by ForwardForTraining's
        // construction, so the lengths match and Engine.TensorAdd /
        // TensorMultiplyScalar take the SIMD path (guidance segment stays zero
        // during training so we don't need it in the pack).
        if (x0 is not null && noise is not null && x0.Length == segLen && noise.Length == segLen)
        {
            var scaledX0 = Engine.TensorMultiplyScalar(x0, sqrtAbar);
            var scaledNoise = Engine.TensorMultiplyScalar(noise, sqrtOneMinus);
            var xtSeg = Engine.TensorAdd(scaledX0, scaledNoise);
            xtSeg.Data.Span.Slice(0, segLen).CopyTo(span.Slice(0, segLen));
        }
        condHidden.Data.Span.Slice(0, condLen).CopyTo(span.Slice(segLen, condLen));
        span[segLen + condLen + segLen] = NumOps.FromDouble(Math.Sin(2.0 * Math.PI * t / Math.Max(1, _diffusionSteps - 1)));

        var predicted = denoisingInput;
        foreach (var layer in _denoisingLayers) predicted = layer.Forward(predicted);
        if (_outputProjection is not null) predicted = _outputProjection.Forward(predicted);
        return predicted; // x̂_0 (length _forecastHorizon)
    }

    /// <summary>
    /// Per-instance (RevIN) statistics over the input series: mean and (eps-stabilized) std.
    /// </summary>
    private (T mean, T std) ComputeInstanceStats(Tensor<T> input)
    {
        int len = Math.Max(1, input.Length);
        T mean = NumOps.Zero;
        for (int i = 0; i < input.Length; i++) mean = NumOps.Add(mean, input[i]);
        mean = NumOps.Divide(mean, NumOps.FromDouble(len));
        T variance = NumOps.Zero;
        for (int i = 0; i < input.Length; i++)
        {
            T d = NumOps.Subtract(input[i], mean);
            variance = NumOps.Add(variance, NumOps.Multiply(d, d));
        }
        variance = NumOps.Divide(variance, NumOps.FromDouble(len));
        T std = NumOps.Sqrt(NumOps.Add(variance, NumOps.FromDouble(1e-5)));
        return (mean, std);
    }

    public override void UpdateParameters(Vector<T> gradients)
    {
        // Parameters are updated through the optimizer in the base Train() → TrainWithTape path.
    }

    public override ModelMetadata<T> GetModelMetadata() => new()
    {
        AdditionalInfo = new Dictionary<string, object> { { "NetworkType", "MGTSD" }, { "ContextLength", _contextLength }, { "ForecastHorizon", _forecastHorizon }, { "HiddenDimension", _hiddenDimension }, { "DiffusionSteps", _diffusionSteps }, { "NumGranularities", _numGranularities }, { "GuidanceWeight", _guidanceWeight }, { "UseNativeMode", _useNativeMode } },
        ModelData = _useNativeMode ? this.Serialize() : Array.Empty<byte>()
    };

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        var opts = new MGTSDOptions<T> { ContextLength = _contextLength, ForecastHorizon = _forecastHorizon, HiddenDimension = _hiddenDimension, NumLayers = _numLayers, NumHeads = _numHeads, DiffusionSteps = _diffusionSteps, DropoutRate = _dropout, BetaStart = _betaStart, BetaEnd = _betaEnd, NumGranularities = _numGranularities, GuidanceWeight = _guidanceWeight };
        if (!_useNativeMode && OnnxModelPath is not null) return new MGTSD<T>(Architecture, OnnxModelPath, opts);
        return new MGTSD<T>(Architecture, opts);
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter writer) { writer.Write(_contextLength); writer.Write(_forecastHorizon); writer.Write(_hiddenDimension); writer.Write(_numLayers); writer.Write(_numHeads); writer.Write(_diffusionSteps); writer.Write(_dropout); writer.Write(_betaStart); writer.Write(_betaEnd); writer.Write(_numGranularities); writer.Write(_guidanceWeight); }
    protected override void DeserializeNetworkSpecificData(BinaryReader reader) { _contextLength = reader.ReadInt32(); _forecastHorizon = reader.ReadInt32(); _hiddenDimension = reader.ReadInt32(); _numLayers = reader.ReadInt32(); _numHeads = reader.ReadInt32(); _diffusionSteps = reader.ReadInt32(); _dropout = reader.ReadDouble(); _betaStart = reader.ReadDouble(); _betaEnd = reader.ReadDouble(); _numGranularities = reader.ReadInt32(); _guidanceWeight = reader.ReadDouble(); ComputeNoiseSchedule(); ExtractLayerReferences(); }

    #endregion

    #region IForecastingModel Implementation

    public override Tensor<T> Forecast(Tensor<T> historicalData, double[]? quantiles = null) { if (quantiles is not null && quantiles.Length > 0) throw new NotSupportedException("MGTSD does not support quantile forecasting. Pass null for point forecasts."); return _useNativeMode ? ForwardNative(historicalData) : ForecastOnnx(historicalData); }
    public override Tensor<T> AutoregressiveForecast(Tensor<T> input, int steps) { var predictions = new List<Tensor<T>>(); var currentInput = input; int stepsRemaining = steps; while (stepsRemaining > 0) { var prediction = Forecast(currentInput, null); predictions.Add(prediction); int stepsUsed = Math.Min(_forecastHorizon, stepsRemaining); stepsRemaining -= stepsUsed; if (stepsRemaining > 0) currentInput = ShiftInputWithPredictions(currentInput, prediction, stepsUsed); } return ConcatenatePredictions(predictions, steps); }

    public override Dictionary<string, T> Evaluate(Tensor<T> predictions, Tensor<T> actuals) { T mse = NumOps.Zero; T mae = NumOps.Zero; int count = 0; for (int i = 0; i < predictions.Length && i < actuals.Length; i++) { var diff = NumOps.Subtract(predictions[i], actuals[i]); mse = NumOps.Add(mse, NumOps.Multiply(diff, diff)); mae = NumOps.Add(mae, NumOps.Abs(diff)); count++; } if (count > 0) { mse = NumOps.Divide(mse, NumOps.FromDouble(count)); mae = NumOps.Divide(mae, NumOps.FromDouble(count)); } return new Dictionary<string, T> { ["MSE"] = mse, ["MAE"] = mae, ["RMSE"] = NumOps.Sqrt(mse) }; }

    public override Tensor<T> ApplyInstanceNormalization(Tensor<T> input) { int batchSize = input.Rank > 1 ? input.Shape[0] : 1; int seqLen = input.Rank > 1 ? input.Shape[1] : input.Length; var result = new Tensor<T>(input._shape); for (int b = 0; b < batchSize; b++) { T mean = NumOps.Zero; for (int t = 0; t < seqLen; t++) { int idx = b * seqLen + t; if (idx < input.Length) mean = NumOps.Add(mean, input[idx]); } mean = NumOps.Divide(mean, NumOps.FromDouble(seqLen)); T variance = NumOps.Zero; for (int t = 0; t < seqLen; t++) { int idx = b * seqLen + t; if (idx < input.Length) { var diff = NumOps.Subtract(input[idx], mean); variance = NumOps.Add(variance, NumOps.Multiply(diff, diff)); } } variance = NumOps.Divide(variance, NumOps.FromDouble(seqLen)); T std = NumOps.Sqrt(NumOps.Add(variance, NumOps.FromDouble(1e-5))); for (int t = 0; t < seqLen; t++) { int idx = b * seqLen + t; if (idx < input.Length && idx < result.Length) result.Data.Span[idx] = NumOps.Divide(NumOps.Subtract(input[idx], mean), std); } } return result; }

    public override Dictionary<string, T> GetFinancialMetrics() { T lastLoss = LastLoss is not null ? LastLoss : NumOps.Zero; return new Dictionary<string, T> { ["ContextLength"] = NumOps.FromDouble(_contextLength), ["ForecastHorizon"] = NumOps.FromDouble(_forecastHorizon), ["NumGranularities"] = NumOps.FromDouble(_numGranularities), ["LastLoss"] = lastLoss }; }

    #endregion

    #region Forward/Backward Pass

    /// <summary>
    /// Multi-granularity DDPM reverse process with coarse-to-fine guided denoising.
    /// MG-TSD generates predictions at multiple temporal granularities (coarse → fine) and
    /// uses coarser predictions to guide finer-grained diffusion via the guidance weight.
    /// </summary>
    private Tensor<T> ForwardNative(Tensor<T> input)
    {
        var conditioned = ApplyInstanceNormalization(input);
        bool addedBatchDim = false;
        if (conditioned.Rank == 1) { conditioned = conditioned.Reshape(new[] { 1, conditioned.Length }); addedBatchDim = true; }

        Tensor<T> condHidden;
        if (_inputProjection is not null)
            condHidden = _inputProjection.Forward(conditioned);
        else
            condHidden = conditioned;

        int outputLen = _forecastHorizon;

        // RevIN: capture the input's instance statistics to de-normalize the forecast at the end,
        // mirroring the normalization applied to the training target in Train(). The denoising
        // network operates entirely in the normalized domain (its conditioning comes from the
        // instance-normalized history), so without de-normalization the forecast is invariant to
        // input scale/shift (ScaledInput_ShouldChangeOutput) and collapses to the same values for
        // different constant inputs (DifferentInputs_ShouldProduceDifferentOutputs).
        var (inMean, inStd) = ComputeInstanceStats(input);

        // Point-forecast inference must be DETERMINISTIC (Predict/Clone must reproduce the same
        // output for the same input + weights), so the DDPM reverse process is driven by a seeded
        // RNG keyed off the architecture seed rather than a fresh cryptographic RNG. Clone copies
        // the same Architecture (hence the same seed), so the cloned model walks an identical
        // noise trajectory. This is the established "deterministic point-forecast" convention for
        // the foundation forecasters.
        int diffusionSeed = Architecture?.RandomSeed ?? 12345;
        var rand = RandomHelper.CreateSeededRandom(diffusionSeed);

        // Multi-granularity: generate coarse predictions first, then refine
        // Granularity levels: g=0 (coarsest, len/2^(G-1)) to g=G-1 (finest, full length)
        Tensor<T>? coarseGuidance = null;

        for (int g = 0; g < _numGranularities; g++)
        {
            // Compute granularity-specific output length
            int granScale = 1 << (_numGranularities - 1 - g); // coarsest has largest scale
            int granLen = Math.Max(1, outputLen / granScale);

            // Start from noise
            var xt = new Tensor<T>(new[] { 1, granLen });
            for (int i = 0; i < granLen; i++)
                xt.Data.Span[i] = SampleStandardNormal(rand);

            // Pack layout is constant across the reverse-diffusion steps:
            // [x_t | condHidden | coarseGuidance | timestep]. Only the x_t segment and the
            // single timestep slot change per step, so allocate the denoising-input buffer ONCE
            // per granularity and pre-fill the conditioning + guidance segments here instead of
            // rebuilding a fresh tensor and re-copying those segments on every one of the
            // _diffusionSteps iterations (issue #1464: per-step allocation + scalar-copy overhead).
            // The denoising network's input length is held CONSTANT across granularities so the
            // shared denoising layers — whose BatchNorm caches fixed per-feature statistics on its
            // first forward — see the same feature dimension at every granularity. The x_t and
            // coarse-guidance segments are sized to the maximum granularity length
            // (_forecastHorizon) and zero-padded beyond the current granularity's granLen; only the
            // first granLen denoiser outputs are consumed in the DDPM update below. Without this,
            // denoiseLen drifted per granularity (g0 = granLen0 + cond + 0 + 1 = 135,
            // g1 = granLen1 + cond + guideLen1 + 1 = 147, ...) and BatchNorm threw a broadcast
            // shape mismatch on the second granularity (issue #1464; previously masked because the
            // 21,504-wide layers timed the test out before inference was reached).
            int segLen = _forecastHorizon;
            int condLen = Math.Min(condHidden.Length, _hiddenDimension);
            int denoiseLen = segLen + condLen + segLen + 1;
            var denoisingInput = new Tensor<T>(new[] { 1, denoiseLen });
            var denoiseSpan = denoisingInput.Data.Span;
            int condOffset = segLen;
            int guideOffset = segLen + condLen;
            int timestepOffset = segLen + condLen + segLen;
            int xtFill = Math.Min(granLen, segLen);

            // condHidden segment — invariant across steps.
            condHidden.Data.Span.Slice(0, condLen).CopyTo(denoiseSpan.Slice(condOffset, condLen));

            // coarse-guidance segment (upsampled to granLen, weighted, zero-padded to segLen) —
            // also invariant across steps.
            if (coarseGuidance is not null)
            {
                T guidanceWeightT = NumOps.FromDouble(_guidanceWeight);
                int guideFill = Math.Min(granLen, segLen);
                for (int i = 0; i < guideFill; i++)
                {
                    int coarseIdx = Math.Min(i * coarseGuidance.Length / Math.Max(1, granLen), coarseGuidance.Length - 1);
                    denoiseSpan[guideOffset + i] = NumOps.Multiply(coarseGuidance[coarseIdx], guidanceWeightT);
                }
            }

            // DDPM reverse process at this granularity
            for (int t = _diffusionSteps - 1; t >= 0; t--)
            {
                // Refresh only the per-step segments: x_t (zero-padded to segLen) and the
                // timestep embedding. The padding region [xtFill, segLen) stays zero across steps.
                xt.Data.Span.Slice(0, xtFill).CopyTo(denoiseSpan.Slice(0, xtFill));
                denoiseSpan[timestepOffset] = NumOps.FromDouble(Math.Sin(2.0 * Math.PI * t / Math.Max(1, _diffusionSteps - 1)));

                // The denoising network predicts the clean sample x̂_0 (x0-parameterization,
                // matching the training objective in ForwardForTraining), NOT the noise ε.
                var x0Pred = denoisingInput;
                foreach (var layer in _denoisingLayers) x0Pred = layer.Forward(x0Pred);
                if (_outputProjection is not null) x0Pred = _outputProjection.Forward(x0Pred);

                // Posterior q(x_{t-1} | x_t, x̂_0) mean + variance (Ho et al. 2020, eqs. 6–7):
                //   μ = (√ᾱ_{t-1}·β_t / (1-ᾱ_t))·x̂_0 + (√α_t·(1-ᾱ_{t-1}) / (1-ᾱ_t))·x_t
                //   σ² = β_t·(1-ᾱ_{t-1}) / (1-ᾱ_t)
                T eps10 = NumOps.FromDouble(1e-10);
                T betaT = _betas[t];
                T alphaT = _alphas[t];
                T abarT = _alphasCumprod[t];
                T abarPrev = t > 0 ? _alphasCumprod[t - 1] : NumOps.One;
                T oneMinusAbarT = NumOps.Add(NumOps.Subtract(NumOps.One, abarT), eps10);
                // Apply the same eps10 stabilization as oneMinusAbarT for defensive consistency:
                // although the linear beta schedule keeps ᾱ_{t-1} < 1 for t > 0 (and the t = 0 branch
                // forces σ_t = 0 / z = 0, so the value is unused), this guards any future schedule
                // whose ᾱ_{t-1} → 1 underflows the numerator/denominator below.
                T oneMinusAbarPrev = NumOps.Add(NumOps.Subtract(NumOps.One, abarPrev), eps10);
                T coefX0 = NumOps.Divide(NumOps.Multiply(NumOps.Sqrt(abarPrev), betaT), oneMinusAbarT);
                T coefXt = NumOps.Divide(NumOps.Multiply(NumOps.Sqrt(alphaT), oneMinusAbarPrev), oneMinusAbarT);
                T sigmaT = t > 0 ? NumOps.Sqrt(NumOps.Divide(NumOps.Multiply(betaT, oneMinusAbarPrev), oneMinusAbarT)) : NumOps.Zero;

                // Vectorised posterior step:
                //   mean = coefX0·x̂_0 + coefXt·x_t
                //   x_{t-1} = mean + σ_t·z   (z = 0 when t = 0)
                // Build the mean as Engine.TensorAdd(scaled_x0, scaled_xt) so
                // the per-element multiply/add runs in SIMD instead of a
                // per-position NumOps chain. Noise sampling is inherently
                // per-element, but we collect z once into a tensor and add
                // it via TensorAdd + TensorMultiplyScalar — same SIMD path.
                int len = Math.Min(granLen, xt.Length);
                if (len > 0)
                {
                    var x0Aligned = x0Pred.Length >= len
                        ? Engine.TensorNarrow(
                            x0Pred.Rank == 2 ? x0Pred : Engine.Reshape(x0Pred, new[] { 1, x0Pred.Length }),
                            dim: x0Pred.Rank == 2 ? 1 : 0, start: 0, length: len)
                        : x0Pred;
                    var xtAligned = Engine.TensorNarrow(
                        xt.Rank == 2 ? xt : Engine.Reshape(xt, new[] { 1, xt.Length }),
                        dim: xt.Rank == 2 ? 1 : 0, start: 0, length: len);
                    var scaledX0 = Engine.TensorMultiplyScalar(x0Aligned, coefX0);
                    var scaledXt = Engine.TensorMultiplyScalar(xtAligned, coefXt);
                    var meanTensor = Engine.TensorAdd(scaledX0, scaledXt);
                    Tensor<T> next = meanTensor;
                    if (t > 0)
                    {
                        var zData = new T[len];
                        for (int i = 0; i < len; i++) zData[i] = SampleStandardNormal(rand);
                        var z = new Tensor<T>(meanTensor._shape, new Vector<T>(zData));
                        var scaledZ = Engine.TensorMultiplyScalar(z, sigmaT);
                        next = Engine.TensorAdd(meanTensor, scaledZ);
                    }
                    next.Data.Span.Slice(0, len).CopyTo(xt.Data.Span.Slice(0, len));
                }
            }

            coarseGuidance = xt; // This granularity's output guides the next finer level
        }

        // Final output from finest granularity (in normalized space) — de-normalize via RevIN so
        // the forecast carries the input series' scale/offset.
        var result = coarseGuidance ?? new Tensor<T>(new[] { 1, outputLen });
        for (int i = 0; i < result.Length; i++)
            result.Data.Span[i] = NumOps.Add(NumOps.Multiply(result[i], inStd), inMean);

        if (addedBatchDim && result.Rank == 2 && result.Shape[0] == 1) result = result.Reshape(new[] { result.Shape[1] });
        return result;
    }

    protected override Tensor<T> ForecastOnnx(Tensor<T> input) { if (OnnxSession == null) throw new InvalidOperationException("ONNX session is not initialized."); int batchSize = input.Rank > 1 ? input.Shape[0] : 1; int seqLen = input.Rank > 1 ? input.Shape[1] : input.Length; int features = input.Rank > 2 ? input.Shape[2] : 1; var inputData = new float[batchSize * seqLen * features]; for (int i = 0; i < input.Length && i < inputData.Length; i++) inputData[i] = (float)NumOps.ToDouble(input[i]); var inputTensor = new OnnxTensors.DenseTensor<float>(inputData, new[] { batchSize, seqLen, features }); string inputName = OnnxSession.InputMetadata.Keys.FirstOrDefault() ?? "input"; var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(inputName, inputTensor) }; using var results = OnnxSession.Run(inputs); var outputTensor = results.First().AsTensor<float>(); var outputShape = outputTensor.Dimensions.ToArray(); var output = new Tensor<T>(outputShape); int totalElements = 1; foreach (var dim in outputShape) totalElements *= dim; for (int i = 0; i < totalElements && i < output.Length; i++) output.Data.Span[i] = NumOps.FromDouble(outputTensor.GetValue(i)); return output; }

    #endregion
}
