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
/// CCDM — Conditional Continuous Diffusion Model for Time Series.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// CCDM extends continuous diffusion models for conditional time series generation,
/// operating in continuous space with a score-matching objective for high-quality
/// probabilistic forecasting.
/// </para>
/// <para><b>For Beginners:</b> CCDM generates future time series values using a diffusion
/// process, similar to how image generators create pictures by gradually refining random
/// noise. Instead of predicting a single future value, it produces a range of probable
/// outcomes, giving you confidence intervals for your forecasts. This is especially
/// useful in finance where understanding uncertainty is as important as the prediction itself.</para>
/// </remarks>
/// <example>
/// <code>
/// // Create a CCDM conditional continuous diffusion model for probabilistic forecasting
/// // Generates future values by refining random noise conditioned on observed history
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.Regression,
///     inputHeight: 512, inputWidth: 1, inputDepth: 1, outputSize: 24);
///
/// // Training mode with score-matching diffusion objective
/// var model = new CCDM&lt;double&gt;(architecture);
///
/// // ONNX inference mode with pre-trained model
/// var onnxModel = new CCDM&lt;double&gt;(architecture, "ccdm.onnx");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Finance)]
[ModelDomain(ModelDomain.TimeSeries)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Diffusion)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.Forecasting)]
[ModelComplexity(ModelComplexity.High)]
[ResearchPaper("Diffusion Variational Autoencoder for Tackling Stochasticity in Multi-Step Regression Stock Price Prediction", "https://arxiv.org/abs/2309.00073")]
    [ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
public partial class CCDM<T> : TimeSeriesFoundationModelBase<T>
{
    #region Fields

    private readonly bool _useNativeMode;
    private ILayer<T>? _inputProjection;
    private readonly List<ILayer<T>> _denoisingLayers = [];
    private ILayer<T>? _outputProjection;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ILossFunction<T> _lossFunction;
    private readonly CCDMOptions<T> _options;

    public override ModelOptions GetOptions() => _options;

    private int _contextLength;
    private int _forecastHorizon;
    private int _hiddenDimension;
    private int _numLayers;
    private int _numHeads;
    private int _diffusionSteps;
    private int _numSamples;
    private double _dropout;
    private double _betaStart;
    private double _betaEnd;

    // DDPM noise schedule (precomputed) - CCDM also uses beta schedule for discrete steps
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

    public CCDM(NeuralNetworkArchitecture<T> architecture, string onnxModelPath,
        CCDMOptions<T>? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null, ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        if (string.IsNullOrWhiteSpace(onnxModelPath)) throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath)) throw new FileNotFoundException($"ONNX model not found: {onnxModelPath}");
        // Tape-based Train() below requires a LossFunctionBase<T>
        // (ComputeTapeLoss is only on the base class, not on the
        // ILossFunction<T> interface). Reject any user-supplied loss
        // that doesn't derive from it at construction time instead of
        // bubbling up a bare InvalidCastException on first Train().
        options ??= new CCDMOptions<T>(); _options = options; Options = _options;
        _useNativeMode = false; OnnxModelPath = onnxModelPath; OnnxSession = new InferenceSession(onnxModelPath);
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this); _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        CopyOptionsToFields(options);
    }

    public CCDM(NeuralNetworkArchitecture<T> architecture,
        CCDMOptions<T>? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null, ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        options ??= new CCDMOptions<T>(); _options = options; Options = _options;
        _useNativeMode = true; OnnxSession = null; OnnxModelPath = null;
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this); _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        CopyOptionsToFields(options); InitializeLayers();
    }

    private void CopyOptionsToFields(CCDMOptions<T> options)
    {
        _contextLength = options.ContextLength; _forecastHorizon = options.ForecastHorizon;
        _hiddenDimension = options.HiddenDimension; _numLayers = options.NumLayers;
        _numHeads = options.NumHeads; _diffusionSteps = options.DiffusionSteps;
        _dropout = options.DropoutRate; _betaStart = options.BetaStart;
        _betaEnd = options.BetaEnd; _numSamples = options.NumSamples;
        ComputeNoiseSchedule();
    }

    private void ComputeNoiseSchedule()
    {
        if (_diffusionSteps <= 0)
            throw new ArgumentOutOfRangeException(nameof(_diffusionSteps), "DiffusionSteps must be positive.");

        // Standard DDPM beta schedule
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
        else if (_useNativeMode) { Layers.AddRange(LayerHelper<T>.CreateDefaultCCDMLayers(Architecture, _contextLength, _forecastHorizon, _hiddenDimension, _numLayers, _numHeads, _dropout)); ExtractLayerReferences(); }
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
    protected override Tensor<T> PredictCore(Tensor<T> input) => _useNativeMode ? ForwardNative(input) : ForecastOnnx(input);

    /// <summary>
    /// One epsilon-prediction training step over the same denoiser graph inference uses.
    /// </summary>
    /// <remarks>
    /// What this replaces trained the Layers as a plain context-to-forecast regression head
    /// while the sampler read the very same Layers as a score network on a differently packed
    /// input. Nothing inference did had ever been trained, which is why loss rose with more
    /// training, why training error exceeded test error, and why the sampler was unbounded.
    /// Following Ho et al. 2020 Algorithm 1: draw t uniformly, draw epsilon, form
    /// x_t = sqrt(alphaBar_t) * y + sqrt(1 - alphaBar_t) * epsilon, and regress the denoiser
    /// output onto epsilon through the configured <see cref="ILossFunction{T}"/>.
    ///
    /// The draw is deliberately NOT seeded from <c>Options.Seed</c>. That seed makes inference
    /// reproducible; reusing it here would hand every call the same t and the same epsilon, so
    /// the model would be fitted at one single noise level out of DiffusionSteps.
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Training is only supported in native mode.");

        if (expectedOutput.Length <= 0) return;

        // Every training entry point on the base puts the layers in training mode for the
        // duration of the step; without it the denoiser's DropoutLayers stay in inference mode
        // and the configured DropoutRate is silently inert.
        SetTrainingMode(true);
        try
        {
            var trainableParams = Training.TapeTrainingStep<T>.CollectParameters(Layers).ToArray();

            var conditioned = ApplyInstanceNormalization(input);
            if (conditioned.Rank == 1)
                conditioned = Engine.Reshape(conditioned, new[] { 1, conditioned.Length });

            // The denoiser is sized for exactly one horizon, so the target is read into a row of
            // that width whatever the caller's framing. A shorter target leaves the tail at zero
            // rather than changing the packed width, which the lazily-sized input projection
            // would bake and then reject on the next call.
            int outputLen = _forecastHorizon;
            var rand = RandomHelper.CreateSecureRandom();
            int t = rand.Next(_diffusionSteps);

            var epsilonTrue = new Tensor<T>(new[] { 1, outputLen });
            var xt = new Tensor<T>(new[] { 1, outputLen });
            T sqrtAlphaBar = _sqrtAlphasCumprod[t];
            T sqrtOneMinus = _sqrtOneMinusAlphasCumprod[t];
            for (int i = 0; i < outputLen; i++)
            {
                T noise = SampleStandardNormal(rand);
                T y = i < expectedOutput.Length ? expectedOutput[i] : NumOps.Zero;
                epsilonTrue.Data.Span[i] = noise;
                xt.Data.Span[i] = NumOps.Add(
                    NumOps.Multiply(sqrtAlphaBar, y),
                    NumOps.Multiply(sqrtOneMinus, noise));
            }

            using var tape = new GradientTape<T>();
            var epsilonPred = DenoiserForward(xt, conditioned, t, samples: 1, outputLen: outputLen);
            var lossTensor = _lossFunction.ComputeTapeLoss(epsilonPred, epsilonTrue);

            // Publish through the base rather than calling tape.ComputeGradients directly.
            // GetParameterGradients() answers from the published surface, and with nothing
            // published it falls back to the per-layer accessors, which fabricate an exact zero
            // for every parameter - indistinguishable from a severed tape.
            var grads = ComputeAndPublishParameterGradients(tape, lossTensor, trainableParams);

            T lossValue = lossTensor.Length > 0 ? lossTensor[0] : NumOps.Zero;
            LastLoss = lossValue;

            // Both closures are pinned to this step's (t, epsilon). A line-searching optimizer
            // that re-drew them would be comparing losses from two different noise levels and
            // reading the difference as progress.
            Tensor<T> ComputeForward(Tensor<T> _, Tensor<T> __) =>
                DenoiserForward(xt, conditioned, t, samples: 1, outputLen: outputLen);
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
    /// Tape-aware forward over the denoiser graph, at the first reverse-diffusion step.
    /// </summary>
    /// <remarks>
    /// This exists so that callers which probe the training graph without a target - gradient
    /// reachability checks, parameter-movement probes - drive the same packed
    /// [x_t | conditioning | sin(t)] input that <see cref="Train"/> and the sampler drive. The
    /// input projection bakes its width on its first forward, so a probe that fed the raw
    /// context here would size the model for a shape neither of the real paths uses.
    /// </remarks>
    public override Tensor<T> ForwardForTraining(Tensor<T> input)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Training is only supported in native mode.");

        var conditioned = ApplyInstanceNormalization(input);
        if (conditioned.Rank == 1)
            conditioned = Engine.Reshape(conditioned, new[] { 1, conditioned.Length });

        var rand = _options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(_options.Seed.Value)
            : RandomHelper.CreateSecureRandom();
        var xt = new Tensor<T>(new[] { 1, _forecastHorizon });
        for (int i = 0; i < _forecastHorizon; i++)
            xt.Data.Span[i] = SampleStandardNormal(rand);

        return DenoiserForward(xt, conditioned, _diffusionSteps - 1, samples: 1, outputLen: _forecastHorizon);
    }

    // UpdateParameters was an empty override, silently dropping every restore. The base
    // distributes the vector over the declared enumeration.
    public override ModelMetadata<T> GetModelMetadata() => new()
    {
        AdditionalInfo = new Dictionary<string, object> { { "NetworkType", "CCDM" }, { "ContextLength", _contextLength }, { "ForecastHorizon", _forecastHorizon }, { "HiddenDimension", _hiddenDimension }, { "DiffusionSteps", _diffusionSteps }, { "NumSamples", _numSamples }, { "UseNativeMode", _useNativeMode } },
        ModelData = _useNativeMode ? this.Serialize() : Array.Empty<byte>()
    };




    #endregion

    #region IForecastingModel Implementation

    public override Tensor<T> Forecast(Tensor<T> historicalData, double[]? quantiles = null) { if (quantiles is not null && quantiles.Length > 0) throw new NotSupportedException("CCDM does not support quantile forecasting. Pass null for point forecasts."); return _useNativeMode ? ForwardNative(historicalData) : ForecastOnnx(historicalData); }
    public override Tensor<T> AutoregressiveForecast(Tensor<T> input, int steps) { var predictions = new List<Tensor<T>>(); var currentInput = input; int stepsRemaining = steps; while (stepsRemaining > 0) { var prediction = Forecast(currentInput, null); predictions.Add(prediction); int stepsUsed = Math.Min(_forecastHorizon, stepsRemaining); stepsRemaining -= stepsUsed; if (stepsRemaining > 0) currentInput = ShiftInputWithPredictions(currentInput, prediction, stepsUsed); } return ConcatenatePredictions(predictions, steps); }

    public override Dictionary<string, T> Evaluate(Tensor<T> predictions, Tensor<T> actuals) { T mse = NumOps.Zero; T mae = NumOps.Zero; int count = 0; for (int i = 0; i < predictions.Length && i < actuals.Length; i++) { var diff = NumOps.Subtract(predictions[i], actuals[i]); mse = NumOps.Add(mse, NumOps.Multiply(diff, diff)); mae = NumOps.Add(mae, NumOps.Abs(diff)); count++; } if (count > 0) { mse = NumOps.Divide(mse, NumOps.FromDouble(count)); mae = NumOps.Divide(mae, NumOps.FromDouble(count)); } return new Dictionary<string, T> { ["MSE"] = mse, ["MAE"] = mae, ["RMSE"] = NumOps.Sqrt(mse) }; }

    public override Tensor<T> ApplyInstanceNormalization(Tensor<T> input)
        // RevIN forward (Kim et al. 2022), delegated to the shared tape-tracked helper. The previous
        // hand-rolled version accumulated mean/variance with scalar NumOps arithmetic and wrote the
        // output through result.Data.Span[...], which the autodiff tape cannot observe: the normalised
        // tensor came back as a LEAF, so no gradient could flow through the normalisation. RevIN is a
        // differentiable layer in the paper, not a preprocessing step.
        => NormalizeInstanceOnTape(input, DefaultRevInEpsilon, out _, out _);

    public override Dictionary<string, T> GetFinancialMetrics() { T lastLoss = LastLoss is not null ? LastLoss : NumOps.Zero; return new Dictionary<string, T> { ["ContextLength"] = NumOps.FromDouble(_contextLength), ["ForecastHorizon"] = NumOps.FromDouble(_forecastHorizon), ["LastLoss"] = lastLoss }; }

    #endregion

    #region Forward/Backward Pass

    /// <summary>
    /// DDPM ancestral reverse process (Ho et al. 2020, Algorithm 2) over the beta schedule
    /// <c>ComputeNoiseSchedule</c> already builds, run for <c>Options.NumSamples</c> paths whose
    /// per-position median is the returned point forecast.
    /// </summary>
    /// <remarks>
    /// This replaces an annealed-Langevin loop over a separate geometric sigma schedule, which
    /// was wrong in two ways at once. It read the Layers as a score network that nothing had
    /// ever fitted for the job - training pushed the same Layers through a plain
    /// context-to-forecast regression head on a differently packed input - and its first step
    /// multiplied that untrained score by sigma_0^2 - sigma_1^2, about 1.2e3 at the old
    /// SigmaMax of 80, so the sample ran away to 4.4e34 before the loop ended. The cited paper
    /// (arXiv:2309.00073) builds on DDPM, the DDPM schedule was already computed here and
    /// simply unused, and every sibling diffusion forecaster in this assembly (CSDI, TSDiff,
    /// DiffusionTS) samples this way, so the sigma schedule was the outlier, not the design.
    /// </remarks>
    private Tensor<T> ForwardNative(Tensor<T> input)
    {
        var conditioned = ApplyInstanceNormalization(input);
        bool addedBatchDim = false;
        if (conditioned.Rank == 1) { conditioned = conditioned.Reshape(new[] { 1, conditioned.Length }); addedBatchDim = true; }

        int outputLen = _forecastHorizon;
        int samples = Math.Max(1, _numSamples);

        // Restart the noise stream at the configured seed so Predict called twice on the same
        // input returns the same answer. The samples within a call stay distinct, so the spread
        // is still a real estimate. With Seed null the draw is secure and not reproducible.
        var rand = _options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(_options.Seed.Value)
            : RandomHelper.CreateSecureRandom();

        // x_T ~ N(0, I), one row per sample path.
        var xt = new Tensor<T>(new[] { samples, outputLen });
        for (int i = 0; i < samples * outputLen; i++)
            xt.Data.Span[i] = SampleStandardNormal(rand);

        T eps10 = NumOps.FromDouble(1e-10);
        for (int t = _diffusionSteps - 1; t >= 0; t--)
        {
            var eps = DenoiserForward(xt, conditioned, t, samples, outputLen);

            T betaT = _betas[t];
            T sqrtAlphaT = NumOps.Sqrt(_alphas[t]);
            T noiseCoeffT = NumOps.Divide(betaT, NumOps.Add(_sqrtOneMinusAlphasCumprod[t], eps10));
            // sigma_t^2 = beta_t, the first of the two variance choices in Ho et al. 2020
            // Section 3.2; the last step is the mean only, so the sample ends denoised.
            T sigmaT = t > 0 ? NumOps.Sqrt(betaT) : NumOps.Zero;

            int epsCols = samples > 0 ? eps.Length / samples : 0;
            for (int s = 0; s < samples; s++)
            {
                for (int i = 0; i < outputLen; i++)
                {
                    int flat = s * outputLen + i;
                    if (flat >= xt.Length) break;
                    int epsIdx = s * epsCols + i;
                    T epsVal = i < epsCols && epsIdx < eps.Length ? eps[epsIdx] : NumOps.Zero;
                    T meanT = NumOps.Divide(
                        NumOps.Subtract(xt[flat], NumOps.Multiply(noiseCoeffT, epsVal)),
                        NumOps.Add(sqrtAlphaT, eps10));
                    T z = t > 0 ? SampleStandardNormal(rand) : NumOps.Zero;
                    xt.Data.Span[flat] = NumOps.Add(meanT, NumOps.Multiply(sigmaT, z));
                }
            }
        }

        var median = MedianAcrossSamples(xt, samples, outputLen);
        if (!addedBatchDim) return Engine.Reshape(median, new[] { 1, outputLen });
        return median;
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

    /// <summary>
    /// Runs the denoiser once over <paramref name="samples"/> rows: packs
    /// [x_t | conditioning | sin(2*pi*t/T)] per row and returns the predicted noise.
    /// </summary>
    /// <remarks>
    /// Training and inference both go through here, which is the whole point of the method
    /// existing. The input projection from <c>LayerHelper&lt;T&gt;.CreateDefaultCCDMLayers</c>
    /// is lazily sized - it bakes its input width on the first forward it sees - so the two
    /// paths packing different widths made the model order-dependent on top of being untrained.
    ///
    /// The packed row is built by a direct span fill rather than traced engine arithmetic
    /// because none of its three parts carries a parameter dependency: x_t is either prior
    /// noise or a noised target, the conditioning is a normalized copy of the input, and the
    /// timestep encoding is a constant. Only the layer chain below needs the tape.
    /// </remarks>
    private Tensor<T> DenoiserForward(Tensor<T> xt, Tensor<T> conditioned, int t, int samples, int outputLen)
    {
        var condFlat = conditioned.Rank == 1
            ? conditioned
            : Engine.Reshape(conditioned, new[] { conditioned.Length });
        int condLen = Math.Min(condFlat.Length, _hiddenDimension);
        int rowLen = outputLen + condLen + 1;

        var packed = new Tensor<T>(new[] { samples, rowLen });
        T sinT = NumOps.FromDouble(Math.Sin(2.0 * Math.PI * t / Math.Max(1, _diffusionSteps - 1)));
        var din = packed.Data.Span;
        for (int s = 0; s < samples; s++)
        {
            int baseIdx = s * rowLen;
            for (int i = 0; i < outputLen; i++)
            {
                int flat = s * outputLen + i;
                din[baseIdx + i] = flat < xt.Length ? xt[flat] : NumOps.Zero;
            }
            for (int i = 0; i < condLen; i++) din[baseIdx + outputLen + i] = condFlat[i];
            din[baseIdx + outputLen + condLen] = sinT;
        }

        var eps = packed;
        if (_inputProjection is not null) eps = _inputProjection.Forward(eps);
        foreach (var layer in _denoisingLayers) eps = layer.Forward(eps);
        if (_outputProjection is not null) eps = _outputProjection.Forward(eps);
        return eps;
    }

    protected override Tensor<T> ForecastOnnx(Tensor<T> input) { if (OnnxSession == null) throw new InvalidOperationException("ONNX session is not initialized."); int batchSize = input.Rank > 1 ? input.Shape[0] : 1; int seqLen = input.Rank > 1 ? input.Shape[1] : input.Length; int features = input.Rank > 2 ? input.Shape[2] : 1; var inputData = new float[batchSize * seqLen * features]; for (int i = 0; i < input.Length && i < inputData.Length; i++) inputData[i] = (float)NumOps.ToDouble(input[i]); var inputTensor = new OnnxTensors.DenseTensor<float>(inputData, new[] { batchSize, seqLen, features }); string inputName = OnnxSession.InputMetadata.Keys.FirstOrDefault() ?? "input"; var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(inputName, inputTensor) }; using var results = OnnxSession.Run(inputs); var outputTensor = results.First().AsTensor<float>(); var outputShape = outputTensor.Dimensions.ToArray(); var output = new Tensor<T>(outputShape); int totalElements = 1; foreach (var dim in outputShape) totalElements *= dim; for (int i = 0; i < totalElements && i < output.Length; i++) output.Data.Span[i] = NumOps.FromDouble(outputTensor.GetValue(i)); return output; }

    #endregion
}
