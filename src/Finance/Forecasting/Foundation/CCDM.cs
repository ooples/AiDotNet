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
using AiDotNet.Tensors.Engines.Autodiff;
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
[ResearchPaper("Diffusion Variational Autoencoder for Tackling Stochasticity in Multi-Step Regression Stock Price Prediction", "https://arxiv.org/abs/2402.06010")]
    [ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
public class CCDM<T> : TimeSeriesFoundationModelBase<T>
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
    private double _dropout;
    private double _betaStart;
    private double _betaEnd;
    private double _sigmaMin;
    private double _sigmaMax;

    // DDPM noise schedule (precomputed) - CCDM also uses beta schedule for discrete steps
    private Vector<T> _betas = Vector<T>.Empty();
    private Vector<T> _alphas = Vector<T>.Empty();
    private Vector<T> _alphasCumprod = Vector<T>.Empty();
    private Vector<T> _sqrtAlphasCumprod = Vector<T>.Empty();
    private Vector<T> _sqrtOneMinusAlphasCumprod = Vector<T>.Empty();
    // Continuous sigma schedule (geometric interpolation from sigmaMax to sigmaMin)
    private Vector<T> _sigmas = Vector<T>.Empty();

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
        _betaEnd = options.BetaEnd; _sigmaMin = options.SigmaMin;
        _sigmaMax = options.SigmaMax;
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

        // Continuous sigma schedule: geometric interpolation from sigmaMax to sigmaMin
        _sigmas = new Vector<T>(_diffusionSteps + 1);
        for (int t = 0; t <= _diffusionSteps; t++)
        {
            double frac = (double)t / Math.Max(1, _diffusionSteps);
            _sigmas[t] = NumOps.FromDouble(_sigmaMax * Math.Pow(_sigmaMin / Math.Max(1e-10, _sigmaMax), frac));
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
    public override Tensor<T> Predict(Tensor<T> input) => _useNativeMode ? ForwardNative(input) : ForecastOnnx(input);

    public override void Train(Tensor<T> input, Tensor<T> target)
    {
        if (!_useNativeMode) throw new InvalidOperationException("Training is only supported in native mode.");

        // Issue #1166 — score-matching train loop rewritten to use the
        // standard GradientTape / TrainWithTape machinery every other
        // model uses. Preserves the research-paper algorithm (Continuous
        // Conditional Diffusion Model, score matching with sigma-weighted
        // MSE) while:
        //
        //   1. Making the loss injectable via the inherited
        //      `LossFunction : ILossFunction<T>` (user can supply any
        //      LossFunctionBase<T> at ctor time; default stays MSE
        //      which gives the paper's σ²·‖score_pred − score_target‖²
        //      once we pre-scale both by σ — see below).
        //   2. Actually running backward through the layer parameters
        //      instead of computing a gradient vector and throwing it
        //      away, then calling _optimizer.UpdateParameters(Layers)
        //      with no backward state populated (the old code path
        //      which produced "Backward pass must be called before
        //      updating parameters" on every train step).
        //
        // Score-matching weighting:
        //   loss = (1/N) · σ² · ‖score_pred − score_target‖²
        //        = (1/N) · ‖σ·score_pred − σ·score_target‖²
        //        = LossFunction.ComputeTapeLoss(σ·score_pred, σ·score_target)
        // when LossFunction is MSE. Other ILossFunction<T> choices
        // (Huber, Charbonnier, …) carry the pre-scaling through their
        // own loss form so users get a robust score-matching variant
        // just by injecting a different loss.

        SetTrainingMode(true);
        try
        {
            var tape = new GradientTape<T>();
            using var _ = tape;

            var rand = RandomHelper.CreateSecureRandom();
            int t = rand.Next(_diffusionSteps);
            T sigmaT = _sigmas[t];
            T eps10 = NumOps.FromDouble(1e-10);
            T invSigma = NumOps.Divide(NumOps.One, NumOps.Add(sigmaT, eps10));

            // Noise and the two derived target tensors are *constants*
            // from a gradient-flow standpoint — they don't need to be
            // tape-aware, only the network-parameters → loss path does.
            // Keep the raw-data construction here, it is correct.
            var noise = new Tensor<T>(target._shape);
            for (int i = 0; i < target.Length; i++)
                noise.Data.Span[i] = SampleStandardNormal(rand);

            var noisyTarget = new Tensor<T>(target._shape);
            var scoreTarget = new Tensor<T>(target._shape);
            for (int i = 0; i < target.Length; i++)
            {
                // x_noisy = x_0 + σ · ε
                noisyTarget.Data.Span[i] = NumOps.Add(target[i], NumOps.Multiply(sigmaT, noise[i]));
                // s_target = −ε / σ  (∇ log p_σ(x))
                scoreTarget.Data.Span[i] = NumOps.Multiply(NumOps.Negate(invSigma), noise[i]);
            }

            // Forward pass — MUST be tape-connected from model parameters
            // to predictedScore, so tape.ComputeGradients can trace back.
            var predictedScore = ForwardTraining(input, noisyTarget, t);

            // σ-pre-scaling: collapses σ²·‖·‖² into a plain MSE/Huber/etc.
            // call on the scaled inputs. `sigmaT` is a scalar, so broadcast
            // it as a same-shape tensor for the pointwise multiply.
            var sigmaTensor = new Tensor<T>(predictedScore._shape);
            for (int i = 0; i < sigmaTensor.Length; i++) sigmaTensor.Data.Span[i] = sigmaT;
            var scaledPred   = Engine.TensorMultiply(predictedScore, sigmaTensor);
            var scaledTarget = Engine.TensorMultiply(scoreTarget,    sigmaTensor);

            var lossTensor = LossFunction is LossFunctionBase<T> lfb
                ? lfb.ComputeTapeLoss(scaledPred, scaledTarget)
                : throw new InvalidOperationException(
                    "LossFunction must derive from LossFunctionBase<T> for tape-based training.");

            LastLoss = lossTensor.Length > 0 ? lossTensor[0] : NumOps.Zero;

            // Collect trainable parameters and compute gradients via the tape.
            // Use the parameterless overload; CollectParameters walks the Layers
            // on every call — fine for per-step training since the graph is
            // stable within a step and this is a diffusion per-step routine
            // rather than a training loop inner.
            var trainableParams = AiDotNet.Training.TapeTrainingStep<T>.CollectParameters(Layers);
            var grads = tape.ComputeGradients(lossTensor, trainableParams);

            T ls = lossTensor.Length > 0 ? lossTensor[0] : NumOps.Zero;
            LastLoss = ls;

            // For re-evaluation inside the optimizer's Step (line-search etc.),
            // rebuild the same forward path. noisyTarget and t stay captured
            // via the closure so the re-eval matches the original step.
            Tensor<T> ComputeForward(Tensor<T> inp, Tensor<T> _) =>
                ForwardTraining(inp, noisyTarget, t);
            Tensor<T> RecomputeLoss(Tensor<T> pred, Tensor<T> _)
            {
                var predScaled2   = Engine.TensorMultiply(pred, sigmaTensor);
                var targetScaled2 = Engine.TensorMultiply(scoreTarget, sigmaTensor);
                return ((LossFunctionBase<T>)LossFunction)
                    .ComputeTapeLoss(predScaled2, targetScaled2);
            }

            var context = new TapeStepContext<T>(
                trainableParams, grads, ls,
                input, target, ComputeForward, RecomputeLoss);
            _optimizer.Step(context);
        }
        finally { SetTrainingMode(false); }
    }

    private Tensor<T> ForwardTraining(Tensor<T> input, Tensor<T> noisyTarget, int t)
    {
        // ApplyInstanceNormalization stays raw-data — its output is not a
        // trainable-parameter-derived tensor; gradients don't need to
        // flow back through it. Same for Reshape (metadata view).
        var conditioned = ApplyInstanceNormalization(input);
        if (conditioned.Rank == 1) conditioned = conditioned.Reshape(new[] { 1, conditioned.Length });

        Tensor<T> condHidden;
        if (_inputProjection is not null)
            condHidden = _inputProjection.Forward(conditioned);   // tape-aware (trainable params)
        else
            condHidden = conditioned;

        // Issue #1166 — denoising-network input is
        //    [ x_t (noisyTarget) | condHidden | log(σ_t) ]
        // concatenated along the feature axis. The old code did this via
        // three raw `.Data.Span[i] = ...` copy loops. That breaks the
        // gradient tape on the `condHidden` slice — any gradient from the
        // downstream loss back toward `_inputProjection.Forward`'s
        // parameters gets cut off at the copy boundary. The
        // `_optimizer.UpdateParameters(Layers)` the old code called after
        // this path therefore ran on layers whose internal gradient state
        // had never been populated — the origin of the
        // "Backward pass must be called before updating parameters" error.
        //
        // Fix: build every row-vector piece as a real Tensor<T> on the
        // correct shape, then combine them with Engine.TensorConcatenate.
        // The concat op records into the GradientTape, so the chain
        // noisyTarget (const) → condHidden (trainable) → concat →
        // denoising layers → loss is fully preserved.
        int targetLen = noisyTarget.Length;
        int condLen = Math.Min(condHidden.Length, _hiddenDimension);

        // noisyTarget arrives as an arbitrary-rank constant — reshape to
        // [1, targetLen] row-vector so axis-1 concatenation works with
        // the [1, H] condHidden.
        var noisyRow = noisyTarget.Rank == 2 && noisyTarget.Shape[0] == 1
            ? noisyTarget
            : noisyTarget.Reshape(new[] { 1, targetLen });

        // Slice condHidden to `_hiddenDimension` columns (tape-aware).
        Tensor<T> condSlice;
        if (condHidden.Rank == 2 && condHidden.Shape[1] == condLen)
        {
            condSlice = condHidden;
        }
        else if (condHidden.Rank == 2)
        {
            condSlice = Engine.TensorSlice(
                condHidden,
                start:  new[] { 0, 0 },
                length: new[] { condHidden.Shape[0], condLen });
        }
        else
        {
            // Rank-1 condHidden: reshape to [1, condLen] via slice of the
            // first `condLen` entries. Use Reshape first (metadata view
            // of the same storage) — cheaper than a full slice + copy,
            // and since this path is hit only when `_inputProjection is
            // null` (so condHidden is not tape-connected anyway), it's
            // always safe.
            condSlice = condHidden.Reshape(new[] { 1, condHidden.Length });
            if (condSlice.Shape[1] != condLen)
            {
                condSlice = Engine.TensorSlice(
                    condSlice,
                    start:  new[] { 0, 0 },
                    length: new[] { 1, condLen });
            }
        }

        // Encode σ level as log(σ), a single-element [1, 1] row-vector
        // constant. No tape history needed.
        var sigmaRow = new Tensor<T>(new[] { 1, 1 });
        sigmaRow.Data.Span[0] = NumOps.FromDouble(
            Math.Log(Math.Max(1e-10, NumOps.ToDouble(_sigmas[t]))));

        var denoisingInput = Engine.TensorConcatenate(
            new[] { noisyRow, condSlice, sigmaRow }, axis: 1);

        var eps = denoisingInput;
        foreach (var layer in _denoisingLayers) eps = layer.Forward(eps);
        if (_outputProjection is not null) eps = _outputProjection.Forward(eps);

        // Slice the leading `_forecastHorizon` entries as a tape-aware op
        // so gradients still flow from the downstream loss back through
        // `_outputProjection` (and the `_denoisingLayers` before it).
        if (eps.Rank == 1)
        {
            int keep = Math.Min(_forecastHorizon, eps.Length);
            return Engine.TensorSlice(
                eps,
                start:  new[] { 0 },
                length: new[] { keep });
        }
        else
        {
            int keep = Math.Min(_forecastHorizon, eps.Shape[eps.Rank - 1]);
            var starts  = new int[eps.Rank];
            var lengths = eps._shape.Clone() as int[] ?? eps._shape;
            lengths[eps.Rank - 1] = keep;
            return Engine.TensorSlice(eps, starts, lengths);
        }
    }

    public override void UpdateParameters(Vector<T> gradients)
    {
        // Parameters are updated through the optimizer in Train()
    }

    public override ModelMetadata<T> GetModelMetadata() => new()
    {
        AdditionalInfo = new Dictionary<string, object> { { "NetworkType", "CCDM" }, { "ContextLength", _contextLength }, { "ForecastHorizon", _forecastHorizon }, { "HiddenDimension", _hiddenDimension }, { "DiffusionSteps", _diffusionSteps }, { "SigmaMin", _sigmaMin }, { "SigmaMax", _sigmaMax }, { "UseNativeMode", _useNativeMode } },
        ModelData = _useNativeMode ? this.Serialize() : Array.Empty<byte>()
    };

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() => new CCDM<T>(Architecture, new CCDMOptions<T> { ContextLength = _contextLength, ForecastHorizon = _forecastHorizon, HiddenDimension = _hiddenDimension, NumLayers = _numLayers, NumHeads = _numHeads, DiffusionSteps = _diffusionSteps, DropoutRate = _dropout, BetaStart = _betaStart, BetaEnd = _betaEnd, SigmaMin = _sigmaMin, SigmaMax = _sigmaMax });

    protected override void SerializeNetworkSpecificData(BinaryWriter writer) { writer.Write(_contextLength); writer.Write(_forecastHorizon); writer.Write(_hiddenDimension); writer.Write(_numLayers); writer.Write(_numHeads); writer.Write(_diffusionSteps); writer.Write(_dropout); writer.Write(_betaStart); writer.Write(_betaEnd); writer.Write(_sigmaMin); writer.Write(_sigmaMax); }
    protected override void DeserializeNetworkSpecificData(BinaryReader reader) { _contextLength = reader.ReadInt32(); _forecastHorizon = reader.ReadInt32(); _hiddenDimension = reader.ReadInt32(); _numLayers = reader.ReadInt32(); _numHeads = reader.ReadInt32(); _diffusionSteps = reader.ReadInt32(); _dropout = reader.ReadDouble(); _betaStart = reader.ReadDouble(); _betaEnd = reader.ReadDouble(); _sigmaMin = reader.ReadDouble(); _sigmaMax = reader.ReadDouble(); ComputeNoiseSchedule(); }

    #endregion

    #region IForecastingModel Implementation

    public override Tensor<T> Forecast(Tensor<T> historicalData, double[]? quantiles = null) { if (quantiles is not null && quantiles.Length > 0) throw new NotSupportedException("CCDM does not support quantile forecasting. Pass null for point forecasts."); return _useNativeMode ? ForwardNative(historicalData) : ForecastOnnx(historicalData); }
    public override Tensor<T> AutoregressiveForecast(Tensor<T> input, int steps) { var predictions = new List<Tensor<T>>(); var currentInput = input; int stepsRemaining = steps; while (stepsRemaining > 0) { var prediction = Forecast(currentInput, null); predictions.Add(prediction); int stepsUsed = Math.Min(_forecastHorizon, stepsRemaining); stepsRemaining -= stepsUsed; if (stepsRemaining > 0) currentInput = ShiftInputWithPredictions(currentInput, prediction, stepsUsed); } return ConcatenatePredictions(predictions, steps); }

    public override Dictionary<string, T> Evaluate(Tensor<T> predictions, Tensor<T> actuals) { T mse = NumOps.Zero; T mae = NumOps.Zero; int count = 0; for (int i = 0; i < predictions.Length && i < actuals.Length; i++) { var diff = NumOps.Subtract(predictions[i], actuals[i]); mse = NumOps.Add(mse, NumOps.Multiply(diff, diff)); mae = NumOps.Add(mae, NumOps.Abs(diff)); count++; } if (count > 0) { mse = NumOps.Divide(mse, NumOps.FromDouble(count)); mae = NumOps.Divide(mae, NumOps.FromDouble(count)); } return new Dictionary<string, T> { ["MSE"] = mse, ["MAE"] = mae, ["RMSE"] = NumOps.Sqrt(mse) }; }

    public override Tensor<T> ApplyInstanceNormalization(Tensor<T> input) { int batchSize = input.Rank > 1 ? input.Shape[0] : 1; int seqLen = input.Rank > 1 ? input.Shape[1] : input.Length; var result = new Tensor<T>(input._shape); for (int b = 0; b < batchSize; b++) { T mean = NumOps.Zero; for (int t = 0; t < seqLen; t++) { int idx = b * seqLen + t; if (idx < input.Length) mean = NumOps.Add(mean, input[idx]); } mean = NumOps.Divide(mean, NumOps.FromDouble(seqLen)); T variance = NumOps.Zero; for (int t = 0; t < seqLen; t++) { int idx = b * seqLen + t; if (idx < input.Length) { var diff = NumOps.Subtract(input[idx], mean); variance = NumOps.Add(variance, NumOps.Multiply(diff, diff)); } } variance = NumOps.Divide(variance, NumOps.FromDouble(seqLen)); T std = NumOps.Sqrt(NumOps.Add(variance, NumOps.FromDouble(1e-5))); for (int t = 0; t < seqLen; t++) { int idx = b * seqLen + t; if (idx < input.Length && idx < result.Length) result.Data.Span[idx] = NumOps.Divide(NumOps.Subtract(input[idx], mean), std); } } return result; }

    public override Dictionary<string, T> GetFinancialMetrics() { T lastLoss = LastLoss is not null ? LastLoss : NumOps.Zero; return new Dictionary<string, T> { ["ContextLength"] = NumOps.FromDouble(_contextLength), ["ForecastHorizon"] = NumOps.FromDouble(_forecastHorizon), ["SigmaMin"] = NumOps.FromDouble(_sigmaMin), ["SigmaMax"] = NumOps.FromDouble(_sigmaMax), ["LastLoss"] = lastLoss }; }

    #endregion

    #region Forward/Backward Pass

    /// <summary>
    /// Continuous diffusion reverse process using annealed Langevin dynamics.
    /// CCDM operates in continuous noise space: at each step, the model predicts the score
    /// (gradient of log probability) and uses it to iteratively denoise from high sigma to low sigma.
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
        var rand = RandomHelper.CreateSecureRandom();

        // Start from noise at highest sigma level
        var xt = new Tensor<T>(new[] { 1, outputLen });
        for (int i = 0; i < outputLen; i++)
            xt.Data.Span[i] = NumOps.Multiply(_sigmas[0], SampleStandardNormal(rand));

        // Annealed Langevin dynamics: iterate from high sigma to low sigma
        for (int t = 0; t < _diffusionSteps; t++)
        {
            T sigmaT = _sigmas[t];
            T sigmaNext = _sigmas[t + 1];

            // Build score network input: [x_t | condHidden | log(sigma)]
            int xtLen = Math.Min(xt.Length, outputLen);
            int condLen = Math.Min(condHidden.Length, _hiddenDimension);
            var scoreInput = new Tensor<T>(new[] { 1, xtLen + condLen + 1 });
            for (int i = 0; i < xtLen; i++) scoreInput.Data.Span[i] = xt[i];
            for (int i = 0; i < condLen; i++) scoreInput.Data.Span[xtLen + i] = condHidden[i];
            scoreInput.Data.Span[xtLen + condLen] = NumOps.FromDouble(Math.Log(Math.Max(1e-10, NumOps.ToDouble(sigmaT))));

            // Predict score: s_theta(x_t, sigma_t)
            var score = scoreInput;
            foreach (var layer in _denoisingLayers) score = layer.Forward(score);
            if (_outputProjection is not null) score = _outputProjection.Forward(score);

            // Langevin step: x_{t+1} = x_t + (sigma_t^2 - sigma_next^2) * score + sqrt(sigma_t^2 - sigma_next^2) * z
            T stepSizeT = NumOps.Subtract(NumOps.Multiply(sigmaT, sigmaT), NumOps.Multiply(sigmaNext, sigmaNext));
            T noiseScaleT = NumOps.Sqrt(NumOps.Add(stepSizeT, NumOps.FromDouble(1e-30))); // clamp to non-negative

            for (int i = 0; i < outputLen && i < xt.Length; i++)
            {
                T scoreVal = i < score.Length ? score[i] : NumOps.Zero;
                T z = (t < _diffusionSteps - 1) ? SampleStandardNormal(rand) : NumOps.Zero;
                xt.Data.Span[i] = NumOps.Add(NumOps.Add(xt[i], NumOps.Multiply(stepSizeT, scoreVal)), NumOps.Multiply(noiseScaleT, z));
            }
        }

        if (addedBatchDim && xt.Rank == 2 && xt.Shape[0] == 1) xt = xt.Reshape(new[] { xt.Shape[1] });
        return xt;
    }

    protected override Tensor<T> ForecastOnnx(Tensor<T> input) { if (OnnxSession == null) throw new InvalidOperationException("ONNX session is not initialized."); int batchSize = input.Rank > 1 ? input.Shape[0] : 1; int seqLen = input.Rank > 1 ? input.Shape[1] : input.Length; int features = input.Rank > 2 ? input.Shape[2] : 1; var inputData = new float[batchSize * seqLen * features]; for (int i = 0; i < input.Length && i < inputData.Length; i++) inputData[i] = (float)NumOps.ToDouble(input[i]); var inputTensor = new OnnxTensors.DenseTensor<float>(inputData, new[] { batchSize, seqLen, features }); string inputName = OnnxSession.InputMetadata.Keys.FirstOrDefault() ?? "input"; var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(inputName, inputTensor) }; using var results = OnnxSession.Run(inputs); var outputTensor = results.First().AsTensor<float>(); var outputShape = outputTensor.Dimensions.ToArray(); var output = new Tensor<T>(outputShape); int totalElements = 1; foreach (var dim in outputShape) totalElements *= dim; for (int i = 0; i < totalElements && i < output.Length; i++) output.Data.Span[i] = NumOps.FromDouble(outputTensor.GetValue(i)); return output; }

    #endregion
}
