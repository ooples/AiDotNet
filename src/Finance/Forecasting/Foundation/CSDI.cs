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
/// CSDI — Conditional Score-based Diffusion Model for Probabilistic Time Series Imputation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// CSDI uses score-based diffusion for non-autoregressive time series imputation and forecasting.
/// It conditions on observed values using a transformer-based denoiser and generates all missing
/// values simultaneously.
/// </para>
/// <para><b>For Beginners:</b> CSDI fills in missing data points in time series and forecasts
/// future values using a diffusion process. Think of it like an artist restoring a damaged
/// painting: it looks at the intact parts and intelligently fills in the gaps. Unlike simpler
/// methods that fill one gap at a time, CSDI fills all missing values simultaneously, which
/// produces more consistent and realistic results.</para>
/// <para>
/// <b>Reference:</b> Tashiro et al., "CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation", NeurIPS 2021.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a CSDI conditional score-based diffusion model for time series imputation
/// // Uses score-matching for probabilistic forecasting and missing value imputation
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.Regression,
///     inputHeight: 512, inputWidth: 1, inputDepth: 1, outputSize: 24);
///
/// // Training mode with conditional score-based diffusion
/// var model = new CSDI&lt;double&gt;(architecture);
///
/// // ONNX inference mode with pre-trained model
/// var onnxModel = new CSDI&lt;double&gt;(architecture, "csdi.onnx");
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
[ResearchPaper("CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation", "https://arxiv.org/abs/2107.03502", Year = 2021, Authors = "Yusuke Tashiro, Jiaming Song, Yang Song, Stefano Ermon")]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 0.001, ReferenceBatchSize = 16,
                Schedule = LearningRateSchedulerType.MultiStep, DecayRate = 0.1,
                MilestoneFractions = [0.75, 0.90],
                Source = "Tashiro et al. 2021, hyperparameters: Adam at learning rate 0.001 decayed to 0.0001 and 0.00001 at 75% and 90% of the total epochs, batch size 16, 200 epochs. The decay points are kept as the fractions the paper states rather than transcribed into the step numbers of a 200-epoch run, which would be wrong at any other length.")]
public partial class CSDI<T> : TimeSeriesFoundationModelBase<T>
{
    #region Fields

    private readonly bool _useNativeMode;
    private ILayer<T>? _inputProjection;
    private readonly List<ILayer<T>> _residualLayers = [];
    private ILayer<T>? _outputProjection;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ILossFunction<T> _lossFunction;
    private readonly CSDIOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    private int _sequenceLength;
    private int _numFeatures;
    private int _hiddenDimension;
    private int _numResidualLayers;
    private int _numDiffusionSteps;

    /// <summary>
    /// Sample paths drawn per prediction, aggregated by median. Tashiro et al.
    /// (arXiv:2107.03502) define CSDI's deterministic output as "the median of 100 generated
    /// samples" - a single path is a draw from the predictive distribution, not the estimate.
    /// </summary>
    private int _numSamples;

    /// <summary>Seed for the sampling noise; null draws securely and is not reproducible.</summary>
    private int? _seed;
    private int _numHeads;
    private int _timeEmbeddingDim;
    private double _dropout;
    private double _betaStart;
    private double _betaEnd;

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

    /// <inheritdoc/>
    public override int SequenceLength => _sequenceLength;
    /// <inheritdoc/>
    public override int PredictionHorizon => _sequenceLength;
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
    /// <inheritdoc/>
    public override FoundationModelSize ModelSize => FoundationModelSize.Small;
    /// <inheritdoc/>
    public override int MaxContextLength => _sequenceLength;
    /// <inheritdoc/>
    public override int MaxPredictionHorizon => _sequenceLength;

    #endregion

    #region Constructors

    public CSDI(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        CSDIOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"ONNX model not found: {onnxModelPath}");

        options ??= new CSDIOptions<T>();
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

    public CSDI(
        NeuralNetworkArchitecture<T> architecture,
        CSDIOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        options ??= new CSDIOptions<T>();
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

    private void CopyOptionsToFields(CSDIOptions<T> options)
    {
        _sequenceLength = options.SequenceLength;
        _numFeatures = options.NumFeatures;
        _hiddenDimension = options.HiddenDimension;
        _numResidualLayers = options.NumResidualLayers;
        _numDiffusionSteps = options.NumDiffusionSteps;
        _numHeads = options.NumHeads;
        _timeEmbeddingDim = options.TimeEmbeddingDim;
        _dropout = options.DropoutRate;
        _betaStart = options.BetaStart;
        _betaEnd = options.BetaEnd;
        _numSamples = Math.Max(1, options.NumSamples);
        _seed = options.Seed;
        ComputeNoiseSchedule();
    }

    private void ComputeNoiseSchedule()
    {
        if (_numDiffusionSteps <= 0)
            throw new ArgumentOutOfRangeException(nameof(CSDIOptions<T>.NumDiffusionSteps), "DiffusionSteps must be positive.");

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
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            ExtractLayerReferences();
        }
        else if (_useNativeMode)
        {
            Layers.AddRange(LayerHelper<T>.CreateDefaultCSDILayers(
                Architecture, _sequenceLength, _numFeatures, _hiddenDimension,
                _numResidualLayers, _numHeads, _timeEmbeddingDim, _dropout));
            ExtractLayerReferences();
        }
    }

    private void ExtractLayerReferences()
    {
        int idx = 0;
        if (idx < Layers.Count)
            _inputProjection = Layers[idx++];
        _residualLayers.Clear();
        while (idx < Layers.Count - 1)
            _residualLayers.Add(Layers[idx++]);
        if (idx < Layers.Count)
            _outputProjection = Layers[idx++];
    }

    #endregion

    #region NeuralNetworkBase Overrides

    public override bool SupportsTraining => _useNativeMode;
    protected override Tensor<T> PredictCore(Tensor<T> input) => _useNativeMode ? ForwardNative(input) : ForecastOnnx(input);

    public override void Train(Tensor<T> input, Tensor<T> target)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Training is only supported in native mode.");

        // CSDI-specific training: denoising-score-matching objective.
        // The original delegation to base.Train routed through
        // ForwardNative — which is the REVERSE-diffusion sampler (start
        // from noise, iteratively denoise). Training that against the
        // clean target is a category error: it supervises the sampler's
        // random noise output against target values instead of teaching
        // the denoiser to predict the noise that was added.
        //
        // Correct DDPM training loop:
        //   1. Sample timestep t ~ Uniform[0, T)
        //   2. Sample noise ε ~ N(0, I) with the same shape as target
        //   3. Form noised x_t = sqrt(α̅_t) * target + sqrt(1-α̅_t) * ε
        //   4. Predict ε_pred = denoiser(x_t, conditioning, t)
        //   5. Loss = MSE(ε_pred, ε)
        // All of steps 3-5 go through Engine ops so the tape records
        // the whole pipeline back into _inputProjection, the residual
        // stack, and _outputProjection.

        var loss = LossFunction;

        var trainableParams = Training.TapeTrainingStep<T>.CollectParameters(Layers).ToArray();

        // One draw of (t, epsilon) feeds whichever path runs below. Hoisting it out of the
        // fused branch is what lets the eager path reuse DenoiserForwardFromSlots, so the two
        // paths share ONE denoiser graph instead of two hand-written ones that can drift
        // apart. Null means a degenerate (zero-length) target, so there is nothing to train.
        var slots = BuildCsdiSlots(input, target);
        if (slots is null)
            return;

        // Every training entry point on the base puts the layers in training mode for the
        // duration of the step. This override is a training entry point too, and without
        // the call the residual stack's DropoutLayer stayed in inference mode for the whole
        // of training, so the configured DropoutRate (0.1 by default, the rate Tashiro et
        // al. 2021 report) was silently inert and the model trained an architecture the
        // caller never asked for.
        SetTrainingMode(true);
        try
        {
            // Preferred fused path: MultiSlotFusedStep with the sampled (noise,
            // xt-scale, sinT) tuple passed as persistent slots. Refreshes per step
            // by host-sampling a fresh (t, eps) pair and copying values into the
            // slot tensors -- the compiled forward reads the CURRENT slot data on
            // every replay. See ooples/AiDotNet#1846.
            if (trainableParams.Length > 0
                && NeuralNetworks.NeuralNetworkBase<T>.TryMapToFusedOptimizerConfig(
                    _optimizer,
                    out var mfsOptType, out var mfsLr, out var mfsB1, out var mfsB2,
                    out var mfsEps, out var mfsWd, out _, out _))
            {
                using var multiSlotStep = new AiDotNet.Training.MultiSlotFusedStep<T>();
                Tensor<T> ForwardFromSlots(IReadOnlyList<Tensor<T>> s) => DenoiserForwardFromSlots(s);
                Tensor<T> ComputeLossFromSlots(Tensor<T> pred, IReadOnlyList<Tensor<T>> s)
                {
                    // s[1] = eps_true (noise slot). Loss = MSE(eps_pred, eps_true) via
                    // the model's LossFunctionBase so custom losses are respected.
                    return loss.ComputeTapeLoss(pred, s[1]);
                }
                if (multiSlotStep.TryStep(
                        parameters: trainableParams,
                        zeroGradAction: null,
                        freshSlotData: slots,
                        forward: ForwardFromSlots,
                        computeLoss: ComputeLossFromSlots,
                        optimizerType: mfsOptType,
                        learningRate: mfsLr,
                        beta1: mfsB1,
                        beta2: mfsB2,
                        epsilon: mfsEps,
                        weightDecay: mfsWd,
                        out T fusedLoss))
                {
                    LastLoss = fusedLoss;
                    return;
                }
            }

            // Eager fallback: runs the same denoiser graph on the tape. Used when the fused
            // path cannot engage (non-fuse-able optimizer, non-GPU host, etc.).
            using var tape = new GradientTape<T>();
            var epsilonPred = DenoiserForwardFromSlots(slots);
            var epsilonTarget = slots[1];

            // Use the model's registered loss (defaults to MSE) so custom
            // loss functions are respected -- the denoising-objective shape
            // matches any per-element loss.
            var lossTensor = loss.ComputeTapeLoss(epsilonPred, epsilonTarget);

            // Publish through the base instead of calling tape.ComputeGradients directly.
            // GetParameterGradients() answers from the published surface, and with nothing
            // published it falls back to the layer accessors, which fabricate a zero for every
            // parameter -- 184804 of them here. A caller then cannot tell a genuinely zero
            // gradient from one that was never written, and a gradient-flow check reads the
            // whole model as severed while the weights visibly move.
            var grads = ComputeAndPublishParameterGradients(tape, lossTensor, trainableParams);

            T lossValue = lossTensor.Length > 0 ? lossTensor[0] : NumOps.Zero;
            LastLoss = lossValue;

            // The optimizer handed to the constructor drives the update. The hand-rolled
            // `param -= 0.001 * grad` this replaces ignored it outright, so an Adam, a
            // configured learning rate, a schedule and any weight decay a caller passed all
            // had no effect whatsoever. Recomputation is pinned to THIS draw of (t, eps):
            // a line-searching optimizer that re-sampled would be comparing losses from two
            // different diffusion timesteps and would read the difference as progress.
            Tensor<T> ComputeForward(Tensor<T> _, Tensor<T> __) => DenoiserForwardFromSlots(slots);
            Tensor<T> RecomputeLoss(Tensor<T> pred, Tensor<T> __) => loss.ComputeTapeLoss(pred, epsilonTarget);

            var context = new AiDotNet.Tensors.Engines.Autodiff.TapeStepContext<T>(
                trainableParams, grads, lossValue,
                input, target, ComputeForward, RecomputeLoss);

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
    /// Assembles the persistent-slot data for the MultiSlotFusedStep wire-up:
    /// samples (t, ε) host-side then packs everything the compiled forward
    /// needs as tensor slots. Slot layout:
    /// <list type="bullet">
    /// <item><description>[0] target — the clean values to denoise (Ho 2020 x_0).</description></item>
    /// <item><description>[1] ε_true — freshly-sampled Gaussian noise, same shape as target.</description></item>
    /// <item><description>[2] sqrtAlphaBar_scalar — cumulative α scaling at the sampled t, shape [1].</description></item>
    /// <item><description>[3] sqrtOneMinusAlphaBar_scalar — noise scaling at t, shape [1].</description></item>
    /// <item><description>[4] sinT_scalar — sin(2π·t/T) timestep encoding, shape [1].</description></item>
    /// <item><description>[5] conditioned — RevIN-normalized observed input, shape [1, seqLen].</description></item>
    /// </list>
    /// Returns null when the target is degenerate (zero length) so the caller
    /// falls back to eager training.
    /// </summary>
    private IReadOnlyList<Tensor<T>>? BuildCsdiSlots(Tensor<T> input, Tensor<T> target)
    {
        int targetLen = target.Length;
        if (targetLen <= 0) return null;

        var rand = RandomHelper.CreateSecureRandom();
        int t = rand.Next(_numDiffusionSteps);
        var noiseData = new T[targetLen];
        for (int i = 0; i < targetLen; i++) noiseData[i] = SampleStandardNormal(rand);
        var epsilonTrue = new Tensor<T>(target._shape, new Vector<T>(noiseData));

        T sqrtAlphaBar = NumOps.Sqrt(_alphasCumprod[t]);
        T sqrtOneMinus = NumOps.Sqrt(NumOps.Subtract(NumOps.One, _alphasCumprod[t]));
        T sinT = NumOps.FromDouble(Math.Sin(2.0 * Math.PI * t / Math.Max(1, _numDiffusionSteps - 1)));

        var sqrtAlphaBarT = new Tensor<T>(new[] { 1 });
        sqrtAlphaBarT[0] = sqrtAlphaBar;
        var sqrtOneMinusT = new Tensor<T>(new[] { 1 });
        sqrtOneMinusT[0] = sqrtOneMinus;
        var sinTT = new Tensor<T>(new[] { 1 });
        sinTT[0] = sinT;

        var conditioned = ApplyInstanceNormalization(input);
        if (conditioned.Rank == 1)
            conditioned = Engine.Reshape(conditioned, new[] { 1, conditioned.Length });
        return new[] { target, epsilonTrue, sqrtAlphaBarT, sqrtOneMinusT, sinTT, conditioned };
    }

    /// <summary>
    /// Fused-path denoiser forward: reads the slot tuple from
    /// <see cref="BuildCsdiSlots"/>, builds x_t via traceable engine
    /// arithmetic (no <c>.Data.Span</c> pack — the previous inline
    /// implementation froze the trace batch's x_t into the compiled plan),
    /// concatenates [x_t | conditioning_slice | sin(t)] via
    /// <see cref="IEngine.TensorConcatenate{T}"/>, and runs the projection +
    /// residual stack + output projection. Returns predicted noise aligned
    /// with the noise slot's shape.
    /// </summary>
    private Tensor<T> DenoiserForwardFromSlots(IReadOnlyList<Tensor<T>> slots)
    {
        var target = slots[0];
        var epsilonTrue = slots[1];
        var sqrtAlphaBarT = slots[2];
        var sqrtOneMinusT = slots[3];
        var sinTT = slots[4];
        var conditioned = slots[5];

        int targetLen = target.Length;
        // x_t = sqrt(α̅_t) * target + sqrt(1-α̅_t) * ε — all traceable.
        var scaledTarget = Engine.TensorMultiplyScalar(target, sqrtAlphaBarT[0]);
        var scaledNoise = Engine.TensorMultiplyScalar(epsilonTrue, sqrtOneMinusT[0]);
        var xt = Engine.TensorAdd(scaledTarget, scaledNoise);
        var xt1d = xt.Rank == 1 ? xt : Engine.Reshape(xt, new[] { targetLen });

        // Conditioning slice: first Min(condLen, hiddenDimension) elements,
        // flattened to 1-D so we can concat with xt1d + sinT.
        var condFlat = conditioned.Rank == 1
            ? conditioned
            : Engine.Reshape(conditioned, new[] { conditioned.Length });
        int condLen = Math.Min(condFlat.Length, _hiddenDimension);
        var condSlice = condLen == condFlat.Length
            ? condFlat
            : Engine.TensorSlice(condFlat, new[] { 0 }, new[] { condLen });

        // Pack [xt | conditioning | sin(t)] via TensorConcatenate — replaces
        // the .Data.Span[i] fill that froze at trace time.
        var packed1D = Engine.TensorConcatenate(new[] { xt1d, condSlice, sinTT }, axis: 0);
        var denoisingInput = Engine.Reshape(packed1D, new[] { 1, targetLen + condLen + 1 });

        var eps = (Tensor<T>)denoisingInput;
        if (_inputProjection is not null)
            eps = _inputProjection.Forward(eps);
        foreach (var layer in _residualLayers)
            eps = layer.Forward(eps);
        if (_outputProjection is not null)
            eps = _outputProjection.Forward(eps);

        if (eps.Rank == 2 && eps.Shape[1] > epsilonTrue.Length)
            eps = Engine.TensorNarrow(eps, dim: 1, start: 0, length: epsilonTrue.Length);
        if (eps.Length != epsilonTrue.Length)
        {
            throw new InvalidOperationException(
                $"CSDI denoising pair: predicted-noise length ({eps.Length}, shape=["
                + $"{string.Join(",", eps._shape)}]) does not match true-noise length ("
                + $"{epsilonTrue.Length}, shape=[{string.Join(",", epsilonTrue._shape)}]).");
        }
        if (!eps._shape.AsEnumerable().SequenceEqual(epsilonTrue._shape))
            eps = Engine.Reshape(eps, epsilonTrue._shape);
        return eps;
    }

    // UpdateParameters was an empty override, silently dropping every restore. The base
    // distributes the vector over the declared enumeration.
    public override ModelMetadata<T> GetModelMetadata() => new()
    {
        AdditionalInfo = new Dictionary<string, object>
        {
            { "NetworkType", "CSDI" }, { "SequenceLength", _sequenceLength },
            { "NumFeatures", _numFeatures }, { "HiddenDimension", _hiddenDimension },
            { "NumDiffusionSteps", _numDiffusionSteps }, { "UseNativeMode", _useNativeMode }
        },
        ModelData = _useNativeMode ? this.Serialize() : Array.Empty<byte>()
    };





    #endregion

    #region IForecastingModel Implementation

    public override Tensor<T> Forecast(Tensor<T> historicalData, double[]? quantiles = null)
    {
        if (quantiles is not null && quantiles.Length > 0)
            throw new NotSupportedException("CSDI does not support quantile forecasting. Pass null for point forecasts.");

        return _useNativeMode ? ForwardNative(historicalData) : ForecastOnnx(historicalData);
    }

    public override Tensor<T> AutoregressiveForecast(Tensor<T> input, int steps)
    {
        // CSDI is non-autoregressive — truncate output to requested steps
        var fullForecast = Forecast(input, null);
        if (steps >= fullForecast.Length) return fullForecast;
        var result = new Tensor<T>(new[] { steps });
        for (int i = 0; i < steps; i++)
            result.Data.Span[i] = fullForecast[i];
        return result;
    }

    public override Dictionary<string, T> Evaluate(Tensor<T> predictions, Tensor<T> actuals)
    {
        T mse = NumOps.Zero; T mae = NumOps.Zero; int count = 0;
        for (int i = 0; i < predictions.Length && i < actuals.Length; i++)
        {
            var diff = NumOps.Subtract(predictions[i], actuals[i]);
            mse = NumOps.Add(mse, NumOps.Multiply(diff, diff));
            mae = NumOps.Add(mae, NumOps.Abs(diff)); count++;
        }
        if (count > 0) { mse = NumOps.Divide(mse, NumOps.FromDouble(count)); mae = NumOps.Divide(mae, NumOps.FromDouble(count)); }
        return new Dictionary<string, T> { ["MSE"] = mse, ["MAE"] = mae, ["RMSE"] = NumOps.Sqrt(mse) };
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Traceable RevIN forward — same pattern as
    /// <see cref="TFC{T}.ApplyInstanceNormalization"/>. Uses
    /// <see cref="IEngine.ReduceMean{T}"/> / <see cref="IEngine.ReduceVariance{T}"/>
    /// / <see cref="IEngine.TensorSqrt{T}"/> / broadcast subtract+divide so the
    /// per-batch stats re-execute on every replay under a compiled fused plan.
    /// </remarks>
    public override Tensor<T> ApplyInstanceNormalization(Tensor<T> input)
    {
        if (input.Length == 0) return input;
        int batchSize = input.Rank > 1 ? input.Shape[0] : 1;
        if (batchSize <= 0) batchSize = 1;
        // Per-sample size is ALL non-batch elements, not just Shape[1]. The previous code used
        // input.Shape[1], which for a rank-3+ input (e.g. [1, 8, 8]) dropped every axis beyond 1 and
        // reshaped 64 elements into [1, 8] (8), throwing "Cannot reshape tensor with 64 elements to
        // shape [1, 8]". Flattening to [batch, perSample] keeps every element and normalizes per
        // sample over the whole sequence, regardless of rank.
        int seqLen = input.Length / batchSize;
        if (seqLen <= 0) return input;

        bool reshaped = !(input.Rank == 2 && input.Shape[1] == seqLen);
        var flat = reshaped ? Engine.Reshape(input, new[] { batchSize, seqLen }) : input;
        var mean = Engine.ReduceMean(flat, new[] { 1 }, keepDims: true);
        var variance = Engine.ReduceVariance(flat, new[] { 1 }, keepDims: true);
        var std = Engine.TensorSqrt(Engine.TensorAddScalar(variance, NumOps.FromDouble(1e-5)));
        var centered = Engine.TensorSubtract(flat, mean);
        var normalized = Engine.TensorDivide(centered, std);
        return reshaped ? Engine.Reshape(normalized, input._shape) : normalized;
    }

    public override Dictionary<string, T> GetFinancialMetrics()
    {
        T lastLoss = LastLoss is not null ? LastLoss : NumOps.Zero;
        return new Dictionary<string, T> { ["SequenceLength"] = NumOps.FromDouble(_sequenceLength), ["NumFeatures"] = NumOps.FromDouble(_numFeatures), ["HiddenDimension"] = NumOps.FromDouble(_hiddenDimension), ["LastLoss"] = lastLoss };
    }

    #endregion

    #region Forward/Backward Pass

    /// <summary>
    /// DDPM reverse process: iteratively denoise from pure noise conditioned on observed values.
    /// </summary>
    /// <summary>
    /// Reverse diffusion, drawing <c>_numSamples</c> paths and returning their per-position median.
    /// </summary>
    /// <remarks>
    /// <para>The paths are drawn as ONE BATCH, not in a loop. xt is [S, outputLen] and the packed
    /// denoiser input is [S, xtLen + condLen + 1], so the reverse process still runs exactly
    /// _numDiffusionSteps times - each step just carries S rows through the same projections and
    /// residual layers. That turns S independent sampling runs into S-wide matrix multiplies, which
    /// is how a batched engine is meant to be fed; a per-sample loop would multiply the layer-call
    /// count by S and leave every GEMM at batch 1.</para>
    ///
    /// <para>The conditioning and the sin(t) encoding are identical across rows - only the noise
    /// differs - so they are broadcast into each row rather than recomputed.</para>
    /// </remarks>
    private Tensor<T> ForwardNative(Tensor<T> input)
    {
        var conditioned = ApplyInstanceNormalization(input);
        bool addedBatchDim = false;
        if (conditioned.Rank == 1) { conditioned = conditioned.Reshape(new[] { 1, conditioned.Length }); addedBatchDim = true; }

        var condFlat = conditioned.Rank == 2
            ? conditioned
            : Engine.Reshape(conditioned, new[] { 1, conditioned.Length });

        int outputLen = _sequenceLength;
        int samples = Math.Max(1, _numSamples);
        var rand = _seed.HasValue
            ? RandomHelper.CreateSeededRandom(_seed.Value)
            : RandomHelper.CreateSecureRandom();

        var xt = new Tensor<T>(new[] { samples, outputLen });
        for (int i = 0; i < samples * outputLen; i++)
            xt.Data.Span[i] = SampleStandardNormal(rand);

        int condLen = Math.Min(condFlat.Length, _hiddenDimension);
        int rowLen = outputLen + condLen + 1;
        T eps10 = NumOps.FromDouble(1e-10);

        for (int t = _numDiffusionSteps - 1; t >= 0; t--)
        {
            var denoisingInput = new Tensor<T>(new[] { samples, rowLen });
            T sinT = NumOps.FromDouble(Math.Sin(2.0 * Math.PI * t / Math.Max(1, _numDiffusionSteps - 1)));
            var din = denoisingInput.Data.Span;
            for (int s = 0; s < samples; s++)
            {
                int baseIdx = s * rowLen;
                for (int i = 0; i < outputLen; i++) din[baseIdx + i] = xt[s * outputLen + i];
                for (int i = 0; i < condLen; i++) din[baseIdx + outputLen + i] = condFlat[i];
                din[baseIdx + outputLen + condLen] = sinT;
            }

            var eps = denoisingInput;
            if (_inputProjection is not null) eps = _inputProjection.Forward(eps);
            foreach (var layer in _residualLayers) eps = layer.Forward(eps);
            if (_outputProjection is not null) eps = _outputProjection.Forward(eps);

            T alphaT = _alphas[t];
            T betaT = _betas[t];
            T sqrtOneMinusAlphaBarT = NumOps.Sqrt(NumOps.Subtract(NumOps.One, _alphasCumprod[t]));
            T noiseCoeffT = NumOps.Divide(betaT, NumOps.Add(sqrtOneMinusAlphaBarT, eps10));
            T sqrtAlphaT = NumOps.Sqrt(alphaT);
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
                    T meanT = NumOps.Divide(NumOps.Subtract(xt[flat], NumOps.Multiply(noiseCoeffT, epsVal)), NumOps.Add(sqrtAlphaT, eps10));
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
    /// Per-position median over the sample axis - the paper's deterministic estimate. The same
    /// sample set is what its 5% and 95% prediction intervals come from.
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

    protected override Tensor<T> ForecastOnnx(Tensor<T> input)
    {
        if (OnnxSession == null) throw new InvalidOperationException("ONNX session is not initialized.");
        int batchSize = input.Rank > 1 ? input.Shape[0] : 1; int seqLen = input.Rank > 1 ? input.Shape[1] : input.Length; int features = input.Rank > 2 ? input.Shape[2] : 1;
        var inputData = new float[batchSize * seqLen * features];
        for (int i = 0; i < input.Length && i < inputData.Length; i++) inputData[i] = (float)NumOps.ToDouble(input[i]);
        var inputTensor = new OnnxTensors.DenseTensor<float>(inputData, new[] { batchSize, seqLen, features });
        string inputName = OnnxSession.InputMetadata.Keys.FirstOrDefault() ?? "input";
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(inputName, inputTensor) };
        using var results = OnnxSession.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();
        var outputShape = outputTensor.Dimensions.ToArray();
        var output = new Tensor<T>(outputShape);
        int totalElements = 1; foreach (var dim in outputShape) totalElements *= dim;
        for (int i = 0; i < totalElements && i < output.Length; i++) output.Data.Span[i] = NumOps.FromDouble(outputTensor.GetValue(i));
        return output;
    }

    #endregion
}
