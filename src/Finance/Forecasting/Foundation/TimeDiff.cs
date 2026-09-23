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
/// TimeDiff — Non-autoregressive Conditional Diffusion Models for Time Series Prediction.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// TimeDiff extends DDPM with future-mixup training augmentation and autoregressive initialization
/// at inference for high-quality non-autoregressive time series forecasting.
/// </para>
/// <para><b>For Beginners:</b> TimeDiff improves diffusion-based forecasting with two clever
/// tricks. During training, it mixes future values into the input (future-mixup) to help the
/// model learn what comes next. During prediction, it uses an initial rough forecast to guide
/// the diffusion process, producing all future values at once rather than one at a time, which
/// is both faster and more consistent.</para>
/// <para>
/// <b>Reference:</b> Shen &amp; Kwok, "Non-autoregressive Conditional Diffusion Models for Time Series Prediction", ICML 2023.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a TimeDiff non-autoregressive conditional diffusion model
/// // Uses future-mixup training and autoregressive initialization for consistent forecasts
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.Regression,
///     inputHeight: 512, inputWidth: 1, inputDepth: 1, outputSize: 24);
///
/// // Training mode with future-mixup augmentation
/// var model = new TimeDiff&lt;double&gt;(architecture);
///
/// // ONNX inference mode with pre-trained model
/// var onnxModel = new TimeDiff&lt;double&gt;(architecture, "timediff.onnx");
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
[ResearchPaper("Non-autoregressive Conditional Diffusion Models for Time Series Prediction", "https://arxiv.org/abs/2306.05043", Year = 2023, Authors = "Lifeng Shen, James Kwok")]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 1e-3, ReferenceBatchSize = 64,
                Source = "Shen and Kwok 2023, Sec. 5: Adam with a learning rate of 1e-3 and a batch "
                        + "size of 64, trained with early stopping for a maximum of 100 epochs.")]
public partial class TimeDiff<T> : TimeSeriesFoundationModelBase<T>, ITrainingObjectiveProvider<T>
{
    #region Fields

    private readonly bool _useNativeMode;
    private ILayer<T>? _inputProjection;
    private readonly List<ILayer<T>> _transformerLayers = [];
    private ILayer<T>? _outputProjection;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ILossFunction<T> _lossFunction;
    private readonly TimeDiffOptions<T> _options;

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
    private int _trainingBatchSize;
    private bool _useFutureMixup;
    private bool _useAutoregressiveInit;

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
    [Buffer]
    private Vector<T> _alphasCumprodPrev = Vector<T>.Empty();

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

    public TimeDiff(NeuralNetworkArchitecture<T> architecture, string onnxModelPath,
        TimeDiffOptions<T>? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null, ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"ONNX model not found: {onnxModelPath}");

        options ??= new TimeDiffOptions<T>();
        _options = options;
        Options = _options;

        _useNativeMode = false;
        OnnxModelPath = onnxModelPath;
        OnnxSession = new InferenceSession(onnxModelPath);

        _optimizer = optimizer
            ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this)
            ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this, new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>> { InitialLearningRate = options.LearningRate });
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();

        CopyOptionsToFields(options);
    }

    public TimeDiff(NeuralNetworkArchitecture<T> architecture,
        TimeDiffOptions<T>? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null, ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        options ??= new TimeDiffOptions<T>();
        _options = options;
        Options = _options;

        _useNativeMode = true;
        OnnxSession = null;
        OnnxModelPath = null;

        _optimizer = optimizer
            ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this)
            ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this, new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>> { InitialLearningRate = options.LearningRate });
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();

        CopyOptionsToFields(options);
        InitializeLayers();
    }

    private void CopyOptionsToFields(TimeDiffOptions<T> options)
    {
        _contextLength = options.ContextLength;
        _forecastHorizon = options.ForecastHorizon;
        _hiddenDimension = options.HiddenDimension;
        _numLayers = options.NumLayers;
        _numHeads = options.NumHeads;
        _diffusionSteps = options.DiffusionSteps;
        _dropout = options.DropoutRate;
        _betaStart = options.BetaStart;
        _betaEnd = options.BetaEnd;
        _trainingBatchSize = options.TrainingBatchSize;
        _useFutureMixup = options.UseFutureMixup;
        _useAutoregressiveInit = options.UseAutoregressiveInit;
        ComputeNoiseSchedule();
    }

    /// <summary>
    /// Builds the cosine variance schedule of Section 4.1, clamped to
    /// <c>[BetaStart, BetaEnd]</c>.
    /// </summary>
    /// <remarks>
    /// Section 4.1 states "K = 100 diffusion steps are used, with a cosine variance schedule
    /// (Rasul et al., 2021) starting from beta_1 = 10^-4 to beta_K = 10^-1". What this replaces
    /// interpolated beta LINEARLY between the two endpoints, which is a different schedule: the
    /// cosine schedule of Nichol and Dhariwal 2021 is defined on alpha_bar,
    /// alpha_bar_k = f(k)/f(0) with f(k) = cos^2(((k/K + s)/(1 + s)) * pi/2), and beta is read
    /// back off it as 1 - alpha_bar_k / alpha_bar_{k-1}. It destroys far less signal in the
    /// early steps than a linear schedule does, which is the whole reason that paper introduced
    /// it and the reason this one cites it.
    ///
    /// The two stated endpoints are the clamp rather than interpolation bounds, the same role
    /// the 0.999 cap plays in Nichol and Dhariwal. alpha_bar is accumulated from the CLAMPED
    /// betas, not from the raw cosine, so the schedule the sampler runs is the one the training
    /// step noises with.
    /// </remarks>
    private void ComputeNoiseSchedule()
    {
        if (_diffusionSteps <= 0)
            throw new ArgumentOutOfRangeException(nameof(_diffusionSteps), "DiffusionSteps must be positive.");

        _betas = new Vector<T>(_diffusionSteps);
        _alphas = new Vector<T>(_diffusionSteps);
        _alphasCumprod = new Vector<T>(_diffusionSteps);
        _alphasCumprodPrev = new Vector<T>(_diffusionSteps);
        _sqrtAlphasCumprod = new Vector<T>(_diffusionSteps);
        _sqrtOneMinusAlphasCumprod = new Vector<T>(_diffusionSteps);

        double cumulative = 1.0;
        double rawPrevious = CosineAlphaBar(0);
        for (int k = 0; k < _diffusionSteps; k++)
        {
            double raw = CosineAlphaBar(k + 1);
            double beta = rawPrevious > 0.0 ? 1.0 - raw / rawPrevious : _betaEnd;
            if (beta < _betaStart) beta = _betaStart;
            if (beta > _betaEnd) beta = _betaEnd;
            rawPrevious = raw;

            _betas[k] = NumOps.FromDouble(beta);
            _alphas[k] = NumOps.FromDouble(1.0 - beta);
            _alphasCumprodPrev[k] = NumOps.FromDouble(cumulative);
            cumulative *= 1.0 - beta;
            _alphasCumprod[k] = NumOps.FromDouble(cumulative);
            _sqrtAlphasCumprod[k] = NumOps.FromDouble(Math.Sqrt(cumulative));
            _sqrtOneMinusAlphasCumprod[k] = NumOps.FromDouble(Math.Sqrt(Math.Max(0.0, 1.0 - cumulative)));
        }
    }

    /// <summary>The unnormalized cosine alpha_bar of Nichol and Dhariwal 2021, Equation 17.</summary>
    private double CosineAlphaBar(int step)
    {
        const double Offset = 0.008;
        double angle = ((double)step / _diffusionSteps + Offset) / (1.0 + Offset) * Math.PI / 2.0;
        double cosine = Math.Cos(angle);
        return cosine * cosine;
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
        else if (_useNativeMode) { Layers.AddRange(LayerHelper<T>.CreateDefaultTimeDiffLayers(Architecture, _contextLength, _forecastHorizon, _hiddenDimension, _numLayers, _numHeads, _dropout)); ExtractLayerReferences(); }
    }

    private void ExtractLayerReferences()
    {
        int idx = 0;
        if (idx < Layers.Count) _inputProjection = Layers[idx++];
        _transformerLayers.Clear();
        while (idx < Layers.Count - 1) _transformerLayers.Add(Layers[idx++]);
        if (idx < Layers.Count) _outputProjection = Layers[idx++];
    }

    #endregion

    #region NeuralNetworkBase Overrides

    public override bool SupportsTraining => _useNativeMode;
    protected override Tensor<T> PredictCore(Tensor<T> input) => _useNativeMode ? ForwardNative(input) : ForecastOnnx(input);

    /// <summary>
    /// One training step of Algorithm 1: sample a batch of diffusion steps k, diffuse the
    /// target, and regress the denoiser's x_0 prediction onto the clean target.
    /// </summary>
    /// <remarks>
    /// What this replaces trained the Layers as a plain context-to-forecast regression head
    /// while the sampler read those very same Layers as a denoiser on a differently packed
    /// input, and its own remarks recorded that as out of scope. Nothing inference did had ever
    /// been trained, which is why the reverse chain ran away to about 1e16.
    ///
    /// Two things here are specific to this paper rather than to DDPM. Equation 19 is
    /// L_k = ||x_0 - x_theta(x_k, k|c)||^2, so the network predicts the DATA and not the noise:
    /// "Note that we predict the data x(x_k, k) for denoising, rather than predicting the noise
    /// epsilon(x_k, k). As time series data usually contain highly irregular noisy components,
    /// estimating the diffusion noise can be more difficult." And the conditioning c is
    /// (z_mix, z_ar) of Equation 13, where future mixup (Equation 14) blends the mapped past
    /// with the ground-truth future under an elementwise Uniform[0,1) mask - available here
    /// precisely because this is the training path.
    ///
    /// The draw is deliberately NOT seeded from <c>Options.Seed</c>. That seed makes inference
    /// reproducible; reusing it here would hand every call the same k and the same epsilon, so
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

            int outputLen = _forecastHorizon;
            var rand = RandomHelper.CreateSecureRandom();
            int rows = Math.Max(1, _trainingBatchSize);

            var autoregressiveInit = BuildAutoregressiveInit(conditioned, outputLen);
            var timesteps = new int[rows];
            var clean = new Tensor<T>(new[] { rows, outputLen });
            var noised = new Tensor<T>(new[] { rows, outputLen });
            var mixed = new Tensor<T>(new[] { rows, outputLen });
            for (int r = 0; r < rows; r++)
            {
                int k = rand.Next(_diffusionSteps);
                timesteps[r] = k;
                T sqrtAlphaBar = _sqrtAlphasCumprod[k];
                T sqrtOneMinus = _sqrtOneMinusAlphasCumprod[k];
                for (int i = 0; i < outputLen; i++)
                {
                    T epsilon = SampleStandardNormal(rand);
                    T y = i < expectedOutput.Length ? expectedOutput[i] : NumOps.Zero;
                    T past = MappedPastValue(conditioned, outputLen, i);
                    int flat = r * outputLen + i;

                    clean.Data.Span[flat] = y;
                    noised.Data.Span[flat] = NumOps.Add(
                        NumOps.Multiply(sqrtAlphaBar, y),
                        NumOps.Multiply(sqrtOneMinus, epsilon));

                    if (_useFutureMixup)
                    {
                        // Equation 14: each element of m_k is drawn from Uniform[0, 1).
                        double m = rand.NextDouble();
                        mixed.Data.Span[flat] = NumOps.Add(
                            NumOps.Multiply(NumOps.FromDouble(m), past),
                            NumOps.Multiply(NumOps.FromDouble(1.0 - m), y));
                    }
                    else
                    {
                        mixed.Data.Span[flat] = past;
                    }
                }
            }

            using var tape = new GradientTape<T>();
            var predicted = DenoiserForward(noised, mixed, autoregressiveInit, timesteps, rows, outputLen);
            var lossTensor = _lossFunction.ComputeTapeLoss(predicted, clean);

            // Publish through the base rather than calling tape.ComputeGradients directly.
            // GetParameterGradients() answers from the published surface, and with nothing
            // published it falls back to the per-layer accessors, which fabricate an exact zero
            // for every parameter - indistinguishable from a severed tape.
            var grads = ComputeAndPublishParameterGradients(tape, lossTensor, trainableParams);

            T lossValue = lossTensor.Length > 0 ? lossTensor[0] : NumOps.Zero;
            LastLoss = lossValue;

            // Both closures are pinned to this step's draws. A line-searching optimizer that
            // re-drew them would be comparing losses from two different noise levels and
            // reading the difference as progress.
            Tensor<T> ComputeForward(Tensor<T> _, Tensor<T> __) =>
                DenoiserForward(noised, mixed, autoregressiveInit, timesteps, rows, outputLen);
            Tensor<T> RecomputeLoss(Tensor<T> pred, Tensor<T> __) =>
                _lossFunction.ComputeTapeLoss(pred, clean);

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
    /// Tape-aware forward over the denoiser graph, at the last diffusion step.
    /// </summary>
    /// <remarks>
    /// This exists so that callers which probe the training graph without a target - gradient
    /// reachability checks, parameter-movement probes - drive the same packed
    /// [x_k | z_mix | z_ar | step embedding] input that <see cref="Train"/> and the sampler
    /// drive. The input projection bakes its width on its first forward, so a probe that fed
    /// the raw context here would size the model for a shape neither of the real paths uses.
    /// </remarks>
    public override Tensor<T> ForwardForTraining(Tensor<T> input)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Training is only supported in native mode.");

        var conditioned = ApplyInstanceNormalization(input);
        if (conditioned.Rank == 1)
            conditioned = Engine.Reshape(conditioned, new[] { 1, conditioned.Length });

        int outputLen = _forecastHorizon;
        var rand = _options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(_options.Seed.Value)
            : RandomHelper.CreateSecureRandom();

        var noised = new Tensor<T>(new[] { 1, outputLen });
        var mixed = new Tensor<T>(new[] { 1, outputLen });
        for (int i = 0; i < outputLen; i++)
        {
            noised.Data.Span[i] = SampleStandardNormal(rand);
            mixed.Data.Span[i] = MappedPastValue(conditioned, outputLen, i);
        }

        return DenoiserForward(
            noised, mixed, BuildAutoregressiveInit(conditioned, outputLen),
            new[] { _diffusionSteps - 1 }, rows: 1, outputLen: outputLen);
    }

    // UpdateParameters was an empty override, silently dropping every restore. The base
    // distributes the vector over the declared enumeration.
    public override ModelMetadata<T> GetModelMetadata() => new()
    {
        AdditionalInfo = new Dictionary<string, object> { { "NetworkType", "TimeDiff" }, { "ContextLength", _contextLength }, { "ForecastHorizon", _forecastHorizon }, { "HiddenDimension", _hiddenDimension }, { "DiffusionSteps", _diffusionSteps }, { "UseFutureMixup", _useFutureMixup }, { "UseAutoregressiveInit", _useAutoregressiveInit }, { "UseNativeMode", _useNativeMode } },
        ModelData = _useNativeMode ? this.Serialize() : Array.Empty<byte>()
    };




    #endregion

    #region IForecastingModel Implementation

    public override Tensor<T> Forecast(Tensor<T> historicalData, double[]? quantiles = null) => _useNativeMode ? ForwardNative(historicalData) : ForecastOnnx(historicalData);
    public override Tensor<T> AutoregressiveForecast(Tensor<T> input, int steps)
    {
        // TimeDiff is non-autoregressive — truncate output to requested steps
        var fullForecast = Forecast(input, null);
        if (steps >= fullForecast.Length) return fullForecast;
        var result = new Tensor<T>(new[] { steps });
        for (int i = 0; i < steps; i++)
            result.Data.Span[i] = fullForecast[i];
        return result;
    }

    public override Dictionary<string, T> Evaluate(Tensor<T> predictions, Tensor<T> actuals) { T mse = NumOps.Zero; T mae = NumOps.Zero; int count = 0; for (int i = 0; i < predictions.Length && i < actuals.Length; i++) { var diff = NumOps.Subtract(predictions[i], actuals[i]); mse = NumOps.Add(mse, NumOps.Multiply(diff, diff)); mae = NumOps.Add(mae, NumOps.Abs(diff)); count++; } if (count > 0) { mse = NumOps.Divide(mse, NumOps.FromDouble(count)); mae = NumOps.Divide(mae, NumOps.FromDouble(count)); } return new Dictionary<string, T> { ["MSE"] = mse, ["MAE"] = mae, ["RMSE"] = NumOps.Sqrt(mse) }; }

    public override Tensor<T> ApplyInstanceNormalization(Tensor<T> input)
        // RevIN forward (Kim et al. 2022), delegated to the shared tape-tracked helper. The previous
        // hand-rolled version accumulated mean/variance with scalar NumOps arithmetic and wrote the
        // output through result.Data.Span[...], which the autodiff tape cannot observe: the normalised
        // tensor came back as a LEAF, so no gradient could flow through the normalisation. RevIN is a
        // differentiable layer in the paper, not a preprocessing step.
        => NormalizeInstanceOnTape(input, DefaultRevInEpsilon, out _, out _);

    public override Dictionary<string, T> GetFinancialMetrics() { T lastLoss = LastLoss is not null ? LastLoss : NumOps.Zero; return new Dictionary<string, T> { ["ContextLength"] = NumOps.FromDouble(_contextLength), ["ForecastHorizon"] = NumOps.FromDouble(_forecastHorizon), ["DiffusionSteps"] = NumOps.FromDouble(_diffusionSteps), ["LastLoss"] = lastLoss }; }

    #endregion

    #region Forward/Backward Pass

    /// <summary>
    /// Algorithm 2: run the reverse chain from x_K ~ N(0, I) down to k = 1 using the
    /// x_0-parameterized denoising step of Equation 18.
    /// </summary>
    /// <remarks>
    /// Two corrections to what this replaces. It ran the epsilon-parameterized DDPM step,
    /// x_{k-1} = (x_k - beta_k / sqrt(1 - alpha_bar_k) * eps) / sqrt(alpha_k), although this
    /// paper predicts the data; and it started the chain from a noised autoregressive guess.
    /// Algorithm 2 line 1 is unconditionally x_K ~ N(0, I) - z_ar enters through the CONDITION
    /// of Equation 13, not through the initial state, which is what the ablation in Table 4
    /// switches on and off.
    ///
    /// Equation 18 is the DDPM posterior mean with x_0 replaced by the prediction:
    ///   x_{k-1} = sqrt(alpha_k)(1 - alpha_bar_{k-1})/(1 - alpha_bar_k) * x_k
    ///           + sqrt(alpha_bar_{k-1}) * beta_k/(1 - alpha_bar_k) * x_theta(x_k, k|c)
    ///           + sigma_k * epsilon.
    /// sigma_k is the posterior standard deviation sqrt(beta_tilde_k), the partner of that
    /// mean, and is zero on the last step so the returned series is denoised. At k = 0,
    /// alpha_bar_{-1} = 1 makes the two coefficients 0 and 1, so the chain ends exactly at the
    /// prediction.
    /// </remarks>
    private Tensor<T> ForwardNative(Tensor<T> input)
    {
        var conditioned = ApplyInstanceNormalization(input);
        bool addedBatchDim = false;
        if (conditioned.Rank == 1) { conditioned = conditioned.Reshape(new[] { 1, conditioned.Length }); addedBatchDim = true; }

        int outputLen = _forecastHorizon;

        // Restart the noise stream at the configured seed so Predict called twice on the same
        // input returns the same answer. With Seed null the draw is secure and not reproducible.
        var rand = _options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(_options.Seed.Value)
            : RandomHelper.CreateSecureRandom();

        var autoregressiveInit = BuildAutoregressiveInit(conditioned, outputLen);

        // Equation 15: with the future unavailable at inference, the mixup branch collapses to
        // the mapped past alone.
        var mixed = new Tensor<T>(new[] { 1, outputLen });
        var xk = new Tensor<T>(new[] { 1, outputLen });
        for (int i = 0; i < outputLen; i++)
        {
            mixed.Data.Span[i] = MappedPastValue(conditioned, outputLen, i);
            xk.Data.Span[i] = SampleStandardNormal(rand);
        }

        T one = NumOps.One;
        T guard = NumOps.FromDouble(1e-10);
        var step = new int[1];
        for (int k = _diffusionSteps - 1; k >= 0; k--)
        {
            step[0] = k;
            var predicted = DenoiserForward(xk, mixed, autoregressiveInit, step, 1, outputLen);

            T alphaBarPrev = _alphasCumprodPrev[k];
            T oneMinusBar = NumOps.Add(NumOps.Subtract(one, _alphasCumprod[k]), guard);
            T coefficientXk = NumOps.Divide(
                NumOps.Multiply(NumOps.Sqrt(_alphas[k]), NumOps.Subtract(one, alphaBarPrev)),
                oneMinusBar);
            T coefficientX0 = NumOps.Divide(
                NumOps.Multiply(NumOps.Sqrt(alphaBarPrev), _betas[k]),
                oneMinusBar);
            T sigma = k > 0
                ? NumOps.Sqrt(NumOps.Divide(
                    NumOps.Multiply(NumOps.Subtract(one, alphaBarPrev), _betas[k]), oneMinusBar))
                : NumOps.Zero;

            for (int i = 0; i < outputLen; i++)
            {
                T x0Hat = i < predicted.Length ? predicted[i] : NumOps.Zero;
                T mean = NumOps.Add(
                    NumOps.Multiply(coefficientXk, xk[i]),
                    NumOps.Multiply(coefficientX0, x0Hat));
                T z = k > 0 ? SampleStandardNormal(rand) : NumOps.Zero;
                xk.Data.Span[i] = NumOps.Add(mean, NumOps.Multiply(sigma, z));
            }
        }

        if (addedBatchDim && xk.Rank == 2 && xk.Shape[0] == 1) xk = xk.Reshape(new[] { xk.Shape[1] });
        return xk;
    }

    protected override Tensor<T> ForecastOnnx(Tensor<T> input) { if (OnnxSession == null) throw new InvalidOperationException("ONNX session is not initialized."); int batchSize = input.Shape[0]; int seqLen = input.Shape.Length > 1 ? input.Shape[1] : input.Length; int features = input.Shape.Length > 2 ? input.Shape[2] : 1; var inputData = new float[batchSize * seqLen * features]; for (int i = 0; i < input.Length && i < inputData.Length; i++) inputData[i] = (float)NumOps.ToDouble(input[i]); var inputTensor = new OnnxTensors.DenseTensor<float>(inputData, new[] { batchSize, seqLen, features }); var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("input", inputTensor) }; using var results = OnnxSession.Run(inputs); var outputTensor = results.First().AsTensor<float>(); var outputShape = outputTensor.Dimensions.ToArray(); var output = new Tensor<T>(outputShape); int totalElements = 1; foreach (var dim in outputShape) totalElements *= dim; for (int i = 0; i < totalElements && i < output.Length; i++) output.Data.Span[i] = NumOps.FromDouble(outputTensor.GetValue(i)); return output; }

    /// <summary>
    /// Runs the denoiser once over <paramref name="rows"/> rows: packs
    /// [x_k | z_mix | z_ar | diffusion-step embedding] per row and returns the predicted x_0.
    /// </summary>
    /// <remarks>
    /// Training and inference both go through here, which is the whole point of the method
    /// existing. The input projection from <c>LayerHelper&lt;T&gt;.CreateDefaultTimeDiffLayers</c>
    /// is lazily sized - it bakes its input width on the first forward it sees - so two paths
    /// packing different widths made the model order-dependent on top of being untrained.
    ///
    /// The row is the paper's own structure rather than the scalar sum it replaces. Section 3.3
    /// concatenates the step embedding p_k with the diffused input's embedding along the
    /// channel dimension and then concatenates the condition c with that, and Equation 13 makes
    /// c the pair (z_mix, z_ar). What was here instead added one scalar,
    /// sin(2*pi*k/(K-1)), to the elementwise sum of x_k and the conditioning: the sum is not
    /// invertible, so the denoiser could not tell which part a value came from, and the scalar
    /// is not injective in k - it gives step 10 and step 90 of a 100-step schedule the same
    /// encoding, so half the schedule was aliased onto the other half. The multi-frequency
    /// embedding of Equation 17 is shared with the sibling diffusion forecasters through
    /// <see cref="TimeSeriesFoundationModelBase{T}.WriteDiffusionTimestepEmbedding"/>.
    ///
    /// The packed row is built by a direct span fill rather than traced engine arithmetic
    /// because none of its parts carries a parameter dependency: x_k is either prior noise or a
    /// noised target, both conditioning components are functions of the normalized input, and
    /// the step encoding is a constant. Only the layer chain below needs the tape.
    /// </remarks>
    private Tensor<T> DenoiserForward(
        Tensor<T> xk, Tensor<T> mixed, T[] autoregressiveInit,
        IReadOnlyList<int> timesteps, int rows, int outputLen)
    {
        int rowLen = outputLen + outputLen + outputLen + DiffusionTimestepEmbeddingDim;
        var packed = new Tensor<T>(new[] { rows, rowLen });
        var destination = packed.Data.Span;
        for (int r = 0; r < rows; r++)
        {
            int baseIndex = r * rowLen;
            for (int i = 0; i < outputLen; i++)
            {
                int flat = r * outputLen + i;
                destination[baseIndex + i] = flat < xk.Length ? xk[flat] : NumOps.Zero;
                destination[baseIndex + outputLen + i] = flat < mixed.Length ? mixed[flat] : NumOps.Zero;
                destination[baseIndex + 2 * outputLen + i] = autoregressiveInit[i];
            }

            int step = timesteps.Count == 1 ? timesteps[0] : timesteps[r];
            WriteDiffusionTimestepEmbedding(
                destination.Slice(baseIndex + 3 * outputLen, DiffusionTimestepEmbeddingDim), step);
        }

        var hidden = packed;
        if (_inputProjection is not null) hidden = _inputProjection.Forward(hidden);
        foreach (var layer in _transformerLayers) hidden = layer.Forward(hidden);
        if (_outputProjection is not null) hidden = _outputProjection.Forward(hidden);
        return hidden;
    }

    /// <summary>
    /// The mapped past F(x_{-L+1:0}) of Equation 15, evaluated at horizon position
    /// <paramref name="position"/>.
    /// </summary>
    /// <remarks>
    /// The paper uses a convolution network for F. Here F is the trailing horizon-length window
    /// of the normalized context, which the learned input projection then maps jointly with the
    /// rest of the packed row - the same arrangement the sibling forecasters use, and the reason
    /// the conditioning is packed RAW rather than pre-projected.
    /// </remarks>
    private T MappedPastValue(Tensor<T> conditioned, int outputLen, int position)
    {
        if (conditioned.Length == 0) return NumOps.Zero;
        int index = conditioned.Length - outputLen + position;
        if (index < 0) return conditioned[0];
        if (index >= conditioned.Length) return conditioned[conditioned.Length - 1];
        return conditioned[index];
    }

    /// <summary>
    /// The autoregressive initial guess z_ar of Equation 16, or zeros when
    /// <c>UseAutoregressiveInit</c> is off.
    /// </summary>
    /// <remarks>
    /// Equation 16 is a linear map from the context columns with trainable W_i and B, pretrained
    /// for a few epochs against the ground-truth future. This implementation uses the fixed
    /// linear extrapolator that map degenerates to for a single series - the last observation
    /// carried forward along its last first difference - so there is no second model to
    /// pretrain and no second optimizer to keep in step. It is still the paper's role for z_ar:
    /// "this simple AR model cannot accurately approximate a complex nonlinear time series in
    /// general, [but] it can still capture simple patterns, such as short-term trends."
    /// </remarks>
    private T[] BuildAutoregressiveInit(Tensor<T> conditioned, int outputLen)
    {
        var result = new T[outputLen];
        if (!_useAutoregressiveInit || conditioned.Length == 0)
        {
            for (int i = 0; i < outputLen; i++) result[i] = NumOps.Zero;
            return result;
        }

        T last = conditioned[conditioned.Length - 1];
        T previous = conditioned.Length > 1 ? conditioned[conditioned.Length - 2] : last;
        T trend = NumOps.Subtract(last, previous);
        for (int i = 0; i < outputLen; i++)
            result[i] = NumOps.Add(last, NumOps.Multiply(trend, NumOps.FromDouble(i + 1)));

        return result;
    }


    #region ITrainingObjectiveProvider

    /// <summary>
    /// The learner is denoising diffusion, not supervised regression of the forecast onto the
    /// target: <see cref="Train"/> minimizes Equation 19, and <see cref="Predict"/> is a
    /// 100-step reverse chain run on top of it.
    /// </summary>
    /// <remarks>
    /// Declaring the objective is what lets a loss-trajectory probe measure the quantity
    /// training actually descends. Judging this model on the sampler instead measures something
    /// the optimizer never sees: the reverse chain compounds whatever bias a partially trained
    /// denoiser still carries over every one of its DiffusionSteps steps, so early in training
    /// that magnified bias moves the sampled path by far more than a handful of optimizer steps
    /// improve it, which reads as a model getting worse while its own objective is falling.
    /// </remarks>
    TrainingObjectiveKind ITrainingObjectiveProvider<T>.TrainingObjectiveKind =>
        TrainingObjectiveKind.DiffusionDenoising;

    /// <summary>The supplied forecast target IS the x_0 the denoiser learns to recover.</summary>
    Tensor<T> ITrainingObjectiveProvider<T>.ResolveTrainingTarget(Tensor<T> input, Tensor<T> proposedTarget)
        => proposedTarget;

    /// <summary>
    /// Equation 19 over a FIXED quadrature of diffusion steps and noise draws, scored through
    /// the model's configured loss function.
    /// </summary>
    /// <remarks>
    /// The quadrature is deterministic so two evaluations of an unchanged model agree, which is
    /// what makes a before/after comparison a statement about the parameters. Future mixup is
    /// left out for the same reason dropout is: Equation 15 already removes it outside training,
    /// it is an augmentation of the conditioning rather than part of the objective's definition,
    /// and mixing the ground-truth future into the measured input would make the number easier
    /// the closer the mask happened to fall to zero.
    /// </remarks>
    T ITrainingObjectiveProvider<T>.EvaluateTrainingObjective(Tensor<T> input, Tensor<T> target)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("The training objective is only defined in native mode.");

        var conditioned = ApplyInstanceNormalization(input);
        if (conditioned.Rank == 1)
            conditioned = Engine.Reshape(conditioned, new[] { 1, conditioned.Length });

        int outputLen = _forecastHorizon;
        var (noised, _, timesteps) = BuildDeterministicDenoisingBatch(
            target, outputLen, _diffusionSteps, _sqrtAlphasCumprod, _sqrtOneMinusAlphasCumprod);

        int rows = timesteps.Length;
        var clean = new Tensor<T>(new[] { rows, outputLen });
        var mixed = new Tensor<T>(new[] { rows, outputLen });
        for (int r = 0; r < rows; r++)
        {
            for (int i = 0; i < outputLen; i++)
            {
                int flat = r * outputLen + i;
                clean.Data.Span[flat] = i < target.Length ? target[i] : NumOps.Zero;
                mixed.Data.Span[flat] = MappedPastValue(conditioned, outputLen, i);
            }
        }

        var predicted = DenoiserForward(
            noised, mixed, BuildAutoregressiveInit(conditioned, outputLen),
            timesteps, rows, outputLen);
        return _lossFunction.ComputeLoss(predicted, clean);
    }

    #endregion
    #endregion
}
