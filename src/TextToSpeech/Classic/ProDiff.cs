using AiDotNet.LearningRateSchedulers;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.TextToSpeech.Interfaces;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.ActivationFunctions;

namespace AiDotNet.TextToSpeech.Classic;

/// <summary>The two training stages of <see cref="ProDiff{T}"/>.</summary>
public enum ProDiffTrainingPhase
{
    /// <summary>The generator-based teacher with <see cref="ProDiffOptions.TeacherDiffusionSteps"/> steps learns to
    /// predict the clean spectrogram (§4.2, Eq. 9).</summary>
    Teacher = 0,

    /// <summary>The student, initialized from the teacher, learns to match two DDIM steps of the frozen teacher with one
    /// of its own (§4.3, Algorithm 1).</summary>
    Distillation = 1,
}

/// <summary>
/// ProDiff: a progressive fast diffusion model that synthesizes a mel spectrogram in two denoising steps by predicting
/// the clean spectrogram directly and distilling a 4-step teacher.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b> "ProDiff: Progressive Fast Diffusion Model for High-Quality Text-to-Speech" (Huang et al.,
/// ACM MM 2022) and its reference implementation (Rongjiehuang/ProDiff) for what the paper leaves unstated.</para>
/// <para>
/// The phoneme encoder and variance adaptor are FastSpeech 2's (§4.4): a pre-net, FFT blocks and a linear projection,
/// then duration, pitch-spectrogram and energy predictors with a length regulator. The spectrogram denoiser
/// f_θ(x_t | t, c) is a non-causal WaveNet conditioned on the adaptor output (<c>ProDiffDenoiser</c>). Diffusion uses a
/// cosine schedule with α_t = √ᾱ_t and σ_t = √(1 − ᾱ_t). The teacher (<see cref="ProDiffTrainingPhase.Teacher"/>)
/// minimizes ‖f_θ(α_t x₀ + σ_t ε) − x₀‖² plus 1 − SSIM (Eq. 9, 11) with T₁ = 4 steps; the distillation phase
/// (Algorithm 1) samples t ∈ S·{1..T₂}, S = T₁/T₂, runs two DDIM steps of the frozen teacher from x_t to x_{t−S},
/// solves for the target x̂₀ that one student step would need, and minimizes ‖f_θ(x_t) − x̂₀‖² + 1 − SSIM (Eq. 10, 11).
/// Both add the variance reconstruction losses with weight 0.1 (Eq. 12). Sampling (Algorithm 2) starts from N(0, I)
/// and alternates x̂₀ = f_θ(x_t) with a draw from the posterior q(x_{t−S} | x_t, x̂₀).
/// </para>
/// <para><b>For Beginners:</b> Diffusion models usually clean up a noisy spectrogram over hundreds of small steps.
/// ProDiff predicts the clean spectrogram outright and learns from a slower teacher, so two steps suffice.</para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "ProDiff: Progressive Fast Diffusion Model for High-Quality Text-to-Speech",
    "https://arxiv.org/abs/2207.06389",
    Year = 2022,
    Authors = "Huang et al."
)]
[PaperOptimizer(OptimizerKind.Adam, Beta1 = 0.9, Beta2 = 0.98, Epsilon = 1e-9,
                ReferenceBatchSize = 64,
                Source = "Huang et al. 2022, Sec. 6.1: the Adam optimizer with beta1 0.9, beta2 0.98 "
                        + "and epsilon 1e-9, trained for 200,000 steps at a batch size of 64 sentences.")]
public partial class ProDiff<T> : VarianceAdaptorTtsModelBase<T>, IAcousticModel<T>
{
    private readonly ProDiffOptions _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;
    private Random _trainingRandom;
    private ProDiffDenoiser<T>? _denoiser;
    private ProDiffDenoiser<T>? _teacher;
    private TrainingDraw? _pendingDraw;

    public override ModelOptions GetOptions() => _options;

    public ProDiff(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        ProDiffOptions? options = null
    )
        : base(architecture)
    {
        _options = options ?? new ProDiffOptions();
        ValidateOptions(_options);
        _useNativeMode = false;
        _trainingRandom = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed);
        base.SampleRate = _options.SampleRate;
        base.MelChannels = _options.MelChannels;
        base.HopSize = _options.HopSize;
        base.HiddenDim = _options.HiddenDim;
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path required.", nameof(modelPath));
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    public ProDiff(
        NeuralNetworkArchitecture<T> architecture,
        ProDiffOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null
    )
        : base(architecture)
    {
        _options = options ?? new ProDiffOptions();
        ValidateOptions(_options);
        _useNativeMode = true;
        _trainingRandom = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed);
        base.SampleRate = _options.SampleRate;
        base.MelChannels = _options.MelChannels;
        base.HopSize = _options.HopSize;
        base.HiddenDim = _options.HiddenDim;
        if (_options.MaxGradientNorm > 0)
            MaxGradNorm = NumOps.FromDouble(_options.MaxGradientNorm);
        _optimizer = optimizer ?? CreateDefaultOptimizer();
        InitializeLayers();
    }

    int ITtsModel<T>.SampleRate => _options.SampleRate;
    public int MaxTextLength => _options.MaxTextLength;
    public new int MelChannels => _options.MelChannels;
    public new int HopSize => _options.HopSize;
    public int FftSize => _options.FftSize;

    /// <summary>
    /// The training stage: <see cref="ProDiffTrainingPhase.Teacher"/> until the teacher converges, then
    /// <see cref="BeginDistillation()"/>. Synthesis uses the teacher's T₁ steps in the teacher phase and the student's T₂
    /// steps once distillation has begun.
    /// </summary>
    public ProDiffTrainingPhase CurrentPhase { get; private set; } = ProDiffTrainingPhase.Teacher;

    /// <summary>Generates a mel spectrogram from text (the paper pairs it with HiFi-GAN).</summary>
    public Tensor<T> TextToMel(string text) => Synthesize(text);

    /// <summary>
    /// Starts distillation from this model's own trained teacher: the student keeps the current parameters ("we first
    /// initialize ProDiff with a copy of the teacher", §4.3) and a frozen copy of the denoiser becomes the teacher.
    /// </summary>
    public void BeginDistillation()
    {
        ThrowIfDisposed();
        if (_denoiser is null)
            throw new NotSupportedException("Distillation needs the paper's denoiser; this model was built from caller-supplied layers.");
        _teacher = CreateDenoiser();
        CopyParameters(_denoiser, _teacher);
        CurrentPhase = ProDiffTrainingPhase.Distillation;
    }

    /// <summary>
    /// Starts distillation from a separately trained teacher: this model is initialized with a copy of the teacher's
    /// parameters (§4.3) and the teacher's denoiser is frozen as the distillation target.
    /// </summary>
    public void BeginDistillation(ProDiff<T> teacher)
    {
        Guard.NotNull(teacher);
        ThrowIfDisposed();
        if (teacher._denoiser is null || _denoiser is null)
            throw new NotSupportedException("Distillation needs the paper's denoiser on both models.");
        SetParameters(teacher.GetParameters());
        BeginDistillation();
    }

    private ProDiffDenoiser<T> CreateDenoiser()
        => new(Engine, _options.MelChannels, _options.HiddenDim, _options.DenoiserChannels, _options.DenoiserLayers,
            _options.DenoiserDilationCycle);

    // Copies a denoiser's parameters layer by layer; one dummy forward first sizes any lazily built layer of either.
    private void CopyParameters(ProDiffDenoiser<T> from, ProDiffDenoiser<T> to)
    {
        using (new NoGradScope<T>())
        {
            var x = new Tensor<T>(new[] { 4, _options.MelChannels });
            var condition = new Tensor<T>(new[] { 4, _options.HiddenDim });
            from.Predict(x, 1, condition);
            to.Predict(x, 1, condition);
        }
        for (int i = 0; i < from.Layers.Count; i++)
            if (from.Layers[i].ParameterCount > 0)
                to.Layers[i].SetParameters(from.Layers[i].GetParameters());
    }

    protected override void InitializeLayers()
    {
        if (!_useNativeMode)
            return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            AddUnsplitLayers(Architecture.Layers);
            return;
        }

        var o = _options;
        // Encoder (§4.4, Table 4): embedding, a 3-layer pre-net, FFT blocks and a final linear projection.
        var encoder = new List<ILayer<T>>
        {
            new EmbeddingLayer<T>(o.VocabSize, o.EncoderDim),
            new ResidualConvReluNormLayer<T>(o.EncoderDim, o.PrenetKernelSize, o.PrenetLayers, o.DropoutRate),
            new PositionalEncodingLayer<T>(o.MaxTextLength, o.EncoderDim),
        };
        for (int i = 0; i < o.NumEncoderLayers; i++)
            encoder.Add(new FeedForwardTransformerBlock<T>(o.EncoderDim, o.NumHeads, o.FftFilterSize, o.FftKernelSizes[0],
                o.FftKernelSizes[1], o.DropoutRate));
        encoder.Add(new DenseLayer<T>(o.HiddenDim, new IdentityActivation<T>() as IActivationFunction<T>));

        var adaptor = new VarianceAdaptorLayer<T>(o.HiddenDim, o.VariancePredictorFilterSize, o.VariancePredictorKernelSize,
            o.VariancePredictorDropout, o.NumPitchBins, o.PitchMinHz, o.PitchMaxHz, o.NumEnergyBins,
            energyMin: 0.0, energyMax: VarianceAdaptorLayer<T>.MaxStftEnergy(o.FftSize));
        AddEncoderDecoderLayers(encoder, new ILayer<T>[] { adaptor });
        _denoiser = CreateDenoiser();
        ComponentLayers.AddRange(_denoiser.Layers);
    }

    private bool HasPaperLayers => _denoiser is not null;

    /// <inheritdoc />
    protected override bool UsesL1MelLoss => false;

    /// <inheritdoc />
    protected override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? TrainingOptimizer => _optimizer;

    /// <inheritdoc />
    protected override double VarianceLossWeight => _options.VarianceLossWeight;

    // ---------------------------------------------------------------- noise schedule

    /// <summary>ᾱ_t for t = 0..T₁ under the cosine schedule (Nichol and Dhariwal 2021, §4.4): ᾱ_t = f(t)/f(0) with
    /// f(t) = cos²(((t/T₁) + s)/(1 + s) · π/2), each β_t clipped to 0.999.</summary>
    private double[] CumulativeAlphas()
    {
        int steps = _options.TeacherDiffusionSteps;
        double s = _options.CosineScheduleOffset;
        double F(double t) => Math.Pow(Math.Cos((t / steps + s) / (1 + s) * Math.PI * 0.5), 2);
        var alphaBar = new double[steps + 1];
        alphaBar[0] = 1.0;
        for (int t = 1; t <= steps; t++)
        {
            double beta = Math.Min(1.0 - F(t) / F(t - 1), 0.999);
            alphaBar[t] = alphaBar[t - 1] * (1.0 - beta);
        }
        return alphaBar;
    }

    private (double Alpha, double Sigma) Schedule(int t)
    {
        double alphaBar = CumulativeAlphas()[t];
        return (Math.Sqrt(alphaBar), Math.Sqrt(1.0 - alphaBar));
    }

    private int StepStride => CurrentPhase == ProDiffTrainingPhase.Distillation
        ? _options.TeacherDiffusionSteps / _options.NumDiffusionSteps
        : 1;

    private Tensor<T> Gaussian(int[] shape, Random random)
    {
        var t = new Tensor<T>(shape);
        for (int i = 0; i < t.Length; i++)
        {
            double u1 = 1.0 - random.NextDouble(), u2 = random.NextDouble();
            t[i] = NumOps.FromDouble(Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2));
        }
        return t;
    }

    // ---------------------------------------------------------------- training

    private sealed record TrainingDraw(int Step, Random Noise);

    // t ~ S · Unif{1..T/S} (Algorithm 1 line 4; S = 1 for the teacher), and the source of ε.
    private TrainingDraw Draw(Random random)
    {
        int stride = StepStride, count = _options.TeacherDiffusionSteps / stride;
        return new TrainingDraw(stride * (1 + random.Next(count)), random);
    }

    /// <inheritdoc />
    protected override T TrainOnSample(TtsTrainingSample<T> sample)
    {
        ThrowIfDisposed();
        if (!HasPaperLayers)
            throw new NotSupportedException("The paper objective needs the paper's denoiser; this model was built from caller-supplied layers.");
        _pendingDraw = Draw(_trainingRandom);
        try
        {
            return base.TrainOnSample(sample);
        }
        finally
        {
            _pendingDraw = null;
        }
    }

    /// <inheritdoc />
    /// <remarks>The step and the noise are fixed by the sampling seed so the same parameters always score the same.</remarks>
    public override T EvaluateTrainingObjective(TtsTrainingSample<T> sample)
    {
        ThrowIfDisposed();
        if (!HasPaperLayers)
            throw new NotSupportedException("The paper objective needs the paper's denoiser; this model was built from caller-supplied layers.");
        _pendingDraw = Draw(AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed));
        try
        {
            using var _ = new NoGradScope<T>();
            return base.EvaluateTrainingObjective(sample);
        }
        finally
        {
            _pendingDraw = null;
        }
    }

    /// <summary>
    /// The sample reconstruction and SSIM losses (Eq. 9–11) for the adaptor output: x_t = α_t x₀ + σ_t ε, and the
    /// denoiser's prediction scored against x₀ (teacher) or against the target x̂₀ that reproduces two DDIM steps of the
    /// frozen teacher (Algorithm 1 lines 6–9).
    /// </summary>
    protected override Tensor<T> MelObjective(Tensor<T> expanded, Tensor<T>? decoderCondition, Tensor<T> mel)
    {
        var draw = _pendingDraw ?? throw new InvalidOperationException("ProDiff's objective runs inside a training or evaluation call.");
        int t = draw.Step;
        var (alpha, sigma) = Schedule(t);
        var noise = Gaussian(mel._shape, draw.Noise);
        var xt = Engine.TensorAdd(Engine.TensorMultiplyScalar(mel, NumOps.FromDouble(alpha)), Engine.TensorMultiplyScalar(noise, NumOps.FromDouble(sigma)));

        Tensor<T> target = mel;
        if (CurrentPhase == ProDiffTrainingPhase.Distillation)
        {
            int stride = StepStride;
            var (alpha1, sigma1) = Schedule(t - stride / 2);
            var (alpha2, sigma2) = Schedule(t - stride);
            using (new NoGradScope<T>())
            {
                var condition = new Tensor<T>(expanded._shape, expanded.ToVector());
                var f = _teacher!.Predict(xt, t, condition);
                var x1 = DdimStep(f, xt, alpha, sigma, alpha1, sigma1);
                var f1 = _teacher.Predict(x1, t - stride / 2, condition);
                var x2 = DdimStep(f1, x1, alpha1, sigma1, alpha2, sigma2);
                double ratio = sigma2 / sigma;
                var solved = Engine.TensorMultiplyScalar(Engine.TensorSubtract(x2, Engine.TensorMultiplyScalar(xt, NumOps.FromDouble(ratio))),
                    NumOps.FromDouble(1.0 / (alpha2 - ratio * alpha)));
                target = new Tensor<T>(solved._shape, solved.ToVector());
            }
        }

        var predicted = _denoiser!.Predict(xt, t, expanded);
        var reconstruction = MeanSquaredError(predicted, target);
        var dissimilarity = Engine.TensorAddScalar(Engine.TensorNegate(SpectrogramSsim.Ssim(Engine, predicted, target)), NumOps.One);
        return Engine.TensorAdd(reconstruction, dissimilarity);
    }

    // DDIM: x_s = α_s f + (σ_s / σ_t)(x_t − α_t f).
    private Tensor<T> DdimStep(Tensor<T> f, Tensor<T> xt, double alphaT, double sigmaT, double alphaS, double sigmaS)
        => Engine.TensorAdd(Engine.TensorMultiplyScalar(f, NumOps.FromDouble(alphaS)),
            Engine.TensorMultiplyScalar(Engine.TensorSubtract(xt, Engine.TensorMultiplyScalar(f, NumOps.FromDouble(alphaT))),
                NumOps.FromDouble(sigmaS / sigmaT)));

    // ---------------------------------------------------------------- inference

    /// <inheritdoc />
    /// <remarks>Algorithm 2: encode and adapt with the predicted durations, pitch and energy, then from x ~ N(0, I) at
    /// the last step alternate x̂₀ = f_θ(x_t | t, c) with a draw from q(x_{t−S} | x_t, x̂₀); the final step returns x̂₀.</remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(input);
        SetTrainingMode(false);
        if (!HasPaperLayers)
            return base.PredictCore(input);
        if (input.Rank == 2 && input.Shape[0] == 1)
            input = Engine.Reshape(input, new[] { input.Shape[1] });
        if (input.Rank != 1)
            throw new ArgumentException($"Expected tokens [tokens], got [{string.Join(", ", input.Shape)}].", nameof(input));

        using var _ = new NoGradScope<T>();
        var condition = input;
        foreach (var layer in Layers) condition = layer.Forward(condition);       // encoder + variance adaptor
        int frames = condition.Shape[0];
        var alphaBar = CumulativeAlphas();
        int stride = StepStride;
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed);
        var x = Gaussian(new[] { frames, _options.MelChannels }, random);
        for (int t = _options.TeacherDiffusionSteps; t > 0; t -= stride)
        {
            var predicted = _denoiser!.Predict(x, t, condition);
            int previous = t - stride;
            if (previous == 0)
                return predicted;
            // q(x_s | x_t, x̂₀) for s = t − S: mean (√ᾱ_s β / (1 − ᾱ_t)) x̂₀ + (√a (1 − ᾱ_s) / (1 − ᾱ_t)) x_t with
            // a = ᾱ_t / ᾱ_s, β = 1 − a, and variance β (1 − ᾱ_s) / (1 − ᾱ_t).
            double a = alphaBar[t] / alphaBar[previous], beta = 1 - a;
            double c0 = Math.Sqrt(alphaBar[previous]) * beta / (1 - alphaBar[t]);
            double ct = Math.Sqrt(a) * (1 - alphaBar[previous]) / (1 - alphaBar[t]);
            double std = Math.Sqrt(beta * (1 - alphaBar[previous]) / (1 - alphaBar[t]));
            x = Engine.TensorAdd(Engine.TensorAdd(
                    Engine.TensorMultiplyScalar(predicted, NumOps.FromDouble(c0)),
                    Engine.TensorMultiplyScalar(x, NumOps.FromDouble(ct))),
                Engine.TensorMultiplyScalar(Gaussian(x._shape, random), NumOps.FromDouble(std)));
        }
        return x;
    }

    protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;

    /// <inheritdoc />
    /// <remarks>In this mode the weights belong to the loaded graph. The base refuses the
    /// write on every parameter surface, so the guard is stated once here instead of being
    /// repeated -- and cannot be applied to one surface and forgotten on another.</remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;

    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "ProDiff-Native" : "ProDiff-ONNX",
            Description = "ProDiff: Progressive Fast Diffusion Model for High-Quality TTS (Huang et al., 2022)",
            FeatureCount = _options.HiddenDim,
            Complexity = _options.NumEncoderLayers + _options.DenoiserLayers,
        };
        m.AdditionalInfo["Architecture"] = "ProDiff";
        m.AdditionalInfo["Phase"] = CurrentPhase.ToString();
        return m;
    }

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> CreateDefaultOptimizer()
    {
        // The reference schedule (rsqrt): lr · min(s / w, 1) · max(w, s)^-0.5 · d^-0.5, the Noam form.
        var scheduler = new NoamSchedule(modelDimension: _options.HiddenDim, warmupSteps: _options.WarmupSteps,
            factor: _options.LearningRate);
        return PaperOptimizerFactory.VerifyHandBuilt(this, new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this,
            new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = scheduler.CurrentLearningRate,
                Beta1 = _options.OptimizerBeta1,
                Beta2 = _options.OptimizerBeta2,
                Epsilon = _options.OptimizerEpsilon,
                UseAdaptiveLearningRate = false,
                UseAdaptiveBetas = false,
                UseAMSGrad = false,
                LearningRateScheduler = scheduler,
                SchedulerStepMode = SchedulerStepMode.StepPerBatch,
            }));
    }

    private static void ValidateOptions(ProDiffOptions options)
    {
        if (options.NumDiffusionSteps <= 0 || options.TeacherDiffusionSteps <= 0
            || options.TeacherDiffusionSteps % options.NumDiffusionSteps != 0
            || (options.TeacherDiffusionSteps / options.NumDiffusionSteps) % 2 != 0 && options.TeacherDiffusionSteps != options.NumDiffusionSteps)
            throw new ArgumentException(
                $"The teacher's steps ({options.TeacherDiffusionSteps}) must be an even multiple of the student's ({options.NumDiffusionSteps}).",
                nameof(options));
        if (options.FftKernelSizes is not { Length: 2 })
            throw new ArgumentException("FftKernelSizes must hold the two FFT-block kernels.", nameof(options));
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(ProDiff<T>));
    }

    protected override void Dispose(bool disposing)
    {
        if (_disposed)
            return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
