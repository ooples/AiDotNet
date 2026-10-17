using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.TextToSpeech.Classic;
using AiDotNet.TextToSpeech.Interfaces;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.ActivationFunctions;

namespace AiDotNet.TextToSpeech.FlowDiffusion;

/// <summary>The two training stages of <see cref="CoMoSpeech{T}"/>.</summary>
public enum CoMoSpeechTrainingPhase
{
    /// <summary>The EDM teacher trains on the duration, prior and denoising losses (§3.1, §3.4).</summary>
    Teacher = 0,

    /// <summary>Consistency distillation (§3.2): only the denoiser trains, towards an EMA target of itself evaluated one
    /// teacher ODE step closer to the data.</summary>
    Distillation = 1,
}

/// <summary>
/// CoMoSpeech: one-step speech synthesis with a consistency model distilled from an EDM diffusion teacher.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b> "CoMoSpeech: One-Step Speech and Singing Voice Synthesis via Consistency Model" (Ye et al.,
/// ACM MM 2023) and its reference implementation (zhenye234/CoMoSpeech) for what the paper leaves unstated.</para>
/// <para>
/// The encoder, duration predictor and monotonic alignment search are Grad-TTS's, giving the prior μ. The denoiser is
/// EDM-preconditioned (Eq. 8–9): <c>D(x, t) = c_skip(t) x + c_out(t) F(c_in(t) x, μ, ln t / 4)</c> with
/// <c>c_skip = σ_d² / ((t − ε)² + σ_d²)</c>, <c>c_out = σ_d (t − ε) / √(σ_d² + t²)</c>, <c>c_in = 1 / √(σ_d² + t²)</c>, and F
/// is Grad-TTS's U-Net. The teacher (<see cref="CoMoSpeechTrainingPhase.Teacher"/>) minimizes the duration loss, the
/// prior loss and the EDM-weighted denoising loss <c>λ(t) ‖D(x₀ + t n, t) − x₀‖²</c>, n ~ N(μ, I), ln t ~ N(−1.2, 1.2²)
/// (Eq. 10), and samples with N Euler steps of <c>dx = (x − D(x, t)) / t dt</c> from <c>t_N x_N</c>, x_N ~ N(μ, I)
/// (Algorithm 1). Distillation (<see cref="CoMoSpeechTrainingPhase.Distillation"/>, Eq. 11–12) freezes the encoder and a
/// copy of the teacher, keeps an EMA target θ⁻, and minimizes <c>‖D_θ(x_{n+1}, t_{n+1}) − D_θ⁻(x̂_n, t_n)‖²</c> where x̂_n
/// is one teacher Euler step from x_{n+1}; the distilled model synthesizes in one step, <c>D_θ(t_N x_N, t_N)</c> (Eq. 13),
/// or in several with Algorithm 2.
/// </para>
/// <para><b>For Beginners:</b> A slow diffusion "teacher" learns to clean noisy spectrograms over many steps; the
/// "student" then learns to jump from any point of that cleaning path straight to the end, so it speaks in one step.</para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Diffusion)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "CoMoSpeech: One-Step Speech and Singing Voice Synthesis via Consistency Model",
    "https://arxiv.org/abs/2305.06908",
    Year = 2023,
    Authors = "Ye et al."
)]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 1e-4, ReferenceBatchSize = 16,
                Source = "Ye et al. 2023, Sec. 4.1.2: the Adam optimizer with learning rate 1e-4 at a batch size of 16 "
                        + "for 1.7 million iterations, for both the teacher and the distilled model.")]
public partial class CoMoSpeech<T> : TtsModelBase<T>, IEndToEndTts<T>, IAcousticModel<T>, ITrainingObjectiveProvider<T>
{
    private readonly CoMoSpeechOptions _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;
    private Random _trainingRandom;

    private EmbeddingLayer<T>? _embedding;
    private ResidualConvReluNormLayer<T>? _prenet;
    private readonly List<RelativePositionTransformerBlock<T>> _encoderBlocks = new();
    private DenseLayer<T>? _meanProjection;
    private VariancePredictorLayer<T>? _durationPredictor;
    private GradTtsScoreEstimator<T>? _denoiser;
    private GradTtsScoreEstimator<T>? _teacher;
    private GradTtsScoreEstimator<T>? _target;

    public override ModelOptions GetOptions() => _options;

    public CoMoSpeech(NeuralNetworkArchitecture<T> architecture, string modelPath, CoMoSpeechOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new CoMoSpeechOptions();
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

    public CoMoSpeech(
        NeuralNetworkArchitecture<T> architecture,
        CoMoSpeechOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new CoMoSpeechOptions();
        _useNativeMode = true;
        _trainingRandom = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed);
        base.SampleRate = _options.SampleRate;
        base.MelChannels = _options.MelChannels;
        base.HopSize = _options.HopSize;
        base.HiddenDim = _options.HiddenDim;
        _optimizer = optimizer
            ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this)
            ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        InitializeLayers();
    }

    int ITtsModel<T>.SampleRate => _options.SampleRate;
    public int MaxTextLength => _options.MaxTextLength;
    public new int HiddenDim => _options.HiddenDim;
    public int NumFlowSteps => _options.NumFlowSteps;
    public new int MelChannels => _options.MelChannels;
    public new int HopSize => _options.HopSize;
    public int FftSize => _options.FftSize;

    /// <summary>The training stage; <see cref="BeginDistillation"/> moves from the teacher to distillation. Synthesis
    /// uses the teacher's Euler sampler (<see cref="CoMoSpeechOptions.TeacherSamplingSteps"/>) in the teacher phase and
    /// the consistency sampler (<see cref="EndToEndTtsOptions.NumFlowSteps"/>, one by default) once distillation has
    /// begun.</summary>
    public CoMoSpeechTrainingPhase CurrentPhase { get; private set; } = CoMoSpeechTrainingPhase.Teacher;

    /// <summary>Generates a mel spectrogram from text (the paper pairs it with HiFi-GAN).</summary>
    public Tensor<T> TextToMel(string text) => Synthesize(text);

    /// <summary>
    /// Starts consistency distillation (§3.2, §3.4): the student keeps the teacher's weights (θ and θ⁻ are "initialized
    /// weights of CoMoSpeech inherited from the teacher model"), a frozen copy of the denoiser becomes the ODE solver φ,
    /// and from now on only the denoiser trains.
    /// </summary>
    public void BeginDistillation()
    {
        ThrowIfDisposed();
        if (_denoiser is null)
            throw new NotSupportedException("Distillation needs the paper's denoiser; this model was built from caller-supplied layers.");
        _teacher = CreateDenoiser();
        _target = CreateDenoiser();
        CopyParameters(_denoiser, _teacher);
        CopyParameters(_denoiser, _target);
        CurrentPhase = CoMoSpeechTrainingPhase.Distillation;
    }

    private GradTtsScoreEstimator<T> CreateDenoiser()
        => new(Engine, _options.FlowDim, _options.DecoderDimMultipliers, _options.TimePositionScale);

    // Copies a denoiser's parameters layer by layer; a dummy evaluation first sizes any lazily built layer of either.
    private void CopyParameters(GradTtsScoreEstimator<T> from, GradTtsScoreEstimator<T> to)
    {
        using (new NoGradScope<T>())
        {
            int frames = CompatibleLength(1);
            var x = new Tensor<T>(new[] { frames, _options.MelChannels });
            var mask = FrameMask(frames, frames);
            from.Estimate(x, x, 0.0, mask);
            to.Estimate(x, x, 0.0, mask);
        }
        for (int i = 0; i < from.Layers.Count; i++)
            if (from.Layers[i].ParameterCount > 0)
                to.Layers[i].SetParameters(from.Layers[i].GetParameters());
    }

    // θ⁻ ← μ θ⁻ + (1 − μ) θ (Eq. 12), before each distillation step as the reference does.
    private void UpdateTarget()
    {
        double mu = _options.EmaDecay;
        for (int i = 0; i < _denoiser!.Layers.Count; i++)
        {
            var layer = _denoiser.Layers[i];
            if (layer.ParameterCount == 0) continue;
            var online = layer.GetParameters();
            var target = _target!.Layers[i].GetParameters();
            var blended = new Vector<T>(online.Length);
            for (int k = 0; k < online.Length; k++)
                blended[k] = NumOps.FromDouble(mu * NumOps.ToDouble(target[k]) + (1 - mu) * NumOps.ToDouble(online[k]));
            _target.Layers[i].SetParameters(blended);
        }
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
        int h = o.HiddenDim;
        _embedding = new EmbeddingLayer<T>(o.VocabSize, h);
        _prenet = new ResidualConvReluNormLayer<T>(h, o.PrenetKernelSize, o.PrenetLayers, o.PrenetDropout);
        for (int i = 0; i < o.NumEncoderLayers; i++)
            _encoderBlocks.Add(new RelativePositionTransformerBlock<T>(h, o.NumHeads, o.FilterChannels, o.EncoderKernelSize,
                o.DropoutRate, o.RelativeWindow));
        _meanProjection = new DenseLayer<T>(o.MelChannels, new IdentityActivation<T>() as IActivationFunction<T>);
        _durationPredictor = new VariancePredictorLayer<T>(h, o.DurationPredictorFilterChannels, 1, o.EncoderKernelSize, o.DropoutRate);
        _denoiser = CreateDenoiser();

        var encoder = new List<ILayer<T>> { _embedding, _prenet };
        encoder.AddRange(_encoderBlocks);
        encoder.Add(_meanProjection);
        AddEncoderDecoderLayers(encoder, Array.Empty<ILayer<T>>());
        ComponentLayers.Add(_durationPredictor);
        ComponentLayers.AddRange(_denoiser.Layers);
    }

    private bool HasPaperLayers => _embedding is not null;

    private (Tensor<T> Hidden, Tensor<T> Means) Encode(Tensor<T> tokens)
    {
        var x = Engine.TensorMultiplyScalar(_embedding!.Forward(tokens), NumOps.FromDouble(Math.Sqrt(_options.HiddenDim)));
        x = _prenet!.Forward(x);
        foreach (var block in _encoderBlocks) x = block.Forward(x);
        return (x, _meanProjection!.Forward(x));
    }

    private Tensor<T> LogDurations(Tensor<T> hidden)
    {
        var detached = new Tensor<T>(hidden._shape, hidden.ToVector());
        return Engine.Reshape(_durationPredictor!.Forward(detached), new[] { hidden.Shape[0] });
    }

    /// <inheritdoc />
    /// <remarks>Each of the encoder and the decoder is clipped to norm 1 on its own, as Grad-TTS's training loop (which the
    /// reference inherits) does.</remarks>
    protected override IReadOnlyList<IReadOnlyList<Tensor<T>>>? GradientClippingGroups(IReadOnlyList<Tensor<T>> trainableParameters)
    {
        if (!HasPaperLayers) return null;
        var encoderLayers = Layers.Take(EncoderLayerCount).Append(_durationPredictor!).Cast<ILayer<T>>().ToList();
        var encoder = Training.TapeTrainingStep<T>.CollectParameters(encoderLayers, -1);
        var decoder = Training.TapeTrainingStep<T>.CollectParameters(_denoiser!.Layers.Cast<ILayer<T>>().ToList(), -1);
        return new IReadOnlyList<Tensor<T>>[] { encoder, decoder };
    }

    /// <inheritdoc />
    /// <remarks>Distillation updates only the denoiser: "the parameters of the encoder are fixed" (§3.4).</remarks>
    protected override IReadOnlyList<Tensor<T>> SelectTrainableParametersForTraining(IReadOnlyList<Tensor<T>> parameters)
    {
        if (!HasPaperLayers || CurrentPhase == CoMoSpeechTrainingPhase.Teacher) return parameters;
        var denoiser = new HashSet<Tensor<T>>(Training.TapeTrainingStep<T>.CollectParameters(_denoiser!.Layers.Cast<ILayer<T>>().ToList(), -1),
            Helpers.TensorReferenceComparer<Tensor<T>>.Instance);
        return parameters.Where(denoiser.Contains).ToList();
    }

    // ---------------------------------------------------------------- EDM

    /// <summary>D(x, t) = c_skip(t) x + c_out(t) F(c_in(t) x, μ, ln t / 4) (Eq. 8–9).</summary>
    private Tensor<T> Denoise(GradTtsScoreEstimator<T> network, Tensor<T> x, double t, Tensor<T> mu, Tensor<T> mask)
    {
        double sd = _options.SigmaData, eps = _options.SigmaMin;
        double cSkip = sd * sd / ((t - eps) * (t - eps) + sd * sd);
        double cOut = (t - eps) * sd / Math.Sqrt(sd * sd + t * t);
        double cIn = 1.0 / Math.Sqrt(sd * sd + t * t);
        var f = network.Estimate(Engine.TensorMultiplyScalar(x, NumOps.FromDouble(cIn)), mu, Math.Log(t) / 4.0, mask);
        return Engine.TensorAdd(Engine.TensorMultiplyScalar(x, NumOps.FromDouble(cSkip)), Engine.TensorMultiplyScalar(f, NumOps.FromDouble(cOut)));
    }

    /// <summary>The Karras discretization t_i = (σ_min^{1/ρ} + i/(n−1)(σ_max^{1/ρ} − σ_min^{1/ρ}))^ρ, i = 0..n−1, ascending.</summary>
    private double[] NoiseLevels(int n)
    {
        double lo = Math.Pow(_options.SigmaMin, 1 / _options.Rho), hi = Math.Pow(_options.SigmaMax, 1 / _options.Rho);
        var levels = new double[n];
        for (int i = 0; i < n; i++) levels[i] = Math.Pow(lo + (n == 1 ? 1.0 : (double)i / (n - 1)) * (hi - lo), _options.Rho);
        return levels;
    }

    private static int CompatibleLength(int length) => (length + 3) / 4 * 4;

    private Tensor<T> FrameMask(int frames, int padded)
    {
        var mask = new Tensor<T>(new[] { 1, 1, 1, padded });
        for (int i = 0; i < frames; i++) mask[0, 0, 0, i] = NumOps.One;
        return mask;
    }

    private Tensor<T> MaskRows(Tensor<T> mask, int padded)
        => Engine.TensorTranspose(Engine.TensorTile(Engine.Reshape(mask, new[] { 1, padded }), new[] { _options.MelChannels, 1 }));

    private Tensor<T> PadFrames(Tensor<T> x, int frames)
    {
        if (x.Shape[0] == frames) return x;
        return Engine.TensorConcatenate(new[] { x, new Tensor<T>(new[] { frames - x.Shape[0], x.Shape[1] }) }, 0);
    }

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

    // ---------------------------------------------------------------- inference

    /// <inheritdoc />
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(input);
        SetTrainingMode(false);
        if (!HasPaperLayers)
        {
            var x = input;
            foreach (var layer in Layers) x = layer.Forward(x);
            return x;
        }
        if (input.Rank == 2 && input.Shape[0] == 1)
            input = Engine.Reshape(input, new[] { input.Shape[1] });
        if (input.Rank != 1)
            throw new ArgumentException($"Expected tokens [tokens], got [{string.Join(", ", input.Shape)}].", nameof(input));
        return SynthesizeOne(input);
    }

    private Tensor<T> SynthesizeOne(Tensor<T> tokens)
    {
        using var _ = new NoGradScope<T>();
        var (hidden, means) = Encode(tokens);
        var logDurations = LogDurations(hidden);
        var durations = new int[logDurations.Length];
        for (int i = 0; i < durations.Length; i++)
            durations[i] = (int)(Math.Ceiling(Math.Exp(NumOps.ToDouble(logDurations[i]))) * _options.LengthScale);
        if (durations.Sum() == 0) durations[durations.Length - 1] = 1;
        int frames = durations.Sum();
        int padded = CompatibleLength(frames);
        var mu = PadFrames(LengthRegulator.Expand(means, durations), padded);
        var mask = FrameMask(frames, padded);
        var maskRows = MaskRows(mask, padded);
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed);
        Tensor<T> NoiseAroundMu() => Engine.TensorAdd(Gaussian(mu._shape, random), mu);

        Tensor<T> x;
        if (CurrentPhase == CoMoSpeechTrainingPhase.Teacher)
        {
            // Algorithm 1: x_N ~ N(μ, I) scaled by t_N, then N Euler steps of dx = (x − D(x, t)) / t dt to t_0 = 0.
            var levels = NoiseLevels(_options.TeacherSamplingSteps);
            x = Engine.TensorMultiplyScalar(NoiseAroundMu(), NumOps.FromDouble(levels[^1]));
            for (int i = levels.Length - 1; i >= 0; i--)
            {
                double t = levels[i], next = i > 0 ? levels[i - 1] : 0.0;
                var d = Engine.TensorMultiplyScalar(Engine.TensorSubtract(x, Denoise(_denoiser!, x, t, mu, mask)), NumOps.FromDouble(1.0 / t));
                x = Engine.TensorMultiply(Engine.TensorAdd(x, Engine.TensorMultiplyScalar(d, NumOps.FromDouble(next - t))), maskRows);
            }
        }
        else
        {
            // Eq. 13 / Algorithm 2: x = D(t_N x_N, t_N); for more steps re-noise to each t_i with √(t_i² − ε²) z,
            // z ~ N(μ, I), and denoise again.
            int steps = Math.Max(1, _options.NumFlowSteps);
            // Enumerable.Reverse explicitly: on net471 an array's .Reverse() binds to the in-place MemoryExtensions.Reverse.
            var levels = Enumerable.Reverse(NoiseLevels(steps + 1)).ToArray();   // σ_max ... σ_min
            x = Denoise(_denoiser!, Engine.TensorMultiplyScalar(NoiseAroundMu(), NumOps.FromDouble(_options.SigmaMax)), _options.SigmaMax, mu, mask);
            for (int i = 1; i < steps; i++)
            {
                double t = levels[i];
                var noisy = Engine.TensorAdd(x, Engine.TensorMultiplyScalar(NoiseAroundMu(),
                    NumOps.FromDouble(Math.Sqrt(t * t - _options.SigmaMin * _options.SigmaMin))));
                x = Denoise(_denoiser!, noisy, t, mu, mask);
            }
            x = Engine.TensorMultiply(x, maskRows);
        }
        return Engine.TensorSlice(x, new[] { 0, 0 }, new[] { frames, _options.MelChannels });
    }

    // ---------------------------------------------------------------- training

    /// <inheritdoc />
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        ThrowIfDisposed();
        if (IsOnnxMode)
            throw new NotSupportedException("Training is not supported in ONNX mode.");
        if (!HasPaperLayers)
        {
            TrainWithTape(input, expectedOutput, _optimizer);
            return;
        }
        var (tokens, mel) = Prepare(input, expectedOutput);
        var draw = DrawTrainingRandomness(mel.Shape[0], _trainingRandom);
        if (CurrentPhase == CoMoSpeechTrainingPhase.Distillation)
            UpdateTarget();
        TrainWithCustomObjective(tokens, mel, (x, y) => Objective(x, y, draw), _optimizer);
    }

    /// <inheritdoc />
    /// <remarks>The model aligns itself with monotonic alignment search, so a token/mel pair is its whole supervision.</remarks>
    protected override T TrainOnSample(TtsTrainingSample<T> sample)
    {
        Train(sample.Tokens, DeriveAcousticTargets(sample).Mel);
        return LastLoss ?? NumOps.Zero;
    }

    /// <inheritdoc />
    public override T EvaluateTrainingObjective(TtsTrainingSample<T> sample)
        => ((ITrainingObjectiveProvider<T>)this).EvaluateTrainingObjective(sample.Tokens, DeriveAcousticTargets(sample).Mel);

    TrainingObjectiveKind ITrainingObjectiveProvider<T>.TrainingObjectiveKind => TrainingObjectiveKind.Supervised;

    Tensor<T> ITrainingObjectiveProvider<T>.ResolveTrainingTarget(Tensor<T> input, Tensor<T> proposedTarget) => proposedTarget;

    /// <remarks>The noise level, the noise and the segment are fixed by the sampling seed so the same parameters always
    /// score the same.</remarks>
    T ITrainingObjectiveProvider<T>.EvaluateTrainingObjective(Tensor<T> input, Tensor<T> target)
    {
        ThrowIfDisposed();
        var (tokens, mel) = Prepare(input, target);
        var draw = DrawTrainingRandomness(mel.Shape[0], AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed));
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            return Objective(tokens, mel, draw)[0];
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    private (Tensor<T> Tokens, Tensor<T> Mel) Prepare(Tensor<T> input, Tensor<T> target)
    {
        var tokens = input.Rank == 2 && input.Shape[0] == 1 ? Engine.Reshape(input, new[] { input.Shape[1] }) : input;
        var mel = target.Rank == 3 ? Engine.Reshape(target, new[] { target.Shape[1], target.Shape[2] }) : target;
        if (tokens.Rank != 1 || mel.Rank != 2 || mel.Shape[1] != _options.MelChannels)
            throw new ArgumentException(
                $"Expected tokens [tokens] and a mel spectrogram [frames, {_options.MelChannels}], got [{string.Join(", ", input.Shape)}] and [{string.Join(", ", target.Shape)}].");
        if (mel.Shape[0] < tokens.Shape[0])
            throw new ArgumentException(
                $"Monotonic alignment gives every token at least one frame: {tokens.Shape[0]} tokens need at least that many frames, got {mel.Shape[0]}.");
        return (tokens, mel);
    }

    private sealed record TrainingDraw(double LogSigma, int Index, int SegmentOffset, Tensor<T> Noise);

    // ln t ~ N(P_mean, P_std²) for the teacher, a discretization index n ∈ {1..N−1} for distillation, a segment offset,
    // and z ~ N(0, I) over the segment.
    private TrainingDraw DrawTrainingRandomness(int frames, Random random)
    {
        double u1 = 1.0 - random.NextDouble(), u2 = random.NextDouble();
        double normal = Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2);
        int index = 1 + random.Next(Math.Max(1, _options.DistillationSteps - 1));
        int segment = Math.Min(frames, _options.SegmentFrames);
        int offset = frames > _options.SegmentFrames ? random.Next(0, frames - _options.SegmentFrames) : 0;
        var noise = Gaussian(new[] { CompatibleLength(segment), _options.MelChannels }, random);
        return new TrainingDraw(_options.LogSigmaMean + _options.LogSigmaStd * normal, index, offset, noise);
    }

    /// <summary>
    /// The objectives (reference <c>compute_loss</c>, <c>EDMLoss</c>, <c>CTLoss_D</c>): the teacher adds the duration
    /// loss against MAS durations, the prior loss <c>Σ ½((y − μ)² + log 2π) / (frames · mel)</c> and the EDM loss
    /// <c>mean λ(t)(D(y + t n, t) − y)²</c> with n = z + μ, λ = (t² + σ_d²)/(t σ_d)²; distillation is
    /// <c>mean (D_θ(y + t_{n+1} n, t_{n+1}) − D_θ⁻(x̂_n, t_n))²</c> with x̂_n one Euler step of the frozen teacher.
    /// </summary>
    private Tensor<T> Objective(Tensor<T> tokens, Tensor<T> mel, TrainingDraw draw)
    {
        int frames = mel.Shape[0], channels = mel.Shape[1];
        bool distilling = CurrentPhase == CoMoSpeechTrainingPhase.Distillation;
        Tensor<T> hidden, means;
        if (distilling)
        {
            using (new NoGradScope<T>()) (hidden, means) = Encode(tokens);
        }
        else (hidden, means) = Encode(tokens);

        int tokenCount = means.Shape[0];
        var scores = new double[tokenCount, frames];
        for (int i = 0; i < tokenCount; i++)
            for (int j = 0; j < frames; j++)
            {
                double sum = 0;
                for (int c = 0; c < channels; c++)
                {
                    double d = NumOps.ToDouble(mel[j, c]) - NumOps.ToDouble(means[i, c]);
                    sum += d * d;
                }
                scores[i, j] = -0.5 * sum;
            }
        int[] durations = MonotonicAlignment.MaximumPath(scores);

        int length = Math.Min(frames, _options.SegmentFrames);
        int padded = draw.Noise.Shape[0];
        var y = PadFrames(Engine.TensorSlice(mel, new[] { draw.SegmentOffset, 0 }, new[] { length, channels }), padded);
        var muFull = LengthRegulator.Expand(means, durations);
        var mu = PadFrames(Engine.TensorSlice(muFull, new[] { draw.SegmentOffset, 0 }, new[] { length, channels }), padded);
        if (distilling) mu = new Tensor<T>(mu._shape, mu.ToVector());
        var mask = FrameMask(length, padded);
        var maskRows = MaskRows(mask, padded);
        double elements = padded * (double)channels;
        var n = Engine.TensorMultiply(Engine.TensorAdd(draw.Noise, mu), maskRows);              // z + μ

        if (distilling)
        {
            var levels = new[] { 0.0 }.Concat(NoiseLevels(_options.DistillationSteps)).ToArray();
            double tNext = levels[draw.Index + 1], t = levels[draw.Index];
            var xNext = Engine.TensorAdd(y, Engine.TensorMultiplyScalar(n, NumOps.FromDouble(tNext)));
            var online = Denoise(_denoiser!, xNext, tNext, mu, mask);
            Tensor<T> emaTarget;
            using (new NoGradScope<T>())
            {
                var d = Engine.TensorMultiplyScalar(Engine.TensorSubtract(xNext, Denoise(_teacher!, xNext, tNext, mu, mask)), NumOps.FromDouble(1.0 / tNext));
                var xHat = Engine.TensorAdd(xNext, Engine.TensorMultiplyScalar(d, NumOps.FromDouble(t - tNext)));
                var ema = Denoise(_target!, xHat, t, mu, mask);
                emaTarget = new Tensor<T>(ema._shape, ema.ToVector());
            }
            var gap = Engine.TensorMultiply(Engine.TensorSubtract(online, emaTarget), maskRows);
            return Engine.TensorMultiplyScalar(Engine.ReduceSum(Engine.TensorMultiply(gap, gap), new[] { 0, 1 }, keepDims: false),
                NumOps.FromDouble(1.0 / elements));
        }

        var durationTarget = new Tensor<T>(new[] { tokenCount });
        for (int i = 0; i < tokenCount; i++) durationTarget[i] = NumOps.FromDouble(Math.Log(1e-8 + durations[i]));
        var durationError = Engine.TensorSubtract(LogDurations(hidden), durationTarget);
        var durationLoss = Engine.TensorMultiplyScalar(
            Engine.ReduceSum(Engine.TensorMultiply(durationError, durationError), new[] { 0 }, keepDims: false),
            NumOps.FromDouble(1.0 / tokenCount));

        double count = length * (double)channels;
        var priorDiff = Engine.TensorSubtract(y, mu);
        var priorSum = Engine.ReduceSum(Engine.TensorMultiply(Engine.TensorMultiply(priorDiff, priorDiff), maskRows), new[] { 0, 1 }, keepDims: false);
        var priorLoss = Engine.TensorAddScalar(Engine.TensorMultiplyScalar(priorSum, NumOps.FromDouble(0.5 / count)),
            NumOps.FromDouble(0.5 * Math.Log(2 * Math.PI)));

        double sigma = Math.Exp(draw.LogSigma), sd = _options.SigmaData;
        double weight = (sigma * sigma + sd * sd) / (sigma * sd * sigma * sd);
        var denoised = Denoise(_denoiser!, Engine.TensorAdd(y, Engine.TensorMultiplyScalar(n, NumOps.FromDouble(sigma))), sigma, mu, mask);
        var error = Engine.TensorMultiply(Engine.TensorSubtract(denoised, y), maskRows);
        var denoisingLoss = Engine.TensorMultiplyScalar(Engine.ReduceSum(Engine.TensorMultiply(error, error), new[] { 0, 1 }, keepDims: false),
            NumOps.FromDouble(weight / elements));

        return Engine.TensorAdd(Engine.TensorAdd(durationLoss, priorLoss), denoisingLoss);
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
            Name = _useNativeMode ? "CoMoSpeech-Native" : "CoMoSpeech-ONNX",
            Description = "CoMoSpeech: One-Step Speech Synthesis via Consistency Model (Ye et al., 2023)",
            FeatureCount = _options.HiddenDim,
            Complexity = _options.NumEncoderLayers + _options.NumFlowSteps,
        };
        m.AdditionalInfo["Architecture"] = "CoMoSpeech";
        m.AdditionalInfo["Phase"] = CurrentPhase.ToString();
        return m;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(CoMoSpeech<T>));
    }

    protected override void Dispose(bool disposing)
    {
        if (_disposed)
            return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
