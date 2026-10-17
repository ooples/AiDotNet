using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.Interfaces;

namespace AiDotNet.TextToSpeech.Vocoders;

/// <summary>
/// The training loop GAN vocoders share: a generator that turns a mel spectrogram into a waveform, discriminators that
/// tell real from generated audio, and alternating discriminator and generator steps on random segments.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Each step crops a segment of <see cref="SegmentSize"/> samples (with its frames when the mel is supplied), updates the
/// discriminators on a detached generation, and updates the generator — in the order the model's reference uses
/// (<see cref="DiscriminatorStepFirst"/>). A model can train its generator alone for its first steps
/// (<see cref="DiscriminatorStartStep"/>, as Parallel WaveGAN and Multi-band MelGAN do). Each group has its own optimizer
/// from <see cref="CreatePaperOptimizer"/> unless one was supplied at construction.
/// </para>
/// <para>A model supplies its generator and discriminators, its input features (<see cref="ComputeInputMel"/>), its
/// losses and its optimizer; the base owns segmenting, the step order, the parameter groups and the training objective
/// the test harness and callers read.</para>
/// <para><b>For Beginners:</b> A generator turns a spectrogram into audio while one or more "critics" learn to tell its
/// audio from real recordings; competing makes the audio realistic.</para>
/// </remarks>
public abstract partial class GanVocoderBase<T> : VocoderBase<T>, ITrainingObjectiveProvider<T>
{
    private readonly VocoderOptions _vocoder;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _suppliedOptimizer;
    private readonly Dictionary<string, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>> _optimizers = new();
    private readonly bool _useNativeMode;
    private bool _disposed;
    private Random _trainingRandom;
    private IReadOnlyList<ILayer<T>>? _trainingLayers;
    private long _trainingSteps;
    private readonly List<LayerBase<T>> _generatorLayers = new();
    private readonly List<LayerBase<T>> _discriminatorLayers = new();
    private Tensor<T>? _rememberedGeneration;
    private string? _trainingGroup;

    /// <summary>Creates a native (trainable) vocoder.</summary>
    protected GanVocoderBase(NeuralNetworkArchitecture<T> architecture, VocoderOptions options,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer, int samplingSeed)
        : base(architecture)
    {
        _vocoder = options ?? throw new ArgumentNullException(nameof(options));
        // The clone plan replays the constructor from members named after its parameters: the options live in the base's
        // Options, as NeuralNetworkBase asks of every derived model.
        Options = _vocoder;
        _useNativeMode = true;
        _suppliedOptimizer = optimizer;
        _trainingRandom = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(samplingSeed);
        ApplyAudioSettings();
        InitializeLayers();
    }

    /// <summary>Creates a vocoder that runs an exported ONNX graph.</summary>
    protected GanVocoderBase(NeuralNetworkArchitecture<T> architecture, string modelPath, VocoderOptions options, int samplingSeed)
        : base(architecture)
    {
        _vocoder = options ?? throw new ArgumentNullException(nameof(options));
        // The clone plan replays the constructor from members named after its parameters: the options live in the base's
        // Options, as NeuralNetworkBase asks of every derived model.
        Options = _vocoder;
        _useNativeMode = false;
        _trainingRandom = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(samplingSeed);
        ApplyAudioSettings();
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path required.", nameof(modelPath));
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _vocoder.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _vocoder.OnnxOptions);
        InitializeLayers();
    }

    private void ApplyAudioSettings()
    {
        base.SampleRate = _vocoder.SampleRate;
        base.MelChannels = _vocoder.MelChannels;
        base.HopSize = _vocoder.HopSize;
    }

    /// <inheritdoc />
    public override ModelOptions GetOptions() => _vocoder;

    /// <summary>The options, available to the hooks the base constructor calls before a derived constructor runs.</summary>
    protected VocoderOptions VocoderSettings => _vocoder;

    /// <summary>Gets the number of training steps taken so far.</summary>
    protected long TrainingSteps => _trainingSteps;

    /// <summary>Gets whether the paper's layers were built (false when the architecture supplied its own layers).</summary>
    protected bool HasPaperLayers => _generatorLayers.Count > 0;

    /// <summary>The generator's layers, in build order.</summary>
    internal IReadOnlyList<LayerBase<T>> GeneratorLayers => _generatorLayers;

    /// <summary>The discriminators' layers, in build order.</summary>
    internal IReadOnlyList<LayerBase<T>> DiscriminatorLayers => _discriminatorLayers;

    /// <inheritdoc />
    public override IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank) => WaveformUpsampleContract(inputRank);

    // ---------------------------------------------------------------- hooks

    /// <summary>Builds the generator, returning its layers.</summary>
    protected abstract IReadOnlyList<LayerBase<T>> CreateGenerator();

    /// <summary>Builds the discriminators, returning their layers.</summary>
    protected abstract IReadOnlyList<LayerBase<T>> CreateDiscriminators();

    /// <summary>The generator: a mel spectrogram <c>[1, mel, frames]</c> to a waveform <c>[1, 1, frames · hop]</c>.</summary>
    protected abstract Tensor<T> Generate(Tensor<T> mel);

    /// <summary>The generator's input features of a waveform <c>[samples]</c>, <c>[1, mel, frames]</c>, as the model's
    /// reference computes them; no gradient.</summary>
    protected abstract Tensor<T> ComputeInputMel(Tensor<T> audio);

    /// <summary>The discriminators' loss on a real segment and a detached generated one (both <c>[samples]</c>).</summary>
    protected abstract Tensor<T> DiscriminatorLoss(Tensor<T> real, Tensor<T> generated);

    /// <summary>The generator's loss for <paramref name="mel"/> against the real segment <paramref name="real"/>
    /// <c>[samples]</c>; <paramref name="adversarial"/> is false while the generator trains alone.</summary>
    protected abstract Tensor<T> GeneratorLoss(Tensor<T> mel, Tensor<T> real, bool adversarial);

    /// <summary>The non-adversarial measure <see cref="ITrainingObjectiveProvider{T}"/> reports for the generation from
    /// <paramref name="mel"/> against the real waveform <paramref name="real"/> <c>[samples]</c> (the adversarial terms are
    /// measured against discriminators trained at the same time, so they do not fall monotonically even when the
    /// generator improves).</summary>
    protected abstract Tensor<T> ReconstructionObjective(Tensor<T> mel, Tensor<T> real);

    /// <summary>The generation for <paramref name="mel"/> and the real waveform, both cut to the shorter length.</summary>
    protected (Tensor<T> Generated, Tensor<T> Real) GeneratedAndReal(Tensor<T> mel, Tensor<T> real)
    {
        var generated = Flat(Generate(mel));
        var target = Flat(real);
        int n = Math.Min(generated.Length, target.Length);
        return (Engine.TensorSlice(generated, new[] { 0 }, new[] { n }), Engine.TensorSlice(target, new[] { 0 }, new[] { n }));
    }

    /// <summary>The optimizer of a training group ("generator" or "discriminator") as the paper configures it.</summary>
    protected abstract IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> CreatePaperOptimizer(string group);

    /// <summary>The training segment in samples.</summary>
    protected abstract int SegmentSize { get; }

    /// <summary>The step from which the discriminators train and the generator's loss is adversarial (0: from the
    /// start).</summary>
    protected virtual long DiscriminatorStartStep => 0;

    /// <summary>The global gradient-norm clip of a training group ("generator" or "discriminator"), 0 for none.</summary>
    protected virtual double GradientClipNorm(string group) => 0.0;

    /// <summary>Sub-networks of a training group whose gradients are clipped separately (BigVGAN clips its period and
    /// resolution discriminators each to its own norm), or null to clip the group as one.</summary>
    protected virtual IReadOnlyList<IReadOnlyList<LayerBase<T>>>? ClippingLayerGroups(string group) => null;

    /// <inheritdoc />
    protected override IReadOnlyList<IReadOnlyList<Tensor<T>>>? GradientClippingGroups(IReadOnlyList<Tensor<T>> trainableParameters)
    {
        if (_trainingGroup is null) return null;
        var groups = ClippingLayerGroups(_trainingGroup);
        if (groups is null) return null;
        var trainable = new HashSet<Tensor<T>>(trainableParameters, Helpers.TensorReferenceComparer<Tensor<T>>.Instance);
        return groups.Select(g => (IReadOnlyList<Tensor<T>>)Training.TapeTrainingStep<T>.CollectParameters(g.Cast<ILayer<T>>().ToList(), -1)
            .Where(trainable.Contains).ToList()).ToList();
    }

    /// <summary>Whether the generator turns F centred frames into (F − 1) · hop samples (an inverse-STFT head with
    /// <c>center=True</c>, Vocos), so a segment's input keeps the extra centred frame.</summary>
    protected virtual bool KeepsCentredFrame => false;

    /// <summary>Whether the discriminator step comes before the generator step (HiFi-GAN, MelGAN) or after it
    /// (Parallel WaveGAN).</summary>
    protected virtual bool DiscriminatorStepFirst => true;

    // ---------------------------------------------------------------- layers

    /// <inheritdoc />
    protected override void InitializeLayers()
    {
        if (!_useNativeMode)
            return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            return;
        }
        _generatorLayers.AddRange(CreateGenerator());
        _discriminatorLayers.AddRange(CreateDiscriminators());
        AddEncoderDecoderLayers(_generatorLayers.Cast<ILayer<T>>().ToList(), Array.Empty<ILayer<T>>());
        ComponentLayers.AddRange(_discriminatorLayers);
    }

    // ---------------------------------------------------------------- synthesis

    /// <inheritdoc />
    public override Tensor<T> MelToWaveform(Tensor<T> melSpectrogram)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(melSpectrogram);
        return Predict(melSpectrogram);
    }

    /// <summary>The generator's input features for <paramref name="audio"/> <c>[samples]</c>, <c>[1, mel, frames]</c>.</summary>
    public Tensor<T> ComputeMel(Tensor<T> audio)
    {
        using var _ = new NoGradScope<T>();
        return ComputeInputMel(Flat(audio));
    }

    /// <inheritdoc />
    protected override Tensor<T> PreprocessText(string text)
        => throw new NotSupportedException($"{GetType().Name} is a vocoder; it converts mel spectrograms (MelToWaveform), not text.");

    /// <inheritdoc />
    protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;

    /// <inheritdoc />
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(input);
        SetTrainingMode(false);
        if (!HasPaperLayers)
        {
            var c = input;
            foreach (var l in Layers) c = l.Forward(c);
            return c;
        }
        using var _ = new NoGradScope<T>();
        return GenerateForInference(MelInput(input));
    }

    /// <summary>The generation at synthesis; by default <see cref="Generate"/>.</summary>
    protected virtual Tensor<T> GenerateForInference(Tensor<T> mel) => Generate(mel);

    // ---------------------------------------------------------------- training

    /// <inheritdoc />
    /// <remarks>A mel spectrogram <paramref name="input"/> <c>[1, mel, frames]</c> aligned with the audio
    /// <paramref name="expectedOutput"/> (frames × hop samples) trains on a frame-aligned segment of both;
    /// <see cref="TtsModelBase{T}.Train(TtsTrainingSample{T})"/> with audio only computes the input features of each
    /// audio segment.</remarks>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        ThrowIfDisposed();
        if (IsOnnxMode)
            throw new NotSupportedException("Training is not supported in ONNX mode.");
        if (!HasPaperLayers)
        {
            TrainWithTape(input, expectedOutput, _suppliedOptimizer);
            return;
        }
        var (mel, audio) = AlignedSegment(MelInput(input), expectedOutput, _trainingRandom);
        TrainStep(mel, audio);
    }

    /// <inheritdoc />
    protected override T TrainOnSample(TtsTrainingSample<T> sample)
    {
        ThrowIfDisposed();
        if (IsOnnxMode)
            throw new NotSupportedException("Training is not supported in ONNX mode.");
        var audio = sample.Audio ?? throw new ArgumentException($"{GetType().Name} trains on recordings; set Audio.", nameof(sample));
        var segment = Segment(audio, _trainingRandom);
        TrainStep(FramesOf(segment), segment);
        return LastLoss ?? NumOps.Zero;
    }

    // The input features of a segment, one frame per hop: a centred STFT's extra last frame is dropped, so frame f
    // describes samples [f·hop, (f + 1)·hop).
    private Tensor<T> FramesOf(Tensor<T> segment)
    {
        var mel = ComputeMel(segment);
        if (KeepsCentredFrame) return mel;
        int frames = Math.Max(1, segment.Length / UpsampleFactor);
        return mel.Shape[2] > frames ? Engine.TensorSlice(mel, new[] { 0, 0, 0 }, new[] { 1, mel.Shape[1], frames }) : mel;
    }

    /// <summary>The input as <c>[1, mel, frames]</c>.</summary>
    protected Tensor<T> MelInput(Tensor<T> input)
    {
        var mel = input.Rank == 2 ? Engine.Reshape(input, new[] { 1, input.Shape[0], input.Shape[1] }) : input;
        if (mel.Rank != 3 || mel.Shape[1] != _vocoder.MelChannels)
            throw new ArgumentException($"Expected a mel spectrogram [1, {_vocoder.MelChannels}, frames], got [{string.Join(", ", input.Shape)}].", nameof(input));
        return mel;
    }

    // A segment of SegmentSize samples (rounded down to whole frames) and its frames, from a random frame on.
    private (Tensor<T> Mel, Tensor<T> Audio) AlignedSegment(Tensor<T> mel, Tensor<T> audio, Random random)
    {
        int hop = UpsampleFactor, frames = mel.Shape[2];
        // A centred head needs two frames for one hop of audio.
        int segmentFrames = Math.Max(KeepsCentredFrame ? Math.Min(2, frames) : 1,
            Math.Min(frames, SegmentSize / hop + (KeepsCentredFrame ? 1 : 0)));
        int start = frames > segmentFrames ? random.Next(0, frames - segmentFrames + 1) : 0;
        var melSegment = Engine.TensorSlice(mel, new[] { 0, 0, start }, new[] { 1, _vocoder.MelChannels, segmentFrames });
        var flat = Flat(audio);
        // F centred frames describe (F − 1) · hop samples (an inverse-STFT head); otherwise frame f covers [f·hop, (f + 1)·hop).
        int samples = (KeepsCentredFrame ? Math.Max(1, segmentFrames - 1) : segmentFrames) * hop, from = start * hop;
        var audioSegment = from + samples <= audio.Length
            ? Engine.TensorSlice(flat, new[] { from }, new[] { samples })
            : Engine.TensorConcatenate(new[] { Engine.TensorSlice(flat, new[] { Math.Min(from, audio.Length) }, new[] { Math.Max(0, audio.Length - from) }),
                new Tensor<T>(new[] { samples - Math.Max(0, audio.Length - from) }) }, 0);
        return (melSegment, audioSegment);
    }

    // SegmentSize samples from a random start, zero-padded when the recording is shorter.
    private Tensor<T> Segment(Tensor<T> audio, Random random)
    {
        int length = audio.Length, size = SegmentSize;
        var flat = Flat(audio);
        if (length >= size)
            return Engine.TensorSlice(flat, new[] { random.Next(0, length - size + 1) }, new[] { size });
        return Engine.TensorConcatenate(new[] { flat, new Tensor<T>(new[] { size - length }) }, 0);
    }

    private void TrainStep(Tensor<T> mel, Tensor<T> segment)
    {
        SetTrainingMode(true);
        _rememberedGeneration = null;
        bool adversarial = _trainingSteps >= DiscriminatorStartStep;
        if (adversarial && DiscriminatorStepFirst) DiscriminatorStep(mel, segment);
        TrainLayers(_generatorLayers, mel, segment, () => GeneratorLoss(mel, segment, adversarial), "generator");
        if (adversarial && !DiscriminatorStepFirst) DiscriminatorStep(mel, segment);
        _trainingSteps++;
    }

    private void DiscriminatorStep(Tensor<T> mel, Tensor<T> segment)
    {
        Tensor<T> generated;
        if (_rememberedGeneration is not null)
        {
            generated = _rememberedGeneration;
            _rememberedGeneration = null;
        }
        else
        {
            using (new NoGradScope<T>())
            {
                var g = Flat(Generate(mel));
                generated = new Tensor<T>(g._shape, g.ToVector());
            }
        }
        TrainLayers(_discriminatorLayers, mel, segment, () => DiscriminatorLoss(segment, generated), "discriminator");
    }

    private void TrainLayers(IReadOnlyList<LayerBase<T>> layers, Tensor<T> mel, Tensor<T> segment, Func<Tensor<T>> loss, string group)
    {
        MaxGradNorm = NumOps.FromDouble(GradientClipNorm(group));
        _trainingLayers = layers.Cast<ILayer<T>>().ToList();
        _trainingGroup = group;
        try
        {
            TrainWithCustomObjective(mel, segment, (_, _) => loss(), OptimizerFor(group));
        }
        finally
        {
            _trainingLayers = null;
            _trainingGroup = null;
        }
    }

    TrainingObjectiveKind ITrainingObjectiveProvider<T>.TrainingObjectiveKind => TrainingObjectiveKind.Supervised;

    Tensor<T> ITrainingObjectiveProvider<T>.ResolveTrainingTarget(Tensor<T> input, Tensor<T> proposedTarget) => proposedTarget;

    T ITrainingObjectiveProvider<T>.EvaluateTrainingObjective(Tensor<T> input, Tensor<T> target)
    {
        ThrowIfDisposed();
        var mel = MelInput(input);
        var audio = Flat(target);
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            return ReconstructionObjective(mel, audio)[0];
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    // ---------------------------------------------------------------- loss helpers

    /// <summary>Keeps the generator step's output (detached) for the discriminator step that follows it, for references
    /// that train the discriminator on the very generation the generator was just updated with (UnivNet) rather than
    /// a fresh one.</summary>
    protected void RememberGeneration(Tensor<T> generated) => _rememberedGeneration = Detached(Flat(generated));

    /// <summary>A waveform as <c>[samples]</c>.</summary>
    protected Tensor<T> Flat(Tensor<T> audio) => Engine.Reshape(audio, new[] { audio.Length });

    /// <summary>The mean of every element, as a scalar tensor.</summary>
    protected Tensor<T> Mean(Tensor<T> x) => Engine.ReduceMean(x, Enumerable.Range(0, x.Rank).ToArray(), keepDims: false);

    /// <summary>A copy of <paramref name="x"/> that carries no gradient.</summary>
    protected static Tensor<T> Detached(Tensor<T> x) => new(x._shape, x.ToVector());

    /// <summary>The sum of <paramref name="terms"/>.</summary>
    protected Tensor<T> Sum(IEnumerable<Tensor<T>> terms)
    {
        Tensor<T>? total = null;
        foreach (var t in terms) total = total is null ? t : Engine.TensorAdd(total, t);
        return total ?? throw new ArgumentException("No terms to sum.", nameof(terms));
    }

    /// <summary>LSGAN discriminator loss <c>mean((1 − D(x))²) + mean(D(G(s))²)</c> (Mao et al. 2017).</summary>
    protected Tensor<T> LeastSquaresDiscriminator(Tensor<T> real, Tensor<T> fake)
    {
        var r = Engine.TensorAddScalar(Engine.TensorNegate(real), NumOps.One);
        return Engine.TensorAdd(Mean(Engine.TensorMultiply(r, r)), Mean(Engine.TensorMultiply(fake, fake)));
    }

    /// <summary>LSGAN generator loss <c>mean((1 − D(G(s)))²)</c>.</summary>
    protected Tensor<T> LeastSquaresGenerator(Tensor<T> fake)
    {
        var d = Engine.TensorAddScalar(Engine.TensorNegate(fake), NumOps.One);
        return Mean(Engine.TensorMultiply(d, d));
    }

    /// <summary>Hinge discriminator loss <c>mean(max(0, 1 − D(x))) + mean(max(0, 1 + D(G(s))))</c> (Lim and Ye 2017).</summary>
    protected Tensor<T> HingeDiscriminator(Tensor<T> real, Tensor<T> fake)
        => Engine.TensorAdd(Mean(Engine.ReLU(Engine.TensorAddScalar(Engine.TensorNegate(real), NumOps.One))),
            Mean(Engine.ReLU(Engine.TensorAddScalar(fake, NumOps.One))));

    /// <summary>Hinge generator loss <c>−mean(D(G(s)))</c>.</summary>
    protected Tensor<T> HingeGenerator(Tensor<T> fake) => Engine.TensorNegate(Mean(fake));

    /// <summary>Feature matching: the sum over layers of the mean L1 distance between the real (detached) and generated
    /// feature maps.</summary>
    protected Tensor<T> FeatureMatching(IReadOnlyList<Tensor<T>> real, IReadOnlyList<Tensor<T>> fake)
        => Sum(Enumerable.Range(0, fake.Count).Select(l => Mean(Engine.TensorAbs(Engine.TensorSubtract(Detached(real[l]), fake[l])))));

    // ---------------------------------------------------------------- groups and optimizers

    /// <inheritdoc />
    /// <remarks>Each step updates only the generator or only the discriminators.</remarks>
    protected override IReadOnlyList<Tensor<T>> SelectTrainableParametersForTraining(IReadOnlyList<Tensor<T>> parameters)
    {
        if (!HasPaperLayers || _trainingLayers is null) return parameters;
        var selected = new HashSet<Tensor<T>>(Training.TapeTrainingStep<T>.CollectParameters(_trainingLayers.ToList(), -1),
            Helpers.TensorReferenceComparer<Tensor<T>>.Instance);
        return parameters.Where(selected.Contains).ToList();
    }

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> OptimizerFor(string group)
    {
        if (_suppliedOptimizer is not null) return _suppliedOptimizer;
        if (!_optimizers.TryGetValue(group, out var optimizer))
            _optimizers[group] = optimizer = CreatePaperOptimizer(group);
        return optimizer;
    }

    /// <inheritdoc />
    /// <remarks>The paper optimizers read the options when they are built, so restored options rebuild them.</remarks>
    protected override void OnMutableConstructorConfigurationRestored()
    {
        base.OnMutableConstructorConfigurationRestored();
        _optimizers.Clear();
    }

    /// <inheritdoc />
    protected override bool SupportsParameterMutation => _useNativeMode;

    /// <summary>Throws when the model has been disposed.</summary>
    protected void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? GetType().Name);
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (_disposed)
            return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
