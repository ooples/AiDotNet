using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.TextToSpeech.Interfaces;
using AiDotNet.TextToSpeech.Vocoders;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.TextToSpeech.EndToEnd;

/// <summary>
/// The VITS family's shared model: a conditional VAE whose prior is a normalizing flow over a text encoding, aligned by
/// monotonic alignment search, with a duration model and a HiFi-GAN decoder trained adversarially on random windows.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Everything VITS (Kim et al. 2021) defines and its successors keep lives here: the relative-position text encoder and
/// its prior projection; the posterior encoder q(z | x) (WaveNet over the linear spectrogram, or over the mel spectrogram
/// when <see cref="PosteriorReadsMel"/>); the residual-coupling flow (optionally with VITS2's Transformer block);
/// monotonic alignment search on log N(f(z); μ, σ) (optionally with VITS2's noise); the KL term through the flow; the
/// HiFi-GAN decoder on a random window; the period and scale discriminators with the LSGAN and feature-matching losses;
/// a speaker table conditioning the posterior, the flow, the decoder and the duration model when there are several
/// speakers; AdamW with the per-epoch decay.
/// </para>
/// <para>
/// A model supplies its duration model: how it is built (<see cref="CreateDurationModel"/>), what it adds to the
/// generator loss (<see cref="JointDurationLoss"/>), and how it predicts durations for synthesis
/// (<see cref="PredictDurations"/>). A model whose paper trains more than the joint generator/discriminator step
/// overrides <see cref="TrainUtterance"/> and uses <see cref="TrainLayers"/> and <see cref="Align"/>.
/// </para>
/// <para><b>For Beginners:</b> A VITS-style model goes straight from text to a waveform in one network: it learns a
/// hidden "voice space" from real audio, learns to predict that space from text, and learns a vocoder that turns it into
/// sound.</para>
/// </remarks>
public abstract partial class VitsTtsModelBase<T> : TtsModelBase<T>, IEndToEndTts<T>, ITrainingObjectiveProvider<T>
{
    private readonly VitsModelOptions _vits;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _suppliedOptimizer;
    private readonly Dictionary<string, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>> _optimizers = new();
    private readonly bool _useNativeMode;
    private bool _disposed;
    private Random _trainingRandom;
    private IReadOnlyList<ILayer<T>>? _trainingLayers;
    private long _trainingSteps;

    private EmbeddingLayer<T>? _embedding;
    private readonly List<RelativePositionTransformerBlock<T>> _encoderBlocks = new();
    private Conv1DLayer<T>? _speakerToEncoder;
    private Conv1DLayer<T>? _priorProjection;
    private EmbeddingLayer<T>? _speakerTable;
    private EmbeddingLayer<T>? _languageTable;
    private VitsPosteriorEncoder<T>? _posterior;
    private VitsFlow<T>? _flow;
    private HiFiGanGenerator<T>? _decoder;
    private HiFiGanDiscriminators<T>? _discriminators;
    private DifferentiableMel<T>? _spectrogram;
    private readonly List<LayerBase<T>> _acousticComponents = new();
    private readonly List<LayerBase<T>> _durationComponents = new();

    /// <summary>Creates a native (trainable) model.</summary>
    protected VitsTtsModelBase(NeuralNetworkArchitecture<T> architecture, VitsModelOptions options,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer)
        : base(architecture)
    {
        _vits = options ?? throw new ArgumentNullException(nameof(options));
        // The clone plan replays the constructor from members named after its parameters: the options live in the base's
        // Options, as NeuralNetworkBase asks of every derived model.
        Options = _vits;
        _useNativeMode = true;
        _suppliedOptimizer = optimizer;
        _trainingRandom = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_vits.SamplingSeed);
        ApplyAudioSettings();
        InitializeLayers();
    }

    /// <summary>Creates a model that runs an exported ONNX graph.</summary>
    protected VitsTtsModelBase(NeuralNetworkArchitecture<T> architecture, string modelPath, VitsModelOptions options)
        : base(architecture)
    {
        _vits = options ?? throw new ArgumentNullException(nameof(options));
        // The clone plan replays the constructor from members named after its parameters: the options live in the base's
        // Options, as NeuralNetworkBase asks of every derived model.
        Options = _vits;
        _useNativeMode = false;
        _trainingRandom = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_vits.SamplingSeed);
        ApplyAudioSettings();
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path required.", nameof(modelPath));
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _vits.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _vits.OnnxOptions);
        InitializeLayers();
    }

    private void ApplyAudioSettings()
    {
        base.SampleRate = _vits.SampleRate;
        base.MelChannels = _vits.MelChannels;
        base.HopSize = _vits.HopSize;
        base.HiddenDim = _vits.HiddenDim;
    }

    /// <inheritdoc />
    public override ModelOptions GetOptions() => _vits;

    /// <summary>The options, available to the hooks the base constructor calls before a derived constructor runs.</summary>
    protected VitsModelOptions VitsOptions => _vits;

    int ITtsModel<T>.SampleRate => _vits.SampleRate;

    /// <summary>Gets the longest text the model accepts.</summary>
    public int MaxTextLength => _vits.MaxTextLength;

    /// <summary>Gets the text encoder's width.</summary>
    public new int HiddenDim => _vits.HiddenDim;

    /// <summary>Gets the number of coupling layers in the flow.</summary>
    public int NumFlowSteps => _vits.NumFlowSteps;

    /// <summary>Gets the number of training steps taken so far.</summary>
    protected long TrainingSteps => _trainingSteps;

    /// <summary>Gets whether the paper's layers were built (false when the architecture supplied its own layers).</summary>
    protected bool HasPaperLayers => _embedding is not null;

    private bool ExternalSpeakers => ExternalSpeakerChannels > 0;

    private bool MultiSpeaker => !ExternalSpeakers && _vits.NumSpeakers > 1;

    private bool MultiLingual => _vits.NumLanguages > 1;

    private int SpeakerChannels => ExternalSpeakers ? ExternalSpeakerChannels : MultiSpeaker ? _vits.SpeakerEmbeddingDim : 0;

    private int LanguageChannels => MultiLingual ? _vits.LanguageEmbeddingDim : 0;

    /// <summary>The text encoder's width: the character embedding plus the language embedding concatenated to it.</summary>
    private int EncoderWidth => _vits.HiddenDim + LanguageChannels;

    private int Hop => _vits.UpsampleRates.Aggregate(1, (a, b) => a * b);

    /// <inheritdoc />
    protected override TtsSupervision RequiredSupervision
        => (MultiSpeaker ? TtsSupervision.SpeakerId : TtsSupervision.None) | (MultiLingual ? TtsSupervision.LanguageId : TtsSupervision.None);

    /// <inheritdoc />
    protected override TtsSupervision RequiredVoice
        => (MultiSpeaker ? TtsSupervision.SpeakerId : ExternalSpeakers ? TtsSupervision.ReferenceRecording : TtsSupervision.None)
           | (MultiLingual ? TtsSupervision.LanguageId : TtsSupervision.None);

    // ---------------------------------------------------------------- hooks

    /// <summary>Builds the duration model for a text encoding of <paramref name="hidden"/> channels, a speaker embedding
    /// of <paramref name="speakerChannels"/> (0 for one speaker) and a language embedding of
    /// <paramref name="languageChannels"/> (0 for one language), returning every layer it owns.</summary>
    protected abstract IEnumerable<LayerBase<T>> CreateDurationModel(int hidden, int speakerChannels, int languageChannels);

    /// <summary>The duration layers trained by the generator step together with the acoustic model (VITS's stochastic
    /// duration predictor); a model that trains its duration model separately returns none.</summary>
    protected abstract IEnumerable<LayerBase<T>> JointDurationLayers { get; }

    /// <summary>The duration term of the generator loss given the text encoding <c>[1, hidden, tokens]</c>, the speaker
    /// and language embeddings <c>[1, channels, 1]</c> (null when absent) and the alignment's durations; null when the
    /// duration model is not trained jointly.</summary>
    protected abstract Tensor<T>? JointDurationLoss(Tensor<T> hidden, Tensor<T>? speaker, Tensor<T>? language, int[] durations, Random random);

    /// <summary>Durations in frames for synthesis, before <see cref="VitsModelOptions.LengthScale"/>: the model's
    /// predicted log-durations, exponentiated.</summary>
    protected abstract double[] PredictDurations(Tensor<T> hidden, Tensor<T>? speaker, Tensor<T>? language, Random random);

    /// <summary>The width of an external speaker embedding computed from a recording (YourTTS's d-vectors), or 0 when
    /// speakers come from <see cref="VitsModelOptions.NumSpeakers"/>' table.</summary>
    protected virtual int ExternalSpeakerChannels => 0;

    /// <summary>The external speaker embedding <c>[1, channels, 1]</c> of a recording <c>[samples]</c>, without
    /// gradient.</summary>
    protected virtual Tensor<T> ExternalSpeaker(Tensor<T> recording)
        => throw new NotSupportedException($"{GetType().Name} has no speaker encoder.");

    /// <summary>A further generator loss term on the decoder window: the real samples and the generated ones, or null
    /// (YourTTS's speaker consistency loss).</summary>
    protected virtual Tensor<T>? ExtraGeneratorLoss(VitsUtterance data, Tensor<T> realSegment, Tensor<T> generatedSegment) => null;

    /// <summary>Whether the posterior encoder reads the mel spectrogram (VITS2) instead of the linear one (VITS).</summary>
    protected virtual bool PosteriorReadsMel => false;

    /// <summary>The scale of the Gaussian noise added to the alignment scores at a training step (VITS2 §2.2); 0 for
    /// none.</summary>
    protected virtual double AlignmentNoiseScale(long step) => 0.0;

    /// <summary>The text encoder block before which the speaker embedding is added (VITS2 §2.4), or −1.</summary>
    protected virtual int SpeakerConditionedEncoderBlock => -1;

    /// <summary>The global gradient-norm clip of every training step, 0 for none (VITS clips nothing; its reference
    /// only measures the norm).</summary>
    protected virtual double GradientClipNorm => 0.0;

    /// <summary>The Transformer block in each flow coupling (VITS2 §2.3): layers (0 for none), heads, feed-forward
    /// kernel and dropout.</summary>
    protected virtual (int Layers, int Heads, int KernelSize, double Dropout) FlowTransformer => (0, 2, 3, 0.1);

    // ---------------------------------------------------------------- layers

    /// <inheritdoc />
    protected override void InitializeLayers()
    {
        if (!_useNativeMode)
            return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            AddUnsplitLayers(Architecture.Layers);
            return;
        }
        var o = _vits;
        int h = o.HiddenDim, inter = o.InterChannels, gin = SpeakerChannels, width = EncoderWidth;
        int posteriorInput = PosteriorReadsMel ? o.MelChannels : o.FftSize / 2 + 1;
        _embedding = new EmbeddingLayer<T>(o.VocabSize, h);
        for (int i = 0; i < o.NumEncoderLayers; i++)
            _encoderBlocks.Add(new RelativePositionTransformerBlock<T>(width, o.NumHeads, o.FilterChannels, o.EncoderKernelSize, o.DropoutRate, o.RelativeWindow));
        _priorProjection = new Conv1DLayer<T>(width, 2 * inter, 1, 1, 1, 0);
        if (MultiLingual)
        {
            _languageTable = new EmbeddingLayer<T>(o.NumLanguages, LanguageChannels);
            _acousticComponents.Add(_languageTable);
        }
        if (MultiSpeaker)
        {
            _speakerTable = new EmbeddingLayer<T>(o.NumSpeakers, gin);
            _acousticComponents.Add(_speakerTable);
        }
        if (gin > 0 && SpeakerConditionedEncoderBlock >= 0)
        {
            if (SpeakerConditionedEncoderBlock >= o.NumEncoderLayers)
                throw new ArgumentException($"The speaker conditions encoder block {SpeakerConditionedEncoderBlock}, but there are {o.NumEncoderLayers}.");
            _speakerToEncoder = new Conv1DLayer<T>(gin, width, 1, 1, 1, 0);
            _acousticComponents.Add(_speakerToEncoder);
        }
        _posterior = new VitsPosteriorEncoder<T>(Engine, _acousticComponents, posteriorInput, inter, h, o.PosteriorKernelSize, o.PosteriorLayers, gin);
        var (tLayers, tHeads, tKernel, tDropout) = FlowTransformer;
        _flow = new VitsFlow<T>(Engine, _acousticComponents, inter, h, o.FlowKernelSize, o.FlowLayers, o.NumFlowSteps, gin, tLayers, tHeads, tKernel, tDropout);
        _decoder = new HiFiGanGenerator<T>(Engine, inter, o.UpsampleInitialChannels, o.UpsampleRates, o.UpsampleKernelSizes,
            o.ResblockKernelSizes, o.ResblockDilationSizes, o.ResblockType == 1, gin, plainEnds: true,
            initialization: AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(o.SamplingSeed + 7));
        _acousticComponents.AddRange(_decoder.Layers);
        _discriminators = new HiFiGanDiscriminators<T>(Engine, o.DiscriminatorPeriods, 1, true, o.DiscriminatorWidthDivisor, spectralFirstScale: false);
        _spectrogram = new DifferentiableMel<T>(Engine, o.SampleRate, o.FftSize, Hop, o.WindowSize, o.MelChannels, 0.0, o.SampleRate / 2.0);
        _durationComponents.AddRange(CreateDurationModel(width, gin, LanguageChannels));

        var encoder = new List<ILayer<T>> { _embedding };
        encoder.AddRange(_encoderBlocks);
        encoder.Add(_priorProjection);
        AddEncoderDecoderLayers(encoder, Array.Empty<ILayer<T>>());
        ComponentLayers.AddRange(_acousticComponents);
        ComponentLayers.AddRange(_durationComponents);
        ComponentLayers.AddRange(_discriminators.Layers);
    }

    /// <summary>The acoustic model's layers: the text encoder, the speaker table, the posterior encoder, the flow and the
    /// decoder.</summary>
    protected IReadOnlyList<ILayer<T>> AcousticLayers => Layers.Concat(_acousticComponents).ToList();

    /// <summary>The period and scale discriminators' layers.</summary>
    protected IReadOnlyList<ILayer<T>> DiscriminatorLayers => _discriminators?.Layers.Cast<ILayer<T>>().ToList() ?? new List<ILayer<T>>();

    // ---------------------------------------------------------------- shared pieces

    /// <summary>The token sequence the text encoder reads for <paramref name="tokens"/>: by default the tokens with a
    /// blank (0) between and around them when <see cref="VitsModelOptions.AddBlank"/> (reference
    /// <c>commons.intersperse</c>).</summary>
    protected virtual Tensor<T> PrepareTokens(Tensor<T> tokens)
    {
        if (!_vits.AddBlank) return tokens;
        var spaced = new Tensor<T>(new[] { 2 * tokens.Length + 1 });
        for (int i = 0; i < tokens.Length; i++) spaced[2 * i + 1] = tokens[i];
        return spaced;
    }

    /// <summary>The speaker embedding <c>[1, gin, 1]</c> from the speaker table, or null for a single-speaker model.</summary>
    protected Tensor<T>? Speaker(int? speakerId)
    {
        if (!MultiSpeaker) return null;
        if (speakerId is null)
            throw new ArgumentException($"{GetType().Name} has {_vits.NumSpeakers} speakers; name the speaker.");
        if (speakerId < 0 || speakerId >= _vits.NumSpeakers)
            throw new ArgumentOutOfRangeException(nameof(speakerId), $"Speaker {speakerId} is outside 0..{_vits.NumSpeakers - 1}.");
        var id = new Tensor<T>(new[] { 1 });
        id[0] = NumOps.FromDouble(speakerId.Value);
        return Engine.Reshape(_speakerTable!.Forward(id), new[] { 1, _vits.SpeakerEmbeddingDim, 1 });
    }

    /// <summary>The language embedding <c>[1, languageChannels, 1]</c>, or null for a monolingual model.</summary>
    protected Tensor<T>? Language(int? languageId)
    {
        if (!MultiLingual) return null;
        if (languageId is null)
            throw new ArgumentException($"{GetType().Name} has {_vits.NumLanguages} languages; name the language.");
        if (languageId < 0 || languageId >= _vits.NumLanguages)
            throw new ArgumentOutOfRangeException(nameof(languageId), $"Language {languageId} is outside 0..{_vits.NumLanguages - 1}.");
        var id = new Tensor<T>(new[] { 1 });
        id[0] = NumOps.FromDouble(languageId.Value);
        return Engine.Reshape(_languageTable!.Forward(id), new[] { 1, LanguageChannels, 1 });
    }

    // The utterance's speaker: its external embedding (computed once, without gradient) or the table's row.
    private Tensor<T>? SpeakerOf(VitsUtterance data) => ExternalSpeakers ? data.ExternalSpeaker : Speaker(data.SpeakerId);

    private Tensor<T>? OverTime(Tensor<T>? speaker, int time)
        => speaker is null ? null : Engine.TensorTile(speaker, new[] { 1, 1, time });

    /// <summary>Text encoder (VITS §2.1, App. A.1): x <c>[1, width, L]</c>, μ and log σ <c>[1, inter, L]</c>; a
    /// multilingual model concatenates the language embedding to every scaled character embedding first (YourTTS §2,
    /// Coqui <c>TextEncoder</c>).</summary>
    protected (Tensor<T> Hidden, Tensor<T> Mean, Tensor<T> LogScale) EncodeText(Tensor<T> tokens, Tensor<T>? speaker, Tensor<T>? language)
    {
        int h = _vits.HiddenDim, inter = _vits.InterChannels, width = EncoderWidth;
        var x = Engine.TensorMultiplyScalar(_embedding!.Forward(tokens), NumOps.FromDouble(Math.Sqrt(h)));
        int length = x.Shape[0];
        if (language is not null)
            x = Engine.TensorConcatenate(new[] { x, Engine.TensorTile(Engine.Reshape(language, new[] { 1, LanguageChannels }), new[] { length, 1 }) }, 1);
        for (int i = 0; i < _encoderBlocks.Count; i++)
        {
            if (i == SpeakerConditionedEncoderBlock && speaker is not null && _speakerToEncoder is not null)
                x = Engine.TensorAdd(x, Engine.TensorTile(Engine.Reshape(_speakerToEncoder.Forward(speaker), new[] { 1, width }), new[] { length, 1 }));
            x = _encoderBlocks[i].Forward(x);
        }
        var channelsFirst = Engine.Reshape(Engine.TensorTranspose(x), new[] { 1, width, length });
        var stats = _priorProjection!.Forward(channelsFirst);
        return (channelsFirst, VitsOps.Slice(Engine, stats, 0, inter), VitsOps.Slice(Engine, stats, inter, inter));
    }

    /// <summary>Standard-normal noise of <paramref name="shape"/> times <paramref name="scale"/>.</summary>
    protected Tensor<T> Gaussian(int[] shape, Random random, double scale = 1.0)
    {
        var t = new Tensor<T>(shape);
        for (int i = 0; i < t.Length; i++)
        {
            double u1 = 1.0 - random.NextDouble(), u2 = random.NextDouble();
            t[i] = NumOps.FromDouble(scale * Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2));
        }
        return t;
    }

    // [1, C, L] -> [1, C, frames] by repeating column l durations[l] times.
    private Tensor<T> Expand(Tensor<T> x, int[] durations)
    {
        int c = x.Shape[1], l = x.Shape[2];
        var rows = Engine.TensorTranspose(Engine.Reshape(x, new[] { c, l }));
        var expanded = LengthRegulator.Expand(rows, durations);
        return Engine.Reshape(Engine.TensorTranspose(expanded), new[] { 1, c, expanded.Shape[0] });
    }

    // ---------------------------------------------------------------- synthesis

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
        var tokens = TokensOf(input);
        using var _ = new NoGradScope<T>();
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_vits.SamplingSeed);
        var voice = RequiredVoice != TtsSupervision.None ? RequireVoice() : null;
        var speaker = ExternalSpeakers ? ExternalSpeaker(voice!.ReferenceAudio!) : Speaker(MultiSpeaker ? voice!.SpeakerId : null);
        var language = Language(MultiLingual ? voice!.LanguageId : null);
        var (hidden, mean, logScale) = EncodeText(PrepareTokens(tokens), speaker, language);
        var predicted = PredictDurations(hidden, speaker, language, random);
        var durations = new int[predicted.Length];
        for (int i = 0; i < durations.Length; i++)
            durations[i] = (int)Math.Ceiling(predicted[i] * _vits.LengthScale);
        if (durations.Sum() == 0) durations[^1] = 1;
        var m = Expand(mean, durations);
        var s = Expand(logScale, durations);
        var zp = Engine.TensorAdd(m, Engine.TensorMultiply(Gaussian(m._shape, random, _vits.NoiseScale), Engine.TensorExp(s)));
        var z = _flow!.Forward(zp, OverTime(speaker, zp.Shape[2]), reverse: true);
        return _decoder!.Forward(z, speaker);                                     // [1, 1, frames · hop]
    }

    /// <inheritdoc />
    protected override Tensor<T> PostprocessAudio(Tensor<T> output) => Engine.Reshape(output, new[] { output.Length });

    private Tensor<T> TokensOf(Tensor<T> input)
    {
        var tokens = input.Rank == 2 && input.Shape[0] == 1 ? Engine.Reshape(input, new[] { input.Shape[1] }) : input;
        if (tokens.Rank != 1)
            throw new ArgumentException($"Expected tokens [tokens], got [{string.Join(", ", input.Shape)}].", nameof(input));
        return tokens;
    }

    // ---------------------------------------------------------------- training

    /// <inheritdoc />
    /// <remarks>A VITS-family model learns its alignment, so text tokens and the recording
    /// (<paramref name="expectedOutput"/>, the waveform) are its whole supervision; a multi-speaker model trains through
    /// <see cref="TtsModelBase{T}.Train(TtsTrainingSample{T})"/> with the speaker named.</remarks>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        ThrowIfDisposed();
        if (IsOnnxMode)
            throw new NotSupportedException("Training is not supported in ONNX mode.");
        ThrowIfTokenMelTrainingUnsupported();
        if (!HasPaperLayers)
        {
            TrainWithTape(input, expectedOutput, _suppliedOptimizer);
            return;
        }
        TrainStep(Prepare(input, expectedOutput, null, null, null));
    }

    /// <inheritdoc />
    protected override T TrainOnSample(TtsTrainingSample<T> sample)
    {
        ThrowIfDisposed();
        if (IsOnnxMode)
            throw new NotSupportedException("Training is not supported in ONNX mode.");
        var audio = sample.Audio ?? throw new ArgumentException($"{GetType().Name} trains on recordings; set Audio.", nameof(sample));
        TrainStep(Prepare(sample.Tokens, audio, sample.SpeakerId, sample.SpeakerReference, sample.LanguageId));
        return LastLoss ?? NumOps.Zero;
    }

    private void TrainStep(VitsUtterance data)
    {
        // The detached generation the discriminators see comes from the training-mode forward, as in the reference.
        SetTrainingMode(true);
        TrainUtterance(data, Draw(data, _trainingRandom));
        _trainingSteps++;
    }

    /// <summary>One training step on an utterance; the default is <see cref="TrainAcoustic"/>.</summary>
    protected virtual void TrainUtterance(VitsUtterance data, VitsTrainingDraw draw) => TrainAcoustic(data, draw);

    /// <summary>VITS's training step (reference <c>train_and_evaluate</c>): the discriminators on the real window and a
    /// detached generation, then the generator (the acoustic model and any jointly trained duration model) on Eq. 9.</summary>
    protected void TrainAcoustic(VitsUtterance data, VitsTrainingDraw draw)
    {
        Tensor<T> generated;
        using (new NoGradScope<T>())
        {
            var g = GeneratorForward(data, draw, training: true).Audio;
            generated = new Tensor<T>(g._shape, g.ToVector());
        }
        TrainLayers(DiscriminatorLayers, data, () => DiscriminatorLoss(data.Segment(draw), generated), "discriminator");
        TrainLayers(AcousticLayers.Concat(JointDurationLayers).ToList(), data, () => GeneratorLoss(data, draw, adversarial: true, training: true), "generator");
    }

    /// <summary>One optimizer step on <paramref name="loss"/> that updates only <paramref name="layers"/>, with the
    /// optimizer kept under <paramref name="group"/> (or the one supplied at construction).</summary>
    protected void TrainLayers(IReadOnlyList<ILayer<T>> layers, VitsUtterance data, Func<Tensor<T>> loss, string group)
    {
        MaxGradNorm = NumOps.FromDouble(GradientClipNorm);
        _trainingLayers = layers;
        try
        {
            TrainWithCustomObjective(data.Tokens, data.Audio, (_, _) => loss(), OptimizerFor(group));
        }
        finally
        {
            _trainingLayers = null;
        }
    }

    TrainingObjectiveKind ITrainingObjectiveProvider<T>.TrainingObjectiveKind => TrainingObjectiveKind.Supervised;

    Tensor<T> ITrainingObjectiveProvider<T>.ResolveTrainingTarget(Tensor<T> input, Tensor<T> proposedTarget) => proposedTarget;

    T ITrainingObjectiveProvider<T>.EvaluateTrainingObjective(Tensor<T> input, Tensor<T> target)
        => EvaluateObjective(Prepare(input, target, null, null, null));

    /// <inheritdoc />
    public override T EvaluateTrainingObjective(TtsTrainingSample<T> sample)
    {
        Guard.NotNull(sample);
        var audio = sample.Audio ?? throw new ArgumentException($"{GetType().Name} trains on recordings; set Audio.", nameof(sample));
        return EvaluateObjective(Prepare(sample.Tokens, audio, sample.SpeakerId, sample.SpeakerReference, sample.LanguageId));
    }

    private T EvaluateObjective(VitsUtterance data)
    {
        ThrowIfDisposed();
        var draw = Draw(data, AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_vits.SamplingSeed));
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            return Objective(data, draw)[0];
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    /// <summary>The objective <see cref="ITrainingObjectiveProvider{T}"/> reports with the draws fixed by the sampling
    /// seed: by default the non-adversarial part of the generator loss (the adversarial terms are measured against
    /// discriminators trained at the same time, so they do not fall monotonically even when the generator improves).</summary>
    protected virtual Tensor<T> Objective(VitsUtterance data, VitsTrainingDraw draw)
        => GeneratorLoss(data, draw, adversarial: false, training: false);

    /// <summary>One utterance: the (blank-interspersed) tokens, the recording cut to whole frames, its spectrograms and
    /// the speaker.</summary>
    protected sealed class VitsUtterance
    {
        internal VitsUtterance(Tensor<T> tokens, Tensor<T> audio, Tensor<T> posteriorInput, Tensor<T> mel, int frames, int hop, int segmentFrames,
            int? speakerId, Tensor<T>? externalSpeaker, int? languageId)
        {
            ExternalSpeaker = externalSpeaker;
            LanguageId = languageId;
            Tokens = tokens;
            Audio = audio;
            PosteriorInput = posteriorInput;
            Mel = mel;
            Frames = frames;
            Hop = hop;
            SegmentFrames = segmentFrames;
            SpeakerId = speakerId;
        }

        /// <summary>The tokens the encoder reads <c>[tokens]</c>.</summary>
        public Tensor<T> Tokens { get; }

        /// <summary>The waveform, a whole number of frames <c>[samples]</c>.</summary>
        public Tensor<T> Audio { get; }

        /// <summary>The posterior encoder's input <c>[1, bins, frames]</c>.</summary>
        public Tensor<T> PosteriorInput { get; }

        /// <summary>The log-mel spectrogram <c>[frames, mel]</c>.</summary>
        public Tensor<T> Mel { get; }

        /// <summary>The number of frames.</summary>
        public int Frames { get; }

        /// <summary>The samples per frame.</summary>
        public int Hop { get; }

        /// <summary>The frames in a decoder training window.</summary>
        public int SegmentFrames { get; }

        /// <summary>The speaker, or null for a single-speaker model.</summary>
        public int? SpeakerId { get; }

        /// <summary>The external speaker embedding <c>[1, channels, 1]</c>, or null.</summary>
        public Tensor<T>? ExternalSpeaker { get; }

        /// <summary>The language, or null for a monolingual model.</summary>
        public int? LanguageId { get; }

        /// <summary>The recording's samples under the draw's decoder window.</summary>
        public Tensor<T> Segment(VitsTrainingDraw draw)
            => AiDotNet.Tensors.Engines.AiDotNetEngine.Current.TensorSlice(Audio, new[] { draw.SegmentStart * Hop }, new[] { SegmentFrames * Hop });
    }

    /// <summary>The random draws of one training step: the posterior noise, the decoder window, and seeds for the
    /// duration model and the alignment noise.</summary>
    protected sealed record VitsTrainingDraw(Tensor<T> PosteriorNoise, int SegmentStart, int DurationSeed, int AlignmentSeed);

    private VitsUtterance Prepare(Tensor<T> input, Tensor<T> target, int? speakerId, Tensor<T>? speakerRecording, int? languageId)
    {
        var tokens = TokensOf(input);
        int hop = Hop;
        int frames = target.Length / hop;
        var spaced = PrepareTokens(tokens);
        if (frames < spaced.Length)
            throw new ArgumentException(
                $"Monotonic alignment gives every token a frame: {spaced.Length} tokens need {spaced.Length * hop} samples, got {target.Length}.", nameof(target));
        if (MultiSpeaker && speakerId is null)
            throw new ArgumentException($"{GetType().Name} has {_vits.NumSpeakers} speakers; set the sample's SpeakerId.", nameof(speakerId));
        if (MultiLingual && languageId is null)
            throw new ArgumentException($"{GetType().Name} has {_vits.NumLanguages} languages; set the sample's LanguageId.", nameof(languageId));
        var audio = Engine.TensorSlice(Engine.Reshape(target, new[] { target.Length }), new[] { 0 }, new[] { frames * hop });
        Tensor<T> posteriorInput, mel;
        using (new NoGradScope<T>())
        {
            var magnitude = _spectrogram!.Magnitude(audio, 1e-6);                                     // spectrogram_torch
            var trimmed = Engine.TensorSlice(magnitude, new[] { 0, 0 }, new[] { frames, magnitude.Shape[1] });
            mel = _spectrogram.LogMel(trimmed);                                                        // spec_to_mel_torch
            var source = PosteriorReadsMel ? mel : trimmed;
            posteriorInput = Engine.Reshape(Engine.TensorTranspose(source), new[] { 1, source.Shape[1], frames });
        }
        int segmentFrames = Math.Max(1, Math.Min(frames, _vits.SegmentSize / hop));
        // An external speaker embedding comes from the given reference recording, or from the utterance itself.
        var external = ExternalSpeakers ? ExternalSpeaker(speakerRecording ?? audio) : null;
        return new VitsUtterance(spaced, audio, posteriorInput, mel, frames, hop, segmentFrames, speakerId, external, languageId);
    }

    private VitsTrainingDraw Draw(VitsUtterance data, Random random)
        => new(Gaussian(new[] { 1, _vits.InterChannels, data.Frames }, random),
            random.Next(0, data.Frames - data.SegmentFrames + 1), random.Next(), random.Next());

    /// <summary>The alignment of one utterance: the text encoding, the prior, the posterior sample and its image under
    /// the flow, and the monotonic alignment's durations.</summary>
    protected sealed record VitsAlignment(Tensor<T> Hidden, Tensor<T> Mean, Tensor<T> LogScale, Tensor<T> Z, Tensor<T> LogScaleQ,
        Tensor<T> Zp, Tensor<T>? Speaker, Tensor<T>? Language, int[] Durations);

    /// <summary>
    /// Encodes the text and the recording and runs monotonic alignment search on
    /// <c>P_ij = log N(f(z_j); μ_i, σ_i)</c> (VITS §2.2.1), adding <c>ε = std(P) · N(0, 1) · scale</c> to the scores when
    /// <see cref="AlignmentNoiseScale"/> is positive at a training step (VITS2 Eq. 4–5).
    /// </summary>
    protected VitsAlignment Align(VitsUtterance data, VitsTrainingDraw draw, bool training)
    {
        var speaker = SpeakerOf(data);
        var language = Language(data.LanguageId);
        var (hidden, mean, logScale) = EncodeText(data.Tokens, speaker, language);
        var condition = OverTime(speaker, data.Frames);
        var (z, _, logScaleQ) = _posterior!.Forward(data.PosteriorInput, condition, draw.PosteriorNoise);
        var zp = _flow!.Forward(z, condition, reverse: false);

        int tokens = mean.Shape[2], frames = data.Frames, channels = _vits.InterChannels;
        var scores = new double[tokens, frames];
        var mHost = mean.ToVector();
        var sHost = logScale.ToVector();
        var zHost = zp.ToVector();
        double sum = 0, sumSquares = 0;
        for (int i = 0; i < tokens; i++)
            for (int j = 0; j < frames; j++)
            {
                double p = 0;
                for (int c = 0; c < channels; c++)
                {
                    double ls = NumOps.ToDouble(sHost[c * tokens + i]), d = NumOps.ToDouble(zHost[c * frames + j]) - NumOps.ToDouble(mHost[c * tokens + i]);
                    p += -0.5 * Math.Log(2 * Math.PI) - ls - 0.5 * d * d * Math.Exp(-2 * ls);
                }
                scores[i, j] = p;
                sum += p;
                sumSquares += p * p;
            }
        double noiseScale = training ? AlignmentNoiseScale(_trainingSteps) : 0.0;
        if (noiseScale > 0)
        {
            int n = tokens * frames;
            double std = n > 1 ? Math.Sqrt(Math.Max(0.0, (sumSquares - sum * sum / n) / (n - 1))) : 0.0;   // torch.std (unbiased)
            var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(draw.AlignmentSeed);
            for (int i = 0; i < tokens; i++)
                for (int j = 0; j < frames; j++)
                {
                    double u1 = 1.0 - random.NextDouble(), u2 = random.NextDouble();
                    scores[i, j] += std * noiseScale * Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2);
                }
        }
        return new VitsAlignment(hidden, mean, logScale, z, logScaleQ, zp, speaker, language, MonotonicAlignment.MaximumPath(scores));
    }

    private sealed record ForwardResult(Tensor<T> Audio, Tensor<T> Kl, Tensor<T>? Duration, Tensor<T>? Speaker);

    /// <summary>The generator's training forward (reference <c>SynthesizerTrn.forward</c>).</summary>
    private ForwardResult GeneratorForward(VitsUtterance data, VitsTrainingDraw draw, bool training)
    {
        var a = Align(data, draw, training);
        var durationLoss = JointDurationLoss(a.Hidden, a.Speaker, a.Language, a.Durations,
            AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(draw.DurationSeed));

        // KL(q(z|x) || p(z|c)) through the flow (reference kl_loss), averaged over frames.
        var mp = Expand(a.Mean, a.Durations);
        var sp = Expand(a.LogScale, a.Durations);
        var diff = Engine.TensorSubtract(a.Zp, mp);
        var klTerms = Engine.TensorAdd(
            Engine.TensorAddScalar(Engine.TensorSubtract(sp, a.LogScaleQ), NumOps.FromDouble(-0.5)),
            Engine.TensorMultiplyScalar(Engine.TensorMultiply(Engine.TensorMultiply(diff, diff), Engine.TensorExp(Engine.TensorMultiplyScalar(sp, NumOps.FromDouble(-2)))),
                NumOps.FromDouble(0.5)));
        var kl = Engine.TensorMultiplyScalar(Engine.ReduceSum(klTerms, new[] { 0, 1, 2 }, keepDims: false), NumOps.FromDouble(1.0 / data.Frames));

        var window = Engine.TensorSlice(a.Z, new[] { 0, 0, draw.SegmentStart }, new[] { 1, _vits.InterChannels, data.SegmentFrames });
        return new ForwardResult(_decoder!.Forward(window, a.Speaker), kl, durationLoss, a.Speaker);
    }

    private Tensor<T> Flat(Tensor<T> x) => Engine.Reshape(x, new[] { x.Length });

    /// <summary>The mean of every element of <paramref name="x"/>, as a scalar tensor.</summary>
    protected Tensor<T> Mean(Tensor<T> x) => Engine.ReduceMean(x, Enumerable.Range(0, x.Rank).ToArray(), keepDims: false);

    /// <summary>VITS Eq. 9: L_recon (mel L1 × 45) + L_kl + the joint duration term, plus L_adv(G) + L_fm(G) when
    /// <paramref name="adversarial"/>.</summary>
    private Tensor<T> GeneratorLoss(VitsUtterance data, VitsTrainingDraw draw, bool adversarial, bool training)
    {
        var result = GeneratorForward(data, draw, training);
        var generatedMel = _spectrogram!.Forward(Flat(result.Audio));
        var targetMel = Engine.TensorSlice(data.Mel, new[] { draw.SegmentStart, 0 }, new[] { data.SegmentFrames, _vits.MelChannels });
        int rows = Math.Min(generatedMel.Shape[0], targetMel.Shape[0]);
        var melLoss = Mean(Engine.TensorAbs(Engine.TensorSubtract(
            Engine.TensorSlice(generatedMel, new[] { 0, 0 }, new[] { rows, _vits.MelChannels }),
            Engine.TensorSlice(targetMel, new[] { 0, 0 }, new[] { rows, _vits.MelChannels }))));
        var loss = Engine.TensorAdd(
            Engine.TensorMultiplyScalar(melLoss, NumOps.FromDouble(_vits.MelLossWeight)),
            Engine.TensorMultiplyScalar(result.Kl, NumOps.FromDouble(_vits.KlLossWeight)));
        if (result.Duration is not null) loss = Engine.TensorAdd(loss, result.Duration);
        var extra = ExtraGeneratorLoss(data, data.Segment(draw), Flat(result.Audio));
        if (extra is not null) loss = Engine.TensorAdd(loss, extra);
        if (!adversarial) return loss;

        var fake = _discriminators!.Forward(Flat(result.Audio));
        List<(Tensor<T> Score, List<Tensor<T>> Features)> real;
        using (new NoGradScope<T>()) real = _discriminators.Forward(data.Segment(draw));
        for (int k = 0; k < fake.Count; k++)
        {
            var d = Engine.TensorAddScalar(Engine.TensorNegate(fake[k].Score), NumOps.One);
            loss = Engine.TensorAdd(loss, Mean(Engine.TensorMultiply(d, d)));
            for (int l = 0; l < fake[k].Features.Count; l++)
            {
                var r = new Tensor<T>(real[k].Features[l]._shape, real[k].Features[l].ToVector());
                // feature_loss returns 2 Σ mean |r − g|.
                loss = Engine.TensorAdd(loss, Engine.TensorMultiplyScalar(Mean(Engine.TensorAbs(Engine.TensorSubtract(r, fake[k].Features[l]))), NumOps.FromDouble(2)));
            }
        }
        return loss;
    }

    private Tensor<T> DiscriminatorLoss(Tensor<T> real, Tensor<T> generated)
    {
        var realScores = _discriminators!.Forward(real);
        var fakeScores = _discriminators.Forward(Flat(generated));
        Tensor<T>? loss = null;
        for (int k = 0; k < realScores.Count; k++)
            loss = Accumulate(loss, LeastSquaresDiscriminatorLoss(realScores[k].Score, fakeScores[k].Score));
        return loss!;
    }

    /// <summary>The LSGAN discriminator loss <c>mean((1 − D(real))²) + mean(D(fake)²)</c> (reference
    /// <c>discriminator_loss</c>).</summary>
    protected Tensor<T> LeastSquaresDiscriminatorLoss(Tensor<T> real, Tensor<T> fake)
    {
        var r = Engine.TensorAddScalar(Engine.TensorNegate(real), NumOps.One);
        return Engine.TensorAdd(Mean(Engine.TensorMultiply(r, r)), Mean(Engine.TensorMultiply(fake, fake)));
    }

    /// <summary>The LSGAN generator loss <c>mean((1 − D(fake))²)</c> (reference <c>generator_loss</c>).</summary>
    protected Tensor<T> LeastSquaresGeneratorLoss(Tensor<T> fake)
    {
        var d = Engine.TensorAddScalar(Engine.TensorNegate(fake), NumOps.One);
        return Mean(Engine.TensorMultiply(d, d));
    }

    private Tensor<T> Accumulate(Tensor<T>? total, Tensor<T> term) => total is null ? term : Engine.TensorAdd(total, term);

    /// <inheritdoc />
    /// <remarks>Each step updates only the layers <see cref="TrainLayers"/> names.</remarks>
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
            _optimizers[group] = optimizer = CreatePaperOptimizer();
        return optimizer;
    }

    // AdamW at the paper's rate, decayed by LearningRateDecay once per epoch of UpdatesPerEpoch steps.
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> CreatePaperOptimizer()
    {
        int perEpoch = Math.Max(1, _vits.UpdatesPerEpoch);
        var scheduler = new AiDotNet.LearningRateSchedulers.LambdaLRScheduler(_vits.LearningRate,
            step => Math.Pow(_vits.LearningRateDecay, step / perEpoch));
        return PaperOptimizerFactory.VerifyHandBuilt(this, new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this,
            new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = _vits.LearningRate,
                Beta1 = _vits.Beta1,
                Beta2 = _vits.Beta2,
                Epsilon = _vits.Epsilon,
                WeightDecay = _vits.WeightDecay,
                LearningRateScheduler = scheduler,
                SchedulerStepMode = AiDotNet.LearningRateSchedulers.SchedulerStepMode.StepPerBatch,
            }));
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
