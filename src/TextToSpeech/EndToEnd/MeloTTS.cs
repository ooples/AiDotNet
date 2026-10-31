using System.Linq;
using System.Text.RegularExpressions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.Interfaces;
using AiDotNet.TextToSpeech.Vocoders;

namespace AiDotNet.TextToSpeech.EndToEnd;

/// <summary>
/// MeloTTS (MyShell, myshell-ai/MeloTTS): a multilingual Bert-VITS2 — VITS with VITS2's speaker-conditioned text
/// encoder, transformer coupling flows, noise-scaled alignment search and duration discriminator, conditioned on per-
/// symbol tones, languages and the features of a frozen per-language BERT, with a parity-tested front end for each of
/// its languages.
/// </summary>
/// <remarks>
/// <para>MeloTTS has no paper; its code is the reference (<c>models.py</c>, <c>train.py</c>, <c>api.py</c>,
/// <c>utils.get_text_for_tts_infer</c>). Synthesis splits the text into sentences as <c>split_utils</c> does, reads
/// each through the language's front end and BERT (<c>hidden_states[-3]</c>, each piece repeated for its phones),
/// generates it with the stochastic and deterministic duration predictors mixed by <see cref="MeloTTSOptions.SdpRatio"/>,
/// and joins the sentences with 50 ms of silence. Training takes MeloTTS's step: the duration discriminator and the
/// multi-period discriminator on a detached generation, then the generator on mel L1 × 45 + KL + duration +
/// adversarial + feature matching (+ the duration adversarial term), each with its own AdamW.</para>
/// <para><b>For Beginners:</b> give it text and it speaks; train it with <see cref="TtsTrainingSample{T}"/>s that carry
/// the recording, its <see cref="TtsTrainingSample{T}.Text"/> and the speaker. Released voices load with
/// <see cref="LoadCheckpoint"/> plus <see cref="LoadBertWeights"/> for the language's BERT.</para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("MeloTTS: High-quality Multi-lingual Multi-accent Text-to-Speech", "https://github.com/myshell-ai/MeloTTS")]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 3e-4, Beta1 = 0.8, Beta2 = 0.99, Epsilon = 1e-9, WeightDecay = 0.01,
                DecayRate = 0.999875, ReferenceBatchSize = 6,
                Source = "myshell-ai/MeloTTS configs/config.json and train.py: AdamW (beta1 0.8, beta2 0.99, eps 1e-9, "
                        + "PyTorch's default weight decay 0.01) at 3e-4 for the generator and each discriminator, decayed by "
                        + "0.999875 every epoch, batch size 6.")]
public partial class MeloTTS<T> : TtsModelBase<T>, IEndToEndTts<T>
{
    private readonly MeloTTSOptions _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _suppliedOptimizer;
    private readonly Dictionary<string, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>> _optimizers = new();
    private readonly List<LayerBase<T>> _generatorLayers = new();
    private readonly List<LayerBase<T>> _durationDiscriminatorLayers = new();
    private readonly List<LayerBase<T>> _bertLayers = new();
    private readonly Random _trainingRandom;
    private readonly bool _useNativeMode;
    private bool _disposed;
    private long _trainingSteps;
    private IReadOnlyList<ILayer<T>>? _trainingLayers;
    private MeloSynthesizer<T>? _synthesizer;
    private MeloBertFeatures<T>? _bert;
    private MeloDurationDiscriminator<T>? _durationDiscriminator;
    private HiFiGanDiscriminators<T>? _discriminators;
    private DifferentiableMel<T>? _spectrogram;

    /// <summary>Creates a MeloTTS that runs an exported ONNX graph.</summary>
    public MeloTTS(NeuralNetworkArchitecture<T> architecture, string modelPath, MeloTTSOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new MeloTTSOptions();
        Options = _options;
        _useNativeMode = false;
        _trainingRandom = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed);
        ConfigureBase();
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path required.", nameof(modelPath));
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath;
        OnnxModel = new AiDotNet.Onnx.OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates a native, trainable MeloTTS.</summary>
    public MeloTTS(NeuralNetworkArchitecture<T> architecture, MeloTTSOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new MeloTTSOptions();
        Options = _options;
        _useNativeMode = true;
        _suppliedOptimizer = optimizer;
        _trainingRandom = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed);
        ConfigureBase();
        InitializeLayers();
    }

    private void ConfigureBase()
    {
        base.SampleRate = _options.SampleRate;
        base.MelChannels = _options.MelChannels;
        base.HopSize = _options.HopSize;
        base.HiddenDim = _options.HiddenDim;
    }

    int ITtsModel<T>.SampleRate => _options.SampleRate;

    /// <inheritdoc />
    public int MaxTextLength => _options.MaxTextLength;

    /// <inheritdoc />
    public new int HiddenDim => _options.HiddenDim;

    /// <inheritdoc />
    public int NumFlowSteps => _options.NumFlowSteps;

    /// <summary>The language this model speaks.</summary>
    public MeloLanguage Language => _options.Language;

    private bool HasPaperLayers => _synthesizer is not null;

    internal MeloSynthesizer<T> Synthesizer => _synthesizer ?? throw new InvalidOperationException("MeloTTS's synthesizer exists in native mode only.");
    internal BertNetwork<T> Bert => (_bert ?? throw new InvalidOperationException("MeloTTS's BERT exists in native mode only.")).Network;
    internal IReadOnlyList<LayerBase<T>> GeneratorLayers => _generatorLayers;
    internal IReadOnlyList<LayerBase<T>> DiscriminatorLayers => _discriminators?.Layers ?? (IReadOnlyList<LayerBase<T>>)Array.Empty<LayerBase<T>>();
    internal IReadOnlyList<LayerBase<T>> DurationDiscriminatorLayers => _durationDiscriminatorLayers;
    internal IReadOnlyList<LayerBase<T>> BertLayers => _bertLayers;

    /// <inheritdoc />
    protected override TtsSupervision RequiredSupervision =>
        TtsSupervision.Recording | TtsSupervision.Text | (_options.NumSpeakers > 1 ? TtsSupervision.SpeakerId : TtsSupervision.None);

    private int Hop => _options.UpsampleRates.Aggregate(1, (a, b) => a * b);

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
        var o = _options;
        if (Hop != o.HopSize)
            throw new ArgumentException($"The upsampling rates give a hop of {Hop} samples, but HopSize is {o.HopSize}.");
        var configuration = new MeloConfiguration(MeloSymbols.Symbols.Count, MeloSymbols.ToneCount, MeloSymbols.LanguageCount,
            o.FftSize / 2 + 1, Math.Max(1, o.SegmentSize / Hop), o.InterChannels, o.HiddenDim, o.FilterChannels, o.NumHeads,
            o.NumEncoderLayers, o.KernelSize, o.DropoutRate, o.ResblockKernelSizes, o.ResblockDilationSizes, o.UpsampleRates,
            o.UpsampleInitialChannels, o.UpsampleKernelSizes, o.NumSpeakers, o.SpeakerChannels, o.NumFlowSteps,
            o.FlowTransformerLayers, o.DurationDropout, o.MasNoiseInitial, o.MasNoiseDecay, o.BertChannels, o.JaBertChannels);
        _synthesizer = new MeloSynthesizer<T>(Engine, _generatorLayers, configuration);
        Layers.AddRange(_generatorLayers);

        _durationDiscriminator = new MeloDurationDiscriminator<T>(Engine, _durationDiscriminatorLayers, o.HiddenDim, o.HiddenDim, 3,
            o.DurationDiscriminatorDropout, o.SpeakerChannels);
        _discriminators = new HiFiGanDiscriminators<T>(Engine, new[] { 2, 3, 5, 7, 11 }, 1, true, o.DiscriminatorWidthDivisor, spectralFirstScale: false, vitsScale: true);
        _spectrogram = new DifferentiableMel<T>(Engine, o.SampleRate, o.FftSize, Hop, o.WindowSize, o.MelChannels, 0.0, o.SampleRate / 2.0);

        bool wide = MeloTextPipeline.UsesWideBert(o.Language);
        var released = MeloTextPipeline.BertShape(o.Language);
        var shape = released with
        {
            Hidden = wide ? o.BertChannels : o.JaBertChannels,
            Layers = o.BertLayers ?? released.Layers,
            Heads = o.BertHeads ?? released.Heads,
            Intermediate = o.BertIntermediate ?? released.Intermediate,
        };
        _bert = new MeloBertFeatures<T>(Engine, _bertLayers, o.Language, shape);

        ComponentLayers.AddRange(_durationDiscriminatorLayers);
        ComponentLayers.AddRange(_discriminators.Layers);
        ComponentLayers.AddRange(_bertLayers);
    }

    // ---------------------------------------------------------------- text

    private sealed record Utterance(MeloText Text, Tensor<T> Bert, Tensor<T> JaBert);

    // get_text_for_tts_infer: the sentence through the front end, its BERT features in the language's slot, zeros in
    // the other.
    private Utterance Prepare(string sentence)
    {
        var text = MeloTextPipeline.Prepare(_options.Language, sentence);
        var features = _bert!.Features(text.Normalized, text.PhonesPerPiece);
        int length = text.Symbols.Length;
        if (features.Shape[2] != length)
            throw new InvalidOperationException($"BERT gave {features.Shape[2]} phone columns for {length} symbols.");
        bool wide = MeloTextPipeline.UsesWideBert(_options.Language);
        var bert = wide ? features : new Tensor<T>(new[] { 1, _options.BertChannels, length });
        var jaBert = wide ? new Tensor<T>(new[] { 1, _options.JaBertChannels, length }) : features;
        return new Utterance(text, bert, jaBert);
    }

    // ---------------------------------------------------------------- synthesis

    /// <inheritdoc />
    /// <remarks>MeloTTS's <c>tts_to_file</c>: the text split into sentences, each synthesized and followed by 50 ms of
    /// silence (shortened by <see cref="MeloTTSOptions.Speed"/>), the waveform <c>[samples]</c>.</remarks>
    public override Tensor<T> Synthesize(string text)
    {
        ThrowIfDisposed();
        if (text is null) throw new ArgumentNullException(nameof(text));
        if (!HasPaperLayers || IsOnnxMode) return base.Synthesize(text);
        var o = _options;
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(o.SamplingSeed);
        int silence = (int)(o.SampleRate * 0.05 / o.Speed);
        var samples = new List<T>();
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            foreach (var piece in MeloTextPipeline.SplitSentence(text, o.Language))
            {
                string sentence = o.Language is MeloLanguage.English or MeloLanguage.ChineseMixEnglish
                    ? Regex.Replace(piece, "([a-z])([A-Z])", "$1 $2")
                    : piece;
                var utterance = Prepare(sentence);
                var (waveform, _) = _synthesizer!.Infer(utterance.Text.Symbols, utterance.Text.Tones, utterance.Text.Languages,
                    utterance.Bert, utterance.JaBert, o.SpeakerId, random, o.NoiseScale, 1.0 / o.Speed, o.NoiseScaleW, o.SdpRatio);
                samples.AddRange(waveform.ToVector());
                for (int i = 0; i < silence; i++) samples.Add(NumOps.Zero);
            }
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
        return new Tensor<T>(new[] { samples.Count }, new Vector<T>(samples.ToArray()));
    }

    /// <inheritdoc />
    /// <remarks>
    /// The tensor path reads <paramref name="input"/> <c>[tokens]</c> as MeloTTS symbol ids (rounded, blanks
    /// interspersed) in the model's language with each language's first tone and no BERT features (MeloTTS's
    /// disable_bert input); text goes through <see cref="Synthesize"/>, which supplies tones and BERT.
    /// </remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        if (!HasPaperLayers) return base.PredictCore(input);
        var o = _options;
        int count = MeloSymbols.Symbols.Count;
        var raw = input.ToVector().Select(v => Math.Max(0, Math.Min(count - 1, (int)Math.Round(NumOps.ToDouble(v))))).ToArray();
        var symbols = MeloSymbols.Intersperse(raw);
        string code = MeloTextPipeline.Code(o.Language);
        var tones = MeloSymbols.Intersperse(Enumerable.Repeat(MeloSymbols.ToneStart(code), raw.Length).ToArray());
        var languages = MeloSymbols.Intersperse(Enumerable.Repeat(MeloSymbols.LanguageId(code), raw.Length).ToArray());
        int length = symbols.Length;
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            var (waveform, _) = _synthesizer!.Infer(symbols, tones, languages, new Tensor<T>(new[] { 1, o.BertChannels, length }),
                new Tensor<T>(new[] { 1, o.JaBertChannels, length }), o.SpeakerId,
                AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(o.SamplingSeed), o.NoiseScale, 1.0 / o.Speed, o.NoiseScaleW, o.SdpRatio);
            return Engine.Reshape(waveform, new[] { waveform.Length });
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    /// <inheritdoc />
    protected override Tensor<T> PostprocessAudio(Tensor<T> output) => Engine.Reshape(output, new[] { output.Length });

    // ---------------------------------------------------------------- training

    /// <inheritdoc />
    protected override bool TryApplyLearningRule(Tensor<T> input, Tensor<T> expectedOutput)
    {
        // A token/waveform pair lacks the sentence MeloTTS's front end and BERT read; Train(TtsTrainingSample) has it.
        ThrowIfTokenMelTrainingUnsupported();
        return false;
    }

    private sealed record TrainingUtterance(Utterance Input, Tensor<T> Audio, Tensor<T> Spectrogram, Tensor<T> Mel, int Frames, int Speaker);

    private TrainingUtterance PrepareTraining(TtsTrainingSample<T> sample)
    {
        string text = sample.Text ?? throw new ArgumentException("MeloTTS trains on the utterance's text; set the sample's Text.", nameof(sample));
        var recording = sample.Audio ?? throw new ArgumentException("MeloTTS trains on recordings; set the sample's Audio.", nameof(sample));
        int speaker = sample.SpeakerId ?? 0;
        if (speaker < 0 || speaker >= _options.NumSpeakers)
            throw new ArgumentOutOfRangeException(nameof(sample), $"Speaker {speaker} is outside the {_options.NumSpeakers}-row speaker table.");
        var input = Prepare(text);
        int hop = Hop, frames = recording.Length / hop;
        if (frames < 1) throw new ArgumentException($"The recording is shorter than one {hop}-sample frame.", nameof(sample));
        var audio = Engine.TensorSlice(Engine.Reshape(recording, new[] { recording.Length }), new[] { 0 }, new[] { frames * hop });
        Tensor<T> spectrogram, mel;
        using (new NoGradScope<T>())
        {
            var magnitude = _spectrogram!.Magnitude(audio, 1e-6);                                         // spectrogram_torch
            var trimmed = Engine.TensorSlice(magnitude, new[] { 0, 0 }, new[] { frames, magnitude.Shape[1] });
            mel = _spectrogram.LogMel(trimmed);                                                            // spec_to_mel_torch
            spectrogram = Engine.Reshape(Engine.TensorTranspose(trimmed), new[] { 1, trimmed.Shape[1], frames });
        }
        return new TrainingUtterance(input, audio, spectrogram, mel, frames, speaker);
    }

    private double MasNoiseScale(long step) => Math.Max(0.0, _options.MasNoiseInitial - _options.MasNoiseDecay * step);

    // SynthesizerTrn.forward with the step's draws fixed by its seed, so the discriminators' detached view and the
    // generator's differentiable one are the same forward (MeloTTS runs it once).
    private MeloTrainingOutputs<T> Forward(TrainingUtterance u, int seed, long step, bool training) =>
        _synthesizer!.Forward(u.Input.Text.Symbols, u.Input.Text.Tones, u.Input.Text.Languages, u.Input.Bert, u.Input.JaBert, u.Spectrogram,
            u.Speaker, MasNoiseScale(step), training, AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed));

    private Tensor<T> Segment(TrainingUtterance u, MeloTrainingOutputs<T> o)
    {
        int segmentFrames = Math.Min(u.Frames, _synthesizer!.Configuration.SegmentFrames);
        return Engine.TensorSlice(u.Audio, new[] { o.SliceStart * Hop }, new[] { segmentFrames * Hop });
    }

    private Tensor<T> Flat(Tensor<T> x) => Engine.Reshape(x, new[] { x.Length });

    private Tensor<T> Mean(Tensor<T> x) => Engine.ReduceMean(x, Enumerable.Range(0, x.Rank).ToArray(), keepDims: false);

    private static Tensor<T> Detached(Tensor<T> x) => new(x._shape, x.ToVector());

    // discriminator_loss: Σ mean((1 − D(real))²) + mean(D(fake)²).
    private Tensor<T> DiscriminatorLoss(IReadOnlyList<Tensor<T>> real, IReadOnlyList<Tensor<T>> fake)
    {
        Tensor<T>? loss = null;
        for (int k = 0; k < real.Count; k++)
        {
            var r = Engine.TensorAddScalar(Engine.TensorNegate(real[k]), NumOps.One);
            var term = Engine.TensorAdd(Mean(Engine.TensorMultiply(r, r)), Mean(Engine.TensorMultiply(fake[k], fake[k])));
            loss = loss is null ? term : Engine.TensorAdd(loss, term);
        }
        return loss!;
    }

    // generator_loss: Σ mean((1 − D(fake))²).
    private Tensor<T> GeneratorAdversarialLoss(IReadOnlyList<Tensor<T>> fake)
    {
        Tensor<T>? loss = null;
        foreach (var f in fake)
        {
            var d = Engine.TensorAddScalar(Engine.TensorNegate(f), NumOps.One);
            var term = Mean(Engine.TensorMultiply(d, d));
            loss = loss is null ? term : Engine.TensorAdd(loss, term);
        }
        return loss!;
    }

    private (Tensor<T> Real, Tensor<T> Generated) DurationScores(MeloTrainingOutputs<T> o, bool detach, Random random)
    {
        // train.py calls net_dur_disc(hidden_x, x_mask, logw, logw_) without the speaker: the predictor's durations take
        // the "real" place and the alignment's the "generated" one, as the reference does.
        var hidden = detach ? Detached(o.Hidden) : o.Hidden;
        var predicted = detach ? Detached(o.PredictedLogDurations) : o.PredictedLogDurations;
        var aligned = detach ? Detached(o.AlignedLogDurations) : o.AlignedLogDurations;
        return _durationDiscriminator!.Forward(hidden, null, predicted, aligned, true, random);
    }

    /// <inheritdoc />
    protected override T TrainOnSample(TtsTrainingSample<T> sample)
    {
        ThrowIfDisposed();
        if (IsOnnxMode)
            throw new NotSupportedException("Training is not supported in ONNX mode.");
        if (!HasPaperLayers)
            throw new NotSupportedException("MeloTTS's training needs its own layers; this model was built from caller-supplied layers.");
        var u = PrepareTraining(sample);
        long step = _trainingSteps;
        int seed = _trainingRandom.Next();
        SetTrainingMode(true);

        MeloTrainingOutputs<T> detachedView;
        using (new NoGradScope<T>()) detachedView = Forward(u, seed, step, training: true);
        var real = Segment(u, detachedView);
        var generated = Detached(Flat(detachedView.Waveform));

        if (_options.UseDurationDiscriminator)
        {
            int durationSeed = _trainingRandom.Next();
            TrainLayers(_durationDiscriminatorLayers, () =>
            {
                var (r, g) = DurationScores(detachedView, detach: true, AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(durationSeed));
                return DiscriminatorLoss(new[] { r }, new[] { g });
            }, "durationDiscriminator", sample);
        }
        TrainLayers(_discriminators!.Layers, () =>
            DiscriminatorLoss(_discriminators.Forward(real).Select(s => s.Score).ToList(),
                _discriminators.Forward(generated).Select(s => s.Score).ToList()), "discriminator", sample);

        int generatorDurationSeed = _trainingRandom.Next();
        T loss = TrainLayers(_generatorLayers, () => GeneratorLoss(u, seed, step, adversarial: true,
            AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(generatorDurationSeed)), "generator", sample);
        _trainingSteps++;
        return loss;
    }

    // loss_gen_all: mel L1 × c_mel + KL × c_kl + Σ duration loss, plus the adversarial, feature-matching and duration
    // adversarial terms when adversarial.
    private Tensor<T> GeneratorLoss(TrainingUtterance u, int seed, long step, bool adversarial, Random durationRandom)
    {
        var o = Forward(u, seed, step, training: true);
        var audio = Flat(o.Waveform);
        var generatedMel = _spectrogram!.Forward(audio);
        int segmentFrames = Math.Min(u.Frames, _synthesizer!.Configuration.SegmentFrames);
        var targetMel = Engine.TensorSlice(u.Mel, new[] { o.SliceStart, 0 }, new[] { segmentFrames, _options.MelChannels });
        int rows = Math.Min(generatedMel.Shape[0], targetMel.Shape[0]);
        var melLoss = Mean(Engine.TensorAbs(Engine.TensorSubtract(
            Engine.TensorSlice(generatedMel, new[] { 0, 0 }, new[] { rows, _options.MelChannels }),
            Engine.TensorSlice(targetMel, new[] { 0, 0 }, new[] { rows, _options.MelChannels }))));
        var loss = Engine.TensorAdd(Engine.TensorMultiplyScalar(melLoss, NumOps.FromDouble(_options.MelLossWeight)),
            Engine.TensorMultiplyScalar(_synthesizer.KlLoss(o), NumOps.FromDouble(_options.KlLossWeight)));
        loss = Engine.TensorAdd(loss, Engine.ReduceSum(o.DurationLoss, Enumerable.Range(0, o.DurationLoss.Rank).ToArray(), keepDims: false));
        if (!adversarial) return loss;

        var fake = _discriminators!.Forward(audio);
        List<(Tensor<T> Score, List<Tensor<T>> Features)> realScores;
        using (new NoGradScope<T>()) realScores = _discriminators.Forward(Segment(u, o));
        loss = Engine.TensorAdd(loss, GeneratorAdversarialLoss(fake.Select(f => f.Score).ToList()));
        for (int k = 0; k < fake.Count; k++)
            for (int l = 0; l < fake[k].Features.Count; l++)
            {
                // feature_loss: 2 Σ mean |r − g|.
                var r = Detached(realScores[k].Features[l]);
                loss = Engine.TensorAdd(loss, Engine.TensorMultiplyScalar(Mean(Engine.TensorAbs(Engine.TensorSubtract(r, fake[k].Features[l]))),
                    NumOps.FromDouble(2)));
            }
        if (_options.UseDurationDiscriminator)
        {
            var (_, generatedDuration) = DurationScores(o, detach: false, durationRandom);
            loss = Engine.TensorAdd(loss, GeneratorAdversarialLoss(new[] { generatedDuration }));
        }
        return loss;
    }

    private T TrainLayers(IReadOnlyList<LayerBase<T>> layers, Func<Tensor<T>> loss, string group, TtsTrainingSample<T> sample)
    {
        // train.py's clip_grad_value_(…, None) measures the gradients and clips nothing.
        MaxGradNorm = NumOps.Zero;
        _trainingLayers = layers.Cast<ILayer<T>>().ToList();
        try
        {
            return TrainWithCustomObjective(sample.Tokens, sample.Tokens, (_, _) => loss(), OptimizerFor(group));
        }
        finally
        {
            _trainingLayers = null;
        }
    }

    /// <inheritdoc />
    public override T EvaluateTrainingObjective(TtsTrainingSample<T> sample)
    {
        ThrowIfDisposed();
        if (!HasPaperLayers)
            throw new NotSupportedException("MeloTTS's objective needs its own layers; this model was built from caller-supplied layers.");
        var u = PrepareTraining(sample);
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed);
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            // The non-adversarial part: the adversarial terms are measured against discriminators trained at the same time.
            return GeneratorLoss(u, random.Next(), _trainingSteps, adversarial: false, random)[0];
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    /// <inheritdoc />
    /// <remarks>Each step updates only the layers its loss belongs to; BERT is never trained.</remarks>
    protected override IReadOnlyList<Tensor<T>> SelectTrainableParametersForTraining(IReadOnlyList<Tensor<T>> parameters)
    {
        if (!HasPaperLayers) return parameters;
        var layers = _trainingLayers ?? _generatorLayers.Cast<ILayer<T>>().ToList();
        var selected = new HashSet<Tensor<T>>(Training.TapeTrainingStep<T>.CollectParameters(layers.ToList(), -1),
            Helpers.TensorReferenceComparer<Tensor<T>>.Instance);
        return parameters.Where(selected.Contains).ToList();
    }

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> OptimizerFor(string group)
    {
        if (_suppliedOptimizer is not null) return _suppliedOptimizer;
        if (!_optimizers.TryGetValue(group, out var optimizer))
        {
            var o = _options;
            int perEpoch = Math.Max(1, o.UpdatesPerEpoch);
            var scheduler = new AiDotNet.LearningRateSchedulers.LambdaLRScheduler(o.LearningRate, s => Math.Pow(o.LearningRateDecay, s / perEpoch));
            optimizer = PaperOptimizerFactory.VerifyHandBuilt(this, new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this,
                new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
                {
                    InitialLearningRate = o.LearningRate,
                    Beta1 = o.Beta1,
                    Beta2 = o.Beta2,
                    Epsilon = o.Epsilon,
                    WeightDecay = o.WeightDecay,
                    LearningRateScheduler = scheduler,
                    SchedulerStepMode = AiDotNet.LearningRateSchedulers.SchedulerStepMode.StepPerBatch,
                }), group);
            _optimizers[group] = optimizer;
        }
        return optimizer;
    }

    /// <inheritdoc />
    /// <remarks>The paper optimizers read the options when they are built, so restored options rebuild them.</remarks>
    protected override void OnMutableConstructorConfigurationRestored()
    {
        base.OnMutableConstructorConfigurationRestored();
        _optimizers.Clear();
    }

    // ---------------------------------------------------------------- weights

    /// <summary>
    /// Loads a released MeloTTS checkpoint's generator (<c>checkpoint.pth</c>'s <c>model</c> state dictionary, or that
    /// dictionary saved as safetensors). Build the model with the checkpoint's language and sizes.
    /// </summary>
    public void LoadCheckpoint(string path)
    {
        var file = new AiDotNet.ComputerVision.Weights.WeightLoader().LoadWeights(path);
        Synthesizer.LoadTorch(name =>
            file.TryGetValue(name, out var t) || file.TryGetValue("model." + name, out t) ? t.ToVector().Select(v => (double)v).ToArray() : null);
    }

    /// <summary>
    /// Loads the language's BERT (its Hugging Face <c>BertForMaskedLM</c> or <c>BertModel</c> weights, e.g.
    /// bert-base-uncased for English; see <see cref="MeloLanguage"/>).
    /// </summary>
    public void LoadBertWeights(string path)
    {
        var file = new AiDotNet.ComputerVision.Weights.WeightLoader().LoadWeights(path);
        Bert.LoadTorchWeights((name, shape) =>
        {
            if (!file.TryGetValue("bert." + name, out var tensor) && !file.TryGetValue(name, out tensor))
                throw new InvalidDataException($"The BERT checkpoint has no tensor '{name}'.");
            if (!tensor.Shape.ToArray().SequenceEqual(shape))
                throw new InvalidDataException($"'{name}' is [{string.Join(", ", tensor.Shape.ToArray())}] in the checkpoint but [{string.Join(", ", shape)}] here.");
            return tensor.ToVector().Select(v => (double)v).ToArray();
        });
    }

    /// <summary>
    /// Loads a MeloTTS multi-period discriminator checkpoint (<c>D_*.pth</c>'s <c>model</c> state dictionary), which
    /// MeloTTS's fine-tuning starts from alongside the generator.
    /// </summary>
    public void LoadDiscriminatorCheckpoint(string path)
    {
        var file = new AiDotNet.ComputerVision.Weights.WeightLoader().LoadWeights(path);
        (_discriminators ?? throw new InvalidOperationException("MeloTTS's discriminators exist in native mode only.")).LoadTorch(name =>
            file.TryGetValue(name, out var t) || file.TryGetValue("model." + name, out t) ? t.ToVector().Select(v => (double)v).ToArray() : null);
    }

    /// <summary>Loads a MeloTTS duration discriminator checkpoint (<c>DUR_*.pth</c>'s <c>model</c> state dictionary).</summary>
    public void LoadDurationDiscriminatorCheckpoint(string path)
    {
        var file = new AiDotNet.ComputerVision.Weights.WeightLoader().LoadWeights(path);
        (_durationDiscriminator ?? throw new InvalidOperationException("MeloTTS's duration discriminator exists in native mode only.")).LoadTorch(name =>
            file.TryGetValue(name, out var t) || file.TryGetValue("model." + name, out t)
                ? t.ToVector().Select(v => (double)v).ToArray()
                : throw new InvalidDataException($"The duration discriminator checkpoint has no tensor '{name}'."), "");
    }

    // ---------------------------------------------------------------- housekeeping

    /// <inheritdoc />
    protected override bool SupportsParameterMutation => _useNativeMode;

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "MeloTTS-Native" : "MeloTTS-ONNX",
            Description = "MeloTTS: multilingual Bert-VITS2 text-to-speech (MyShell)",
            FeatureCount = _options.HiddenDim,
        };
        m.AdditionalInfo["Language"] = _options.Language.ToString();
        m.AdditionalInfo["SampleRate"] = _options.SampleRate.ToString();
        return m;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(MeloTTS<T>));
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
