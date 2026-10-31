using AiDotNet.Audio.Resampling;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.FrontEnd;

namespace AiDotNet.TextToSpeech.CodecBased;

public partial class GPTSoVITS<T>
{
    private readonly Dictionary<string, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>> _optimizers = new();
    private List<ILayer<T>>? _trainingLayers;
    private Random? _trainingRandom;

    /// <inheritdoc />
    /// <remarks>GPT-SoVITS trains on recordings: the recording at <see cref="TtsModelOptions.SampleRate"/>
    /// (<see cref="TtsTrainingSample{T}.Audio"/>) and its transcript (<see cref="TtsTrainingSample{T}.Text"/>, in
    /// <see cref="GPTSoVITSOptions.TextLanguage"/>; without it, <see cref="TtsTrainingSample{T}.Tokens"/> are the phone
    /// ids, read with zero BERT features).</remarks>
    protected override TtsSupervision RequiredSupervision => TtsSupervision.Recording;

    /// <inheritdoc />
    public override bool TrainsOnSampleObjective => true;

    /// <inheritdoc />
    protected internal override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? TrainingOptimizer =>
        HasPaperLayers ? OptimizerFor(_options.CurrentStage switch
        {
            GptSoVitsStage.TextToSemantic => "t2s",
            GptSoVitsStage.SoVits => "sovits",
            _ => "cfm",
        }) : null;

    /// <inheritdoc />
    protected override bool TryApplyLearningRule(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        Train(new TtsTrainingSample<T> { Tokens = input, Audio = expectedOutput });
        return true;
    }

    // A training recording's features, as the release's dataset preparation writes them (prepare_datasets/1 to 3).
    private sealed class Recording
    {
        public int[] Phones = Array.Empty<int>();
        public double[][] Bert = Array.Empty<double[]>();
        public Tensor<T> Audio = null!;       // the loudness-normalized 32 kHz recording (5-wav32k) [samples]
        public Tensor<T> Ssl = null!;         // chinese-hubert features, padded to an even length [1, 768, T]
        public int[] Semantic = Array.Empty<int>();
        public Tensor<T> Spectrogram = null!; // [1, 1025, frames]
        public Tensor<T> Mel = null!;         // the GAN stage's 128-band log-mel [frames, 128]; the flow stage's normalized mel [1, 100, M]
        public Tensor<T>? SpeakerEmbedding;   // v2Pro

        public Recording WithoutText() => new()
        {
            Audio = Audio, Ssl = Ssl, Semantic = Semantic, Spectrogram = Spectrogram, Mel = Mel, SpeakerEmbedding = SpeakerEmbedding,
        };
    }

    private readonly Dictionary<Tensor<T>, Recording> _recordings = new(TensorReferenceComparer<Tensor<T>>.Instance);

    /// <summary>The release's dataset language code for <see cref="GPTSoVITSOptions.TextLanguage"/> (each line of its
    /// list file names one).</summary>
    private string DatasetLanguage => _options.TextLanguage switch
    {
        GptSoVitsTextLanguage.Chinese or GptSoVitsTextLanguage.ChineseEnglish => "zh",
        GptSoVitsTextLanguage.Japanese or GptSoVitsTextLanguage.JapaneseEnglish => "ja",
        GptSoVitsTextLanguage.English => "en",
        GptSoVitsTextLanguage.Korean or GptSoVitsTextLanguage.KoreanEnglish => "ko",
        GptSoVitsTextLanguage.Cantonese or GptSoVitsTextLanguage.CantoneseEnglish => "yue",
        _ => throw new InvalidOperationException("Training transcripts name one language; set TextLanguage to a single language."),
    };

    private Recording RecordingOf(TtsTrainingSample<T> sample)
    {
        var audio = sample.Audio ?? throw new ArgumentException($"{GetType().Name} trains on recordings; set the sample's Audio.", nameof(sample));
        using var _ = new NoGradScope<T>();
        // The audio's features are cached per recording; the text is read for each sample.
        var r = _recordings.TryGetValue(audio, out var cached) ? cached.WithoutText() : AudioFeatures(audio, sample);
        if (sample.Text is { } text)
        {
            // 1-get-text.py: clean_text of the transcript with % read as - and ￥ as a comma, BERT features for Chinese.
            string language = DatasetLanguage;
            var (phones, wordToPhones, normalized) = Text.CleanText(text.Replace("%", "-").Replace("\uffe5", ","), language);
            r.Phones = Text.Ids(phones);
            r.Bert = language == "zh" ? BertFeatures(normalized, wordToPhones!) : r.Phones.Select(_ => new double[_options.BertHidden]).ToArray();
        }
        else
        {
            int vocabulary = TextToSemantic.Configuration.PhonemeVocab;
            r.Phones = new int[sample.Tokens.Length];
            for (int i = 0; i < r.Phones.Length; i++)
            {
                r.Phones[i] = (int)Math.Round(Ops.ToDouble(sample.Tokens.GetFlat(i)));
                if (r.Phones[i] < 0 || r.Phones[i] >= vocabulary)
                    throw new ArgumentOutOfRangeException(nameof(sample), $"Without Text, the sample's Tokens are phone ids in [0, {vocabulary}).");
            }
            r.Bert = r.Phones.Select(_ => new double[_options.BertHidden]).ToArray();
        }
        if (r.Phones.Length == 0) throw new ArgumentException("The sample has no phones.", nameof(sample));
        return r;
    }

    private Recording AudioFeatures(Tensor<T> audio, TtsTrainingSample<T> sample)
    {
        var r = new Recording();

        // 2-get-hubert-wav32k.py: the recording at 32 kHz, refused above a peak of 2.2, normalized to a 0.95 peak mixed
        // half and half with itself; the int16 copy is the training audio, a copy scaled to 1145.14 is resampled by
        // librosa to 16 kHz for chinese-hubert.
        var a = Doubles(ResampleTorchaudio(Doubles(audio), _options.SampleRate, 32000));
        double peak = a.Length == 0 ? 0 : a.Max(Math.Abs);
        if (peak == 0) throw new ArgumentException("The recording is silent.", nameof(sample));
        if (peak > 2.2) throw new ArgumentException($"The recording peaks at {peak}, above the 2.2 the release filters out.", nameof(sample));
        const double maxx = 0.95, alpha = 0.5;
        var audio32 = a.Select(v => (double)unchecked((short)(int)(v / peak * (maxx * alpha * 32768) + (1 - alpha) * 32768 * v)) / 32768.0).ToArray();
        var hubertInput = a.Select(v => v / peak * (maxx * alpha * 1145.14) + (1 - alpha) * 1145.14 * v).ToArray();
        var ssl = SslFeatures(SoxrResampler.LibrosaResample(hubertInput, 32000, 16000));
        // 3-get-semantic.py: the synthesizer's extract_latent.
        r.Semantic = SemanticTokens(ssl);
        r.Audio = Wave(audio32);
        r.Spectrogram = LinearSpectrogram(r.Audio);
        int frames = r.Spectrogram.Shape[2];
        // The dataset replicates the last SSL frame when one short; the collate pads to 2·(⌊T/2⌋ + 1) with zeros.
        if (ssl.Shape[2] < frames)
            ssl = Engine.TensorConcatenate(new[] { ssl, Engine.TensorNarrow(ssl, 2, ssl.Shape[2] - 1, 1) }, 2);
        int padded = 2 * (ssl.Shape[2] / 2 + 1);
        r.Ssl = Engine.TensorConcatenate(new[] { ssl, new Tensor<T>(new[] { 1, ssl.Shape[1], padded - ssl.Shape[2] }) }, 2);
        if (IsV2Pro) r.SpeakerEmbedding = SpeakerEmbedding(ResampleTorchaudio(audio32, 32000, 16000));                 // 2-get-sv.py
        if (IsCfm)
        {
            var cfm = FlowSynthesizer;
            r.Mel = cfm.ReferenceMel(_options.Version == GptSoVitsVersion.V3 ? ResampleTorchaudio(audio32, 32000, 24000) : r.Audio);
        }
        else
        {
            var magnitude = SpectrogramTransform.Magnitude(r.Audio, 1e-8);
            r.Mel = SpectrogramTransform.LogMel(magnitude);                                                           // spec_to_mel_torch
        }
        if (_recordings.Count > 64) _recordings.Clear();
        _recordings[audio] = r;
        return r.WithoutText();
    }

    /// <inheritdoc />
    /// <remarks>
    /// One step of <see cref="GPTSoVITSOptions.CurrentStage"/>: s1's cross-entropy of the semantic tokens after the
    /// phones (plus the reference-free DPO loss with <see cref="GPTSoVITSOptions.UseDpo"/>), ScaledAdam stepping every
    /// fourth call; s2's GAN step (the multi-period discriminator's LSGAN loss, then the generator's mel L1 × 45, KL, feature
    /// matching and adversarial losses over a 32-frame segment); or the flow-matching synthesizer's loss.
    /// </remarks>
    protected override T TrainOnSample(TtsTrainingSample<T> sample)
    {
        ThrowIfDisposed();
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        if (!HasPaperLayers) throw new NotSupportedException("The objective needs the release's networks; this model was built from caller-supplied layers.");
        var random = _trainingRandom ??= AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.TrainingSeed);
        var r = RecordingOf(sample);
        switch (_options.CurrentStage)
        {
            case GptSoVitsStage.TextToSemantic:
            {
                int seed = random.Next();
                return TrainLayers(_t2sLayers, () => T2SLoss(r, AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed), training: true), "t2s", sample);
            }
            case GptSoVitsStage.FlowMatching:
            {
                int seed = random.Next();
                return TrainLayers(FlowTrainedLayers(), () => FlowLoss(r, AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed), training: true),
                    "cfm", sample);
            }
            default:
                return GanStep(r, sample, random);
        }
    }

    private Tensor<T> T2SLoss(Recording r, Random random, bool training)
    {
        var example = new GptSoVitsT2S<T>.Example(r.Phones, BertTensor(r.Bert), r.Semantic);
        return TextToSemantic.Loss(new[] { example }, _options.UseDpo, training, random);
    }

    private Tensor<T> FlowLoss(Recording r, Random random, bool training) =>
        FlowSynthesizer.Loss(r.Ssl, r.Spectrogram, r.Mel, r.Phones, freezeQuantizer: true, random, training);

    /// <summary>The flow-matching stage's trainable layers: with LoRA (s2_train_v3_lora.py) the adapters plus the style
    /// encoder, bridge and wns1; without (s2_train_v3.py) everything but the frozen SSL projection and text encoder.</summary>
    private IReadOnlyList<LayerBase<T>> FlowTrainedLayers()
    {
        var cfm = FlowSynthesizer;
        return cfm.Cfm.Estimator.HasLora
            ? cfm.AdapterLayers.Concat(_loraLayers).ToList()
            : cfm.AdapterLayers.Concat(cfm.EstimatorLayers).ToList();
    }

    // ---------------------------------------------------------------- the GAN stage

    private int SegmentFrames => _options.SegmentSize / _options.HopSize;

    private GptSoVitsSynthesizer<T>.TrainingPass GanForward(Recording r, int seed, bool training)
    {
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed);
        int frames = r.Spectrogram.Shape[2];
        var noise = new Tensor<T>(new[] { 1, _options.InterChannels, frames });
        for (int i = 0; i < noise.Length; i++) noise.SetFlat(i, Ops.FromDouble(AiDotNet.TextToSpeech.Latest.IndexTtsOps.Gaussian(random)));
        return SoVits.Forward(r.Ssl, r.Spectrogram, r.Phones, r.SpeakerEmbedding, SegmentFrames, freezeQuantizer: true, noise, random);
    }

    // slice_segments of the padded recording: SegmentSize samples from the segment's first frame.
    private Tensor<T> RealSegment(Recording r, int start)
    {
        int from = start * _options.HopSize, count = Math.Min(_options.SegmentSize, r.Audio.Length - from);
        var real = Engine.TensorSlice(r.Audio, new[] { from }, new[] { count });
        return count == _options.SegmentSize ? real : Engine.TensorConcatenate(new[] { real, new Tensor<T>(new[] { _options.SegmentSize - count }) }, 0);
    }

    private Tensor<T> Mean(Tensor<T> x) => Engine.ReduceMean(x, Enumerable.Range(0, x.Rank).ToArray(), keepDims: false);

    private static Tensor<T> Detached(Tensor<T> x) => new(x._shape, x.ToVector());

    // discriminator_loss: Σ mean((1 − D(real))²) + mean(D(fake)²).
    private Tensor<T> DiscriminatorLoss(IReadOnlyList<Tensor<T>> real, IReadOnlyList<Tensor<T>> fake)
    {
        Tensor<T>? loss = null;
        for (int k = 0; k < real.Count; k++)
        {
            var d = Engine.TensorAddScalar(Engine.TensorNegate(real[k]), NumOps.One);
            var term = Engine.TensorAdd(Mean(Engine.TensorMultiply(d, d)), Mean(Engine.TensorMultiply(fake[k], fake[k])));
            loss = loss is null ? term : Engine.TensorAdd(loss, term);
        }
        return loss!;
    }

    private T GanStep(Recording r, TtsTrainingSample<T> sample, Random random)
    {
        int seed = random.Next();
        GptSoVitsSynthesizer<T>.TrainingPass view;
        using (new NoGradScope<T>()) view = GanForward(r, seed, training: true);
        var real = RealSegment(r, view.SliceStart);
        var generated = Detached(view.Audio);
        var discriminators = _discriminators!;
        TrainLayers(discriminators.Layers, () => DiscriminatorLoss(discriminators.Forward(real).Select(s => s.Score).ToList(),
            discriminators.Forward(generated).Select(s => s.Score).ToList()), "discriminator", sample);
        return TrainLayers(GeneratorLayers(), () => GeneratorLoss(r, seed, adversarial: true), "sovits", sample);
    }

    /// <summary>loss_gen_all: the adversarial loss, feature matching, mel L1 × c_mel, the commitment loss and KL × c_kl
    /// (the adversarial terms only when <paramref name="adversarial"/>).</summary>
    private Tensor<T> GeneratorLoss(Recording r, int seed, bool adversarial)
    {
        var o = _options;
        var pass = GanForward(r, seed, training: true);
        var generatedMel = SpectrogramTransform.LogMel(SpectrogramTransform.Magnitude(pass.Audio, 1e-8));        // mel_spectrogram_torch
        int frames = SegmentFrames, available = Math.Min(frames, r.Mel.Shape[0] - pass.SliceStart);
        var targetMel = Engine.TensorSlice(r.Mel, new[] { pass.SliceStart, 0 }, new[] { available, o.MelChannels });
        // slice_segments reads zeros past a short utterance's mel.
        if (available < frames) targetMel = Engine.TensorConcatenate(new[] { targetMel, new Tensor<T>(new[] { frames - available, o.MelChannels }) }, 0);
        int rows = Math.Min(generatedMel.Shape[0], targetMel.Shape[0]);
        var melLoss = Mean(Engine.TensorAbs(Engine.TensorSubtract(Engine.TensorSlice(targetMel, new[] { 0, 0 }, new[] { rows, o.MelChannels }),
            Engine.TensorSlice(generatedMel, new[] { 0, 0 }, new[] { rows, o.MelChannels }))));
        var loss = Engine.TensorAdd(Engine.TensorMultiplyScalar(melLoss, NumOps.FromDouble(o.MelLossWeight)),
            Engine.TensorMultiplyScalar(SoVits.KlLoss(pass), NumOps.FromDouble(o.KlLossWeight)));
        loss = Engine.TensorAdd(loss, Engine.ReduceSum(pass.Commitment, null));
        if (!adversarial) return loss;
        var discriminators = _discriminators!;
        var fake = discriminators.Forward(pass.Audio);
        List<(Tensor<T> Score, List<Tensor<T>> Features)> realScores;
        using (new NoGradScope<T>()) realScores = discriminators.Forward(RealSegment(r, pass.SliceStart));
        for (int k = 0; k < fake.Count; k++)
        {
            // generator_loss: mean((1 − D(fake))²); feature_loss: 2 Σ mean |r − g|.
            var d = Engine.TensorAddScalar(Engine.TensorNegate(fake[k].Score), NumOps.One);
            loss = Engine.TensorAdd(loss, Mean(Engine.TensorMultiply(d, d)));
            for (int l = 0; l < fake[k].Features.Count; l++)
                loss = Engine.TensorAdd(loss, Engine.TensorMultiplyScalar(Mean(Engine.TensorAbs(Engine.TensorSubtract(Detached(realScores[k].Features[l]),
                    fake[k].Features[l]))), NumOps.FromDouble(2)));
        }
        return loss;
    }

    /// <summary>The generator's trainable layers: the synthesizer but the frozen SSL projection (its quantizer, in
    /// evaluation, passes no gradient back to it).</summary>
    private IReadOnlyList<LayerBase<T>> GeneratorLayers() => _synthesizerLayers.Where(l => !ReferenceEquals(l, SoVits.SslProjection)).ToList();

    // ---------------------------------------------------------------- shared

    /// <inheritdoc />
    /// <remarks>The current stage's objective at a fixed draw (the training seed's first): for the GAN stage, its
    /// non-adversarial part, as the adversarial terms are measured against discriminators trained alongside.</remarks>
    public override T EvaluateTrainingObjective(TtsTrainingSample<T> sample)
    {
        ThrowIfDisposed();
        Validation.Guard.NotNull(sample);
        if (!HasPaperLayers) throw new NotSupportedException("The objective needs the release's networks; this model was built from caller-supplied layers.");
        var r = RecordingOf(sample);
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.TrainingSeed);
            return _options.CurrentStage switch
            {
                GptSoVitsStage.TextToSemantic => T2SLoss(r, random, training: false)[0],
                GptSoVitsStage.FlowMatching => FlowLoss(r, random, training: false)[0],
                _ => GeneratorLoss(r, random.Next(), adversarial: false)[0],
            };
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    private T TrainLayers(IReadOnlyList<LayerBase<T>> layers, Func<Tensor<T>> loss, string group, TtsTrainingSample<T> sample)
    {
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
    /// <remarks>Each step updates only its stage's layers; chinese-hubert, chinese-roberta, ERes2NetV2 and the vocoder are
    /// frozen.</remarks>
    protected override IReadOnlyList<Tensor<T>> SelectTrainableParametersForTraining(IReadOnlyList<Tensor<T>> parameters)
    {
        if (!HasPaperLayers) return parameters;
        var layers = _trainingLayers ?? (_options.CurrentStage switch
        {
            GptSoVitsStage.TextToSemantic => _t2sLayers,
            GptSoVitsStage.FlowMatching => FlowTrainedLayers(),
            _ => GeneratorLayers(),
        }).Cast<ILayer<T>>().ToList();
        var selected = new HashSet<Tensor<T>>(Training.TapeTrainingStep<T>.CollectParameters(layers, -1), TensorReferenceComparer<Tensor<T>>.Instance);
        return parameters.Where(selected.Contains).ToList();
    }

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> OptimizerFor(string group)
    {
        if (_optimizers.TryGetValue(group, out var cached)) return cached;
        var o = _options;
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> optimizer;
        if (group == "t2s")
        {
            // WarmupCosineLRSchedule never sets the rate before its first step and locks it at 0.002 after.
            ScaledAdamOptimizer<T, Tensor<T>, Tensor<T>>? built = null;
            double locked = o.T2SLockedLearningRate / o.T2SLearningRate;
            built = new ScaledAdamOptimizer<T, Tensor<T>, Tensor<T>>(this, new ScaledAdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = o.T2SLearningRate, Beta1 = 0.9, Beta2 = 0.95, ClippingScale = 2.0, ClippingUpdatePeriod = 1000,
                GradientAccumulationPeriod = o.T2SAccumulation,
                LearningRateScheduler = new AiDotNet.LearningRateSchedulers.LambdaLRScheduler(o.T2SLearningRate, _ => built!.StepCount == 0 ? 1.0 : locked),
                SchedulerStepMode = AiDotNet.LearningRateSchedulers.SchedulerStepMode.StepPerBatch,
            });
            optimizer = PaperOptimizerFactory.VerifyHandBuilt(this, built, group);
        }
        else
        {
            var layers = group switch
            {
                "sovits" => GeneratorLayers(),
                "discriminator" => _discriminators!.Layers,
                _ => FlowTrainedLayers(),
            };
            var parameters = Training.TapeTrainingStep<T>.CollectParameters(layers.Cast<ILayer<T>>().ToList(), -1).ToList();
            Vector<T>? rateMask = null;
            if (group == "sovits")
            {
                // s2_train.py's parameter groups: text_embedding, encoder_text and MRTE at text_low_lr_rate.
                var low = new HashSet<Tensor<T>>(Training.TapeTrainingStep<T>.CollectParameters(SoVits.LowRateLayers.Cast<ILayer<T>>().ToList(), -1),
                    TensorReferenceComparer<Tensor<T>>.Instance);
                rateMask = new Vector<T>(parameters.SelectMany(p => Enumerable.Repeat(Ops.FromDouble(low.Contains(p) ? o.TextLowLearningRate : 1.0), p.Length)).ToArray());
            }
            int? perEpoch = o.UpdatesPerEpoch;
            optimizer = PaperOptimizerFactory.VerifyHandBuilt(this, new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this,
                new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
                {
                    InitialLearningRate = o.SynthesizerLearningRate, Beta1 = o.Beta1, Beta2 = o.Beta2, Epsilon = o.Epsilon, WeightDecay = o.WeightDecay,
                    MaskParameters = parameters, LearningRateMask = rateMask,
                    LearningRateScheduler = new AiDotNet.LearningRateSchedulers.LambdaLRScheduler(o.SynthesizerLearningRate,
                        s => perEpoch is int n && n > 0 ? Math.Pow(o.LearningRateDecay, s / n) : 1.0),
                    SchedulerStepMode = AiDotNet.LearningRateSchedulers.SchedulerStepMode.StepPerBatch,
                }), group);
        }
        _optimizers[group] = optimizer;
        return optimizer;
    }

    /// <inheritdoc />
    /// <remarks>The optimizers read the options when they are built, so restored options rebuild them.</remarks>
    protected override void OnMutableConstructorConfigurationRestored()
    {
        base.OnMutableConstructorConfigurationRestored();
        _optimizers.Clear();
    }
}
