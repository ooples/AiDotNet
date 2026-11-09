using AiDotNet.Audio.Codecs;
using AiDotNet.Audio.Pitch;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.FlowDiffusion;
using AiDotNet.TextToSpeech.Vocoders;

namespace AiDotNet.TextToSpeech.CodecBased;

public partial class NaturalSpeech3<T>
{
    private readonly Dictionary<string, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>> _optimizers = new();
    private List<ILayer<T>>? _trainingLayers;
    private Random? _trainingRandom;
    private HiFiGanDiscriminators<T>? _periods;
    private readonly List<DacBandDiscriminator<T>> _bands = new();
    private readonly List<LayerBase<T>> _bandLayers = new();
    private List<(CenteredComplexStft<T> Stft, Tensor<T> Filterbank)>? _melScales;

    private IEnumerable<LayerBase<T>> DiscriminatorLayers => (_periods?.Layers ?? Array.Empty<LayerBase<T>>()).Concat(_bandLayers);

    /// <inheritdoc />
    /// <remarks>The recording and its phones' durations in codec frames (Codec stage: also the speaker's index for the
    /// speaker-reversal head; the pitch is extracted with WORLD when absent).</remarks>
    protected override TtsSupervision RequiredSupervision => TtsSupervision.Durations | TtsSupervision.Recording;

    /// <inheritdoc />
    public override bool TrainsOnSampleObjective => true;

    /// <inheritdoc />
    protected internal override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? TrainingOptimizer => HasPaperLayers ? OptimizerFor(GeneratorGroup) : null;

    private string GeneratorGroup => _options.CurrentStage switch
    {
        NaturalSpeech3Stage.Codec => "codec",
        NaturalSpeech3Stage.Redecoder => "redecoder",
        _ => "diffusion",
    };

    /// <inheritdoc />
    protected override bool TryApplyLearningRule(Tensor<T> input, Tensor<T> expectedOutput)
    {
        // A token/audio pair does not carry the durations this model trains on; Train(TtsTrainingSample) does.
        ThrowIfTokenMelTrainingUnsupported();
        return false;
    }

    private void BuildDiscriminators()
    {
        var o = _options;
        _periods = new HiFiGanDiscriminators<T>(Engine, o.DiscriminatorPeriods, 0, useScaleDiscriminator: false, widthDivisor: o.DiscriminatorWidthDivisor);
        foreach (int w in o.DiscriminatorWindows)
            _bands.Add(new DacBandDiscriminator<T>(Engine, w, new[] { 0.0, 0.1, 0.25, 0.5, 0.75, 1.0 }, Math.Max(1, 32 / o.DiscriminatorWidthDivisor), _bandLayers));
        if (o.MelWindows.Length != o.MelBins.Length) throw new ArgumentException("Every mel window needs a mel bin count.");
        _melScales = new List<(CenteredComplexStft<T>, Tensor<T>)>();
        for (int i = 0; i < o.MelWindows.Length; i++)
        {
            int w = o.MelWindows[i];
            var stft = new CenteredComplexStft<T>(Engine, w, w / 4, w);
            var basis = new TacotronSpectrogram(o.SampleRate, w, w / 4, w, o.MelBins[i], 0.0, o.SampleRate / 2.0).MelBasis;
            var fb = new Tensor<T>(new[] { stft.Bins, o.MelBins[i] });
            for (int m = 0; m < o.MelBins[i]; m++)
                for (int k = 0; k < stft.Bins; k++) fb[k, m] = Ops.FromDouble(basis[m, k]);
            _melScales.Add((stft, fb));
        }
    }

    /// <inheritdoc />
    protected override T TrainOnSample(TtsTrainingSample<T> sample)
    {
        ThrowIfDisposed();
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        if (!HasPaperLayers) throw NotNative();
        _trainingRandom ??= AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(SeedValue);
        int seed = _trainingRandom.Next();
        SetTrainingMode(true);
        switch (_options.CurrentStage)
        {
            case NaturalSpeech3Stage.Diffusion:
            {
                var u = PrepareDiffusion(sample);
                return TrainLayers(_diffusionLayers, () => DiffusionLoss(u, Seeded(seed), training: true), "diffusion", sample);
            }
            default:
            {
                bool redecoder = _options.CurrentStage == NaturalSpeech3Stage.Redecoder;
                if (redecoder && Codec.RedecoderEmbeddings is null) throw new InvalidOperationException("Build the model with IncludeRedecoder to train the redecoder.");
                var u = PrepareCodec(sample);
                Tensor<T> generated;
                using (new NoGradScope<T>()) generated = Detached(redecoder ? Redecoded(u) : CodecForward(u, Seeded(seed), training: true).Audio);
                TrainLayers(DiscriminatorLayers.ToList(), () => DiscriminatorLoss(u.Audio, generated), "discriminator", sample);
                return redecoder
                    ? TrainLayers(_redecoderLayers, () => RedecoderLoss(u), "redecoder", sample)
                    : TrainLayers(_codecEncoderLayers.Concat(_codecDecoderLayers).Concat(_codecHeadLayers).ToList(),
                        () => CodecLoss(u, Seeded(seed)), "codec", sample);
            }
        }
    }

    /// <inheritdoc />
    public override T EvaluateTrainingObjective(TtsTrainingSample<T> sample)
    {
        ThrowIfDisposed();
        if (!HasPaperLayers) throw NotNative();
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            var random = Seeded(SeedValue);
            return _options.CurrentStage switch
            {
                NaturalSpeech3Stage.Diffusion => DiffusionLoss(PrepareDiffusion(sample), random, training: false)[0],
                NaturalSpeech3Stage.Redecoder => MelLoss(PrepareCodec(sample).Audio, Redecoded(PrepareCodec(sample)))[0],
                _ => CodecObjective(PrepareCodec(sample), random)[0],
            };
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    private static Random Seeded(int seed) => AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed);

    private static Tensor<T> Detached(Tensor<T> x) => new(x._shape, x.ToVector());

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
    protected override IReadOnlyList<Tensor<T>> SelectTrainableParametersForTraining(IReadOnlyList<Tensor<T>> parameters)
    {
        if (!HasPaperLayers) return parameters;
        var layers = _trainingLayers ?? _diffusionLayers.Cast<ILayer<T>>().ToList();
        var selected = new HashSet<Tensor<T>>(Training.TapeTrainingStep<T>.CollectParameters(layers.ToList(), -1), TensorReferenceComparer<Tensor<T>>.Instance);
        return parameters.Where(selected.Contains).ToList();
    }

    private Tensor<T> Mean(Tensor<T> x) => Engine.TensorMultiplyScalar(Engine.ReduceSum(x, null), Ops.FromDouble(1.0 / x.Length));

    private Tensor<T> Sum(IEnumerable<Tensor<T>> terms) => terms.Aggregate(Engine.TensorAdd);

    private int[] PhonemesOf(TtsTrainingSample<T> sample) =>
        sample.Text is { } text ? TextToPhonemes(text) : PhonemeIds(sample.Tokens, nameof(sample));

    // ---------------------------------------------------------------- the codec (App. B.1)

    // A training clip: the waveform, its frame-level phone labels, normalized F0 and voicing, and the speaker.
    private sealed record CodecUtterance(Tensor<T> Audio, int[] Phones, double[] F0, bool[] Voiced, int? Speaker, Tensor<T>? ProsodyFeature);

    private CodecUtterance PrepareCodec(TtsTrainingSample<T> sample)
    {
        var o = _options;
        var recording = sample.Audio ?? throw new ArgumentException($"{GetType().Name} trains on recordings; set the sample's Audio.", nameof(sample));
        var phonemes = PhonemesOf(sample);
        var durations = (int[])(sample.Durations ?? throw new ArgumentException("FACodec's content supervision reads frame-level phones; set Durations.", nameof(sample))).Clone();
        if (durations.Length != phonemes.Length) throw new ArgumentException($"{durations.Length} durations for {phonemes.Length} phonemes.", nameof(sample));
        int frames = recording.Length / o.HopSize;
        if (frames < 1) throw new ArgumentException("The recording is shorter than one codec frame.", nameof(sample));
        Align(durations, frames);
        var phones = new int[frames];
        int frame = 0;
        for (int j = 0; j < durations.Length; j++)
            for (int r = 0; r < durations[j] && frame < frames; r++) phones[frame++] = phonemes[j];
        for (; frame < frames; frame++) phones[frame] = phonemes[^1];
        var audio = Engine.TensorSlice(Engine.Reshape(recording, new[] { recording.Length }), new[] { 0 }, new[] { frames * o.HopSize });
        var pitch = sample.Pitch is { } given ? given : new WorldPitchDetector<T>(o.SampleRate).EstimateF0(audio, 1000.0 * o.HopSize / o.SampleRate).F0;
        var f0 = Enumerable.Range(0, frames).Select(f => f < pitch.Length ? pitch[f] : 0.0).ToArray();
        var voiced = f0.Select(v => v > 0).ToArray();
        // The z-score of F0 over the voiced frames (§3.2.2's normalized F0).
        var values = f0.Where(v => v > 0).ToArray();
        double mean = values.Length > 0 ? values.Average() : 0, std = values.Length > 1 ? Math.Sqrt(values.Sum(v => (v - mean) * (v - mean)) / values.Length) : 1;
        var normalized = f0.Select(v => v > 0 ? (v - mean) / Math.Max(std, 1e-8) : 0.0).ToArray();
        var prosody = o.MelProsody ? Codec.ProsodyFeature(Doubles(audio)) : null;
        return new CodecUtterance(audio, phones, normalized, voiced, sample.SpeakerId, prosody);
    }

    private sealed record CodecPass(Tensor<T> Audio, Tensor<T> Auxiliary);

    /// <summary>
    /// The codec's training forward (the release's <c>forward</c>): factors, the supervision and reversal heads' losses, the
    /// decoder's input stop(z_p) + stop(z_c) + m·z_d with the detail dropout's mask m, the waveform.
    /// </summary>
    private CodecPass CodecForward(CodecUtterance u, Random random, bool training)
    {
        var o = _options;
        var codec = Codec;
        foreach (var q in codec.Quantizers) q.SetTrainingMode(training);
        var latent = codec.Encoder.Forward(AsWave(u.Audio));
        var f = codec.Factorize(latent, u.ProsodyFeature, training, random);
        var terms = new List<Tensor<T>>();
        foreach (var q in codec.Quantizers)
        {
            if (q.CodebookLoss is { } book) terms.Add(Engine.TensorMultiplyScalar(book, Ops.FromDouble(o.CodebookWeight)));
            if (q.CommitmentLoss is { } commit) terms.Add(Engine.TensorMultiplyScalar(commit, Ops.FromDouble(o.CommitmentWeight)));
        }
        Tensor<T> Weighted(Tensor<T> loss, double weight) => Engine.TensorMultiplyScalar(loss, Ops.FromDouble(weight));
        Tensor<T> Reversed(Tensor<T> x) => FACodecWeights.Reverse(Engine, x, 1.0);
        var (zp, zc, zd) = (f.Groups[0], f.Groups[1], f.Groups[2]);
        terms.Add(Weighted(F0Loss(codec.F0Predictor.Forward(zp), u), o.F0Weight));
        terms.Add(Weighted(PhoneLoss(codec.PhonePredictor.Forward(zc)[0], u.Phones), o.PhoneWeight));
        if (codec.DetailF0Predictor is not null) terms.Add(Weighted(F0Loss(codec.DetailF0Predictor.Forward(Reversed(zd)), u), o.F0ReversalWeight));
        if (codec.DetailPhonePredictor is not null) terms.Add(Weighted(PhoneLoss(codec.DetailPhonePredictor.Forward(Reversed(zd))[0], u.Phones), o.PhoneReversalWeight));
        if (codec.ContentF0Predictor is not null) terms.Add(Weighted(F0Loss(codec.ContentF0Predictor.Forward(Reversed(zc)), u), o.F0ReversalWeight));
        if (codec.ProsodyPhonePredictor is not null) terms.Add(Weighted(PhoneLoss(codec.ProsodyPhonePredictor.Forward(Reversed(zp))[0], u.Phones), o.PhoneReversalWeight));

        var known = o.DetachProsodyContent ? Engine.TensorAdd(Ns2Ops.Detach(zp), Ns2Ops.Detach(zc)) : Engine.TensorAdd(zp, zc);
        bool keepDetail = !training || random.NextDouble() >= o.DetailDropout;
        var x = keepDetail ? Engine.TensorAdd(known, zd) : known;
        if (codec.TimbrePredictor is not null)
        {
            int speaker = u.Speaker ?? throw new ArgumentException("The speaker-reversal head classifies the speaker; set the sample's SpeakerId (or turn TimbreReversal off).");
            terms.Add(Weighted(PhoneLoss(codec.TimbrePredictor.Forward(Reversed(x))[0], new[] { speaker }), o.SpeakerReversalWeight));
        }
        var audio = codec.Decode(x, f.Speaker);
        return new CodecPass(Engine.Reshape(audio, new[] { audio.Length }), Sum(terms));
    }

    // L_f0: L1 on the normalized F0 of voiced frames and the voicing's binary cross-entropy (the release's f0 and uv heads;
    // the paper names only the F0 loss).
    private Tensor<T> F0Loss(List<Tensor<T>> heads, CodecUtterance u)
    {
        int frames = u.F0.Length;
        var voicedRows = Enumerable.Range(0, frames).Where(i => u.Voiced[i]).ToArray();
        var terms = new List<Tensor<T>>();
        if (voicedRows.Length > 0)
        {
            var predicted = Engine.TensorIndexSelect(Engine.Reshape(heads[0], new[] { frames, 1 }), new Tensor<int>(new[] { voicedRows.Length }, new Vector<int>(voicedRows)), 0);
            var target = new Tensor<T>(new[] { voicedRows.Length, 1 }, new Vector<T>(voicedRows.Select(i => Ops.FromDouble(u.F0[i])).ToArray()));
            terms.Add(Mean(Engine.TensorAbs(Engine.TensorSubtract(predicted, target))));
        }
        var logits = Engine.Reshape(heads[1], new[] { frames });
        var labels = new Tensor<T>(new[] { frames }, new Vector<T>(u.Voiced.Select(v => v ? Ops.One : Ops.Zero).ToArray()));
        // BCE with logits: softplus(x) − y·x.
        terms.Add(Mean(Engine.TensorSubtract(Engine.Softplus(logits), Engine.TensorMultiply(labels, logits))));
        return Sum(terms);
    }

    // Cross-entropy of logits [rows, classes] against one target per row.
    private Tensor<T> PhoneLoss(Tensor<T> logits, IReadOnlyList<int> targets)
    {
        int n = logits.Shape[0], v = logits.Shape[1];
        var picks = Enumerable.Range(0, n).Select(i => i * v + Math.Min(targets[Math.Min(i, targets.Count - 1)], v - 1)).ToArray();
        var logProbabilities = Engine.Reshape(Engine.TensorLogSoftmax(logits, axis: 1), new[] { n * v, 1 });
        return Engine.TensorMultiplyScalar(Engine.ReduceSum(Engine.TensorIndexSelect(logProbabilities, new Tensor<int>(new[] { n }, new Vector<int>(picks)), 0), null),
            Ops.FromDouble(-1.0 / n));
    }

    private Tensor<T> CodecObjective(CodecUtterance u, Random random)
    {
        var pass = CodecForward(u, random, training: false);
        return Engine.TensorAdd(Engine.TensorMultiplyScalar(MelLoss(u.Audio, pass.Audio), Ops.FromDouble(_options.ReconstructionWeight)), pass.Auxiliary);
    }

    /// <summary>The generator's loss (App. B.1): λ_rec L_rec + λ_adv L_adv + λ_feat L_feat + the codebook, commitment,
    /// supervision and gradient-reversal terms.</summary>
    private Tensor<T> CodecLoss(CodecUtterance u, Random random)
    {
        var o = _options;
        var pass = CodecForward(u, random, training: true);
        return Sum(new[]
        {
            Engine.TensorMultiplyScalar(MelLoss(u.Audio, pass.Audio), Ops.FromDouble(o.ReconstructionWeight)),
            AdversarialLoss(u.Audio, pass.Audio),
            pass.Auxiliary,
        });
    }

    private Tensor<T> Redecoded(CodecUtterance u)
    {
        FACodecFactors<T> f;
        using (new NoGradScope<T>())
        {
            foreach (var q in Codec.Quantizers) q.SetTrainingMode(false);
            f = Codec.Factorize(Codec.Encoder.Forward(AsWave(u.Audio)), u.ProsodyFeature, false, null);
        }
        var speaker = Ns2Ops.Detach(f.Speaker);
        var wave = Codec.Redecode(Codec.RedecoderEmbed(f.Codes, speaker, detail: true, training: IsTrainingMode), speaker);
        return Engine.Reshape(wave, new[] { wave.Length });
    }

    // The redecoder reconstructs a frozen codec's codes in their own timbre with the codec's reconstruction and adversarial
    // losses (the release does not show how it was trained).
    private Tensor<T> RedecoderLoss(CodecUtterance u)
    {
        var generated = Redecoded(u);
        return Engine.TensorAdd(Engine.TensorMultiplyScalar(MelLoss(u.Audio, generated), Ops.FromDouble(_options.ReconstructionWeight)), AdversarialLoss(u.Audio, generated));
    }

    // DAC's discriminator input: centred and peak-normalized to 0.8.
    private Tensor<T> Normalize(Tensor<T> audio)
    {
        var mean = Mean(audio);
        var centred = Engine.TensorSubtract(audio, Engine.TensorBroadcastTo(Engine.Reshape(mean, new[] { 1 }), audio._shape));
        var peak = Engine.ReduceMax(Engine.TensorAbs(centred), new[] { 0 }, keepDims: true);
        return Engine.TensorMultiplyScalar(Engine.TensorDivide(centred, Engine.TensorBroadcastTo(Engine.TensorAddScalar(peak, Ops.FromDouble(1e-9)), centred._shape)),
            Ops.FromDouble(0.8));
    }

    private List<(Tensor<T> Logits, List<Tensor<T>> Features)> Discriminate(Tensor<T> audio)
    {
        var x = Normalize(audio);
        var outputs = _periods!.Forward(x).Select(p => (p.Score, p.Features)).ToList();
        foreach (var band in _bands) outputs.Add(band.Forward(x));
        return outputs;
    }

    // λ_adv · hinge (DAC's) + λ_feat · the relative feature matching (EnCodec's: ‖D(x) − D(x̂)‖₁ / mean ‖D(x)‖₁ per layer).
    private Tensor<T> AdversarialLoss(Tensor<T> real, Tensor<T> generated)
    {
        var o = _options;
        var fake = Discriminate(generated);
        List<(Tensor<T> Logits, List<Tensor<T>> Features)> reference;
        using (new NoGradScope<T>()) reference = Discriminate(real);
        var adversarial = Sum(fake.Select(f => Engine.TensorNegate(Mean(f.Logits))));
        var features = new List<Tensor<T>>();
        for (int k = 0; k < fake.Count; k++)
            for (int l = 0; l < fake[k].Features.Count - 1; l++)
            {
                var r = Detached(reference[k].Features[l]);
                double scale = r.ToArray().Average(v => Math.Abs(Ops.ToDouble(v)));
                features.Add(Engine.TensorMultiplyScalar(Mean(Engine.TensorAbs(Engine.TensorSubtract(fake[k].Features[l], r))), Ops.FromDouble(1.0 / Math.Max(scale, 1e-8))));
            }
        var feature = features.Count == 0 ? new Tensor<T>(new[] { 1 }) : Engine.TensorMultiplyScalar(Sum(features), Ops.FromDouble(1.0 / features.Count));
        return Engine.TensorAdd(Engine.TensorMultiplyScalar(adversarial, Ops.FromDouble(o.AdversarialWeight)), Engine.TensorMultiplyScalar(feature, Ops.FromDouble(o.FeatureWeight)));
    }

    // The hinge loss Σ_k mean(max(0, 1 − D_k(x))) + mean(max(0, 1 + D_k(x̂))).
    private Tensor<T> DiscriminatorLoss(Tensor<T> real, Tensor<T> generated)
    {
        var r = Discriminate(real);
        var g = Discriminate(generated);
        return Sum(Enumerable.Range(0, r.Count).Select(k => Engine.TensorAdd(
            Mean(Engine.ReLU(Engine.TensorAddScalar(Engine.TensorNegate(r[k].Logits), Ops.One))),
            Mean(Engine.ReLU(Engine.TensorAddScalar(g[k].Logits, Ops.One))))));
    }

    // L_rec (DAC's multi-scale mel distance): Σ over windows of mean |log10 max(mel(x), 1e-5) − log10 max(mel(x̂), 1e-5)|.
    private Tensor<T> MelLoss(Tensor<T> real, Tensor<T> generated)
    {
        int n = Math.Min(real.Length, generated.Length);
        var x = Engine.TensorSlice(Engine.Reshape(real, new[] { real.Length }), new[] { 0 }, new[] { n });
        var y = Engine.TensorSlice(Engine.Reshape(generated, new[] { generated.Length }), new[] { 0 }, new[] { n });
        var terms = new List<Tensor<T>>();
        foreach (var (stft, filterbank) in _melScales!)
        {
            Tensor<T> target;
            using (new NoGradScope<T>()) target = Detached(LogMel(stft, filterbank, x));
            terms.Add(Mean(Engine.TensorAbs(Engine.TensorSubtract(target, LogMel(stft, filterbank, y)))));
        }
        return Sum(terms);
    }

    private Tensor<T> LogMel(CenteredComplexStft<T> stft, Tensor<T> filterbank, Tensor<T> audio)
    {
        var (re, im) = stft.Forward(audio);
        var magnitude = Engine.TensorPow(Engine.TensorAddScalar(Engine.TensorAdd(Engine.TensorMultiply(re, re), Engine.TensorMultiply(im, im)), Ops.FromDouble(1e-12)),
            Ops.FromDouble(0.5));
        var rows = Engine.TensorTranspose(Engine.Reshape(magnitude, new[] { magnitude.Shape[1], magnitude.Shape[2] }));
        var mel = Engine.TensorMatMul(rows, filterbank);
        var clamped = Engine.TensorAddScalar(Engine.ReLU(Engine.TensorAddScalar(mel, Ops.FromDouble(-1e-5))), Ops.FromDouble(1e-5));
        return Engine.TensorMultiplyScalar(Engine.TensorLog(clamped), Ops.FromDouble(1.0 / Math.Log(10.0)));
    }

    // ---------------------------------------------------------------- the factorized diffusion (§3.3)

    // An utterance's phones, durations (aligned to its codes) and codes [frames, codebooks].
    private sealed record DiffusionUtterance(int[] Phonemes, int[] Durations, int[,] Codes);

    private DiffusionUtterance PrepareDiffusion(TtsTrainingSample<T> sample)
    {
        var phonemes = PhonemesOf(sample);
        var durations = (int[])(sample.Durations ?? throw new ArgumentException($"{GetType().Name} trains with each phoneme's duration in codec frames; set Durations.", nameof(sample))).Clone();
        if (durations.Length != phonemes.Length) throw new ArgumentException($"{durations.Length} durations for {phonemes.Length} phonemes.", nameof(sample));
        var audio = sample.Audio ?? throw new ArgumentException($"{GetType().Name} trains on recordings; set the sample's Audio.", nameof(sample));
        var codes = Factorize(audio).Codes;
        Align(durations, codes.GetLength(1));
        int frames = durations.Sum();
        if (frames == 0) throw new ArgumentException("The durations cover no codec frame.", nameof(sample));
        return new DiffusionUtterance(phonemes, durations, Transposed(codes, frames));
    }

    // MaskGCT's prompt draw: none with p_cfg, else uniform in [min(n // 4, 5), high); at most n − 1.
    private int PromptLength(int n, int high, Random random)
    {
        if (random.NextDouble() < _options.PromptDropProbability) return 0;
        int low = Math.Min(n / 4, 5);
        high = Math.Min(high, n - 1);
        return high <= low ? Math.Max(0, Math.Min(low, n - 1)) : random.Next(low, high);
    }

    // The positions masked with probability σ(t) past the prompt; at least one.
    private static bool[] Mask(int n, int prompt, double p, Random random)
    {
        var mask = new bool[n];
        bool any = false;
        for (int i = prompt; i < n; i++) any |= mask[i] = random.NextDouble() < p;
        if (!any && prompt < n) mask[prompt] = true;
        return mask;
    }

    // MaskGCT's codebook draw: weights Q − i over the codebooks.
    private static int Layer(int codebooks, Random random)
    {
        double total = codebooks * (codebooks + 1.0) / 2.0, draw = random.NextDouble() * total, cumulative = 0;
        for (int i = 0; i < codebooks; i++)
        {
            cumulative += codebooks - i;
            if (draw < cumulative) return i;
        }
        return codebooks - 1;
    }

    // L_mask (§3.3.2): the cross-entropy of the masked tokens, averaged over them.
    private Tensor<T> MaskedCrossEntropy(Tensor<T> logits, IReadOnlyList<int> targets, IReadOnlyList<bool> masked)
    {
        int n = logits.Shape[0], v = logits.Shape[1];
        var picks = Enumerable.Range(0, n).Where(i => masked[i]).Select(i => i * v + targets[i]).ToArray();
        var logProbabilities = Engine.Reshape(Engine.TensorLogSoftmax(logits, axis: 1), new[] { n * v, 1 });
        return Engine.TensorMultiplyScalar(Engine.ReduceSum(Engine.TensorIndexSelect(logProbabilities, new Tensor<int>(new[] { picks.Length }, new Vector<int>(picks)), 0), null),
            Ops.FromDouble(-1.0 / picks.Length));
    }

    private Tensor<T> Term(Ns3MaskedDiffusion<T> model, Tensor<T> condition, int[,] codes, int prompt, Random random, bool training)
    {
        int n = codes.GetLength(0);
        double t = Math.Min(1.0, Math.Max(1e-5, random.NextDouble()));
        int layer = Layer(model.Codebooks, random);
        var mask = Mask(n, prompt, MaskGctSampling.MaskProbability(t), random);
        var targets = Enumerable.Range(0, n).Select(i => codes[i, layer]).ToArray();
        return MaskedCrossEntropy(model.TrainingLogits(condition, codes, layer, prompt, mask, t, training, random), targets, mask);
    }

    /// <summary>
    /// The factorized diffusion's loss for one utterance: L_mask of the duration, (phoneme-level prosody,) prosody, content
    /// and detail diffusions, each with its own step, masked codebook and mask, after a prompt of the utterance's first
    /// phones (dropped with p_cfg); the conditions use the ground truth.
    /// </summary>
    private Tensor<T> DiffusionLoss(DiffusionUtterance u, Random random, bool training)
    {
        var o = _options;
        var core = Core;
        int n = u.Phonemes.Length;
        int promptPhonemes = PromptLength(n, (int)(n * 0.4), random);
        // A prompt leaves at least one frame to predict (trailing phonemes may have no frames).
        int frames = u.Codes.GetLength(0);
        while (promptPhonemes > 0 && u.Durations.Take(promptPhonemes).Sum() >= frames) promptPhonemes--;
        int promptFrames = u.Durations.Take(promptPhonemes).Sum();
        var cph = core.Phonemes(u.Phonemes, training, random);
        var terms = new List<Tensor<T>>();
        int[]? phonemeProsody = null;
        if (core.PhonemeProsody is not null)
        {
            phonemeProsody = PhonemeProsodyCodes(u.Codes, u.Durations);
            terms.Add(Term(core.PhonemeProsody, cph, Column(phonemeProsody), promptPhonemes, random, training));
        }
        var phonemeCondition = core.WithPhonemeProsody(cph, phonemeProsody);
        terms.Add(Term(core.Duration, phonemeCondition, Column(u.Durations.Select(d => Math.Min(d, o.MaxDuration)).ToArray()), promptPhonemes, random, training));
        int np = o.ProsodyQuantizers, nc = o.ContentQuantizers, nd = o.DetailQuantizers;
        terms.Add(Term(core.Prosody, core.Expand(phonemeCondition, u.Durations), Columns(u.Codes, 0, np), promptFrames, random, training));
        var contentCondition = core.Plus(core.Expand(cph, u.Durations), u.Codes, 0, core.ProsodyCondition);
        terms.Add(Term(core.Content, contentCondition, Columns(u.Codes, np, nc), promptFrames, random, training));
        var detailCondition = core.Plus(contentCondition, u.Codes, np, core.ContentCondition);
        terms.Add(Term(core.Detail, detailCondition, Columns(u.Codes, np + nc, nd), promptFrames, random, training));
        return Sum(terms);
    }

    // ---------------------------------------------------------------- optimizers

    /// <summary>
    /// The diffusion's AdamW (1e-4, β 0.9/0.98, 5k warm-up then the inverse square root; App. A.2) and FACodec's Adam (2e-4,
    /// β 0.5/0.9; App. B.1) for the codec, the redecoder and the discriminators.
    /// </summary>
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> OptimizerFor(string group)
    {
        if (_optimizers.TryGetValue(group, out var cached)) return cached;
        var o = _options;
        var layers = group switch
        {
            "codec" => _codecEncoderLayers.Concat(_codecDecoderLayers).Concat(_codecHeadLayers).ToList(),
            "redecoder" => _redecoderLayers,
            "discriminator" => DiscriminatorLayers.ToList(),
            _ => _diffusionLayers,
        };
        var parameters = Training.TapeTrainingStep<T>.CollectParameters(layers.Cast<ILayer<T>>().ToList(), -1).ToList();
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> optimizer;
        if (group == "diffusion")
        {
            double warmup = Math.Max(1, o.WarmupSteps);
            optimizer = PaperOptimizerFactory.VerifyHandBuilt(this, new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this, new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = o.LearningRate, Beta1 = 0.9, Beta2 = o.DiffusionBeta2, WeightDecay = o.WeightDecay, MaskParameters = parameters,
                LearningRateScheduler = new AiDotNet.LearningRateSchedulers.LambdaLRScheduler(o.LearningRate,
                    s => s + 1 < warmup ? (s + 1) / warmup : Math.Sqrt(warmup / (s + 1))),
                SchedulerStepMode = AiDotNet.LearningRateSchedulers.SchedulerStepMode.StepPerBatch,
            }));
        }
        else
        {
            optimizer = new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this, new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = o.CodecLearningRate, Beta1 = o.CodecBeta1, Beta2 = o.CodecBeta2, WeightDecay = 0.0, MaskParameters = parameters,
            });
        }
        _optimizers[group] = optimizer;
        return optimizer;
    }

    /// <inheritdoc />
    protected override void OnMutableConstructorConfigurationRestored()
    {
        base.OnMutableConstructorConfigurationRestored();
        _optimizers.Clear();
    }

    // ---------------------------------------------------------------- weights

    /// <summary>
    /// Loads Amphion's released FACodec (amphion/naturalspeech3_facodec): <c>ns3_facodec_encoder.bin</c> and
    /// <c>ns3_facodec_decoder.bin</c> (the <c>_v2</c> files with <see cref="NaturalSpeech3Options.MelProsody"/>), and
    /// optionally <c>ns3_facodec_redecoder.bin</c> (with <see cref="NaturalSpeech3Options.IncludeRedecoder"/>).
    /// </summary>
    public void LoadFACodec(string encoderPath, string decoderPath, string? redecoderPath = null)
    {
        ThrowIfDisposed();
        if (!HasPaperLayers) throw NotNative();
        using (var encoder = AiDotNet.ModelLoading.TorchCheckpointFile.Open(encoderPath)) Codec.LoadEncoder(encoder);
        using (var decoder = AiDotNet.ModelLoading.TorchCheckpointFile.Open(decoderPath)) Codec.LoadDecoder(decoder);
        if (redecoderPath is not null)
        {
            if (Codec.RedecoderEmbeddings is null) throw new InvalidOperationException("Build the model with IncludeRedecoder to load the redecoder.");
            using var redecoder = AiDotNet.ModelLoading.TorchCheckpointFile.Open(redecoderPath);
            Codec.LoadRedecoder(redecoder);
        }
    }
}
