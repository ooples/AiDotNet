using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.Vocoders;

namespace AiDotNet.TextToSpeech.CodecBased;

public partial class NaturalSpeech<T>
{
    private readonly Dictionary<string, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>> _optimizers = new();
    private List<ILayer<T>>? _trainingLayers;
    private Random? _trainingRandom;
    private bool _memoryInitialized;

    /// <inheritdoc />
    /// <remarks>NaturalSpeech trains on recordings at <see cref="TtsModelOptions.SampleRate"/> (<see cref="TtsTrainingSample{T}.Audio"/>)
    /// with their phonemes (<see cref="TtsTrainingSample{T}.Tokens"/>, from <see cref="TextToPhonemes"/>, or
    /// <see cref="TtsTrainingSample{T}.Text"/>); pre-training reads text alone.</remarks>
    protected override TtsSupervision RequiredSupervision => TtsSupervision.Recording;

    /// <inheritdoc />
    public override bool TrainsOnSampleObjective => true;

    /// <inheritdoc />
    protected internal override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? TrainingOptimizer => HasPaperLayers ? OptimizerFor("generator") : null;

    /// <inheritdoc />
    protected override bool TryApplyLearningRule(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        Train(new TtsTrainingSample<T> { Tokens = input, Audio = expectedOutput });
        return true;
    }

    private sealed record Utterance(int[] Phonemes, int[]? SupPhonemes, Tensor<T> Audio, Tensor<T> Spectrogram, Tensor<T> Mel, int Frames);

    private DifferentiableMel<T> Spectrogram => _spectrogram ??=
        new DifferentiableMel<T>(Engine, _options.SampleRate, _options.FftSize, _options.HopSize, _options.FftSize, _options.MelChannels, 0, _options.SampleRate / 2.0);

    private int[] PhonemesOf(TtsTrainingSample<T> sample) =>
        sample.Text is { } text ? TextToPhonemes(text) : PhonemeIds(sample.Tokens, nameof(sample));

    private Utterance Prepare(TtsTrainingSample<T> sample)
    {
        var phonemes = PhonemesOf(sample);
        var recording = sample.Audio ?? throw new ArgumentException($"{GetType().Name} trains on recordings; set the sample's Audio.", nameof(sample));
        int hop = _options.HopSize, frames = recording.Length / hop;
        if (frames < phonemes.Length) throw new ArgumentException($"The recording has {frames} frames at hop {hop}, fewer than the text's {phonemes.Length} phonemes.", nameof(sample));
        var audio = Engine.TensorSlice(Engine.Reshape(recording, new[] { recording.Length }), new[] { 0 }, new[] { frames * hop });
        using var _ = new NoGradScope<T>();
        // spectrogram_torch (ε 1e-6) and spec_to_mel_torch, frames trimmed to whole hops.
        var magnitude = Spectrogram.Magnitude(audio, 1e-6);
        // Symmetric STFT reflection can yield one fewer frame at odd hops; the posterior reads the actual analysis.
        frames = Math.Min(frames, magnitude.Shape[0]);
        if (frames < phonemes.Length)
            throw new ArgumentException($"The analysis has {frames} frames, fewer than the text's {phonemes.Length} phonemes.", nameof(sample));
        var trimmed = Engine.TensorSlice(magnitude, new[] { 0, 0 }, new[] { frames, magnitude.Shape[1] });
        var mel = Spectrogram.LogMel(trimmed);
        var spectrogram = Engine.Reshape(Engine.TensorTranspose(trimmed), new[] { 1, trimmed.Shape[1], frames });
        return new Utterance(phonemes, SupPhonemeIds(phonemes), audio, spectrogram, mel, frames);
    }

    /// <inheritdoc />
    /// <remarks>
    /// One step of <see cref="NaturalSpeechOptions.CurrentStage"/>. Pre-training minimizes the masked phoneme (and
    /// sup-phoneme) cross-entropy. Warm-up, joint training and tuning take a discriminator step and a generator step: the
    /// generator minimizes Eq. 5's L_bwd + L_fwd + L_rec + L_e2e (warm-up: L_bwd only of the KLs, plus the duration loss on
    /// alignment-search durations; tuning: L_e2e alone, the durator alone).
    /// </remarks>
    protected override T TrainOnSample(TtsTrainingSample<T> sample)
    {
        ThrowIfDisposed();
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        if (!HasPaperLayers) throw new NotSupportedException("NaturalSpeech's training needs its own layers; this model was built from caller-supplied layers.");
        var random = _trainingRandom ??= AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(SeedValue);
        if (_options.CurrentStage == NaturalSpeechStage.PhonemePretraining)
        {
            var phonemes = PhonemesOf(sample);
            int seed = random.Next();
            return TrainLayers(_phonemeLayers.Concat(_pretrainingLayers).ToList(),
                () => PretrainingLoss(phonemes, AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed), training: true), "pretraining", sample);
        }
        if (_options.CurrentStage == NaturalSpeechStage.Joint && !_memoryInitialized)
            throw new InvalidOperationException("The joint stage reads the memory bank; initialize it with InitializeMemoryBank after warm-up.");
        Core.UseMemory = _options.CurrentStage != NaturalSpeechStage.Warmup;
        var u = Prepare(sample);
        int stepSeed = random.Next();
        SetTrainingMode(true);
        Pass view;
        using (new NoGradScope<T>()) view = Forward(u, stepSeed, training: true);
        var discriminators = _discriminators!;
        var real = Segment(u.Audio, view.SliceStart);
        var realE2e = Segment(u.Audio, view.SliceStartE2e);
        var generated = view.Audio is null ? null : Detached(view.Audio);
        var generatedE2e = Detached(view.AudioE2e);
        TrainLayers(discriminators.Layers, () =>
        {
            var loss = Engine.TensorMultiplyScalar(DiscriminatorLoss(discriminators, realE2e, generatedE2e), Ops.FromDouble(_options.EndToEndWeight));
            return generated is null ? loss : Engine.TensorAdd(DiscriminatorLoss(discriminators, real, generated), loss);
        }, "discriminator", sample);
        return TrainLayers(StageLayers(), () => GeneratorLoss(u, stepSeed, adversarial: true), "generator", sample);
    }

    private sealed record Pass(Tensor<T>? Audio, int SliceStart, Tensor<T> AudioE2e, int SliceStartE2e, Tensor<T> Loss);

    private static Tensor<T> Detached(Tensor<T> x) => new(x._shape, x.ToVector());

    private Tensor<T> Mean(Tensor<T> x) => Engine.ReduceMean(x, Enumerable.Range(0, x.Rank).ToArray(), keepDims: false);

    private Tensor<T> Segment(Tensor<T> audio, int startFrame)
    {
        int hop = _options.HopSize, length = _options.SegmentFrames * hop, from = startFrame * hop;
        int count = Math.Max(0, Math.Min(length, audio.Length - from));
        var part = Engine.TensorSlice(audio, new[] { from }, new[] { count });
        return count == length ? part : Engine.TensorConcatenate(new[] { part, new Tensor<T>(new[] { length - count }) }, 0);
    }

    private Tensor<T> FramesSlice(Tensor<T> z, int start)
    {
        int frames = z.Shape[2], length = _options.SegmentFrames, count = Math.Max(0, Math.Min(length, frames - start));
        var part = Engine.TensorSlice(z, new[] { 0, 0, start }, new[] { 1, z.Shape[1], count });
        return count == length ? part : Engine.TensorConcatenate(new[] { part, new Tensor<T>(new[] { 1, z.Shape[1], length - count }) }, 2);
    }

    // discriminator_loss: Σ mean((1 − D(real))²) + mean(D(fake)²).
    private Tensor<T> DiscriminatorLoss(HiFiGanDiscriminators<T> discriminators, Tensor<T> real, Tensor<T> fake)
    {
        var r = discriminators.Forward(real).Select(s => s.Score).ToList();
        var f = discriminators.Forward(fake).Select(s => s.Score).ToList();
        Tensor<T>? loss = null;
        for (int k = 0; k < r.Count; k++)
        {
            var d = Engine.TensorAddScalar(Engine.TensorNegate(r[k]), NumOps.One);
            var term = Engine.TensorAdd(Mean(Engine.TensorMultiply(d, d)), Mean(Engine.TensorMultiply(f[k], f[k])));
            loss = loss is null ? term : Engine.TensorAdd(loss, term);
        }
        return loss!;
    }

    // generator_loss: Σ mean((1 − D(fake))²), with feature_loss 2 Σ mean |r − g| when real is given.
    private Tensor<T> AdversarialLoss(HiFiGanDiscriminators<T> discriminators, Tensor<T> fake, Tensor<T>? real)
    {
        var f = discriminators.Forward(fake);
        List<(Tensor<T> Score, List<Tensor<T>> Features)>? r = null;
        if (real is not null)
            using (new NoGradScope<T>()) r = discriminators.Forward(real);
        Tensor<T>? loss = null;
        for (int k = 0; k < f.Count; k++)
        {
            var d = Engine.TensorAddScalar(Engine.TensorNegate(f[k].Score), NumOps.One);
            var term = Mean(Engine.TensorMultiply(d, d));
            if (r is not null)
                for (int l = 0; l < f[k].Features.Count; l++)
                    term = Engine.TensorAdd(term, Engine.TensorMultiplyScalar(Mean(Engine.TensorAbs(Engine.TensorSubtract(Detached(r[k].Features[l]), f[k].Features[l]))),
                        Ops.FromDouble(2)));
            loss = loss is null ? term : Engine.TensorAdd(loss, term);
        }
        return loss!;
    }

    /// <summary>
    /// The model's forward for one utterance with its draws fixed by <paramref name="seed"/> (so the discriminator's view
    /// and the generator's step see the same pass): the reconstruction from the posterior, the end-to-end waveform from the
    /// prior, and the non-adversarial losses.
    /// </summary>
    private Pass Forward(Utterance u, int seed, bool training)
    {
        var o = _options;
        var core = Core;
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed);
        var stage = o.CurrentStage;
        var hidden = core.Encode(u.Phonemes, u.SupPhonemes, training, random);
        var logw = core.LogDurations(hidden, training, random);
        var noise = Gaussian(new[] { 1, o.HiddenDim, u.Frames }, random);
        var (z, _, logsQ) = core.Posterior.Forward(u.Spectrogram, null, noise);
        var zPrime = core.Flow(z, backward: true);                                                     // z′ = f⁻¹(z)
        Tensor<T> loss = new Tensor<T>(new[] { 1 });
        Tensor<T> durations;
        if (stage == NaturalSpeechStage.Warmup)
        {
            // Monotonic alignment search between the phoneme-level prior and the simplified posterior (App. F, Eqs. 17-18).
            var (meanPh, logsPh) = core.PhonemePrior(hidden);
            var alignment = Alignment(Detached(zPrime), Detached(meanPh), Detached(logsPh));
            durations = new Tensor<T>(new[] { alignment.Length }, new Vector<T>(alignment.Select(d => Ops.FromDouble(d)).ToArray()));
            // The phoneme-level prior learns from the aligned posterior (VITS's KL with the hard-expanded prior; the paper
            // only says it is "obtained by two linear layers").
            var expandedMean = Expand(meanPh, alignment);
            var expandedLogs = Expand(logsPh, alignment);
            var phonemeKl = Kl(Detached(zPrime), Detached(logsQ), expandedMean, expandedLogs);
            loss = Engine.TensorAdd(loss, phonemeKl);
        }
        else
        {
            durations = Engine.TensorExp(logw);                                                        // the predicted durations
        }
        if (stage == NaturalSpeechStage.Warmup || o.DurationLossAfterWarmup)
        {
            if (stage != NaturalSpeechStage.Warmup)
            {
                // heatz123 keeps the duration loss against alignment-search labels after warm-up.
                var (meanPh, logsPh) = core.PhonemePrior(hidden);
                var labels = Alignment(Detached(zPrime), Detached(meanPh), Detached(logsPh));
                loss = Engine.TensorAdd(loss, DurationLoss(logw, labels));
            }
            else
            {
                var labels = durations.ToArray().Select(v => (int)Math.Round(Ops.ToDouble(v))).ToArray();
                loss = Engine.TensorAdd(loss, DurationLoss(logw, labels));
            }
        }
        var (meanP, logsP, priorFrames) = core.Upsampler.Forward(hidden, durations, training, random);
        if (stage != NaturalSpeechStage.Tuning)
        {
            // L_bwd: KL[q(z′|x) ‖ p(z′|y)].
            var bwd = KlLoss(zPrime, logsQ, meanP, logsP, u.Frames, priorFrames);
            loss = Engine.TensorAdd(loss, Engine.TensorMultiplyScalar(bwd, Ops.FromDouble(o.BackwardKlWeight)));
        }
        // The prior's sample through f: the enhanced prior p(z|y).
        var priorNoise = Gaussian(meanP.Shape.ToArray(), random);
        var zFromPrior = core.Flow(Engine.TensorAdd(meanP, Engine.TensorMultiply(priorNoise, Engine.TensorExp(logsP))), backward: false);
        if (stage == NaturalSpeechStage.Joint)
        {
            // L_fwd: KL[p(z|y) ‖ q(z|x)] (sample-based, with the posterior's mean and log-scale).
            var (_, meanQ, logsQ2) = core.Posterior.Forward(u.Spectrogram, null, noise);
            var fwd = KlLoss(zFromPrior, logsP, meanQ, logsQ2, priorFrames, u.Frames);
            loss = Engine.TensorAdd(loss, Engine.TensorMultiplyScalar(fwd, Ops.FromDouble(o.ForwardKlWeight)));
        }
        Tensor<T>? audio = null;
        int start = 0;
        if (stage != NaturalSpeechStage.Tuning)
        {
            start = (int)(random.NextDouble() * Math.Max(u.Frames - o.SegmentFrames + 1, 1));
            audio = core.Decode(FramesSlice(z, start));
        }
        int startE2e = (int)(random.NextDouble() * Math.Max(Math.Min(priorFrames, u.Frames) - o.SegmentFrames + 1, 1));
        var audioE2e = core.Decode(FramesSlice(zFromPrior, startE2e));
        return new Pass(audio, start, audioE2e, startE2e, loss);
    }

    private Tensor<T> GeneratorLoss(Utterance u, int seed, bool adversarial)
    {
        var o = _options;
        var pass = Forward(u, seed, training: true);
        var loss = pass.Loss;
        if (pass.Audio is not null)
        {
            // L_rec's mel L1 against the recording's segment.
            var generatedMel = Spectrogram.LogMel(Spectrogram.Magnitude(pass.Audio, 1e-6));
            int frames = o.SegmentFrames, available = Math.Max(0, Math.Min(frames, u.Mel.Shape[0] - pass.SliceStart));
            var targetMel = Engine.TensorSlice(u.Mel, new[] { pass.SliceStart, 0 }, new[] { available, o.MelChannels });
            int rows = Math.Min(generatedMel.Shape[0], available);
            var melLoss = Mean(Engine.TensorAbs(Engine.TensorSubtract(Engine.TensorSlice(targetMel, new[] { 0, 0 }, new[] { rows, o.MelChannels }),
                Engine.TensorSlice(generatedMel, new[] { 0, 0 }, new[] { rows, o.MelChannels }))));
            loss = Engine.TensorAdd(loss, Engine.TensorMultiplyScalar(melLoss, Ops.FromDouble(o.MelLossWeight)));
        }
        if (!adversarial) return loss;
        var discriminators = _discriminators!;
        if (pass.Audio is not null) loss = Engine.TensorAdd(loss, AdversarialLoss(discriminators, pass.Audio, Segment(u.Audio, pass.SliceStart)));
        // L_e2e: the GAN loss only.
        loss = Engine.TensorAdd(loss, Engine.TensorMultiplyScalar(AdversarialLoss(discriminators, pass.AudioE2e, null), Ops.FromDouble(o.EndToEndWeight)));
        return loss;
    }

    // The KL between frames: soft-DTW over the pairwise matrix (App. D), or VITS's frame-wise KL when the lengths allow.
    private Tensor<T> KlLoss(Tensor<T> z, Tensor<T> logsQ, Tensor<T> meanP, Tensor<T> logsP, int zFrames, int pFrames)
    {
        var o = _options;
        if (o.UseSoftDtw)
        {
            var matrix = NaturalSpeechOps.KlMatrix(Engine, z, logsQ, meanP, logsP);
            return Engine.TensorMultiplyScalar(NaturalSpeechOps.SoftDtw(Engine, matrix, o.SoftDtwGamma, o.SoftDtwWarp), Ops.FromDouble(1.0 / pFrames));
        }
        // The plain KL over the overlapping frames (heatz123's kl_loss).
        int frames = Math.Min(zFrames, pFrames), c = o.HiddenDim;
        Tensor<T> Cut(Tensor<T> x) => Engine.TensorSlice(x, new[] { 0, 0, 0 }, new[] { 1, c, frames });
        return Kl(Cut(z), Cut(logsQ), Cut(meanP), Cut(logsP));
    }

    // kl_loss(z_p, logs_q, m_p, logs_p): Σ (logs_p − logs_q − ½ + ½ (z_p − m_p)² e^{−2 logs_p}) / frames.
    private Tensor<T> Kl(Tensor<T> z, Tensor<T> logsQ, Tensor<T> meanP, Tensor<T> logsP)
    {
        var kl = Engine.TensorAddScalar(Engine.TensorSubtract(logsP, logsQ), Ops.FromDouble(-0.5));
        var diff = Engine.TensorSubtract(z, meanP);
        kl = Engine.TensorAdd(kl, Engine.TensorMultiplyScalar(Engine.TensorMultiply(Engine.TensorMultiply(diff, diff),
            Engine.TensorExp(Engine.TensorMultiplyScalar(logsP, Ops.FromDouble(-2.0)))), Ops.FromDouble(0.5)));
        return Engine.TensorMultiplyScalar(Engine.ReduceSum(kl, null), Ops.FromDouble(1.0 / z.Shape[2]));
    }

    // The duration loss: Σ (log w − log(d + 1e-6))² over the phonemes, averaged (heatz123's l_length).
    private Tensor<T> DurationLoss(Tensor<T> logw, int[] durations)
    {
        var target = new Tensor<T>(new[] { durations.Length }, new Vector<T>(durations.Select(d => Ops.FromDouble(Math.Log(d + 1e-6))).ToArray()));
        var diff = Engine.TensorSubtract(logw, target);
        return Engine.TensorMultiplyScalar(Engine.ReduceSum(Engine.TensorMultiply(diff, diff), null), Ops.FromDouble(_options.DurationLossWeight / durations.Length));
    }

    // Monotonic alignment search: log N(z′_i; µ_j, σ_j) summed over channels, the most likely monotonic path.
    private int[] Alignment(Tensor<T> zPrime, Tensor<T> mean, Tensor<T> logs)
    {
        int c = zPrime.Shape[1], frames = zPrime.Shape[2], n = mean.Shape[2];
        var scores = new double[n, frames];
        for (int j = 0; j < n; j++)
            for (int i = 0; i < frames; i++)
            {
                double s = 0;
                for (int k = 0; k < c; k++)
                {
                    double ls = Ops.ToDouble(logs[0, k, j]), diff = Ops.ToDouble(zPrime[0, k, i]) - Ops.ToDouble(mean[0, k, j]);
                    s += -0.5 * Math.Log(2 * Math.PI) - ls - 0.5 * diff * diff * Math.Exp(-2 * ls);
                }
                scores[j, i] = s;
            }
        return MonotonicAlignment.MaximumPath(scores);
    }

    // [1, h, n] repeated by integer durations to [1, h, Σd].
    private Tensor<T> Expand(Tensor<T> x, int[] durations)
    {
        var index = new List<int>();
        for (int j = 0; j < durations.Length; j++)
            for (int r = 0; r < durations[j]; r++) index.Add(j);
        int h = x.Shape[1], n = x.Shape[2];
        var picked = Engine.TensorIndexSelect(Engine.Reshape(x, new[] { h, n }), new Tensor<int>(new[] { index.Count }, new Vector<int>(index.ToArray())), 1);
        return Engine.Reshape(picked, new[] { 1, h, index.Count });
    }

    // ---------------------------------------------------------------- phoneme pre-training

    /// <summary>
    /// Mixed-phoneme pre-training's loss for one phoneme sequence: 15% of the sup-phonemes (or of the phonemes, without
    /// sup-phonemes) masked with their phonemes (BERT's 80/10/10 replacement), and the cross-entropy of the masked
    /// phonemes and sup-phonemes from the encoder's output.
    /// </summary>
    private Tensor<T> PretrainingLoss(int[] phonemes, Random random, bool training)
    {
        var o = _options;
        var core = Core;
        var sup = SupPhonemeIds(phonemes);
        int n = phonemes.Length, maskPhoneme = o.VocabSize, maskSup = SupPhonemes?.VocabSize ?? 0;
        // The units to mask: each sup-phoneme span (consecutive phonemes sharing an id inside a word), or each phoneme.
        var spans = new List<(int Start, int Length)>();
        for (int i = 0; i < n;)
        {
            int j = i + 1;
            if (sup is not null && phonemes[i] != SpaceId && phonemes[i] != 0)
                while (j < n && sup[j] == sup[i] && phonemes[j] != SpaceId && phonemes[j] != 0) j++;
            spans.Add((i, j - i));
            i = j;
        }
        var masked = new bool[n];
        var inputPhonemes = (int[])phonemes.Clone();
        var inputSup = sup is null ? null : (int[])sup.Clone();
        foreach (var (start, length) in spans)
        {
            if (random.NextDouble() >= o.MaskRatio) continue;
            double roll = random.NextDouble();
            int randomPhoneme = random.Next(o.VocabSize);
            for (int k = start; k < start + length; k++)
            {
                masked[k] = true;
                if (roll < 0.8)
                {
                    inputPhonemes[k] = maskPhoneme;
                    if (inputSup is not null) inputSup[k] = maskSup;
                }
                else if (roll < 0.9) inputPhonemes[k] = randomPhoneme;
            }
        }
        if (!masked.Any(m => m)) masked[random.Next(n)] = true;
        var hidden = core.Encode(inputPhonemes, inputSup, training, random);                           // [1, h, n]
        var rows = Engine.TensorTranspose(Engine.Reshape(hidden, new[] { o.HiddenDim, n }));
        var positions = Enumerable.Range(0, n).Where(i => masked[i]).ToArray();
        var picked = Engine.TensorIndexSelect(rows, new Tensor<int>(new[] { positions.Length }, new Vector<int>(positions)), 0);
        var loss = CrossEntropy(core.PhonemeHead.Forward(picked), positions.Select(i => phonemes[i]).ToArray());
        if (core.SupHead is not null && sup is not null)
            loss = Engine.TensorAdd(loss, CrossEntropy(core.SupHead.Forward(picked), positions.Select(i => sup[i]).ToArray()));
        return loss;
    }

    private Tensor<T> CrossEntropy(Tensor<T> logits, IReadOnlyList<int> targets)
    {
        int n = logits.Shape[0], v = logits.Shape[1];
        var logProbabilities = Engine.Reshape(Engine.TensorLogSoftmax(logits, axis: 1), new[] { n * v, 1 });
        var picks = new Tensor<int>(new[] { n }, new Vector<int>(Enumerable.Range(0, n).Select(i => i * v + targets[i]).ToArray()));
        return Engine.TensorMultiplyScalar(Engine.ReduceSum(Engine.TensorIndexSelect(logProbabilities, picks, 0), null), Ops.FromDouble(-1.0 / n));
    }

    // ---------------------------------------------------------------- the memory bank

    /// <summary>
    /// Initializes the memory bank (App. F): posterior latents z ~ q(z|x) of every frame of the recordings (at
    /// <see cref="TtsModelOptions.SampleRate"/>), clustered by K-means (k-means++ seeding, Lloyd iterations) into
    /// <see cref="NaturalSpeechOptions.MemorySize"/> centres. Call it after warm-up, before the joint stage.
    /// </summary>
    public void InitializeMemoryBank(IEnumerable<Tensor<T>> recordings)
    {
        ThrowIfDisposed();
        if (recordings is null) throw new ArgumentNullException(nameof(recordings));
        if (!HasPaperLayers) throw NotNative();
        var o = _options;
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(SeedValue + 1);
        var points = new List<double[]>();
        using (new NoGradScope<T>())
            foreach (var recording in recordings)
            {
                var u = Prepare(new TtsTrainingSample<T> { Tokens = new Tensor<T>(new[] { 1 }), Audio = recording, Text = null });
                var (z, _, _) = Core.Posterior.Forward(u.Spectrogram, null, Gaussian(new[] { 1, o.HiddenDim, u.Frames }, random));
                for (int t = 0; t < u.Frames; t++)
                {
                    var p = new double[o.HiddenDim];
                    for (int c = 0; c < o.HiddenDim; c++) p[c] = Ops.ToDouble(z[0, c, t]);
                    points.Add(p);
                }
            }
        if (points.Count < o.MemorySize) throw new ArgumentException($"K-means into {o.MemorySize} centres needs at least that many frames; got {points.Count}.", nameof(recordings));
        Core.Memory.SetBank(KMeans(points, o.MemorySize, o.MemoryKMeansIterations, random));
        _memoryInitialized = true;
    }

    private static double[][] KMeans(List<double[]> points, int k, int iterations, Random random)
    {
        int dim = points[0].Length;
        double Distance(double[] a, double[] b)
        {
            double s = 0;
            for (int i = 0; i < dim; i++) s += (a[i] - b[i]) * (a[i] - b[i]);
            return s;
        }
        // k-means++ seeding.
        var centres = new List<double[]> { (double[])points[random.Next(points.Count)].Clone() };
        var nearest = points.Select(p => Distance(p, centres[0])).ToArray();
        while (centres.Count < k)
        {
            double total = nearest.Sum(), u = random.NextDouble() * total, acc = 0;
            int chosen = points.Count - 1;
            for (int i = 0; i < points.Count; i++) { acc += nearest[i]; if (acc >= u) { chosen = i; break; } }
            var centre = (double[])points[chosen].Clone();
            centres.Add(centre);
            for (int i = 0; i < points.Count; i++) nearest[i] = Math.Min(nearest[i], Distance(points[i], centre));
        }
        var assignment = new int[points.Count];
        for (int it = 0; it < iterations; it++)
        {
            for (int i = 0; i < points.Count; i++)
            {
                double best = double.MaxValue;
                for (int c = 0; c < k; c++)
                {
                    double d = Distance(points[i], centres[c]);
                    if (d < best) { best = d; assignment[i] = c; }
                }
            }
            var sums = new double[k][];
            var counts = new int[k];
            for (int c = 0; c < k; c++) sums[c] = new double[dim];
            for (int i = 0; i < points.Count; i++)
            {
                counts[assignment[i]]++;
                for (int d = 0; d < dim; d++) sums[assignment[i]][d] += points[i][d];
            }
            for (int c = 0; c < k; c++)
                if (counts[c] > 0)
                    for (int d = 0; d < dim; d++) centres[c][d] = sums[c][d] / counts[c];
        }
        return centres.ToArray();
    }

    // ---------------------------------------------------------------- shared

    /// <summary>The layers the current stage's generator step trains.</summary>
    private IReadOnlyList<LayerBase<T>> StageLayers() => _options.CurrentStage switch
    {
        // Tuning (App. F): the posterior encoder, decoder, phoneme encoder and flow frozen; the durator alone updated.
        NaturalSpeechStage.Tuning => _duratorLayers,
        NaturalSpeechStage.Warmup => _phonemeLayers.Concat(_duratorLayers).Concat(_flowLayers).Concat(_posteriorLayers).Concat(_decoderLayers).ToList(),
        _ => _phonemeLayers.Concat(_duratorLayers).Concat(_flowLayers).Concat(_posteriorLayers).Concat(_decoderLayers).Concat(_memoryLayers).ToList(),
    };

    /// <inheritdoc />
    /// <remarks>The current stage's objective at a fixed draw; for the GAN stages, its non-adversarial part.</remarks>
    public override T EvaluateTrainingObjective(TtsTrainingSample<T> sample)
    {
        ThrowIfDisposed();
        Validation.Guard.NotNull(sample);
        if (!HasPaperLayers) throw new NotSupportedException("NaturalSpeech's objective needs its own layers; this model was built from caller-supplied layers.");
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(SeedValue);
            if (_options.CurrentStage == NaturalSpeechStage.PhonemePretraining)
                return PretrainingLoss(PhonemesOf(sample), random, training: false)[0];
            Core.UseMemory = _options.CurrentStage != NaturalSpeechStage.Warmup && _memoryInitialized;
            return GeneratorLoss(Prepare(sample), random.Next(), adversarial: false)[0];
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
    protected override IReadOnlyList<Tensor<T>> SelectTrainableParametersForTraining(IReadOnlyList<Tensor<T>> parameters)
    {
        if (!HasPaperLayers) return parameters;
        var layers = _trainingLayers ?? StageLayers().Cast<ILayer<T>>().ToList();
        var selected = new HashSet<Tensor<T>>(Training.TapeTrainingStep<T>.CollectParameters(layers.ToList(), -1), TensorReferenceComparer<Tensor<T>>.Instance);
        return parameters.Where(selected.Contains).ToList();
    }

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> OptimizerFor(string group)
    {
        if (_optimizers.TryGetValue(group, out var cached)) return cached;
        var o = _options;
        int? perEpoch = o.UpdatesPerEpoch;
        var optimizer = PaperOptimizerFactory.VerifyHandBuilt(this, new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this,
            new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = o.LearningRate, Beta1 = o.Beta1, Beta2 = o.Beta2, Epsilon = o.Epsilon, WeightDecay = o.WeightDecay,
                EnableGradientClipping = false, // The released loop calls clip_grad_value_(..., None).
                LearningRateScheduler = new AiDotNet.LearningRateSchedulers.LambdaLRScheduler(o.LearningRate,
                    s => perEpoch is int n && n > 0 ? Math.Pow(o.LearningRateDecay, Math.Truncate(s / (double)n)) : 1.0),
                SchedulerStepMode = AiDotNet.LearningRateSchedulers.SchedulerStepMode.StepPerBatch,
            }));
        _optimizers[group] = optimizer;
        return optimizer;
    }

    /// <inheritdoc />
    protected override void OnMutableConstructorConfigurationRestored()
    {
        base.OnMutableConstructorConfigurationRestored();
        _optimizers.Clear();
    }

    /// <summary>Loads a heatz123/naturalspeech generator checkpoint (<c>G_*.pth</c>'s <c>model</c>); build the model with
    /// <see cref="NaturalSpeechOptions.Heatz123"/> and the checkpoint's vocabulary.</summary>
    public void LoadHeatz123Checkpoint(string path)
    {
        ThrowIfDisposed();
        if (!HasPaperLayers) throw NotNative();
        var file = new AiDotNet.ComputerVision.Weights.WeightLoader().LoadWeights(path);
        Core.LoadHeatz123(name => file.TryGetValue(name, out var t) || file.TryGetValue("model." + name, out t)
            ? t.ToVector().Select(v => (double)v).ToArray() : null);
        _memoryInitialized |= file.Keys.Any(k => k.EndsWith("memory_bank.memory_bank", StringComparison.Ordinal));
    }
}
