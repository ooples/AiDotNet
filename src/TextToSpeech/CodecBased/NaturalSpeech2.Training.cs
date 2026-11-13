using AiDotNet.Audio.Pitch;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.TextToSpeech.CodecBased;

public partial class NaturalSpeech2<T>
{
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private Random? _trainingRandom;

    /// <inheritdoc />
    /// <remarks>The phonemes' durations in codec frames (from a forced aligner; the paper's internal alignment tool), and
    /// the recording (encoded by the codec) or its codec tokens; the pitch per frame is optional (extracted from the
    /// recording with WORLD, the paper's PyWorld, when absent).</remarks>
    protected override TtsSupervision RequiredSupervision => TtsSupervision.Durations | TtsSupervision.CodecTokens;

    /// <inheritdoc />
    public override bool TrainsOnSampleObjective => true;

    /// <inheritdoc />
    public override int CodecTokenCodebooks => HasPaperLayers ? Codebooks : base.CodecTokenCodebooks;

    /// <inheritdoc />
    public override int CodecTokenVocabulary => HasPaperLayers ? Codec.CodebookSize : base.CodecTokenVocabulary;

    /// <inheritdoc />
    protected internal override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? TrainingOptimizer => HasPaperLayers ? Optimizer() : null;

    /// <inheritdoc />
    protected override bool TryApplyLearningRule(Tensor<T> input, Tensor<T> expectedOutput)
    {
        // A token/audio pair does not carry the durations this model trains on; Train(TtsTrainingSample) does.
        ThrowIfTokenMelTrainingUnsupported();
        return false;
    }

    /// <inheritdoc />
    protected override T TrainOnSample(TtsTrainingSample<T> sample)
    {
        ThrowIfDisposed();
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        if (!HasPaperLayers) throw NotNative();
        _trainingRandom ??= AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(SeedValue);
        var utterance = Prepare(sample, _trainingRandom);
        int seed = _trainingRandom.Next();
        SetTrainingMode(true);
        return TrainWithCustomObjective(sample.Tokens, sample.Tokens,
            (_, _) => Loss(utterance, AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed), training: true), Optimizer());
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
            var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(SeedValue);
            return Loss(Prepare(sample, random), random, training: false)[0];
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    /// <inheritdoc />
    protected override IReadOnlyList<Tensor<T>> SelectTrainableParametersForTraining(IReadOnlyList<Tensor<T>> parameters)
    {
        if (!HasPaperLayers) return parameters;
        var selected = new HashSet<Tensor<T>>(Training.TapeTrainingStep<T>.CollectParameters(TrainedLayers(), -1), TensorReferenceComparer<Tensor<T>>.Instance);
        return parameters.Where(selected.Contains).ToList();
    }

    // The codec is trained before, on its own, and stays frozen.
    private List<ILayer<T>> TrainedLayers() => _priorLayers.Concat(_promptLayers).Concat(_denoiserLayers).Cast<ILayer<T>>().ToList();

    // The utterance with its prompt cut out: the target's phonemes, durations, codes and pitch, and the prompt's codes.
    private sealed record Utterance(int[] Phonemes, int[] Durations, int[,] Codes, Tensor<T> Pitch, int[,] PromptCodes);

    private int[] PhonemesOf(TtsTrainingSample<T> sample) =>
        sample.Text is { } text ? TextToPhonemes(text) : PhonemeIds(sample.Tokens, nameof(sample));

    private Utterance Prepare(TtsTrainingSample<T> sample, Random random)
    {
        var o = _options;
        var phonemes = PhonemesOf(sample);
        var durations = (int[])(sample.Durations ?? throw new ArgumentException(
            $"{GetType().Name} trains with each phoneme's duration in codec frames; set the sample's Durations.", nameof(sample))).Clone();
        if (durations.Length != phonemes.Length) throw new ArgumentException($"{durations.Length} durations for {phonemes.Length} phonemes.", nameof(sample));
        if (durations.Any(d => d < 0)) throw new ArgumentException("Durations are not negative.", nameof(sample));
        var codes = sample.CodecTokens is { } tokens ? CodesOf(tokens, nameof(sample))
            : Encode(sample.Audio ?? throw new ArgumentException($"{GetType().Name} trains on the recording or its codec tokens.", nameof(sample)));
        var pitch = sample.Pitch is { } given ? Engine.Reshape(given, new[] { given.Length })
            : sample.Audio is { } audio ? ExtractPitch(audio)
            : throw new ArgumentException($"{GetType().Name} needs the pitch per frame, or the recording to extract it from.", nameof(sample));

        // align_length: the codes, pitch and durations cut to the shorter of the codes and the durations' total, the
        // excess taken from the last durations (Amphion takes it from the last one alone).
        int frames = Math.Min(codes.GetLength(1), durations.Sum());
        codes = Take(codes, 0, frames);
        // The pitch cut, or held at its last frame, to the codes' frames.
        pitch = pitch.Length == 0 ? new Tensor<T>(new[] { frames })
            : Engine.TensorIndexSelect(pitch, Engine.TensorClamp(Engine.TensorArange<int>(0, frames, 1), 0, pitch.Length - 1), 0);
        for (int excess = durations.Sum() - frames, j = durations.Length - 1; excess > 0 && j >= 0; j--)
        {
            int cut = Math.Min(excess, durations[j]);
            durations[j] -= cut;
            excess -= cut;
        }
        pitch = Interpolated(pitch);

        // The prompt (§3.3): a span of 10-50 % of the phonemes, anywhere (Amphion's "mid") or at the start.
        int n = phonemes.Length;
        if (n < 2) throw new ArgumentException("A training utterance needs two or more phonemes: one span becomes the prompt.", nameof(sample));
        int lowest = Math.Max(1, (int)(n * o.PromptMinFraction)), highest = Math.Min(n - 1, Math.Max(lowest, (int)(n * o.PromptMaxFraction)));
        int clip = Math.Min(Math.Max(1, random.Next((int)(n * o.PromptMinFraction), (int)(n * o.PromptMaxFraction) + 1)), n - 1);
        // Spans whose prompt and remaining target both have frames (zero durations can empty either); the drawn length
        // first, then the others.
        var starts = ValidStarts(durations, clip, frames, o.PromptPosition);
        for (int c = lowest; starts.Count == 0 && c <= highest; c++)
            if (c != clip && (starts = ValidStarts(durations, c, frames, o.PromptPosition)).Count > 0) clip = c;
        if (starts.Count == 0) throw new ArgumentException("No span of the utterance leaves frames for both the prompt and the target.", nameof(sample));
        int start = o.PromptPosition == NaturalSpeech2PromptPosition.Start ? starts[0] : starts[random.Next(starts.Count)];
        int end = start + clip, startFrame = durations.Take(start).Sum(), endFrame = durations.Take(end).Sum();
        var promptCodes = Take(codes, startFrame, endFrame - startFrame);
        var targetCodes = Concatenate(Take(codes, 0, startFrame), Take(codes, endFrame, frames - endFrame));
        var targetFrames = Enumerable.Range(0, startFrame).Concat(Enumerable.Range(endFrame, frames - endFrame)).ToArray();
        var targetPitch = Engine.TensorIndexSelect(pitch, new Tensor<int>(targetFrames, new[] { targetFrames.Length }), 0);
        var targetDurations = durations.Take(start).Concat(durations.Skip(end)).ToArray();
        var targetPhonemes = phonemes.Take(start).Concat(phonemes.Skip(end)).ToArray();
        return new Utterance(targetPhonemes, targetDurations, targetCodes, targetPitch, promptCodes);
    }

    // Where a prompt of `clip` phonemes may start: anywhere (Amphion's "mid": randint(0, n − clip)) or at the start (phoneme
    // 1 when the first has no frames and the prompt is one phoneme), keeping frames in both the prompt and the target.
    private static List<int> ValidStarts(int[] durations, int clip, int frames, NaturalSpeech2PromptPosition position)
    {
        var candidates = position == NaturalSpeech2PromptPosition.Start
            ? new List<int> { durations[0] == 0 && clip == 1 ? 1 : 0 }
            : Enumerable.Range(0, Math.Max(1, durations.Length - clip)).ToList();
        return candidates.Where(s =>
        {
            int count = durations.Skip(s).Take(clip).Sum();
            return s + clip <= durations.Length && count > 0 && count < frames;
        }).ToList();
    }

    private static int[,] Take(int[,] codes, int from, int count)
    {
        int q = codes.GetLength(0);
        var part = new int[q, count];
        for (int k = 0; k < q; k++)
            for (int f = 0; f < count; f++) part[k, f] = codes[k, from + f];
        return part;
    }

    private static int[,] Concatenate(int[,] a, int[,] b)
    {
        int q = a.GetLength(0), na = a.GetLength(1), nb = b.GetLength(1);
        var joined = new int[q, na + nb];
        for (int k = 0; k < q; k++)
        {
            for (int f = 0; f < na; f++) joined[k, f] = a[k, f];
            for (int f = 0; f < nb; f++) joined[k, na + f] = b[k, f];
        }
        return joined;
    }

    /// <summary>The F0 per codec frame by WORLD's DIO and StoneMask (PyWorld, §4.1).</summary>
    private Tensor<T> ExtractPitch(Tensor<T> audio)
    {
        var detector = new WorldPitchDetector<T>(_options.SampleRate);
        var (f0, _) = detector.EstimateF0(Engine.Reshape(audio, new[] { audio.Length }), 1000.0 * _options.HopSize / _options.SampleRate);
        return Engine.Reshape(f0, new[] { f0.Length });
    }

    /// <summary>
    /// Unvoiced frames (F0 0) filled by linear interpolation between voiced ones, held at the ends: the pitch target is a
    /// log-F0 and the bucket embedding reads every frame (Amphion's recipe does not show how it handles them).
    /// </summary>
    private Tensor<T> Interpolated(Tensor<T> pitch)
    {
        // F0 is never negative: a voiced frame is a positive one.
        var voicedOnly = Engine.TensorWhere(Engine.TensorGreaterThan(pitch, Ops.Zero), pitch, new Tensor<T>(pitch.Shape.ToArray()));
        if (Engine.TensorCountNonzero(voicedOnly) == 0) throw new ArgumentException("The pitch has no voiced frame.");
        return PitchWaveletTransform.FillUnvoiced(voicedOnly);
    }

    // ---------------------------------------------------------------- the losses (§3.2)

    /// <summary>L = L_diff + L_dur + L_pitch for one utterance with its draws fixed by <paramref name="random"/>.</summary>
    private Tensor<T> Loss(Utterance u, Random random, bool training)
    {
        var o = _options;
        var core = Core;
        var (prompt, query) = core.Prompt(Latent(u.PromptCodes), training, random);
        var (hidden, logDurations) = core.Phonemes(u.Phonemes, prompt, training, random);
        var (condition, logPitch) = core.Condition(hidden, u.Durations, u.Pitch, prompt, training, random);

        // log_dur_loss and log_pitch_loss: L1 against log(1 + d) and log F0.
        var durationTarget = Engine.TensorLog(Engine.TensorAddScalar(new Tensor<int>(u.Durations, new[] { u.Durations.Length }).Cast<T>(), Ops.One));
        var loss = MeanAbs(Engine.TensorSubtract(logDurations, durationTarget));
        loss = Engine.TensorAdd(loss, MeanAbs(Engine.TensorSubtract(logPitch, Engine.TensorLog(u.Pitch))));

        var x0 = Latent(u.Codes);
        double t = Math.Min(Math.Max(random.NextDouble(), o.TimeEpsilon), 1 - o.TimeEpsilon);
        var z = Gaussian(x0.Shape.ToArray(), random);
        Tensor<T> x0Prediction;
        if (o.Diffusion == NaturalSpeech2Diffusion.FlowMatching)
        {
            var xt = Engine.TensorAdd(Engine.TensorMultiplyScalar(x0, Ops.FromDouble(1 - t)), Engine.TensorMultiplyScalar(z, Ops.FromDouble(t)));
            var flow = core.Denoiser.Forward(xt, t, condition, query, training, random);
            x0Prediction = Engine.TensorSubtract(z, flow);
            loss = Engine.TensorAdd(loss, Distance(flow, Engine.TensorSubtract(z, x0), o.DataLossL1));
        }
        else
        {
            double cumBeta = NaturalSpeech2Core<T>.CumulativeBeta(t, o.BetaMin, o.BetaMax), decay = Math.Exp(-0.5 * cumBeta);
            double variance = 1 - Math.Exp(-cumBeta);
            var xt = Engine.TensorAdd(Engine.TensorMultiplyScalar(x0, Ops.FromDouble(decay)), Engine.TensorMultiplyScalar(z, Ops.FromDouble(Math.Sqrt(variance))));
            x0Prediction = core.Denoiser.Forward(xt, t, condition, query, training, random);
            loss = Engine.TensorAdd(loss, Distance(x0Prediction, x0, o.DataLossL1));
            // ε̂ = (z_t − ρ(ẑ₀, t)) / √Σ_t; Eq. 6's score term is the same error divided by √Σ_t again.
            var noise = Engine.TensorMultiplyScalar(Engine.TensorSubtract(xt, Engine.TensorMultiplyScalar(x0Prediction, Ops.FromDouble(decay))),
                Ops.FromDouble(1 / Math.Sqrt(variance)));
            var target = z;
            if (o.ScoreLoss == NaturalSpeech2ScoreLoss.ScoreSpace)
            {
                noise = Engine.TensorMultiplyScalar(noise, Ops.FromDouble(1 / Math.Sqrt(variance)));
                target = Engine.TensorMultiplyScalar(z, Ops.FromDouble(1 / Math.Sqrt(variance)));
            }
            loss = Engine.TensorAdd(loss, Engine.TensorMultiplyScalar(Distance(noise, target, o.NoiseLossL1), Ops.FromDouble(o.NoiseLossWeight)));
        }
        if (o.CeLossWeight > 0)
            loss = Engine.TensorAdd(loss, Engine.TensorMultiplyScalar(RvqCrossEntropy(x0Prediction, u.Codes), Ops.FromDouble(o.CeLossWeight)));
        return loss;
    }

    private Tensor<T> Mean(Tensor<T> x) => Engine.TensorMultiplyScalar(Engine.ReduceSum(x, null), Ops.FromDouble(1.0 / x.Length));

    private Tensor<T> MeanAbs(Tensor<T> x) => Mean(Engine.TensorAbs(x));

    private Tensor<T> Distance(Tensor<T> prediction, Tensor<T> target, bool l1)
    {
        var diff = Engine.TensorSubtract(prediction, target);
        return l1 ? MeanAbs(diff) : Mean(Engine.TensorMultiply(diff, diff));
    }

    /// <summary>
    /// L_ce-rvq (§3.2): for each quantizer j, the residual of ẑ₀ after the earlier quantizers' embeddings (the ground
    /// truth's, or Amphion's: those ẑ₀ itself quantizes to), the negative squared L2 distances to quantizer j's codebook
    /// as logits, and the cross-entropy with the ground-truth code; the mean over quantizers and frames.
    /// </summary>
    internal Tensor<T> RvqCrossEntropy(Tensor<T> prediction, int[,] codes)
    {
        var quantizer = Quantizer;
        int quantizers = codes.GetLength(0), frames = codes.GetLength(1);
        var residual = Ns2Ops.Rows(Engine, prediction);                                                    // [T, dim]
        Tensor<T>? total = null;
        for (int q = 0; q < quantizers; q++)
        {
            var codebook = Detached(quantizer.Codebook(q));                                         // [bins, dim]
            int bins = codebook.Shape[0];
            var cross = Engine.TensorMatMul(residual, Engine.TensorTranspose(codebook));                   // [T, bins]
            var rowNorms = Engine.ReduceSum(Engine.TensorMultiply(residual, residual), new[] { 1 }, keepDims: true);
            var codeNorms = Engine.Reshape(Engine.ReduceSum(Engine.TensorMultiply(codebook, codebook), new[] { 1 }, keepDims: false), new[] { 1, bins });
            var logits = Engine.TensorSubtract(Engine.TensorMultiplyScalar(cross, Ops.FromDouble(2)),
                Engine.TensorAdd(Engine.TensorBroadcastTo(rowNorms, cross._shape), Engine.TensorBroadcastTo(codeNorms, cross._shape)));
            var logProbabilities = Engine.Reshape(Engine.TensorLogSoftmax(logits, axis: 1), new[] { frames * bins, 1 });
            var picks = new Tensor<int>(new[] { frames }, new Vector<int>(Enumerable.Range(0, frames).Select(f => f * bins + codes[q, f]).ToArray()));
            var term = Engine.TensorMultiplyScalar(Engine.ReduceSum(Engine.TensorIndexSelect(logProbabilities, picks, 0), null), Ops.FromDouble(-1.0 / frames));
            total = total is null ? term : Engine.TensorAdd(total, term);
            if (q == quantizers - 1) break;
            int[] used;
            if (_options.CeResidual == NaturalSpeech2CeResidual.GroundTruth) used = Enumerable.Range(0, frames).Select(f => codes[q, f]).ToArray();
            else
            {
                used = Engine.TensorArgMax(Detached(logits), 1).ToArray();
            }
            var subtract = Engine.TensorIndexSelect(codebook, new Tensor<int>(new[] { frames }, new Vector<int>(used)), 0);
            residual = Engine.TensorSubtract(residual, subtract);
        }
        if (total is null) throw new ArgumentException("The codes have no quantizer.", nameof(codes));
        return Engine.TensorMultiplyScalar(total, Ops.FromDouble(1.0 / quantizers));
    }

    private static Tensor<T> Detached(Tensor<T> x) => new(x._shape, x.ToVector());

    // ---------------------------------------------------------------- the optimizer (§5.1)

    /// <summary>
    /// AdamW at <see cref="TtsModelOptions.LearningRate"/> with a linear warm-up over
    /// <see cref="NaturalSpeech2Options.WarmupSteps"/>, then the inverse square root (the paper's), or Amphion's cosine
    /// decay to zero at <see cref="NaturalSpeech2Options.TrainingSteps"/>.
    /// </summary>
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> Optimizer()
    {
        if (_optimizer is not null) return _optimizer;
        var o = _options;
        double warmup = Math.Max(1, o.WarmupSteps), total = Math.Max(o.TrainingSteps, o.WarmupSteps + 1);
        // Step s is the (s + 1)-th update.
        Func<int, double> schedule = o.CosineSchedule
            ? s => s + 1 < warmup ? (s + 1) / warmup : Math.Max(0.0, 0.5 * (1 + Math.Cos(Math.PI * (s + 1 - warmup) / (total - warmup))))
            : s => s + 1 < warmup ? (s + 1) / warmup : Math.Sqrt(warmup / (s + 1));
        var parameters = Training.TapeTrainingStep<T>.CollectParameters(TrainedLayers(), -1).ToList();
        _optimizer = PaperOptimizerFactory.VerifyHandBuilt(this, new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this,
            new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = o.LearningRate, WeightDecay = o.WeightDecay, MaskParameters = parameters,
                LearningRateScheduler = new AiDotNet.LearningRateSchedulers.LambdaLRScheduler(o.LearningRate, schedule),
                SchedulerStepMode = AiDotNet.LearningRateSchedulers.SchedulerStepMode.StepPerBatch,
            }));
        return _optimizer;
    }

    /// <inheritdoc />
    protected override void OnMutableConstructorConfigurationRestored()
    {
        base.OnMutableConstructorConfigurationRestored();
        _optimizer = null;
    }
}
