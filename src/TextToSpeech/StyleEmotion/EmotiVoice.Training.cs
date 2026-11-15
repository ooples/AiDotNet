using AiDotNet.Helpers;
using AiDotNet.Audio.Pitch;
using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.Vocoders;

namespace AiDotNet.TextToSpeech.StyleEmotion;

public partial class EmotiVoice<T>
{
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _generatorOptimizer;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _discriminatorOptimizer;
    private bool _discriminatorStep;
    private TacotronSpectrogram? _targetMel;
    private CenteredComplexStft<T>? _energyStft;
    private Tensor<T>? _lossCos;
    private Tensor<T>? _lossSin;
    private Tensor<T>? _lossMel;

    /// <summary>Training updates taken.</summary>
    public int TrainingUpdates { get; private set; }

    /// <summary>
    /// Pre-trains the style encoder on one prompt (the release's style encoder training, <c>simbert.py</c>'s
    /// <c>StylePretrainLoss</c>): the cross-entropy of its pitch, speed, energy and emotion classifiers on the pooled
    /// SimBERT output.
    /// </summary>
    /// <returns>The step's loss.</returns>
    public T TrainStyleEncoder(IReadOnlyList<int> promptIds, int pitch, int speed, int energy, int emotion)
    {
        ThrowIfDisposed();
        if (!HasPaperLayers) throw new NotSupportedException("The style encoder exists in native mode only.");
        var o = _options;
        void Check(int label, int count, string name)
        {
            if (label < 0 || label >= count) throw new ArgumentOutOfRangeException(name, $"Labels lie in [0, {count}).");
        }
        Check(pitch, o.PitchLabels, nameof(pitch));
        Check(speed, o.SpeedLabels, nameof(speed));
        Check(energy, o.EnergyLabels, nameof(energy));
        Check(emotion, o.EmotionLabels, nameof(emotion));
        var ids = Ids(promptIds);
        _styleStep = true;
        try
        {
            return TrainWithCustomObjective(ids, ids, (_, _) =>
            {
                // ClassificationHead: dropout 0.1 on the pooled output, then the linear classifier.
                var pooled = CodecBased.T5Seq2Seq<T>.Dropout(Engine, Pooled(promptIds), 0.1, _trainingRandom);
                Tensor<T> CrossEntropy(DenseLayer<T> head, int label, int count)
                {
                    var logp = Engine.TensorLogSoftmax(head.Forward(pooled), axis: -1);
                    return Engine.TensorNegate(Engine.TensorSlice(Engine.Reshape(logp, new[] { count }), new[] { label }, new[] { 1 }));
                }
                return Engine.TensorAdd(Engine.TensorAdd(CrossEntropy(_pitchHead!, pitch, o.PitchLabels), CrossEntropy(_speedHead!, speed, o.SpeedLabels)),
                    Engine.TensorAdd(CrossEntropy(_energyHead!, energy, o.EnergyLabels), CrossEntropy(_emotionHead!, emotion, o.EmotionLabels)));
            }, StyleOptimizer());
        }
        finally
        {
            _styleStep = false;
        }
    }

    private bool _styleStep;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _styleOptimizer;

    // The release ships the style loss (simbert.py's StylePretrainLoss) but not its training loop; the step uses the
    // released Adam settings.
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> StyleOptimizer() => _styleOptimizer ??= new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this,
        new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
        {
            InitialLearningRate = _options.LearningRate, Beta1 = _options.Beta1, Beta2 = _options.Beta2, Epsilon = _options.Epsilon,
            EnableGradientClipping = false,
        });

    /// <inheritdoc />
    /// <remarks>A token/waveform pair trains with speaker 0 and an empty prompt; use the typed sample for the speaker and
    /// prompt.</remarks>
    protected override bool TryApplyLearningRule(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        Train(new TtsTrainingSample<T> { Tokens = input, Audio = expectedOutput, SpeakerId = 0, Description = Ids(StyleTokens(string.Empty)) });
        return true;
    }

    /// <summary>What a step reads that its trained modules do not produce.</summary>
    internal sealed class Step
    {
        public int Seed;
        public int[] Phonemes = Array.Empty<int>();
        public int Speaker;
        public Tensor<T> Style = default!;
        public Tensor<T> Content = default!;
        public Tensor<T> Mel = default!;       // [T, mels] the alignment's target
        public Tensor<T> Pitch = new(new[] { 0 });     // [T], normalized
        public Tensor<T> Energy = new(new[] { 0 });
        public int Start;                      // the vocoder segment's first frame
        public Tensor<T> Wave = default!;      // [segment · hop]
    }

    /// <inheritdoc />
    protected override T TrainOnSample(TtsTrainingSample<T> sample)
    {
        ThrowIfDisposed();
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        if (!HasPaperLayers) throw new NotSupportedException("The paper objective needs the paper's layers; this model was built from caller-supplied layers.");
        Validation.Guard.NotNull(sample);
        var p = Prepare(sample, _trainingRandom.Next(), training: true);
        var fake = Generate(p, training: true, withLosses: false).Wave;
        _discriminatorStep = true;
        try
        {
            var real = p.Wave;
            var detached = new Tensor<T>(fake._shape, new Vector<T>(fake.ToArray()));
            TrainWithCustomObjective(sample.Tokens, sample.Tokens, (_, _) => DiscriminatorLoss(Discriminators.Forward(real), Discriminators.Forward(detached)),
                Optimizer(discriminator: true));
        }
        finally
        {
            _discriminatorStep = false;
        }
        var loss = TrainWithCustomObjective(sample.Tokens, sample.Tokens, (_, _) => Generate(p, training: true, withLosses: true).Loss!, Optimizer(discriminator: false));
        TrainingUpdates++;
        return loss;
    }

    /// <inheritdoc />
    /// <remarks>The generator objective without its adversarial terms, with dropout off and fixed draws.</remarks>
    public override T EvaluateTrainingObjective(TtsTrainingSample<T> sample)
    {
        ThrowIfDisposed();
        Validation.Guard.NotNull(sample);
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            var p = Prepare(sample, _options.SamplingSeed, training: false);
            return Generate(p, training: false, withLosses: true, adversarialTerms: false).Loss![0];
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    // ---------------------------------------------------------------- data, as the released loader prepares it

    private Step Prepare(TtsTrainingSample<T> sample, int seed, bool training)
    {
        var o = _options;
        var recording = sample.Audio ?? throw new ArgumentException($"{GetType().Name} trains on the recording; set Audio.", nameof(sample));
        var audio = Engine.Reshape(recording, new[] { recording.Length });
        if (o.TrimSilence) audio = Trim(audio);
        var p = new Step
        {
            Seed = seed,
            Phonemes = ToIds(sample.Tokens, nameof(sample)),
            Speaker = sample.SpeakerId ?? 0,
        };
        p.Style = StyleEmbedding(DescriptionIds(sample.Description ?? throw new ArgumentException($"{GetType().Name} trains with the emotion prompt; set Description.", nameof(sample))));
        p.Content = StyleEmbedding(sample.Transcript is null ? StyleTokens(string.Empty) : DescriptionIds(sample.Transcript));
        _targetMel ??= new TacotronSpectrogram(o.SampleRate, o.FftSize, o.HopSize, o.WindowLength, o.MelChannels, o.MelMinFrequency, o.MelMaxFrequency);
        p.Mel = _targetMel.LogMel(audio);                                      // [frames, mels]
        int frames = p.Mel.Shape[0];
        p.Pitch = Standardized(Fit(ContinuousPitch(new WorldPitchDetector<T>(o.SampleRate).EstimateF0(audio, 1000.0 * o.HopSize / o.SampleRate).F0), frames),
            o.PitchMean, o.PitchStd);
        p.Energy = Standardized(Fit(FrameEnergy(audio), frames), o.EnergyMean, o.EnergyStd);
        // The vocoder's random segment (get_random_segments: ⌊U · (T − segment)⌋), zero-padded when the utterance is
        // shorter.
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed);
        int segment = o.SegmentFrames;
        p.Start = training ? (int)(random.NextDouble() * Math.Max(0, frames - segment)) : 0;
        int samples = segment * o.HopSize, from = p.Start * o.HopSize;
        int available = Math.Max(0, Math.Min(samples, audio.Length - from));
        p.Wave = Fit(available > 0 ? Engine.TensorSlice(audio, new[] { from }, new[] { available }) : new Tensor<T>(new[] { 0 }), samples);
        return p;
    }

    private int[] DescriptionIds(Tensor<T> ids) => Engine.TensorRound(ids).Cast<int>().ToArray();

    /// <summary>The loader's trim: from the first to the last sample above 0.5 % of the peak, then 50 ms of silence each side.</summary>
    internal Tensor<T> Trim(Tensor<T> audio)
    {
        var magnitude = Engine.TensorAbs(Engine.Reshape(audio, new[] { audio.Length }));
        T threshold = NumOps.Multiply(Engine.TensorMaxValue(magnitude), NumOps.FromDouble(0.005));
        int[] above = Engine.TensorNonzero(Engine.TensorGreaterThan(magnitude, threshold)).ToArray();
        if (above.Length == 0) return audio;
        int start = above[0], end = above[above.Length - 1];
        int pad = _options.SampleRate / 20;
        // wav[start:end] — the last sample above the threshold excluded, as the reference slices.
        var kept = end > start ? Engine.TensorSlice(Engine.Reshape(audio, new[] { audio.Length }), new[] { start }, new[] { end - start })
            : new Tensor<T>(new[] { 0 });
        return Engine.PadNd(kept, new[] { pad, pad }, AiDotNet.Tensors.Engines.PadMode.Constant, NumOps.Zero);
    }

    /// <summary>The pitch made continuous (feats.Pitch: leading and trailing zeros take the first and last voiced value,
    /// gaps linearly interpolated).</summary>
    internal static Tensor<T> ContinuousPitch(Tensor<T> pitch) => PitchWaveletTransform.FillUnvoiced(pitch);

    /// <summary>feats.Energy: the L2 norm of each frame of librosa's centred STFT magnitude.</summary>
    /// <remarks>The power, its sum and the 1e-10 floor are formed in double and only the norm is narrowed to T: in
    /// half precision the squares of a scaled spectrum underflow to zero for quiet frames, and so does the floor.
    /// </remarks>
    internal Tensor<T> FrameEnergy(Tensor<T> audio)
    {
        var o = _options;
        _energyStft ??= new CenteredComplexStft<T>(Engine, o.FftSize, o.HopSize, o.WindowLength);
        using var _ = new NoGradScope<T>();
        var (re, im) = _energyStft.Forward(Engine.Reshape(audio, new[] { audio.Length }));
        int frames = re.Shape[2];
        var wideRe = TensorCopyHelper.CastLogical<T, double>(re);
        var wideIm = TensorCopyHelper.CastLogical<T, double>(im);
        var power = Engine.TensorAdd(Engine.TensorSquare(wideRe), Engine.TensorSquare(wideIm));
        var sum = Engine.ReduceSum(power, new[] { 1 }, false);
        var norm = Engine.TensorSqrt(Engine.TensorMax(sum, 1e-10));
        return Engine.Reshape(norm.Cast<T>(), new[] { frames });
    }

    // The collated targets: truncated or zero-padded to the mel frames.
    private Tensor<T> Fit(Tensor<T> values, int frames)
    {
        int length = values.Length, kept = Math.Min(frames, length);
        var flat = Engine.Reshape(values, new[] { length });
        if (kept < length) flat = Engine.TensorSlice(flat, new[] { 0 }, new[] { kept });
        return kept < frames ? Engine.PadNd(flat, new[] { 0, frames - kept }, AiDotNet.Tensors.Engines.PadMode.Constant, NumOps.Zero) : flat;
    }

    // (v − mean) / std.
    private Tensor<T> Standardized(Tensor<T> values, double mean, double std)
        => Engine.TensorDivideScalar(Engine.TensorSubtractScalar(values, NumOps.FromDouble(mean)), NumOps.FromDouble(std));

    // ---------------------------------------------------------------- the generator's pass and losses

    internal sealed class Pass
    {
        public Tensor<T> Wave = default!;
        public Tensor<T>? Loss;
        public Tensor<T> Conditioned = default!;
        public Tensor<T> LogP = default!;
        public int[] Durations = Array.Empty<int>();
        public Tensor<T> PitchTargets = new(new[] { 0 });
        public Tensor<T> EnergyTargets = new(new[] { 0 });
        public Tensor<T> PitchOut = default!, EnergyOut = default!, DurationOut = default!;
        public Tensor<T> Latent = default!;
        public Tensor<T>? Mel, Duration, Pitch, Energy, ForwardSum, Binarization;
    }

    /// <summary>
    /// The joint generator's step (<c>JETSGenerator</c> and <c>train_am_vocoder_joint.py</c>): the alignment's
    /// log-probabilities, its Viterbi durations and the binarization and forward-sum losses; pitch and energy averaged
    /// over each phoneme's frames; the predictors; the latent from the teacher-forced durations, pitch and energy; HiFi-GAN
    /// on a 32-frame segment of it; the losses mel L1 × 45, log-duration L1, pitch and energy L1, forward-sum × 2,
    /// binarization × 2, and HiFi-GAN's adversarial and feature-matching losses.
    /// </summary>
    internal Pass Generate(Step p, bool training, bool withLosses, bool adversarialTerms = true)
    {
        var o = _options;
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(unchecked(p.Seed * 31 + 1));
        var x = Encode(p.Phonemes, p.Speaker, p.Style, p.Content, training, random);
        int n = p.Phonemes.Length, frames = p.Mel.Shape[0];
        var logP = AlignmentNetwork.Forward(x, p.Mel);
        var path = EmotiVoiceAlignment<T>.Viterbi(logP);
        var durations = EmotiVoiceAlignment<T>.Durations(path, n);
        // Each phoneme's mean pitch and energy over its frames, accumulated as Σ v_f / d in frame order.
        int covered = durations.Sum();
        var owner = new int[covered];
        for (int i = 0, f = 0; i < n; i++)
            for (int k = 0; k < durations[i]; k++) owner[f++] = i;
        var ownerIndex = new Tensor<int>(owner, new[] { covered });
        var frameDuration = Engine.TensorIndexSelect(new Tensor<int>(durations, new[] { n }), ownerIndex, 0).Cast<T>();
        Tensor<T> PhonemeMean(Tensor<T> values) => Engine.TensorIndexPut(new Tensor<T>(new[] { n }), new[] { ownerIndex },
            Engine.TensorDivide(Engine.TensorSlice(Engine.Reshape(values, new[] { values.Length }), new[] { 0 }, new[] { covered }), frameDuration), true);
        var pitchTargets = PhonemeMean(p.Pitch);
        var energyTargets = PhonemeMean(p.Energy);
        var durationValues = new Tensor<int>(durations, new[] { n }).Cast<T>();
        var pitchOut = PitchPredictor.Forward(x, training, random);
        var energyOut = EnergyPredictor.Forward(x, training, random);
        var durationOut = DurationPredictor.Forward(x, training, random);
        var latent = DecodeLatent(x, pitchTargets, energyTargets, durationValues, frames, training, random);
        // The segment of the latent the vocoder reads, zero-padded past the end.
        int segment = o.SegmentFrames;
        int available = Math.Min(segment, frames - p.Start);
        var slice = Engine.TensorSlice(latent, new[] { p.Start, 0 }, new[] { available, o.MelChannels });
        if (available < segment) slice = Engine.TensorConcatenate(new[] { slice, new Tensor<T>(new[] { segment - available, o.MelChannels }) }, 0);
        var wave = Vocoder.Forward(Engine.Reshape(Engine.TensorTranspose(slice), new[] { 1, o.MelChannels, segment }));
        wave = Engine.Reshape(wave, new[] { wave.Length });
        var pass = new Pass
        {
            Wave = wave, Conditioned = x, LogP = logP, Durations = durations, PitchTargets = pitchTargets, EnergyTargets = energyTargets,
            PitchOut = pitchOut, EnergyOut = energyOut, DurationOut = durationOut, Latent = latent,
        };
        if (!withLosses) return pass;

        var mel = L1(TrainingMel(wave), TrainingMel(p.Wave));
        var logDurations = Engine.TensorLog(Engine.TensorAddScalar(durationValues, NumOps.One));
        var duration = L1(durationOut, logDurations);
        var pitch = L1(pitchOut, pitchTargets);
        var energy = L1(energyOut, energyTargets);
        var forwardSum = AlignmentNetwork.ForwardSumLoss(logP);
        var binarization = AlignmentNetwork.BinarizationLoss(logP, path);
        (pass.Mel, pass.Duration, pass.Pitch, pass.Energy, pass.ForwardSum, pass.Binarization) = (mel, duration, pitch, energy, forwardSum, binarization);
        var total = Engine.TensorAdd(Engine.TensorAdd(Scaled(mel, o.MelLossWeight), duration), Engine.TensorAdd(pitch, energy));
        total = Engine.TensorAdd(total, Engine.TensorAdd(Scaled(forwardSum, o.ForwardSumWeight), Scaled(binarization, o.BinarizationWeight)));
        if (adversarialTerms)
        {
            List<(Tensor<T> Score, List<Tensor<T>> Features)> real;
            using (new NoGradScope<T>()) real = Discriminators.Forward(p.Wave);
            var generated = Discriminators.Forward(wave);
            Tensor<T>? adversarial = null;
            foreach (var (score, _) in generated)
            {
                var term = Mean(Square(Engine.TensorAddScalar(Engine.TensorNegate(score), NumOps.One)));
                adversarial = adversarial is null ? term : Engine.TensorAdd(adversarial, term);
            }
            Tensor<T>? matching = null;
            for (int dIndex = 0; dIndex < real.Count; dIndex++)
                for (int l = 0; l < real[dIndex].Features.Count; l++)
                {
                    var term = L1(generated[dIndex].Features[l], real[dIndex].Features[l]);
                    matching = matching is null ? term : Engine.TensorAdd(matching, term);
                }
            total = Engine.TensorAdd(total, Engine.TensorAdd(adversarial!, Scaled(matching!, 2.0)));
        }
        pass.Loss = Engine.Reshape(total, new[] { 1 });
        return pass;
    }

    private Tensor<T> Scaled(Tensor<T> x, double factor) => Engine.TensorMultiplyScalar(x, NumOps.FromDouble(factor));
    private Tensor<T> Square(Tensor<T> x) => Engine.TensorMultiply(x, x);

    private Tensor<T> Mean(Tensor<T> x) =>
        Engine.TensorMultiplyScalar(Engine.ReduceSum(Engine.Reshape(x, new[] { x.Length }), new[] { 0 }, keepDims: true), NumOps.FromDouble(1.0 / x.Length));

    private Tensor<T> L1(Tensor<T> a, Tensor<T> b) =>
        Mean(Engine.TensorAbs(Engine.TensorSubtract(Engine.Reshape(a, new[] { a.Length }), Engine.Reshape(b, new[] { b.Length }))));

    /// <summary>The training mel (<c>mel_process.mel_spectrogram_torch</c>): reflect padding of (n_fft − hop)/2, an
    /// uncentred Hann STFT, magnitude √(re² + im² + 1e-6), librosa's Slaney mel basis and log(max(·, 1e-5)) — on the
    /// tape, <c>[mels, frames]</c>.</summary>
    internal Tensor<T> TrainingMel(Tensor<T> audio)
    {
        var o = _options;
        int fft = o.FftSize, hop = o.HopSize, bins = fft / 2 + 1;
        if (_lossCos is null)
        {
            // The windowed DFT tables [fft, bins] (fixed constants).
            var cos = new double[fft, bins];
            var sin = new double[fft, bins];
            int offset = (fft - o.WindowLength) / 2;
            for (int i = 0; i < o.WindowLength; i++)
            {
                double w = 0.5 - 0.5 * Math.Cos(2 * Math.PI * i / o.WindowLength);
                for (int k = 0; k < bins; k++)
                {
                    double a = 2 * Math.PI * (offset + i) * k / fft;
                    cos[offset + i, k] = w * Math.Cos(a);
                    sin[offset + i, k] = -w * Math.Sin(a);
                }
            }
            _lossCos = AiDotNet.Audio.Features.EngineStft<T>.Table(cos);
            _lossSin = AiDotNet.Audio.Features.EngineStft<T>.Table(sin);
            _lossMel = AiDotNet.Audio.Features.EngineStft<T>.Table(new TacotronSpectrogram(o.SampleRate, fft, hop, o.WindowLength, o.MelChannels,
                o.MelMinFrequency, o.MelMaxFrequency).MelBasis);
        }
        int pad = (fft - hop) / 2;
        var padded = VocoderOps.ReflectPad(Engine, Engine.Reshape(audio, new[] { 1, 1, audio.Length }), pad, pad);
        int length = padded.Shape[2], frames = 1 + (length - fft) / hop;
        var starts = new int[frames * fft];
        for (int f = 0; f < frames; f++)
            for (int i = 0; i < fft; i++) starts[f * fft + i] = f * hop + i;
        var index = new Tensor<int>(starts, new[] { frames * fft });
        var framed = Engine.Reshape(Engine.TensorIndexSelect(Engine.Reshape(padded, new[] { length, 1 }), index, 0), new[] { frames, fft });
        var re = Engine.TensorMatMul(framed, _lossCos);
        var im = Engine.TensorMatMul(framed, _lossSin!);
        var magnitude = Engine.TensorSqrt(Engine.TensorAddScalar(Engine.TensorAdd(Square(re), Square(im)), NumOps.FromDouble(1e-6)));
        var mel = Engine.TensorMatMul(_lossMel!, Engine.TensorTranspose(magnitude));
        return Engine.TensorLog(Engine.TensorMax(mel, NumOps.FromDouble(1e-5)));
    }

    /// <summary>HiFi-GAN's discriminator loss: Σ mean (1 − D(y))² + mean D(ŷ)² over the period and scale discriminators.</summary>
    private Tensor<T> DiscriminatorLoss(List<(Tensor<T> Score, List<Tensor<T>> Features)> real, List<(Tensor<T> Score, List<Tensor<T>> Features)> fake)
    {
        Tensor<T>? loss = null;
        for (int d = 0; d < real.Count; d++)
        {
            var term = Engine.TensorAdd(Mean(Square(Engine.TensorAddScalar(Engine.TensorNegate(real[d].Score), NumOps.One))), Mean(Square(fake[d].Score)));
            loss = loss is null ? term : Engine.TensorAdd(loss, term);
        }
        return Engine.Reshape(loss!, new[] { 1 });
    }

    // ---------------------------------------------------------------- optimizers

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> Optimizer(bool discriminator)
    {
        var o = _options;
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> Build(string component)
        {
            var options = new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = o.LearningRate,
                Beta1 = o.Beta1,
                Beta2 = o.Beta2,
                Epsilon = o.Epsilon,
                EnableGradientClipping = false,
            };
            if (o.UpdatesPerEpoch > 0)
            {
                // ExponentialLR stepped once per epoch: the rate times γ after every epoch of updates.
                options.LearningRateScheduler = new StepLRScheduler(o.LearningRate, o.UpdatesPerEpoch, o.LearningRateDecay);
                options.SchedulerStepMode = SchedulerStepMode.StepPerBatch;
            }
            return PaperOptimizerFactory.VerifyHandBuilt(this, new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this, options), component);
        }
        return discriminator ? _discriminatorOptimizer ??= Build("discriminator") : _generatorOptimizer ??= Build("generator");
    }

    /// <inheritdoc />
    /// <remarks>The discriminator's step trains the period and scale discriminators; the generator's step the acoustic
    /// model and HiFi-GAN; the style encoder's pre-training SimBERT and its classifiers. The style encoder is frozen
    /// during TTS training, as the release precomputes its embeddings.</remarks>
    protected override IReadOnlyList<Tensor<T>> SelectTrainableParametersForTraining(IReadOnlyList<Tensor<T>> parameters)
    {
        if (!HasPaperLayers) return parameters;
        IEnumerable<LayerBase<T>> layers = _styleStep ? _styleLayers
            : _discriminatorStep ? _discriminatorLayers
            : _acousticLayers.Concat(_decoderLayers).Concat(_vocoderLayers);
        var trained = new HashSet<Tensor<T>>(Training.TapeTrainingStep<T>.CollectParameters(layers.Cast<ILayer<T>>().ToList(), -1),
            Helpers.TensorReferenceComparer<Tensor<T>>.Instance);
        return parameters.Where(trained.Contains).ToList();
    }
}
