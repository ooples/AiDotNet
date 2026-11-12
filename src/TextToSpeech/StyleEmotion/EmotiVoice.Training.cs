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
        public double[] Pitch = Array.Empty<double>();   // [T], normalized
        public double[] Energy = Array.Empty<double>();
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
        var audio = (sample.Audio ?? throw new ArgumentException($"{GetType().Name} trains on the recording; set Audio.", nameof(sample)))
            .ToVector().Select(v => NumOps.ToDouble(v)).ToArray();
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
        var mel = _targetMel.LogMel(audio);
        int frames = mel.GetLength(0);
        p.Mel = new Tensor<T>(new[] { frames, o.MelChannels });
        for (int f = 0; f < frames; f++)
            for (int m = 0; m < o.MelChannels; m++) p.Mel[f, m] = NumOps.FromDouble(mel[f, m]);
        p.Pitch = Fit(ContinuousPitch(new WorldPitchDetector<T>(o.SampleRate).EstimateF0(audio, 1000.0 * o.HopSize / o.SampleRate).F0), frames)
            .Select(v => (v - o.PitchMean) / o.PitchStd).ToArray();
        p.Energy = Fit(FrameEnergy(audio), frames).Select(v => (v - o.EnergyMean) / o.EnergyStd).ToArray();
        // The vocoder's random segment (get_random_segments: ⌊U · (T − segment)⌋), zero-padded when the utterance is
        // shorter.
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed);
        int segment = o.SegmentFrames;
        p.Start = training ? (int)(random.NextDouble() * Math.Max(0, frames - segment)) : 0;
        p.Wave = new Tensor<T>(new[] { segment * o.HopSize });
        for (int i = 0; i < segment * o.HopSize; i++)
        {
            int index = p.Start * o.HopSize + i;
            if (index < audio.Length) p.Wave[i] = NumOps.FromDouble(audio[index]);
        }
        return p;
    }

    private int[] DescriptionIds(Tensor<T> ids) => ids.ToVector().Select(v => (int)Math.Round(NumOps.ToDouble(v))).ToArray();

    /// <summary>The loader's trim: from the first to the last sample above 0.5 % of the peak, then 50 ms of silence each side.</summary>
    internal double[] Trim(double[] audio)
    {
        double peak = audio.Max(Math.Abs), threshold = peak * 0.005;
        int start = Array.FindIndex(audio, v => Math.Abs(v) > threshold), end = Array.FindLastIndex(audio, v => Math.Abs(v) > threshold);
        if (start < 0) return audio;
        int pad = _options.SampleRate / 20;
        var trimmed = new double[end - start + 2 * pad];
        // wav[start:end] — the last sample above the threshold excluded, as the reference slices.
        Array.Copy(audio, start, trimmed, pad, end - start);
        return trimmed;
    }

    /// <summary>The pitch made continuous (feats.Pitch: leading and trailing zeros take the first and last voiced value,
    /// gaps linearly interpolated).</summary>
    internal static double[] ContinuousPitch(double[] pitch)
    {
        if (pitch.All(v => v == 0)) return pitch;
        int first = Array.FindIndex(pitch, v => v != 0), last = Array.FindLastIndex(pitch, v => v != 0);
        var result = (double[])pitch.Clone();
        for (int i = 0; i < first; i++) result[i] = pitch[first];
        for (int i = last + 1; i < pitch.Length; i++) result[i] = pitch[last];
        int previous = first;
        for (int i = first + 1; i <= last; i++)
        {
            if (pitch[i] == 0) continue;
            for (int j = previous + 1; j < i; j++) result[j] = pitch[previous] + (pitch[i] - pitch[previous]) * (j - previous) / (i - previous);
            previous = i;
        }
        return result;
    }

    /// <summary>feats.Energy: the L2 norm of each frame of librosa's centred STFT magnitude.</summary>
    internal double[] FrameEnergy(double[] audio)
    {
        var o = _options;
        _energyStft ??= new CenteredComplexStft<T>(Engine, o.FftSize, o.HopSize, o.WindowLength);
        using var _ = new NoGradScope<T>();
        var (re, im) = _energyStft.Forward(new Tensor<T>(new[] { audio.Length }, new Vector<T>(audio.Select(v => NumOps.FromDouble(v)).ToArray())));
        int bins = re.Shape[1], frames = re.Shape[2];
        var energy = new double[frames];
        for (int f = 0; f < frames; f++)
        {
            double sum = 0;
            for (int k = 0; k < bins; k++)
            {
                double a = NumOps.ToDouble(re[0, k, f]), b = NumOps.ToDouble(im[0, k, f]);
                sum += a * a + b * b;
            }
            energy[f] = Math.Sqrt(Math.Max(sum, 1e-10));
        }
        return energy;
    }

    // The collated targets: truncated or zero-padded to the mel frames.
    private static double[] Fit(double[] values, int frames)
    {
        var fitted = new double[frames];
        Array.Copy(values, fitted, Math.Min(frames, values.Length));
        return fitted;
    }

    // ---------------------------------------------------------------- the generator's pass and losses

    internal sealed class Pass
    {
        public Tensor<T> Wave = default!;
        public Tensor<T>? Loss;
        public Tensor<T> Conditioned = default!;
        public Tensor<T> LogP = default!;
        public int[] Durations = Array.Empty<int>();
        public double[] PitchTargets = Array.Empty<double>();
        public double[] EnergyTargets = Array.Empty<double>();
        public Tensor<T> PitchOut = default!, EnergyOut = default!, DurationOut = default!;
        public Tensor<T> Latent = default!;
        public Tensor<T>? Mel, Duration, Pitch, Energy, ForwardSum, Binarization;
    }

    private Tensor<T> ValuesOf(double[] values) => new(new[] { values.Length }, new Vector<T>(values.Select(v => NumOps.FromDouble(v)).ToArray()));

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
        var matrix = new double[frames, n];
        for (int t = 0; t < frames; t++)
            for (int i = 0; i < n; i++) matrix[t, i] = NumOps.ToDouble(logP[t, i]);
        var path = EmotiVoiceAlignment<T>.Viterbi(matrix);
        var durations = EmotiVoiceAlignment<T>.Durations(path, n);
        var pitchTargets = new double[n];
        var energyTargets = new double[n];
        int frame = 0;
        for (int i = 0; i < n; i++)
        {
            for (int f = 0; f < durations[i]; f++)
            {
                pitchTargets[i] += p.Pitch[frame + f] / durations[i];
                energyTargets[i] += p.Energy[frame + f] / durations[i];
            }
            frame += durations[i];
        }
        var pitchOut = PitchPredictor.Forward(x, training, random);
        var energyOut = EnergyPredictor.Forward(x, training, random);
        var durationOut = DurationPredictor.Forward(x, training, random);
        var latent = DecodeLatent(x, ValuesOf(pitchTargets), ValuesOf(energyTargets), ValuesOf(durations.Select(v => (double)v).ToArray()), frames, training, random);
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
        var logDurations = ValuesOf(durations.Select(v => Math.Log(v + 1.0)).ToArray());
        var duration = L1(durationOut, logDurations);
        var pitch = L1(pitchOut, ValuesOf(pitchTargets));
        var energy = L1(energyOut, ValuesOf(energyTargets));
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
            _lossCos = new Tensor<T>(new[] { fft, bins });
            _lossSin = new Tensor<T>(new[] { fft, bins });
            int offset = (fft - o.WindowLength) / 2;
            for (int i = 0; i < o.WindowLength; i++)
            {
                double w = 0.5 - 0.5 * Math.Cos(2 * Math.PI * i / o.WindowLength);
                for (int k = 0; k < bins; k++)
                {
                    double a = 2 * Math.PI * (offset + i) * k / fft;
                    _lossCos[offset + i, k] = NumOps.FromDouble(w * Math.Cos(a));
                    _lossSin![offset + i, k] = NumOps.FromDouble(-w * Math.Sin(a));
                }
            }
            var basis = new TacotronSpectrogram(o.SampleRate, fft, hop, o.WindowLength, o.MelChannels, o.MelMinFrequency, o.MelMaxFrequency).MelBasis;
            _lossMel = new Tensor<T>(new[] { o.MelChannels, bins });
            for (int m = 0; m < o.MelChannels; m++)
                for (int k = 0; k < bins; k++) _lossMel[m, k] = NumOps.FromDouble(basis[m, k]);
        }
        int pad = (fft - hop) / 2;
        var padded = VocoderOps.ReflectPad(Engine, Engine.Reshape(audio, new[] { 1, 1, audio.Length }), pad, pad);
        int length = padded.Shape[2], frames = 1 + (length - fft) / hop;
        var index = new Tensor<int>(new[] { frames * fft });
        for (int f = 0; f < frames; f++)
            for (int i = 0; i < fft; i++) index[f * fft + i] = f * hop + i;
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
