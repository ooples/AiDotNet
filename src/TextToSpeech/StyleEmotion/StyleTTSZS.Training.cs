using AiDotNet.Audio.Features;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.Vocoders;

namespace AiDotNet.TextToSpeech.StyleEmotion;

/// <summary>The modules a StyleTTS-ZS training step updates (Li et al. 2024, §III and §IV-A).</summary>
public enum StyleTtsZsTrainingStage
{
    /// <summary>The acoustic synthesizer, the prosody autoencoder, the style diffusion denoiser and the student's
    /// pre-training (predicting the time-varying style at σ = 0 from zero noise), against the discriminators.</summary>
    Training,

    /// <summary>Distillation of the denoiser's guided DDIM sampling into the one-step student (Eq. 6).</summary>
    Distillation,
}

public partial class StyleTTSZS<T>
{
    private StyleTts2Losses<T>? _losses;
    private StyleTts2ModuleOptimizer<T>? _generatorOptimizer;
    private StyleTts2ModuleOptimizer<T>? _discriminatorOptimizer;
    private StyleTtsZsTrainingStage _optimizerStage;
    private Dictionary<Tensor<T>, string>? _moduleOfParameter;
    private HashSet<string> _steppedModules = new();
    private CenteredComplexStft<T>? _melStft;
    private Tensor<T>? _melBasis;

    /// <summary>The modules a training step updates.</summary>
    public StyleTtsZsTrainingStage CurrentStage { get; set; } = StyleTtsZsTrainingStage.Training;

    /// <summary>Training updates taken (they decide when the aligner's objectives and the adversarial terms start).</summary>
    public int TrainingUpdates { get; private set; }

    /// <summary>Samples skipped because their segment was shorter than <see cref="StyleTtsBaseOptions.MinTrainingFrames"/>.</summary>
    public int SkippedSamples { get; private set; }

    private StyleTts2Losses<T> Losses => _losses ??= new StyleTts2Losses<T>(Engine, _options.SampleRate, _options.SlmSampleRate);

    private bool TmaActive => TrainingUpdates >= _options.TmaStartStep;

    private static readonly string[] GeneratorModules =
    {
        "prompt_text_encoder", "decoder", "bert", "bert_encoder", "prosody_encoder", "duration_decoder", "prosody_decoder", "diffusion", "student",
    };

    private static readonly string[] DiscriminatorModules = { "mpd", "msd", "waveform_discriminator", "duration_discriminator", "prosody_discriminator" };

    // ---------------------------------------------------------------- entry points

    /// <inheritdoc />
    /// <remarks>A token/waveform pair is a full StyleTTS-ZS training sample (the prompt is a segment of it).</remarks>
    protected override bool TryApplyLearningRule(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        Train(new TtsTrainingSample<T> { Tokens = input, Audio = expectedOutput });
        return true;
    }

    /// <inheritdoc />
    protected override T TrainOnSample(TtsTrainingSample<T> sample)
    {
        ThrowIfDisposed();
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        if (!HasPaperLayers) throw new NotSupportedException("The paper objective needs the paper's layers; this model was built from caller-supplied layers.");
        Validation.Guard.NotNull(sample);
        var ids = ToIds(sample.Tokens, nameof(sample));
        var wave = Padded(sample.Audio ?? throw new ArgumentException($"{GetType().Name} trains on the recording; set Audio.", nameof(sample)));
        var mel = MelOf(wave);
        int seed = _trainingRandom.Next();
        var prepared = Prepare(ids, wave, mel, seed, training: true);
        if (prepared is null)
        {
            SkippedSamples++;
            return NumOps.Zero;
        }
        if (CurrentStage == StyleTtsZsTrainingStage.Distillation)
            return RunStep(sample.Tokens, new[] { "student" }, discriminator: false, () => DistillationLoss(prepared, training: true));

        // The discriminators see the generator's outputs from this step's draws, before its update.
        (Tensor<T> Wave, Tensor<T> Durations, Tensor<T> Prosody, Tensor<T> Style)? fake;
        using (new NoGradScope<T>()) fake = GeneratorPass(prepared, training: true, outputsOnly: true).Fakes;
        RunStep(sample.Tokens, DiscriminatorModules, discriminator: true, () => DiscriminatorLoss(prepared, fake!.Value));
        var modules = new List<string>(GeneratorModules);
        if (TmaActive) modules.AddRange(new[] { "text_aligner", "pitch_extractor" });
        var loss = RunStep(sample.Tokens, modules, discriminator: false, () => GeneratorPass(prepared, training: true).Loss!);
        TrainingUpdates++;
        return loss;
    }

    /// <inheritdoc />
    /// <remarks>The current stage's generator objective without its adversarial terms, with dropout off and fixed draws.</remarks>
    public override T EvaluateTrainingObjective(TtsTrainingSample<T> sample)
    {
        ThrowIfDisposed();
        Validation.Guard.NotNull(sample);
        var ids = ToIds(sample.Tokens, nameof(sample));
        var wave = Padded(sample.Audio ?? throw new ArgumentException($"{GetType().Name} trains on the recording; set Audio.", nameof(sample)));
        var mel = MelOf(wave);
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            var prepared = Prepare(ids, wave, mel, _options.SamplingSeed, training: false)
                ?? throw new ArgumentException($"{GetType().Name} evaluates segments of at least {_options.MinTrainingFrames} frames; this one has {mel.Shape[1]}.", nameof(sample));
            return CurrentStage == StyleTtsZsTrainingStage.Distillation
                ? DistillationLoss(prepared, training: false)[0]
                : GeneratorPass(prepared, training: false, adversarialTerms: false).Loss![0];
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    // ---------------------------------------------------------------- data

    private Tensor<T> Padded(Tensor<T> audio)
    {
        int pad = _options.WavePadding, length = audio.Length;
        var wave = new Tensor<T>(new[] { length + 2 * pad });
        for (int i = 0; i < length; i++) wave[pad + i] = audio[i];
        return wave;
    }

    private Tensor<T> MelOf(Tensor<T> wave)
    {
        var mel = ComputeMel(wave);
        int frames = mel.Shape[1] / 2 * 2;
        return frames == mel.Shape[1] ? mel : Engine.TensorSlice(mel, new[] { 0, 0 }, new[] { _options.MelChannels, frames });
    }

    /// <summary>The model's normalized log-mel spectrogram <c>[mels, frames]</c> of a waveform, on the tape.</summary>
    private Tensor<T> MelOnTape(Tensor<T> audio)
    {
        var o = _options;
        _melStft ??= new CenteredComplexStft<T>(Engine, o.FftSize, o.HopSize, o.WindowLength);
        if (_melBasis is null)
        {
            var basis = MelFilterbanks.Htk(o.SampleRate, o.FftSize, o.MelChannels, 0.0, Math.Floor(o.SampleRate / 2.0));
            _melBasis = new Tensor<T>(new[] { o.MelChannels, o.FftSize / 2 + 1 });
            for (int m = 0; m < o.MelChannels; m++)
                for (int k = 0; k < o.FftSize / 2 + 1; k++) _melBasis[m, k] = NumOps.FromDouble(basis[m, k]);
        }
        var (re, im) = _melStft.Forward(Engine.Reshape(audio, new[] { audio.Length }));
        int bins = re.Shape[1], frames = re.Shape[2];
        var power = Engine.TensorAdd(Engine.TensorMultiply(re, re), Engine.TensorMultiply(im, im));
        var mel = Engine.TensorMatMul(_melBasis, Engine.Reshape(power, new[] { bins, frames }));
        var log = Engine.TensorLog(Engine.TensorAddScalar(mel, NumOps.FromDouble(1e-5)));
        return Engine.TensorMultiplyScalar(Engine.TensorAddScalar(log, NumOps.FromDouble(-o.MelMean)), NumOps.FromDouble(1.0 / o.MelStd));
    }

    private Random Fork(int seed, int stream) => AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(unchecked(seed * 31 + stream));

    /// <summary>What a step reads that its trained modules do not produce.</summary>
    private sealed class Step
    {
        public int Seed;
        public bool Hard;
        public int[] Ids = Array.Empty<int>();
        public Tensor<T> Mel = default!;            // [mels, F]
        public Tensor<T> Prompt = default!;         // [mels, P]
        public int[] Durations = Array.Empty<int>(); // aligner frames per phoneme of the monotonic alignment
        public Tensor<T> Pitch = default!;          // [F] whole utterance
        public Tensor<T> Energy = default!;         // [F]
        public int[] MaskedFrames = Array.Empty<int>(); // the prosody encoder's per-frame phoneme (−1 masked)
        public Tensor<T> EncoderPitch = default!;   // [F], zero where masked
        public Tensor<T> EncoderEnergy = default!;
        public int Start, Length;                   // aligner frames of the acoustic segment
        public Tensor<T> Wave = default!;           // [2·Length·hop]
        public Tensor<T> SegmentPitch = default!;   // [1, 2·Length]
        public Tensor<T> SegmentEnergy = default!;
        public Tensor<T> SegmentMel = default!;     // [mels, 2·Length]
        public double Tau;
        public Tensor<T> Noise = default!;          // [K, d] diffusion noise
        public bool DropConditions;
        public double DistillationGuidance;
    }

    private Tensor<T> Row(Tensor<T> x) => Engine.Reshape(x, new[] { x.Length });

    private Step? Prepare(int[] ids, Tensor<T> wave, Tensor<T> mel, int seed, bool training)
    {
        var o = _options;
        var random = Fork(seed, 0);
        int half = mel.Shape[1] / 2;
        int length = Math.Min(half - 1, o.MaxTrainingFrames / 2);
        if (length <= 0 || 2 * length < o.MinTrainingFrames) return null;
        using var _ = new NoGradScope<T>();
        var p = new Step { Seed = seed, Ids = ids, Mel = mel, Length = length, Hard = !training || random.NextDouble() < o.HardAlignmentProbability };
        // Ground-truth durations: the aligner's monotonic path (evaluation mode, no token masking).
        var (logits, energies) = AlignerNetwork.Decode(mel, ids, false, random);
        var soft = Engine.Softmax(Engine.TensorSlice(energies, new[] { 1, 0 }, new[] { ids.Length, energies.Shape[1] }), axis: 1);
        var hard = StyleTTS<T>.HardPath(ToMatrix(soft));
        p.Durations = Enumerable.Range(0, ids.Length).Select(i => Enumerable.Range(0, hard.GetLength(1)).Count(f => hard[i, f] > 0)).ToArray();
        int frames = 2 * p.Durations.Sum();
        var fullMel = Engine.TensorSlice(mel, new[] { 0, 0 }, new[] { o.MelChannels, frames });
        p.Pitch = Row(PitchExtractorNetwork.Forward(fullMel, false, random));
        p.Energy = Row(FrontEnd.LogNorm(fullMel));
        // The prosody encoder's input, a random span masked (§III-B).
        p.MaskedFrames = PhonemeOfFrame(p.Durations, 2);
        p.EncoderPitch = new Tensor<T>(new[] { frames });
        p.EncoderEnergy = new Tensor<T>(new[] { frames });
        int maskStart = 0, maskEnd = 0;
        if (training && random.NextDouble() < o.ProsodyMaskProbability)
        {
            int span = (int)(random.NextDouble() * frames);
            maskStart = random.Next(0, frames - span + 1);
            maskEnd = maskStart + span;
        }
        for (int f = 0; f < frames; f++)
        {
            bool masked = f >= maskStart && f < maskEnd;
            if (masked) p.MaskedFrames[f] = -1;
            p.EncoderPitch[f] = masked ? NumOps.Zero : p.Pitch[f];
            p.EncoderEnergy[f] = masked ? NumOps.Zero : p.Energy[f];
        }
        // The acoustic segment (at most MaxTrainingFrames mel frames) and the prompt, another segment of the utterance.
        p.Start = random.Next(0, half - length);
        p.SegmentMel = MelCrop(mel, 2 * p.Start, 2 * length);
        p.Wave = Engine.TensorSlice(wave, new[] { 2 * p.Start * o.HopSize }, new[] { 2 * length * o.HopSize });
        p.SegmentPitch = Engine.Reshape(PitchExtractorNetwork.Forward(p.SegmentMel, training && TmaActive, random), new[] { 1, 2 * length });
        p.SegmentEnergy = FrontEnd.LogNorm(p.SegmentMel);
        int promptFrames = Math.Min(o.PromptFrames, mel.Shape[1]);
        int promptStart = random.Next(0, mel.Shape[1] - promptFrames + 1);
        p.Prompt = MelCrop(mel, promptStart, promptFrames);
        p.Tau = training ? random.NextDouble() : 0.5;
        p.Noise = Gaussian(new[] { o.StyleLength, o.ModelDim }, random);
        p.DropConditions = training && random.NextDouble() < o.ConditionDropout;
        p.DistillationGuidance = o.DistillationGuidanceMin + random.NextDouble() * (o.DistillationGuidanceMax - o.DistillationGuidanceMin);
        return p;
    }

    private Tensor<T> MelCrop(Tensor<T> mel, int start, int frames) =>
        Engine.TensorSlice(mel, new[] { 0, start }, new[] { _options.MelChannels, frames });

    private double[,] ToMatrix(Tensor<T> tensor)
    {
        var m = new double[tensor.Shape[0], tensor.Shape[1]];
        for (int i = 0; i < m.GetLength(0); i++)
            for (int j = 0; j < m.GetLength(1); j++) m[i, j] = NumOps.ToDouble(tensor[i, j]);
        return m;
    }

    private Tensor<T> FromMatrix(double[,] m)
    {
        var t = new Tensor<T>(new[] { m.GetLength(0), m.GetLength(1) });
        for (int i = 0; i < m.GetLength(0); i++)
            for (int j = 0; j < m.GetLength(1); j++) t[i, j] = NumOps.FromDouble(m[i, j]);
        return t;
    }

    private Tensor<T> Scaled(Tensor<T> x, double factor) => Engine.TensorMultiplyScalar(x, NumOps.FromDouble(factor));

    private Tensor<T> Total(params (Tensor<T>? Term, double Weight)[] terms)
    {
        Tensor<T>? total = null;
        foreach (var (term, weight) in terms)
        {
            if (term is null || weight == 0) continue;
            var scaled = Engine.Reshape(Scaled(Engine.Reshape(term, new[] { term.Length }), weight), new[] { 1 });
            total = total is null ? scaled : Engine.TensorAdd(total, scaled);
        }
        return total ?? new Tensor<T>(new[] { 1 });
    }

    private Tensor<T> Mean(Tensor<T> x) =>
        Engine.TensorMultiplyScalar(Engine.ReduceSum(Engine.Reshape(x, new[] { x.Length }), new[] { 0 }, keepDims: true), NumOps.FromDouble(1.0 / x.Length));

    private Tensor<T> OneMinusSquared(Tensor<T> score)
    {
        var d = Engine.TensorAddScalar(Engine.TensorNegate(score), NumOps.One);
        return Mean(Engine.TensorMultiply(d, d));
    }

    private Tensor<T> FeatureMatching(List<Tensor<T>> real, List<Tensor<T>> fake) =>
        real.Zip(fake, (r, f) => Losses.L1(f, r)).Aggregate((a, b) => Engine.TensorAdd(a, b));

    // ---------------------------------------------------------------- the generator's pass

    private sealed class Pass
    {
        public Tensor<T>? Loss;
        public (Tensor<T> Wave, Tensor<T> Durations, Tensor<T> Prosody, Tensor<T> Style)? Fakes;
    }

    /// <summary>The waveform discriminator's input: WavLM's stacked hidden states, frames first <c>[frames, layers · hidden]</c>.</summary>
    private Tensor<T> WavLmRows(Tensor<T> wave)
    {
        var stacked = Losses.Stacked(Losses.SlmStates(SlmNetwork, wave));                                      // [1, C, F]
        return Engine.TensorTranspose(Engine.Reshape(stacked, new[] { stacked.Shape[1], stacked.Shape[2] }));
    }

    /// <summary>The waveform discriminator's conditions per mel frame (Fig. 2e): the aligned prompt-aligned text embeddings,
    /// the global style, pitch, energy and the frame's phoneme duration.</summary>
    private Tensor<T> WaveformConditions(Step p, Tensor<T> segmentText, Tensor<T> style, int[] segmentDurations)
    {
        int frames = 2 * p.Length;
        var phonemeOfFrame = new int[frames];
        var duration = new Tensor<T>(new[] { frames, 1 });
        for (int f = 0; f < frames; f++)
        {
            phonemeOfFrame[f] = f / 2;
            duration[f, 0] = NumOps.FromDouble(segmentDurations[f / 2]);
        }
        var text = Upsample(segmentText, phonemeOfFrame);
        return Engine.TensorConcatenate(new[] { text, Engine.TensorTile(style, new[] { frames, 1 }),
            Engine.TensorTranspose(p.SegmentPitch), Engine.TensorTranspose(p.SegmentEnergy), duration }, 1);
    }

    /// <summary>
    /// One training step's generator forward and objective (App. C): the acoustic synthesizer's mel L1, TMA, adversarial,
    /// feature-matching and speaker-embedding feature-matching losses; the prosody autoencoder's duration, pitch and energy
    /// L1, quantizer and multimodal adversarial losses; the denoiser's v-objective L1 (Eq. 3) and the student's
    /// pre-training L1. With <paramref name="outputsOnly"/> only the outputs the discriminators judge.
    /// </summary>
    private Pass GeneratorPass(Step p, bool training, bool outputsOnly = false, bool adversarialTerms = true)
    {
        var o = _options;
        var random = Fork(p.Seed, 1);
        // Acoustic synthesizer.
        var (soft, logits) = AlignerFor(p, training, random);
        var hard = StyleTTS<T>.HardPath(ToMatrix(soft));
        var alignment = p.Hard ? FromMatrix(hard) : soft;
        var (text, style) = PromptTextNetwork.Forward(p.Ids, p.Prompt, training, random);
        var styleIn = training && o.StyleDropout > 0 ? CodecBased.T5Seq2Seq<T>.Dropout(Engine, style, o.StyleDropout, random) : style;
        var asr = Engine.Reshape(Engine.TensorMatMul(Engine.TensorTranspose(text), alignment), new[] { 1, o.ModelDim, alignment.Shape[1] });
        var en = Engine.TensorSlice(asr, new[] { 0, 0, p.Start }, new[] { 1, o.ModelDim, p.Length });
        var wave = DecoderNetwork.Forward(en, p.SegmentPitch, p.SegmentEnergy, styleIn, false, random);
        // Prosody autoencoder.
        var e = BertEncoderLayer.Forward(BertNetwork.Encode(p.Ids, training, random));
        var continuous = ProsodyEncoderNetwork.Forward(p.MaskedFrames, p.EncoderPitch, p.EncoderEnergy, training, random);
        var quantized = Quantizer.Forward(PhonemesFirst(continuous));
        var h = RowsFirst(quantized);                                                                         // [K, d]
        var durations = DurationDecoderNetwork.Forward(e, h, training, random);                               // [N, 1]
        var frameText = Upsample(e, PhonemeOfFrame(p.Durations, 2));
        var prosody = ProsodyDecoderNetwork.Forward(frameText, h, training, random);                          // [F, 2]
        var pass = new Pass { Fakes = (wave, durations, prosody, h) };
        if (outputsOnly) return pass;

        var segmentDurations = p.Durations.Select(_ => 0).ToArray();
        int frame = 0;
        for (int i = 0; i < p.Durations.Length; i++)
            for (int f = 0; f < p.Durations[i]; f++, frame++)
                if (frame >= p.Start && frame < p.Start + p.Length) segmentDurations[i]++;
        var segmentText = Engine.TensorTranspose(Engine.Reshape(en, new[] { o.ModelDim, p.Length }));

        // The centred STFT gives 1 + samples / hop frames; the segment has 2·Length.
        var mel = Losses.L1(MelCrop(MelOnTape(wave), 0, 2 * p.Length), p.SegmentMel);
        Tensor<T>? s2s = null, mono = null;
        if (TmaActive)
        {
            int length = p.Ids.Length, tokens = o.VocabSize;
            var head = Engine.TensorSlice(logits, new[] { 0, 0 }, new[] { length, tokens });
            var selection = new Tensor<T>(new[] { length, tokens });
            for (int i = 0; i < length; i++) selection[i, p.Ids[i]] = NumOps.FromDouble(-1.0 / length);
            s2s = Engine.ReduceSum(Engine.TensorMultiply(Engine.TensorLogSoftmax(head, axis: -1), selection), new[] { 0, 1 }, keepDims: false);
            mono = Losses.L1(soft, FromMatrix(hard));
        }
        Tensor<T>? adversarial = null, speaker = null;
        if (adversarialTerms && TmaActive)
        {
            List<(Tensor<T>, List<Tensor<T>>)> real;
            Tensor<T> realRows, conditions;
            using (new NoGradScope<T>())
            {
                real = WaveDiscriminators.Forward(p.Wave);
                realRows = WavLmRows(p.Wave);
                conditions = WaveformConditions(p, segmentText, style, segmentDurations);
            }
            var wavePart = Losses.Generator(real, WaveDiscriminators.Forward(wave), relativistic: false);
            List<Tensor<T>> realFeatures;
            using (new NoGradScope<T>()) realFeatures = WaveformDiscriminator.Forward(realRows, conditions, training, Fork(p.Seed, 8)).Features;
            var (score, features) = WaveformDiscriminator.Forward(WavLmRows(wave), conditions, training, Fork(p.Seed, 8));
            adversarial = Engine.TensorAdd(wavePart, Engine.TensorAdd(OneMinusSquared(score), FeatureMatching(realFeatures, features)));
            // Speaker-embedding feature matching (Eq. 14) over the ResNet's stem and stages, at 16 kHz.
            IReadOnlyList<Tensor<T>> realMaps;
            using (new NoGradScope<T>()) realMaps = SpeakerNetwork.Forward(Losses.Resample(p.Wave)).Maps;
            var fakeMaps = SpeakerNetwork.Forward(Losses.Resample(wave)).Maps;
            speaker = FeatureMatching(realMaps.ToList(), fakeMaps.ToList());
        }

        var durationTarget = new Tensor<T>(new[] { p.Durations.Length, 1 });
        for (int i = 0; i < p.Durations.Length; i++) durationTarget[i, 0] = NumOps.FromDouble(p.Durations[i]);
        int frames = prosody.Shape[0];
        var pitch = Losses.L1(Engine.TensorSlice(prosody, new[] { 0, 0 }, new[] { frames, 1 }), p.Pitch);
        var energy = Losses.L1(Engine.TensorSlice(prosody, new[] { 0, 1 }, new[] { frames, 1 }), p.Energy);
        var duration = Losses.L1(durations, durationTarget);
        Tensor<T>? prosodyAdversarial = null;
        if (adversarialTerms)
        {
            var styleCondition = Engine.TensorConcatenate(new[] { h, e }, 0);
            var frameCondition = Engine.TensorConcatenate(new[] { h, frameText }, 0);
            var realProsody = Engine.TensorConcatenate(new[] { Engine.Reshape(p.Pitch, new[] { frames, 1 }), Engine.Reshape(p.Energy, new[] { frames, 1 }) }, 1);
            List<Tensor<T>> realDuration, realPitch;
            using (new NoGradScope<T>())
            {
                realDuration = DurationDiscriminator.Forward(durationTarget, styleCondition, training, Fork(p.Seed, 9)).Features;
                realPitch = ProsodyDiscriminator.Forward(realProsody, frameCondition, training, Fork(p.Seed, 10)).Features;
            }
            var (durationScore, durationFeatures) = DurationDiscriminator.Forward(durations, styleCondition, training, Fork(p.Seed, 9));
            var (pitchScore, pitchFeatures) = ProsodyDiscriminator.Forward(prosody, frameCondition, training, Fork(p.Seed, 10));
            prosodyAdversarial = Engine.TensorAdd(Engine.TensorAdd(OneMinusSquared(durationScore), FeatureMatching(realDuration, durationFeatures)),
                Engine.TensorAdd(OneMinusSquared(pitchScore), FeatureMatching(realPitch, pitchFeatures)));
        }
        var quantizer = Total((Quantizer.CodebookLoss, o.CodebookWeight), (Quantizer.CommitmentLoss, o.CommitmentWeight));

        // The denoiser on the time-varying style (detached; Eq. 3) and the student's pre-training.
        Tensor<T> h0;
        using (new NoGradScope<T>()) h0 = Engine.TensorMultiplyScalar(h, NumOps.One);
        var (alpha, sigma) = Schedule(p.Tau);
        var noisy = Engine.TensorAdd(Scaled(h0, alpha), Scaled(p.Noise, sigma));
        var velocity = Engine.TensorSubtract(Scaled(p.Noise, alpha), Scaled(h0, sigma));
        var predicted = TeacherNetwork.Forward(noisy, sigma, e, p.Prompt, p.DropConditions, training, random);
        var diffusion = Losses.L1(predicted, velocity);
        var student = Losses.L1(StudentNetwork.Forward(new Tensor<T>(h0._shape), 0.0, e, p.Prompt, false, training, random), h0);

        pass.Loss = Total((mel, 1.0), (s2s, o.S2sLossWeight), (mono, o.MonotonicLossWeight), (adversarial, 1.0), (speaker, 1.0),
            (duration, 1.0), (pitch, o.F0LossWeight), (energy, o.NormLossWeight), (prosodyAdversarial, 1.0), (quantizer, 1.0),
            (diffusion, 1.0), (student, 1.0));
        return pass;
    }

    private (Tensor<T> Soft, Tensor<T> Logits) AlignerFor(Step p, bool training, Random random)
    {
        var (logits, energies) = AlignerNetwork.Decode(p.Mel, p.Ids, training, random);
        var soft = Engine.Softmax(Engine.TensorSlice(energies, new[] { 1, 0 }, new[] { p.Ids.Length, energies.Shape[1] }), axis: 1);
        return (soft, logits);
    }

    /// <summary>The discriminators' objective: least-squares on the multi-period, multi-resolution and multimodal
    /// waveform discriminators (once the adversarial terms start) and the two multimodal prosody discriminators.</summary>
    private Tensor<T> DiscriminatorLoss(Step p, (Tensor<T> Wave, Tensor<T> Durations, Tensor<T> Prosody, Tensor<T> Style) fake)
    {
        var o = _options;
        var random = Fork(p.Seed, 1);
        Tensor<T> e, frameText, segmentText, style;
        int[] segmentDurations;
        using (new NoGradScope<T>())
        {
            e = BertEncoderLayer.Forward(BertNetwork.Encode(p.Ids, false, random));
            frameText = Upsample(e, PhonemeOfFrame(p.Durations, 2));
            var (text, global) = PromptTextNetwork.Forward(p.Ids, p.Prompt, false, random);
            style = global;
            var alignment = FromMatrix(StyleTTS<T>.HardPath(ToMatrix(AlignerFor(p, false, random).Soft)));
            var asr = Engine.TensorMatMul(Engine.TensorTranspose(text), alignment);
            segmentText = Engine.TensorTranspose(Engine.TensorSlice(asr, new[] { 0, p.Start }, new[] { o.ModelDim, p.Length }));
            segmentDurations = p.Durations.Select(_ => 0).ToArray();
            int frame = 0;
            for (int i = 0; i < p.Durations.Length; i++)
                for (int f = 0; f < p.Durations[i]; f++, frame++)
                    if (frame >= p.Start && frame < p.Start + p.Length) segmentDurations[i]++;
        }
        Tensor<T> LeastSquares(Tensor<T> real, Tensor<T> generated) => Engine.TensorAdd(OneMinusSquared(real), Mean(Engine.TensorMultiply(generated, generated)));
        var terms = new List<Tensor<T>>();
        if (TmaActive)
        {
            terms.Add(Losses.Discriminator(WaveDiscriminators.Forward(p.Wave), WaveDiscriminators.Forward(fake.Wave), relativistic: false));
            Tensor<T> realRows, fakeRows, conditions;
            using (new NoGradScope<T>())
            {
                realRows = WavLmRows(p.Wave);
                fakeRows = WavLmRows(fake.Wave);
                conditions = WaveformConditions(p, segmentText, style, segmentDurations);
            }
            terms.Add(LeastSquares(WaveformDiscriminator.Forward(realRows, conditions, true, Fork(p.Seed, 8)).Score,
                WaveformDiscriminator.Forward(fakeRows, conditions, true, Fork(p.Seed, 8)).Score));
        }
        var durationTarget = new Tensor<T>(new[] { p.Durations.Length, 1 });
        for (int i = 0; i < p.Durations.Length; i++) durationTarget[i, 0] = NumOps.FromDouble(p.Durations[i]);
        int frames = fake.Prosody.Shape[0];
        var realProsody = Engine.TensorConcatenate(new[] { Engine.Reshape(p.Pitch, new[] { frames, 1 }), Engine.Reshape(p.Energy, new[] { frames, 1 }) }, 1);
        var styleCondition = Engine.TensorConcatenate(new[] { fake.Style, e }, 0);
        var frameCondition = Engine.TensorConcatenate(new[] { fake.Style, frameText }, 0);
        terms.Add(LeastSquares(DurationDiscriminator.Forward(durationTarget, styleCondition, true, Fork(p.Seed, 9)).Score,
            DurationDiscriminator.Forward(fake.Durations, styleCondition, true, Fork(p.Seed, 9)).Score));
        terms.Add(LeastSquares(ProsodyDiscriminator.Forward(realProsody, frameCondition, true, Fork(p.Seed, 10)).Score,
            ProsodyDiscriminator.Forward(fake.Prosody, frameCondition, true, Fork(p.Seed, 10)).Score));
        return terms.Aggregate((a, b) => Engine.TensorAdd(a, b));
    }

    /// <summary>
    /// Distillation (Eq. 6): from the same noise and a guidance scale ω ~ U(1, 15), the student's one-step style and the
    /// teacher's guided DDIM style, compared through the prosody decoder (durations, and pitch and energy upsampled by
    /// the teacher's rounded durations), L1.
    /// </summary>
    private Tensor<T> DistillationLoss(Step p, bool training)
    {
        var random = Fork(p.Seed, 11);
        Tensor<T> e, teacher, teacherDurations, teacherProsody;
        int[] phonemeOfFrame;
        using (new NoGradScope<T>())
        {
            e = BertEncoderLayer.Forward(BertNetwork.Encode(p.Ids, false, random));
            teacher = TeacherSample(p.Noise, e, p.Prompt, p.DistillationGuidance, false, random);
            teacherDurations = DurationDecoderNetwork.Forward(e, teacher, false, random);
            phonemeOfFrame = PhonemeOfFrame(FramesOf(teacherDurations), 2);
            teacherProsody = ProsodyDecoderNetwork.Forward(Upsample(e, phonemeOfFrame), teacher, false, random);
        }
        var student = StudentSample(p.Noise, e, p.Prompt, p.DistillationGuidance, training, random);
        var studentDurations = DurationDecoderNetwork.Forward(e, student, false, random);
        var studentProsody = ProsodyDecoderNetwork.Forward(Upsample(e, phonemeOfFrame), student, false, random);
        return Total((Losses.L1(studentDurations, teacherDurations), 1.0), (Losses.L1(studentProsody, teacherProsody), 1.0));
    }

    // ---------------------------------------------------------------- optimizers

    private Dictionary<Tensor<T>, string> ModuleOfParameter()
    {
        if (_moduleOfParameter is not null) return _moduleOfParameter;
        var map = new Dictionary<Tensor<T>, string>(Helpers.TensorReferenceComparer<Tensor<T>>.Instance);
        foreach (var entry in _modules)
        {
            if (entry.Key is "wavlm" or "speaker") continue;
            foreach (var tensor in Training.TapeTrainingStep<T>.CollectParameters(entry.Value.Cast<ILayer<T>>().ToList(), -1)) map[tensor] = entry.Key;
        }
        return _moduleOfParameter = map;
    }

    private AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>> ModuleOptions(string key)
    {
        var o = _options;
        bool bert = key == "bert";
        return new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
        {
            InitialLearningRate = bert ? o.BertLearningRate : o.LearningRate,
            Beta1 = bert ? o.BertBeta1 : o.Beta1,
            Beta2 = o.Beta2,
            Epsilon = o.Epsilon,
            WeightDecay = bert ? o.BertWeightDecay : o.WeightDecay,
            EnableGradientClipping = false,
        };
    }

    private StyleTts2ModuleOptimizer<T> NewOptimizer(string component)
    {
        var map = ModuleOfParameter();
        var optimizer = new StyleTts2ModuleOptimizer<T>(this, ModuleOptions(component == "discriminator" ? "mpd" : "decoder"),
            p => map.TryGetValue(p, out var key) ? key : null, ModuleOptions);
        PaperOptimizerFactory.VerifyHandBuilt(this, optimizer, component);
        return optimizer;
    }

    private T RunStep(Tensor<T> tokens, IEnumerable<string> modules, bool discriminator, Func<Tensor<T>> objective)
    {
        if (_generatorOptimizer is null || _discriminatorOptimizer is null || _optimizerStage != CurrentStage)
        {
            _optimizerStage = CurrentStage;
            _generatorOptimizer = NewOptimizer("generator");
            _discriminatorOptimizer = NewOptimizer("discriminator");
        }
        _steppedModules = new HashSet<string>(modules);
        try
        {
            return TrainWithCustomObjective(tokens, tokens, (_, _) =>
            {
                // WavLM and the speaker network are frozen and read in evaluation mode.
                foreach (var layer in Module("wavlm").Concat(Module("speaker"))) layer.SetTrainingMode(false);
                return objective();
            }, discriminator ? _discriminatorOptimizer : _generatorOptimizer);
        }
        finally
        {
            _steppedModules = new HashSet<string>();
        }
    }

    /// <inheritdoc />
    /// <remarks>The modules the current step updates; WavLM and the speaker network are never trained.</remarks>
    protected override IReadOnlyList<Tensor<T>> SelectTrainableParametersForTraining(IReadOnlyList<Tensor<T>> parameters)
    {
        if (!HasPaperLayers) return parameters;
        var map = ModuleOfParameter();
        return parameters.Where(p => map.TryGetValue(p, out var key) && _steppedModules.Contains(key)).ToList();
    }
}
