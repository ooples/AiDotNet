using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.Vocoders;

namespace AiDotNet.TextToSpeech.MultiModal;

public partial class SpeechT5<T>
{
    private readonly Dictionary<SpeechT5Stage, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>> _optimizers = new();
    private Random? _trainingRandom;
    private int _updates;
    private bool _encoderFrozenThisStep;

    /// <summary>Updates taken so far in the current stage (the official <c>num_updates</c>, which the encoder freeze and the
    /// Gumbel temperature read).</summary>
    public int Updates => _updates;

    /// <inheritdoc />
    /// <remarks>Text-to-speech fine-tuning trains on a 16 kHz recording (<see cref="TtsTrainingSample{T}.Audio"/>, or its
    /// log-mel spectrogram <c>[frames, mel]</c> in <see cref="TtsTrainingSample{T}.Mel"/>) and its text's token ids
    /// (<see cref="TtsTrainingSample{T}.Tokens"/>, from <see cref="TokenizeText"/>); the speaker's x-vector comes from
    /// <see cref="TtsTrainingSample{T}.SpeakerReference"/> (the recording itself when absent). Pre-training reads either
    /// a recording with its HuBERT labels (<see cref="TtsTrainingSample{T}.CodecTokens"/>: <c>[frames]</c> or
    /// <c>[frames, sets]</c> at <see cref="SpeechT5Options.LabelRate"/>, as the label dictionary indexes them) or text
    /// alone (its token ids); as either serves, pre-training requires neither up front and the objective checks the
    /// sample it gets.</remarks>
    protected override TtsSupervision RequiredSupervision =>
        _options.CurrentStage == SpeechT5Stage.Pretraining ? TtsSupervision.None : TtsSupervision.Recording;

    /// <inheritdoc />
    public override bool TrainsOnSampleObjective => true;

    /// <inheritdoc />
    /// <remarks>The current stage's Adam: pre-training's (2e-4 peak, linear warm-up over 8% and linear decay, decoupled
    /// weight decay 0.01) or text-to-speech's (4e-4 peak, 10k warm-up steps, inverse square root).</remarks>
    protected internal override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? TrainingOptimizer => HasPaperLayers ? Optimizer() : null;

    /// <inheritdoc />
    protected override bool TryApplyLearningRule(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        Train(new TtsTrainingSample<T> { Tokens = input, Audio = expectedOutput });
        return true;
    }

    /// <inheritdoc />
    /// <remarks>
    /// One update of <see cref="SpeechT5Options.CurrentStage"/>'s objective. Text-to-speech: the official
    /// <c>TexttoSpeechLoss</c> — L1 of the frames before and after the post-net, the stop token's BCE (positive weight 5)
    /// and the guided attention loss — with the shared encoder (and the text pre-net feeding it) frozen for the first
    /// <see cref="SpeechT5Options.FreezeEncoderUpdates"/> updates. Pre-training: a recording with HuBERT labels, randomly
    /// cropped to <see cref="SpeechT5Options.MaxSpeechSamples"/>, is span-masked and trained on the official
    /// <c>SpeechPretrainCriterion</c>; text alone is noised by BART's infilling and trained on <c>TextPretrainCriterion</c>;
    /// both mix the encoder states with the Gumbel codebook.
    /// </remarks>
    protected override T TrainOnSample(TtsTrainingSample<T> sample)
    {
        ThrowIfDisposed();
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        if (!HasPaperLayers) throw new NotSupportedException("The objective needs the paper's networks; this model was built from caller-supplied layers.");
        var random = _trainingRandom ??= AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.TrainingSeed);
        _encoderFrozenThisStep = _options.CurrentStage == SpeechT5Stage.TextToSpeech && _updates < _options.FreezeEncoderUpdates;
        var objective = Objective(sample, random, training: true);
        var loss = TrainWithCustomObjective(sample.Tokens, sample.Tokens, (_, _) => objective(), Optimizer());
        _updates++;
        return loss;
    }

    /// <inheritdoc />
    /// <remarks>The objective at a fixed draw (the training seed's first) with dropout off, so repeated evaluations
    /// compare.</remarks>
    public override T EvaluateTrainingObjective(TtsTrainingSample<T> sample)
    {
        ThrowIfDisposed();
        Validation.Guard.NotNull(sample);
        if (!HasPaperLayers) throw new NotSupportedException("The objective needs the paper's networks; this model was built from caller-supplied layers.");
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            return Objective(sample, AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.TrainingSeed), training: false)()[0];
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    // ---------------------------------------------------------------- data

    /// <summary>The official <c>logmelfilterbank</c> <c>[frames, mel]</c> of 16 kHz audio (librosa's centred STFT, n_fft
    /// 1024, hop 256, Slaney mel filters over 80 Hz – 7.6 kHz, log10 floored at 1e-10).</summary>
    internal Tensor<T> LogMel(Tensor<T> audio)
    {
        var o = _options;
        _mel ??= new CenteredLogMel<T>(Engine, o.SampleRate, o.FftSize, o.HopSize, o.FftSize, o.MelChannels, 80, 7600, 1e-10);
        using var _ = new NoGradScope<T>();
        var mel = _mel.Forward(Engine.Reshape(audio, new[] { audio.Length }));
        return Engine.TensorTranspose(Engine.Reshape(mel, new[] { o.MelChannels, mel.Shape[2] }));
    }

    private readonly Dictionary<Tensor<T>, Tensor<T>> _trainingSpeakers = new(TensorReferenceComparer<Tensor<T>>.Instance);

    private Tensor<T> TrainingSpeaker(Tensor<T> audio)
    {
        if (_trainingSpeakers.TryGetValue(audio, out var speaker)) return speaker;
        if (_trainingSpeakers.Count > 64) _trainingSpeakers.Clear();
        return _trainingSpeakers[audio] = XVector(audio, AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed));
    }

    private Tensor<T> MelOf(TtsTrainingSample<T> sample)
    {
        if (sample.Audio is { } audio) return LogMel(audio);
        var mel = sample.Mel ?? throw new ArgumentException($"{GetType().Name} trains on a recording; set Audio (or Mel).", nameof(sample));
        int m = _options.MelChannels;
        if (mel.Rank == 2 && mel.Shape[1] == m) return mel;
        if (mel.Rank == 2 && mel.Shape[0] == m) return Engine.TensorTranspose(mel);
        throw new ArgumentException($"The spectrogram is [frames, {m}].", nameof(sample));
    }

    private int[][] Labels(Tensor<T> codes)
    {
        int frames = codes.Shape[0], sets = codes.Rank == 1 ? 1 : codes.Shape[1];
        if (sets != _options.HubertClasses.Length) throw new ArgumentException($"Give {_options.HubertClasses.Length} label sets.");
        var labels = new int[sets][];
        for (int s = 0; s < sets; s++)
        {
            labels[s] = new int[frames];
            for (int f = 0; f < frames; f++)
            {
                int label = (int)Math.Round(Ops.ToDouble(codes.GetFlat(f * sets + s)));
                if (label < 0 || label >= _options.HubertClasses[s]) throw new ArgumentException($"HuBERT labels of set {s} lie in [0, {_options.HubertClasses[s]}).");
                labels[s][f] = label;
            }
        }
        return labels;
    }

    // ---------------------------------------------------------------- the objectives

    private Func<Tensor<T>> Objective(TtsTrainingSample<T> sample, Random random, bool training)
    {
        Validation.Guard.NotNull(sample);
        var o = _options;
        var core = Core;
        var tokens = TokenIds(sample.Tokens, nameof(sample));
        if (o.CurrentStage == SpeechT5Stage.TextToSpeech)
        {
            if (tokens.Length == 0) throw new ArgumentException("The sample has no text tokens.", nameof(sample));
            var mel = MelOf(sample);
            if (mel.Shape[0] < o.ReductionFactor) throw new ArgumentException("The recording is shorter than one decoder step.", nameof(sample));
            if (mel.Shape[0] / o.ReductionFactor > o.MaxMelLength) throw new ArgumentException("The recording has more decoder steps than the model has positions.", nameof(sample));
            var speakerAudio = sample.SpeakerReference ?? sample.Audio
                ?? throw new ArgumentException("The x-vector needs a recording: set SpeakerReference or Audio.", nameof(sample));
            var speaker = TrainingSpeaker(speakerAudio);
            bool frozen = training && _encoderFrozenThisStep;
            return () =>
            {
                Tensor<T> memory;
                if (frozen)
                    using (new NoGradScope<T>()) memory = core.EncodeText(tokens, training, random);
                else memory = core.EncodeText(tokens, training, random);
                var outputs = core.DecodeSpeech(memory, core.DecoderInputs(mel), speaker, training, random);
                return core.SpeechLoss(outputs, mel, tokens.Length, o.GuidedAttentionHeads, o.GuidedAttentionSigma, o.StopPositiveWeight);
            };
        }

        core.Quantizer.Temperature = Math.Max(2.0 * Math.Pow(0.999995, _updates), 0.5);
        var draws = new SpeechT5RandomDraws(random);
        if (sample.Audio is { } recording)
        {
            var codes = sample.CodecTokens ?? throw new ArgumentException("Speech pre-training needs the recording's HuBERT labels in CodecTokens.", nameof(sample));
            var labels = Labels(codes);
            var speaker = TrainingSpeaker(recording);
            var fullMel = LogMel(recording);
            var (wave, mel, cropped) = Crop(recording, fullMel, labels, random);
            var c = core.Configuration;
            double ratio = o.LabelRate * c.FeatureLayers.Aggregate(1, (a, l) => a * l.Stride) / o.SampleRate;
            int frames = core.SpeechEncoderPrenet.Frames(wave.Length), shortest = cropped.Min(l => l.Length);
            if (ratio * frames > shortest) frames = (int)(shortest / ratio);
            if (frames < 1 || mel.Shape[0] < o.ReductionFactor) throw new ArgumentException("The recording is too short to pre-train on.", nameof(sample));
            var mask = SpeechT5Noising.SpanMask(frames, o.MaskProbability, o.MaskLength, 2, draws);
            var permutation = draws.Permutation(frames);
            return () => core.SpeechPretrainLoss(wave, cropped, mel, speaker, mask, permutation, o.CodebookProbability, o.LabelRate, o.SampleRate,
                training, random, o.FeaturePenaltyWeight, o.DiversityWeight).Total;
        }

        var text = tokens.Length > 0 && tokens[tokens.Length - 1] == o.EosId ? tokens.Take(tokens.Length - 1) : tokens;
        var source = new[] { o.BosId }.Concat(text).Concat(new[] { o.EosId }).ToArray();
        if (source.Length < 3) throw new ArgumentException("Text pre-training needs at least one token.", nameof(sample));
        if (source.Length > o.MaxTextLength) throw new ArgumentException($"The text has more than {o.MaxTextLength} tokens.", nameof(sample));
        var noised = SpeechT5Noising.TextInfilling(source, o.TextMaskRatio, o.TextRandomRatio, o.PoissonLambda, o.MaskId, o.VocabSize, draws);
        var order = draws.Permutation(noised.Length);
        return () => core.TextPretrainLoss(noised, source, order, o.CodebookProbability, training, random, o.DiversityWeight);
    }

    /// <summary>The official random crop to <see cref="SpeechT5Options.MaxSpeechSamples"/>: the waveform, its log-mel
    /// frames and its labels cut at the matching offsets.</summary>
    private (Tensor<T> Wave, Tensor<T> Mel, int[][] Labels) Crop(Tensor<T> recording, Tensor<T> mel, int[][] labels, Random random)
    {
        var o = _options;
        int size = recording.Length, target = Math.Min(size, o.MaxSpeechSamples), start = 0;
        if (size > target) start = random.Next(0, size - target + 1);
        var wave = start == 0 && target == size ? Engine.Reshape(recording, new[] { size }) : Engine.TensorNarrow(Engine.Reshape(recording, new[] { size }), 0, start, target);
        int melFrames = mel.Shape[0];
        double samplesPerFrame = (double)size / melFrames;
        int melStart = (int)(start / samplesPerFrame), melEnd = Math.Min(melStart + (int)(target / samplesPerFrame), melFrames);
        var croppedMel = Engine.TensorNarrow(mel, 0, melStart, melEnd - melStart);
        double s2f = o.LabelRate / o.SampleRate;
        int labelStart = (int)Math.Round(start * s2f, MidpointRounding.ToEven), labelSize = (int)Math.Round(target * s2f, MidpointRounding.ToEven);
        labelSize = Math.Min(labelSize, labels.Min(l => l.Length - labelStart));
        if (labelSize < 1) throw new ArgumentException("The recording's labels do not cover its crop.");
        return (wave, croppedMel, labels.Select(l => l.Skip(labelStart).Take(labelSize).ToArray()).ToArray());
    }

    // ---------------------------------------------------------------- optimizers

    private IReadOnlyList<LayerBase<T>> StageLayers()
    {
        if (_options.CurrentStage == SpeechT5Stage.Pretraining) return _layers.Concat(_pretrainLayers).ToList();
        if (!_encoderFrozenThisStep) return _layers;
        var frozen = new HashSet<LayerBase<T>>(Core.TextEncoderLayers);
        return _layers.Where(l => !frozen.Contains(l)).ToList();
    }

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> Optimizer()
    {
        var stage = _options.CurrentStage;
        if (_optimizers.TryGetValue(stage, out var cached)) return cached;
        var o = _options;
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> optimizer;
        if (stage == SpeechT5Stage.Pretraining)
        {
            int warmup = (int)(o.PretrainUpdates * o.PretrainWarmupFraction);
            var scheduler = new FairseqPolynomialDecaySchedule(o.PretrainLearningRate, warmup, o.PretrainUpdates);
            var parameters = Training.TapeTrainingStep<T>.CollectParameters(_layers.Concat(_pretrainLayers).Cast<ILayer<T>>().ToList(), -1).ToList();
            // fairseq's Adam decays the weights decoupled from the gradient (its "fixed" weight decay), on every parameter.
            optimizer = PaperOptimizerFactory.VerifyHandBuilt(this, new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this, new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                // The peak, which the paper declares; the optimizer starts at the schedule's own first rate.
                InitialLearningRate = o.PretrainLearningRate, Beta1 = o.OptimizerBeta1, Beta2 = o.OptimizerBeta2, Epsilon = o.PretrainEpsilon,
                WeightDecay = o.PretrainWeightDecay, MaskParameters = parameters,
                WeightDecayMask = new Vector<T>(Enumerable.Repeat(Ops.One, parameters.Sum(p => p.Length)).ToArray()),
                EnableGradientClipping = o.PretrainMaxGradientNorm > 0, MaxGradientNorm = o.PretrainMaxGradientNorm > 0 ? o.PretrainMaxGradientNorm : 1.0,
                LearningRateScheduler = scheduler, SchedulerStepMode = SchedulerStepMode.StepPerBatch,
            }));
        }
        else
        {
            // fairseq's inverse_sqrt: linear warm-up from 0 to the peak, then peak · √(warmup / step) — Noam's curve
            // scaled to the stated peak.
            if (o.WarmupUpdates <= 0) throw new ArgumentOutOfRangeException(nameof(o.WarmupUpdates), "WarmupUpdates must be positive.");
            double factor = o.LearningRate * Math.Sqrt(o.HiddenDim * (double)o.WarmupUpdates);
            var scheduler = new NoamSchedule(o.HiddenDim, o.WarmupUpdates, factor);
            optimizer = new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this, new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = scheduler.CurrentLearningRate, Beta1 = o.OptimizerBeta1, Beta2 = o.OptimizerBeta2, Epsilon = o.OptimizerEpsilon,
                EnableGradientClipping = o.MaxGradientNorm > 0, MaxGradientNorm = o.MaxGradientNorm > 0 ? o.MaxGradientNorm : 1.0,
                UseAdaptiveLearningRate = false, UseAdaptiveBetas = false, UseAMSGrad = false,
                LearningRateScheduler = scheduler, SchedulerStepMode = SchedulerStepMode.StepPerBatch,
            });
        }
        _optimizers[stage] = optimizer;
        return optimizer;
    }

    /// <inheritdoc />
    /// <remarks>Training updates the current stage's layers: every network but the x-vector extractor and HiFi-GAN in
    /// pre-training; the text-to-speech layers in fine-tuning, without the shared encoder and the text pre-net while they
    /// are frozen.</remarks>
    protected override IReadOnlyList<Tensor<T>> SelectTrainableParametersForTraining(IReadOnlyList<Tensor<T>> parameters)
    {
        if (!HasPaperLayers) return parameters;
        var trained = new HashSet<Tensor<T>>(Training.TapeTrainingStep<T>.CollectParameters(StageLayers().Cast<ILayer<T>>().ToList(), -1),
            TensorReferenceComparer<Tensor<T>>.Instance);
        return parameters.Where(trained.Contains).ToList();
    }
}
