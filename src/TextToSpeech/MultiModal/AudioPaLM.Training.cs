using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.Latest;

namespace AiDotNet.TextToSpeech.MultiModal;

/// <summary>
/// The schedule of §5.4.8's PaLM-2 experiments: a linear ramp to the peak, then an exponential decay to the final rate.
/// </summary>
internal sealed class RampExponentialSchedule : LearningRateSchedulerBase
{
    private readonly int _warmup;
    private readonly int _decay;
    private readonly double _final;

    public RampExponentialSchedule(double peak, double final, int warmup, int decay)
        : base(peak)
    {
        if (warmup < 0 || decay <= 0) throw new ArgumentOutOfRangeException(nameof(decay), "The ramp needs a non-negative warm-up and a positive decay.");
        _warmup = warmup;
        _decay = decay;
        _final = final;
        _currentLearningRate = ComputeLearningRate(0);
    }

    protected override double ComputeLearningRate(int step)
    {
        if (step < _warmup) return _baseLearningRate * (step + 1) / _warmup;
        double progress = Math.Min(1.0, (double)(step - _warmup) / _decay);
        return _baseLearningRate * Math.Pow(_final / _baseLearningRate, progress);
    }
}

public partial class AudioPaLM<T>
{
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private Random? _trainingRandom;

    /// <inheritdoc />
    /// <remarks>AudioPaLM's TTS task trains on a 16 kHz recording (<see cref="TtsTrainingSample{T}.Audio"/>), whose audio
    /// tokens are the target, and its text's token ids (<see cref="TtsTrainingSample{T}.Tokens"/>, from
    /// <see cref="TokenizeText"/>); other tasks train through <see cref="TrainTask"/>.</remarks>
    protected override TtsSupervision RequiredSupervision => TtsSupervision.Recording;

    /// <inheritdoc />
    public override bool TrainsOnSampleObjective => true;

    /// <inheritdoc />
    protected internal override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? TrainingOptimizer => HasPaperLayers ? Optimizer() : null;

    /// <inheritdoc />
    protected override bool TryApplyLearningRule(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        Train(new TtsTrainingSample<T> { Tokens = input, Audio = expectedOutput });
        return true;
    }

    private (int[] Prefix, int[] Output) TtsExample(TtsTrainingSample<T> sample)
    {
        Validation.Guard.NotNull(sample);
        var text = TokenIds(sample.Tokens, nameof(sample));
        if (text.Length == 0) throw new ArgumentException("The sample has no text tokens.", nameof(sample));
        if (text.Any(id => id >= _options.VocabSize)) throw new ArgumentException("The sample's text tokens include audio ids.", nameof(sample));
        var audio = sample.Audio ?? throw new ArgumentException($"{GetType().Name} trains on the recording; set Audio.", nameof(sample));
        int[] codes;
        using (new NoGradScope<T>()) codes = AudioTokensOf(audio);
        if (codes.Length == 0) throw new ArgumentException("The recording has no audio tokens.", nameof(sample));
        return (TaskPrefix(TtsTag, text), Core.AudioIds(codes));
    }

    /// <inheritdoc />
    /// <remarks>One update of the TTS task <c>[TTS language] text → audio tokens</c>: the cross-entropy of the audio
    /// tokens and the end token with the input's positions masked (§3.6), plus PaLM's z-loss.</remarks>
    protected override T TrainOnSample(TtsTrainingSample<T> sample)
    {
        ThrowIfDisposed();
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        if (!HasPaperLayers) throw new NotSupportedException("The objective needs the paper's networks; this model was built from caller-supplied layers.");
        var (prefix, output) = TtsExample(sample);
        return TrainSequence(sample.Tokens, prefix, output);
    }

    /// <summary>
    /// One update of any task (§3.4): <paramref name="tag"/> (such as <c>[ASR French]</c> or
    /// <c>[ASR AST S2ST English French]</c>) and <paramref name="input"/> in, <paramref name="output"/> and the end token
    /// out, all as decoder ids (text ids, or audio codes offset by <see cref="AudioOffset"/>). Returns the loss.
    /// </summary>
    public T TrainTask(string tag, IReadOnlyList<int> input, IReadOnlyList<int> output)
    {
        ThrowIfDisposed();
        if (output is null) throw new ArgumentNullException(nameof(output));
        if (!HasPaperLayers) throw NotNative();
        var prefix = TaskPrefix(tag, input);
        var ids = IndexTtsOps.Ids<T>(prefix);
        return TrainSequence(ids, prefix, output.ToArray());
    }

    private T TrainSequence(Tensor<T> anchor, int[] prefix, int[] output)
    {
        var random = _trainingRandom ??= AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.TrainingSeed);
        return TrainWithCustomObjective(anchor, anchor, (_, _) => Loss(prefix, output, true, random), Optimizer());
    }

    /// <inheritdoc />
    /// <remarks>The TTS objective with dropout off.</remarks>
    public override T EvaluateTrainingObjective(TtsTrainingSample<T> sample)
    {
        ThrowIfDisposed();
        if (!HasPaperLayers) throw new NotSupportedException("The objective needs the paper's networks; this model was built from caller-supplied layers.");
        var (prefix, output) = TtsExample(sample);
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            return Loss(prefix, output, false, null)[0];
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    internal Tensor<T> Loss(int[] prefix, int[] output, bool training, Random? random) => Core.Loss(prefix, output, training, random);

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> Optimizer()
    {
        if (_optimizer is not null) return _optimizer;
        var o = _options;
        // PaLM's Adafactor (Chowdhery et al. 2022 §5): no factorization, β1 0.9, β2 = 1 − k^−0.8, the step scaled by
        // the parameters' RMS, global gradient clipping at 1.0; AudioPaLM's constant 5e-5 (§3.6), or §5.4.8's ramp.
        var options = new AdafactorOptimizerOptions<T, Tensor<T>, Tensor<T>>
        {
            UseRelativeStepSize = false, MultiplyByParameterScale = true, Factored = false, Beta1 = o.Beta1,
            EnableGradientClipping = o.MaxGradientNorm > 0, MaxGradientNorm = o.MaxGradientNorm > 0 ? o.MaxGradientNorm : 1.0,
        };
        if (o.RampWarmupSteps > 0 || o.RampDecaySteps > 0)
        {
            options.InitialLearningRate = 0;
            options.LearningRateScheduler = new RampExponentialSchedule(o.RampPeakLearningRate, o.RampFinalLearningRate, o.RampWarmupSteps, o.RampDecaySteps);
            options.SchedulerStepMode = SchedulerStepMode.StepPerBatch;
            _optimizer = new AdafactorOptimizer<T, Tensor<T>, Tensor<T>>(this, options);
        }
        else
        {
            options.InitialLearningRate = o.LearningRate;
            _optimizer = PaperOptimizerFactory.VerifyHandBuilt(this, new AdafactorOptimizer<T, Tensor<T>, Tensor<T>>(this, options));
        }
        return _optimizer;
    }

    /// <inheritdoc />
    /// <remarks>All the decoder's parameters train (§3.2: "train all model parameters"); the speech encoders and the
    /// detokenizer are separate, frozen models.</remarks>
    protected override IReadOnlyList<Tensor<T>> SelectTrainableParametersForTraining(IReadOnlyList<Tensor<T>> parameters)
    {
        if (!HasPaperLayers) return parameters;
        var trained = new HashSet<Tensor<T>>(Training.TapeTrainingStep<T>.CollectParameters(_decoderLayers.Cast<ILayer<T>>().ToList(), -1),
            TensorReferenceComparer<Tensor<T>>.Instance);
        return parameters.Where(trained.Contains).ToList();
    }
}
