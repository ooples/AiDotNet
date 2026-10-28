using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.TextToSpeech.CodecBased;

public partial class Dia<T>
{
    /// <inheritdoc />
    protected override bool TryApplyLearningRule(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        Train(new TtsTrainingSample<T> { Tokens = input, Audio = expectedOutput });
        return true;
    }

    /// <inheritdoc />
    /// <remarks>One script and its recording, scored as transformers' DiaForConditionalGeneration scores them
    /// (<see cref="Loss"/>).</remarks>
    protected override T TrainOnSample(TtsTrainingSample<T> sample)
    {
        ThrowIfDisposed();
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        if (!HasPaperLayers) throw new NotSupportedException("The objective needs the release's networks; this model was built from caller-supplied layers.");
        var (text, codes) = TrainingPair(sample);
        return TrainWithCustomObjective(sample.Tokens, sample.Tokens, (_, _) => Loss(text, codes), Optimizer());
    }

    /// <inheritdoc />
    public override T EvaluateTrainingObjective(TtsTrainingSample<T> sample)
    {
        ThrowIfDisposed();
        Validation.Guard.NotNull(sample);
        if (!HasPaperLayers) throw new NotSupportedException("The objective needs the release's networks; this model was built from caller-supplied layers.");
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            var (text, codes) = TrainingPair(sample);
            return Loss(text, codes)[0];
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    // Recordings' codes (DAC runs once per recording).
    private readonly Dictionary<Tensor<T>, int[,]> _codes = new(Helpers.TensorReferenceComparer<Tensor<T>>.Instance);

    private (int[] Text, int[,] Codes) TrainingPair(TtsTrainingSample<T> sample)
    {
        Validation.Guard.NotNull(sample);
        var text = sample.Text is not null ? Tokenize(sample.Text, _options.MaxTextLength) : Tokenize(TextOf(sample.Tokens), _options.MaxTextLength);
        if (text.Length == 0) throw new ArgumentException("The script has no bytes.", nameof(sample));
        int[,] codes;
        if (sample.CodecTokens is not null) codes = CodesOf(sample.CodecTokens);
        else
        {
            var audio = sample.Audio ?? throw new ArgumentException($"{GetType().Name} trains on the recording or its codes; set Audio or CodecTokens.", nameof(sample));
            if (!_codes.TryGetValue(audio, out codes!))
            {
                codes = FramesFirst(Codec.Encode(audio));
                if (_codes.Count > 64) _codes.Clear();
                _codes[audio] = codes;
            }
        }
        if (codes.GetLength(0) == 0) throw new ArgumentException("The recording has no frames.", nameof(sample));
        return (text, codes);
    }

    /// <summary>
    /// The decoder's training layout (DiaProcessor with <c>output_labels</c>): the start frame, the codes and the end
    /// frame, delayed (start tokens before each channel's delay, padding after the end), as <c>[steps, channels]</c>.
    /// </summary>
    internal int[,] DelayedSequence(int[,] codes)
    {
        var o = _options;
        int frames = codes.GetLength(0), channels = o.Channels, maxDelay = o.DelayPattern.Max(), steps = frames + 2 + maxDelay;
        var delayed = new int[steps, channels];
        for (int t = 0; t < steps; t++)
            for (int c = 0; c < channels; c++)
            {
                int source = t - o.DelayPattern[c];
                delayed[t, c] = source < 0 || source == 0 ? o.BeginToken : source <= frames ? codes[source - 1, c] : source == frames + 1 ? o.EndToken : o.PadToken;
            }
        return delayed;
    }

    /// <summary>
    /// transformers' DiaForConditionalGeneration loss: the encoder reads the script; the decoder reads the delayed sequence
    /// but its last step; every channel's next token is scored by cross-entropy (<c>ForMaskedLMLoss</c>), padding and start
    /// tokens excluded, averaged over all of them.
    /// </summary>
    internal Tensor<T> Loss(int[] text, int[,] codes)
    {
        var o = _options;
        var delayed = DelayedSequence(codes);
        int steps = delayed.GetLength(0) - 1, channels = o.Channels, vocab = o.AudioVocabSize;
        var ids = new int[1, text.Length];
        for (int j = 0; j < text.Length; j++) ids[0, j] = text[j];
        var inputs = new int[1, steps, channels];
        for (int t = 0; t < steps; t++)
            for (int c = 0; c < channels; c++) inputs[0, t, c] = delayed[t, c];
        var logits = Network.Decode(inputs, Network.StartDecoding(Network.Encode(ids)));                  // [1, steps, channels, vocab]
        var picks = new List<int>();
        for (int t = 0; t < steps; t++)
            for (int c = 0; c < channels; c++)
            {
                int label = delayed[t + 1, c];
                if (label == o.PadToken || label == o.BeginToken) continue;
                picks.Add((t * channels + c) * vocab + label);
            }
        if (picks.Count == 0) throw new ArgumentException("The sample has nothing to score.", nameof(codes));
        var logProbabilities = Engine.Reshape(Engine.TensorLogSoftmax(Engine.Reshape(logits, new[] { steps * channels, vocab }), axis: 1),
            new[] { steps * channels * vocab, 1 });
        var picked = Engine.TensorIndexSelect(logProbabilities, new Tensor<int>(new[] { picks.Count }, new Vector<int>(picks.ToArray())), 0);
        return Engine.TensorMultiplyScalar(Engine.ReduceSum(picked, null), Ops.FromDouble(-1.0 / picks.Count));
    }

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> Optimizer()
    {
        if (_optimizer is not null) return _optimizer;
        var o = _options;
        var options = new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
        {
            InitialLearningRate = o.LearningRate, Beta1 = o.Beta1, Beta2 = o.Beta2, Epsilon = o.Epsilon, WeightDecay = o.WeightDecay,
            EnableGradientClipping = o.MaxGradNorm > 0, MaxGradientNorm = o.MaxGradNorm,
        };
        if (o.TrainingSteps > 0)
        {
            options.LearningRateScheduler = new LinearWarmupScheduler(o.LearningRate, 0, o.TrainingSteps, 0.0, LinearWarmupScheduler.DecayMode.Linear);
            options.SchedulerStepMode = SchedulerStepMode.StepPerBatch;
        }
        options.MaskParameters = Training.TapeTrainingStep<T>.CollectParameters(_modelLayers.Cast<ILayer<T>>().ToList(), -1).ToList();
        return _optimizer = PaperOptimizerFactory.VerifyHandBuilt(this, new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this, options));
    }

    /// <inheritdoc />
    /// <remarks>Training updates the encoder and the decoder; DAC is frozen.</remarks>
    protected override IReadOnlyList<Tensor<T>> SelectTrainableParametersForTraining(IReadOnlyList<Tensor<T>> parameters)
    {
        if (!HasPaperLayers) return parameters;
        var trained = new HashSet<Tensor<T>>(Training.TapeTrainingStep<T>.CollectParameters(_modelLayers.Cast<ILayer<T>>().ToList(), -1),
            Helpers.TensorReferenceComparer<Tensor<T>>.Instance);
        return parameters.Where(trained.Contains).ToList();
    }
}
