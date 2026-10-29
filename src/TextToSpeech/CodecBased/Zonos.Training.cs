using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.TextToSpeech.CodecBased;

public partial class Zonos<T>
{
    private Random? _trainingRandom;

    /// <inheritdoc />
    protected override bool TryApplyLearningRule(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        Train(new TtsTrainingSample<T> { Tokens = input, Audio = expectedOutput });
        return true;
    }

    /// <inheritdoc />
    /// <remarks>One utterance scored by <see cref="Loss"/>, each optional conditioner given replaced by its unconditional
    /// vector with probability <see cref="ZonosOptions.ConditionerDropout"/>.</remarks>
    protected override T TrainOnSample(TtsTrainingSample<T> sample)
    {
        ThrowIfDisposed();
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        if (!HasPaperLayers) throw new NotSupportedException("The objective needs the release's networks; this model was built from caller-supplied layers.");
        _trainingRandom ??= AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed);
        var (inputs, codes) = TrainingInputs(sample);
        var kept = DropConditioners(inputs, _trainingRandom);
        return TrainWithCustomObjective(sample.Tokens, sample.Tokens, (_, _) => Loss(kept, codes), Optimizer());
    }

    /// <inheritdoc />
    /// <remarks>The objective with every conditioner the sample gives (no dropout).</remarks>
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
            var (inputs, codes) = TrainingInputs(sample);
            return Loss(inputs, codes)[0];
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    // Recordings' codes and speaker embeddings (each model runs once per recording).
    private readonly Dictionary<Tensor<T>, int[,]> _codes = new(Helpers.TensorReferenceComparer<Tensor<T>>.Instance);
    private readonly Dictionary<Tensor<T>, double[]> _speakers = new(Helpers.TensorReferenceComparer<Tensor<T>>.Instance);

    /// <summary>
    /// A sample's conditioning and codes <c>[codebooks, frames]</c>: its phonemes (from <see cref="TtsTrainingSample{T}.Text"/>,
    /// else <see cref="TtsTrainingSample{T}.Tokens"/> as phoneme ids), its speaker's embedding when the speaker model is loaded
    /// and the sample has <see cref="TtsTrainingSample{T}.SpeakerReference"/>, and its language when
    /// <see cref="TtsTrainingSample{T}.LanguageId"/> is set; the conditioners a sample has no label for (emotion, pitch, rate,
    /// bandwidth, quality) stay unconditional.
    /// </summary>
    private (Dictionary<string, object> Inputs, int[,] Codes) TrainingInputs(TtsTrainingSample<T> sample)
    {
        Validation.Guard.NotNull(sample);
        var ids = sample.Text is not null
            ? ZonosPrefixConditioner<T>.PhonemeIds(Phonemes(sample.Text, _options.Language))
            : sample.Tokens.ToArray().Select(v => (int)Math.Round(Ops.ToDouble(v))).ToArray();
        if (ids.Length == 0) throw new ArgumentException("The sample has no text.", nameof(sample));
        var inputs = new Dictionary<string, object>(StringComparer.Ordinal) { ["espeak"] = ids };
        if (sample.SpeakerReference is not null && _speakerEncoder is not null)
        {
            if (!_speakers.TryGetValue(sample.SpeakerReference, out var speaker))
            {
                speaker = SpeakerEmbedding(sample.SpeakerReference, _options.SampleRate);
                if (_speakers.Count > 64) _speakers.Clear();
                _speakers[sample.SpeakerReference] = speaker;
            }
            inputs["speaker"] = speaker;
        }
        if (sample.LanguageId is int language) inputs["language_id"] = new[] { (double)language };

        int[,] codes;
        if (sample.CodecTokens is not null) codes = ByCodebook(CodesOf(sample.CodecTokens));
        else
        {
            var audio = sample.Audio ?? throw new ArgumentException($"{GetType().Name} trains on the recording or its codes; set Audio or CodecTokens.", nameof(sample));
            if (!_codes.TryGetValue(audio, out codes!))
            {
                codes = Codec.Encode(audio);
                if (_codes.Count > 64) _codes.Clear();
                _codes[audio] = codes;
            }
        }
        if (codes.GetLength(1) == 0) throw new ArgumentException("The recording has no frames.", nameof(sample));
        return (inputs, codes);
    }

    // Each optional conditioner given is left out (its unconditional vector used) with the dropout probability.
    private Dictionary<string, object> DropConditioners(Dictionary<string, object> inputs, Random random)
    {
        var required = Conditioner.RequiredNames.ToHashSet(StringComparer.Ordinal);
        return inputs.Where(kv => required.Contains(kv.Key) || random.NextDouble() >= _options.ConditionerDropout)
            .ToDictionary(kv => kv.Key, kv => kv.Value, StringComparer.Ordinal);
    }

    /// <summary>The training layout of codes <c>[codebooks, frames]</c>: the codes and a frame of end tokens, delayed as
    /// generation delays them (codebook k by k + 1 steps, mask tokens before and after), <c>[codebooks, frames + 10]</c>.</summary>
    internal int[,] DelayedSequence(int[,] codes)
    {
        int books = codes.GetLength(0), frames = codes.GetLength(1), length = frames + 1 + books;
        var delayed = new int[books, length];
        for (int k = 0; k < books; k++)
            for (int t = 0; t < length; t++)
            {
                int source = t - k - 1;
                delayed[k, t] = source < 0 || source > frames ? _options.MaskToken : source == frames ? _options.EndToken : codes[k, source];
            }
        return delayed;
    }

    /// <summary>
    /// The objective (no training code was released; the blog's "simple autoregression task"): the prefix of the
    /// conditioning, then the delayed codes but the last step; each head's logits at each step scored by cross-entropy on its
    /// codebook's next delayed token, the mask tokens unscored, averaged — the predictions generation samples.
    /// </summary>
    internal Tensor<T> Loss(IReadOnlyDictionary<string, object> inputs, int[,] codes)
    {
        var o = _options;
        var delayed = DelayedSequence(codes);
        int books = delayed.GetLength(0), steps = delayed.GetLength(1) - 1, v = o.Codec.CodebookSize + 1;
        var prefix = Conditioner.Forward(inputs);                                                                     // [1, P, width]
        var codesIn = new int[1, books, steps];
        for (int k = 0; k < books; k++)
            for (int t = 0; t < steps; t++) codesIn[0, k, t] = delayed[k, t];
        var hidden = Network.Forward(Engine.TensorConcatenate(new[] { prefix, Network.EmbedCodes(codesIn) }, 1), null);
        int p = prefix.Shape[1];
        var steps_ = Engine.TensorSlice(hidden, new[] { 0, p, 0 }, new[] { 1, steps, o.HiddenDim });
        var logits = Network.ApplyHeads(steps_);                                                                       // [1, steps, books, v]
        var picks = new List<int>();
        for (int t = 0; t < steps; t++)
            for (int k = 0; k < books; k++)
            {
                int target = delayed[k, t + 1];
                if (target != o.MaskToken) picks.Add((t * books + k) * v + target);
            }
        var logProbabilities = Engine.Reshape(Engine.TensorLogSoftmax(Engine.Reshape(logits, new[] { steps * books, v }), axis: 1), new[] { steps * books * v, 1 });
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
    /// <remarks>Training updates the backbone, the code tables, the heads and the conditioners; DAC and the speaker model are
    /// frozen, and the Fourier frequencies are fixed buffers.</remarks>
    protected override IReadOnlyList<Tensor<T>> SelectTrainableParametersForTraining(IReadOnlyList<Tensor<T>> parameters)
    {
        if (!HasPaperLayers) return parameters;
        var trained = new HashSet<Tensor<T>>(Training.TapeTrainingStep<T>.CollectParameters(_modelLayers.Cast<ILayer<T>>().ToList(), -1),
            Helpers.TensorReferenceComparer<Tensor<T>>.Instance);
        return parameters.Where(trained.Contains).ToList();
    }
}
