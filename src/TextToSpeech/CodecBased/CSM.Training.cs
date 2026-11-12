using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>A training row: the frames, codebook 0's label per frame (-100 where not scored), and the frames the audio
/// decoder learns from.</summary>
internal sealed record CsmTrainingRow(IReadOnlyList<CsmFrame> Frames, int[] Labels, int[] DecoderFrames);

public partial class CSM<T>
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
    /// <remarks>One turn — the speaker's text and the recording's frames with the frame of zeros that ends them — scored as
    /// Hugging Face's CSM port scores it: codebook 0 of every audio frame from the backbone's previous position, and
    /// codebooks 1–31 of a random sixteenth of the frames (<see cref="CSMOptions.DecoderFrameFraction"/>) from the audio
    /// decoder.</remarks>
    protected override T TrainOnSample(TtsTrainingSample<T> sample)
    {
        ThrowIfDisposed();
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        if (!HasPaperLayers) throw new NotSupportedException("The objective needs the release's networks; this model was built from caller-supplied layers.");
        _trainingRandom ??= AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed);
        var row = TrainingRow(sample, _trainingRandom);
        return TrainWithCustomObjective(sample.Tokens, sample.Tokens, (_, _) => Loss(row), Optimizer());
    }

    /// <inheritdoc />
    /// <remarks>The objective with the decoder's frames drawn from a fresh generator seeded by
    /// <see cref="CSMOptions.SamplingSeed"/>, so repeated evaluations agree.</remarks>
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
            return Loss(TrainingRow(sample, AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed)))[0];
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    // Recordings' codes (Mimi runs once per recording).
    private readonly Dictionary<Tensor<T>, int[,]> _codes = new(Helpers.TensorReferenceComparer<Tensor<T>>.Instance);

    internal int[,] SampleCodes(TtsTrainingSample<T> sample)
    {
        if (sample.CodecTokens is not null) return CodesOf(sample.CodecTokens, _options.MimiCodebookSize);
        var audio = sample.Audio ?? throw new ArgumentException($"{GetType().Name} trains on the recording or its codes; set Audio or CodecTokens.", nameof(sample));
        if (!_codes.TryGetValue(audio, out var codes))
        {
            codes = Codec.Encode(Doubles(audio));
            if (_codes.Count > 64) _codes.Clear();
            _codes[audio] = codes;
        }
        return codes;
    }

    /// <summary>
    /// The training row of a sample: the turn's text frames, its audio frames and the frame of zeros; labels (codebook 0)
    /// on the audio frames only, as the processor labels the audio token and not the audio end; and the decoder's frames,
    /// those left after the processor removes ⌊n · (1 − fraction)⌋ of the n audio frames at random.
    /// </summary>
    internal CsmTrainingRow TrainingRow(TtsTrainingSample<T> sample, Random random)
    {
        Validation.Guard.NotNull(sample);
        var codes = SampleCodes(sample);
        if (codes.GetLength(0) == 0) throw new ArgumentException("The recording has no frames.", nameof(sample));
        var frames = TextFrames(sample.Text ?? TextOf(sample.Tokens), sample.SpeakerId ?? 0);
        int firstAudio = frames.Count, audioFrames = codes.GetLength(0);
        frames.AddRange(AudioFrames(codes));
        if (frames.Count > _options.MaxSequenceLength)
            throw new ArgumentException($"The turn has {frames.Count} frames; CSM reads at most {_options.MaxSequenceLength}.", nameof(sample));
        var labels = frames.Select((f, p) => p >= firstAudio && p < firstAudio + audioFrames ? f.Codes![0] : -100).ToArray();
        // torch.randperm(n)[: int(n * (1 − ratio))] skipped; the rest train the decoder.
        int skipped = (int)(audioFrames * (1 - _options.DecoderFrameFraction));
        var order = Enumerable.Range(0, audioFrames).ToArray();
        for (int i = order.Length - 1; i > 0; i--)
        {
            int j = random.Next(i + 1);
            (order[i], order[j]) = (order[j], order[i]);
        }
        var decoderFrames = order.Skip(skipped).Select(i => firstAudio + i).OrderBy(p => p).ToArray();
        return new CsmTrainingRow(frames, labels, decoderFrames);
    }

    /// <summary>
    /// Hugging Face's <c>CsmForConditionalGeneration</c> loss: the backbone's codebook-0 logits scored against the next
    /// frame's label (<c>ForCausalLMLoss</c>), plus the decoder's: for each chosen frame, the backbone's state at the
    /// position before it and the embeddings of codebooks 0–30, projected and decoded, position j scored on codebook j + 1
    /// by its own head, averaged over all of them.
    /// </summary>
    internal Tensor<T> Loss(CsmTrainingRow row)
    {
        var o = _options;
        var frames = row.Frames;
        var hidden = Backbone.Forward(EmbedFrames(frames)).Hidden;                                   // [1, L, hidden]
        var labels = new int[1, row.Labels.Length];
        for (int i = 0; i < row.Labels.Length; i++) labels[0, i] = row.Labels[i];
        var backboneLoss = CausalLanguageModelLoss.Compute(Engine, Codebook0Logits(hidden), labels);

        int n = row.DecoderFrames.Length, h = o.HiddenDim, books = o.AudioCodebooks;
        if (n == 0) return backboneLoss;
        var previous = new Tensor<int>(new[] { n }, new Vector<int>(row.DecoderFrames.Select(p => p - 1).ToArray()));
        var states = Engine.Reshape(Engine.TensorIndexSelect(Engine.Reshape(hidden, new[] { frames.Count, h }), previous, 0), new[] { n, 1, h });
        var codeIds = new List<int>();
        foreach (int p in row.DecoderFrames)
            for (int k = 0; k + 1 < books; k++) codeIds.Add(k * o.AudioVocabSize + frames[p].Codes![k]);
        var embeddings = Engine.Reshape(Rows(_audioEmbeddings!, h, codeIds), new[] { n, books - 1, h });
        var decoded = DecoderHidden(Engine.TensorConcatenate(new[] { states, embeddings }, 1), null);   // [n, books, decoder hidden]
        var logits = new List<Tensor<T>>();
        var picks = new List<int>();
        for (int k = 1; k < books; k++)
        {
            var at = Engine.Reshape(Engine.TensorSlice(decoded, new[] { 0, k, 0 }, new[] { n, 1, o.DecoderHidden }), new[] { n, o.DecoderHidden });
            logits.Add(CodebookLogits(k, at));
            for (int i = 0; i < n; i++) picks.Add(((k - 1) * n + i) * o.AudioVocabSize + frames[row.DecoderFrames[i]].Codes![k]);
        }
        var all = Engine.TensorConcatenate(logits.ToArray(), 0);                                       // [(books − 1) n, vocab]
        var logProbabilities = Engine.Reshape(Engine.TensorLogSoftmax(all, axis: 1), new[] { all.Length, 1 });
        var picked = Engine.TensorIndexSelect(logProbabilities, new Tensor<int>(new[] { picks.Count }, new Vector<int>(picks.ToArray())), 0);
        var decoderLoss = Engine.TensorMultiplyScalar(Engine.ReduceSum(picked, null), Ops.FromDouble(-1.0 / picks.Count));
        return Engine.TensorAdd(backboneLoss, decoderLoss);
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
    /// <remarks>Training updates the backbone, the decoder, the embeddings and the heads; Mimi is frozen.</remarks>
    protected override IReadOnlyList<Tensor<T>> SelectTrainableParametersForTraining(IReadOnlyList<Tensor<T>> parameters)
    {
        if (!HasPaperLayers) return parameters;
        var trained = new HashSet<Tensor<T>>(Training.TapeTrainingStep<T>.CollectParameters(_modelLayers.Cast<ILayer<T>>().ToList(), -1),
            Helpers.TensorReferenceComparer<Tensor<T>>.Instance);
        return parameters.Where(trained.Contains).ToList();
    }
}
