using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.TextToSpeech.CodecBased;

public abstract partial class FishSpeechBase<T>
{
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private Random? _trainingRandom;
    private FishDualAr<T>? _preferenceReference;

    /// <inheritdoc />
    /// <remarks>Fish Speech trains on Firefly-GAN's codes of a recording (<see cref="TtsTrainingSample{T}.CodecTokens"/>,
    /// <c>[frames, codebooks]</c>; the release extracts them ahead of training with <c>extract_vq.py</c>) or on the recording
    /// itself at the codec's rate (<see cref="TtsTrainingSample{T}.Audio"/>), with its text
    /// (<see cref="TtsTrainingSample{T}.Text"/>, or the cleaned text's token ids in
    /// <see cref="TtsTrainingSample{T}.Tokens"/>).</remarks>
    protected override TtsSupervision RequiredSupervision => TtsSupervision.CodecTokens;

    /// <inheritdoc />
    public override int CodecTokenCodebooks => _options.NumCodebooks;

    /// <inheritdoc />
    /// <remarks>The codec's codes (the quantizer's levels multiplied: 1,000 in the release).</remarks>
    public override int CodecTokenVocabulary => _options.CodecLevels.Aggregate(1, (a, b) => a * b);

    /// <inheritdoc />
    public override bool TrainsOnSampleObjective => true;

    /// <inheritdoc />
    /// <remarks>AdamW with the recipe's schedule (<see cref="FishSpeechFamilyOptions.Recipe"/>).</remarks>
    protected internal override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? TrainingOptimizer => HasPaperLayers ? Optimizer() : null;

    /// <inheritdoc />
    protected override bool TryApplyLearningRule(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        Train(new TtsTrainingSample<T> { Tokens = input, Audio = expectedOutput });
        return true;
    }

    private Random TrainingRandom => _trainingRandom ??= AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.TrainingSeed);

    private void RequireTraining()
    {
        ThrowIfDisposed();
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        if (!HasPaperLayers) throw new NotSupportedException("The objective needs the release's networks; this model was built from caller-supplied layers.");
    }

    /// <inheritdoc />
    /// <remarks>One step of <c>lit_module</c>'s objective: the token cross-entropy plus the codebook cross-entropy of the
    /// sample packed as the release's dataset packs it.</remarks>
    protected override T TrainOnSample(TtsTrainingSample<T> sample)
    {
        RequireTraining();
        var (tokens, labels) = Packed(sample);
        var random = TrainingRandom;
        return TrainWithCustomObjective(sample.Tokens, sample.Tokens, (_, _) => CrossEntropy(Transformer, tokens, labels, true, random), Optimizer());
    }

    /// <inheritdoc />
    public override T EvaluateTrainingObjective(TtsTrainingSample<T> sample)
    {
        RequireTraining();
        var (tokens, labels) = Packed(sample);
        return Evaluate(() => CrossEntropy(Transformer, tokens, labels, false, null));
    }

    private T Evaluate(Func<Tensor<T>> objective)
    {
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            return objective()[0];
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    private Tensor<T> CrossEntropy(FishDualAr<T> net, int[,] tokens, int[,] labels, bool training, Random? random)
    {
        var (token, codebook) = FishSpeechObjective<T>.Losses(Engine, net, Format, tokens, labels, training, random);
        return Engine.TensorAdd(token, codebook);
    }

    // Recordings' codes (Firefly-GAN runs once per recording).
    private readonly Dictionary<Tensor<T>, int[,]> _codes = new(TensorReferenceComparer<Tensor<T>>.Instance);

    private int[,] CodesOf(TtsTrainingSample<T> sample)
    {
        if (sample.CodecTokens is { } given)
        {
            if (given.Rank != 2 || given.Shape[1] != _options.NumCodebooks)
                throw new ArgumentException($"CodecTokens are [frames, {_options.NumCodebooks}].", nameof(sample));
            var codes = new int[given.Shape[1], given.Shape[0]];
            for (int t = 0; t < given.Shape[0]; t++)
                for (int b = 0; b < given.Shape[1]; b++)
                {
                    codes[b, t] = (int)Math.Round(Ops.ToDouble(given[t, b]));
                    if (codes[b, t] < 0 || codes[b, t] >= CodecTokenVocabulary)
                        throw new ArgumentOutOfRangeException(nameof(sample), $"Codes lie in [0, {CodecTokenVocabulary}).");
                }
            return codes;
        }
        var audio = sample.Audio ?? throw new ArgumentException($"{GetType().Name} trains on the recording or its codes; set Audio or CodecTokens.", nameof(sample));
        if (!_codes.TryGetValue(audio, out var encoded))
        {
            encoded = EncodeAudio(audio);
            if (_codes.Count > 64) _codes.Clear();
            _codes[audio] = encoded;
        }
        return encoded;
    }

    /// <summary>A sample as the release's dataset packs it: the cleaned text (or the given token ids) and the codes.</summary>
    private (int[,] Tokens, int[,] Labels) Packed(TtsTrainingSample<T> sample)
    {
        Validation.Guard.NotNull(sample);
        var codes = CodesOf(sample);
        if (codes.GetLength(1) == 0) throw new ArgumentException("The recording has no frames.", nameof(sample));
        var packed = sample.Text is { } text
            ? Format.TrainingSample(CleanText(text), codes)
            : Format.TrainingSample(TextIds(sample.Tokens, nameof(sample)), codes);
        if (packed.Tokens.GetLength(1) > _options.MaxSequenceLength)
            throw new ArgumentException($"The packed sample is longer than the context ({_options.MaxSequenceLength}).", nameof(sample));
        return packed;
    }

    // ---------------------------------------------------------------- preference training

    /// <summary>
    /// Takes the current transformer as standard DPO's frozen reference (Rafailov et al. 2023). <see cref="TrainPreference"/>
    /// takes it at its first step when this has not been called.
    /// </summary>
    public void FreezePreferenceReference()
    {
        RequireTraining();
        var referenceLayers = new List<LayerBase<T>>();
        var reference = new FishDualAr<T>(Engine, referenceLayers, TransformerConfiguration(),
            AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(0));
        var source = Training.TapeTrainingStep<T>.CollectParameters(_netLayers.Cast<ILayer<T>>().ToList(), -1).ToList();
        var target = Training.TapeTrainingStep<T>.CollectParameters(referenceLayers.Cast<ILayer<T>>().ToList(), -1).ToList();
        if (source.Count != target.Count) throw new InvalidOperationException("The reference transformer's parameters do not line up with the model's.");
        for (int k = 0; k < source.Count; k++)
        {
            if (source[k].Length != target[k].Length) throw new InvalidOperationException("The reference transformer's parameters do not line up with the model's.");
            for (int i = 0; i < source[k].Length; i++) target[k][i] = source[k][i];
        }
        _preferenceReference = reference;
    }

    /// <summary>
    /// One step of standard DPO on a preference pair for the same text: <c>-log σ(β((log π(chosen) − log π_ref(chosen)) −
    /// (log π(rejected) − log π_ref(rejected))))</c> over each sequence's scored tokens and codes, β being
    /// <see cref="FishSpeechFamilyOptions.DpoBeta"/>.
    /// </summary>
    public T TrainPreference(TtsTrainingSample<T> chosen, TtsTrainingSample<T> rejected)
    {
        RequireTraining();
        if (_preferenceReference is null) FreezePreferenceReference();
        var reference = _preferenceReference ?? throw new InvalidOperationException("The preference reference was not taken.");
        var (chosenTokens, chosenLabels) = Packed(chosen);
        var (rejectedTokens, rejectedLabels) = Packed(rejected);
        double referenceChosen, referenceRejected;
        using (new NoGradScope<T>())
        {
            referenceChosen = Ops.ToDouble(FishSpeechObjective<T>.SequenceLogProbability(Engine, reference, Format, chosenTokens, chosenLabels)[0]);
            referenceRejected = Ops.ToDouble(FishSpeechObjective<T>.SequenceLogProbability(Engine, reference, Format, rejectedTokens, rejectedLabels)[0]);
        }
        var random = TrainingRandom;
        Tensor<T> Objective()
        {
            var c = FishSpeechObjective<T>.SequenceLogProbability(Engine, Transformer, Format, chosenTokens, chosenLabels, true, random);
            var r = FishSpeechObjective<T>.SequenceLogProbability(Engine, Transformer, Format, rejectedTokens, rejectedLabels, true, random);
            return FishSpeechObjective<T>.Dpo(Engine, c, r, referenceChosen, referenceRejected, _options.DpoBeta);
        }
        return TrainWithCustomObjective(chosen.Tokens, chosen.Tokens, (_, _) => Objective(), Optimizer());
    }

    /// <summary>
    /// One step of the release's reference-free preference objective (git 8093258): the chosen sample's cross-entropy plus
    /// <c>-log σ(β(log π(chosen codes) − log π(rejected codes)))</c> averaged over positions, β being
    /// <see cref="FishSpeechFamilyOptions.ReferenceFreeDpoBeta"/>.
    /// </summary>
    public T TrainPreferenceReferenceFree(TtsTrainingSample<T> chosen, TtsTrainingSample<T> rejected)
    {
        RequireTraining();
        var (chosenTokens, chosenLabels) = Packed(chosen);
        var (rejectedTokens, rejectedLabels) = Packed(rejected);
        var random = TrainingRandom;
        Tensor<T> Objective() => Engine.TensorAdd(CrossEntropy(Transformer, chosenTokens, chosenLabels, true, random),
            FishSpeechObjective<T>.ReferenceFreeDpo(Engine, Transformer, Format, chosenTokens, chosenLabels, rejectedTokens, rejectedLabels,
                _options.ReferenceFreeDpoBeta, true, random));
        return TrainWithCustomObjective(chosen.Tokens, chosen.Tokens, (_, _) => Objective(), Optimizer());
    }

    // ---------------------------------------------------------------- optimizer

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> Optimizer()
    {
        if (_optimizer is not null) return _optimizer;
        var o = _options;
        var net = Transformer;
        // lit_module's parameter groups: the token embeddings and every norm's gain are not decayed (".embeddings." and
        // "norm.weight"); the codebook and fast embeddings are.
        var exempt = new HashSet<ILayer<T>>(new ILayer<T>[] { net.Embeddings, net.SlowNorm, net.FastNorm }
            .Concat(net.Slow.Concat(net.Fast).SelectMany(b => new ILayer<T>[] { b.AttentionNorm, b.FeedForwardNorm })));
        var parameters = new List<Tensor<T>>();
        var decays = new List<T>();
        foreach (var layer in _netLayers)
            foreach (var p in Training.TapeTrainingStep<T>.CollectParameters(new ILayer<T>[] { layer }, -1))
            {
                parameters.Add(p);
                var decay = exempt.Contains(layer) ? Ops.Zero : Ops.One;
                for (int i = 0; i < p.Length; i++) decays.Add(decay);
            }
        int warmup = o.WarmupSteps, total = o.TrainingSteps;
        bool cosine = o.CosineDecay;
        double floor = o.FinalLearningRateRatio;
        var options = new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
        {
            InitialLearningRate = o.LearningRate, Beta1 = o.Beta1, Beta2 = o.Beta2, Epsilon = o.Epsilon, WeightDecay = o.WeightDecay,
            EnableGradientClipping = o.MaxGradientNorm > 0, MaxGradientNorm = o.MaxGradientNorm,
            // fish_speech/scheduler.py: linear warm-up, then constant, or a half cosine to the floor.
            LearningRateScheduler = new LambdaLRScheduler(o.LearningRate, step =>
            {
                if (step < warmup) return step / (double)Math.Max(1, warmup);
                if (!cosine) return 1.0;
                double progress = (step - warmup) / (double)Math.Max(1, total - warmup);
                return Math.Max(floor, 0.5 * (1.0 + Math.Cos(Math.PI * progress)));
            }),
            SchedulerStepMode = SchedulerStepMode.StepPerBatch,
            WeightDecayMask = new Vector<T>(decays.ToArray()),
            MaskParameters = parameters,
        };
        _optimizer = PaperOptimizerFactory.VerifyHandBuilt(this, new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this, options));
        return _optimizer;
    }

    /// <inheritdoc />
    /// <remarks>Training updates the Dual-AR transformer; Firefly-GAN is frozen.</remarks>
    protected override IReadOnlyList<Tensor<T>> SelectTrainableParametersForTraining(IReadOnlyList<Tensor<T>> parameters)
    {
        if (!HasPaperLayers) return parameters;
        var trained = new HashSet<Tensor<T>>(Training.TapeTrainingStep<T>.CollectParameters(_netLayers.Cast<ILayer<T>>().ToList(), -1),
            TensorReferenceComparer<Tensor<T>>.Instance);
        return parameters.Where(trained.Contains).ToList();
    }
}
