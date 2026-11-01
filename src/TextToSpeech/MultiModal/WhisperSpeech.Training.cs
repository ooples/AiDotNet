using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.Latest;

namespace AiDotNet.TextToSpeech.MultiModal;

public partial class WhisperSpeech<T>
{
    private readonly Dictionary<WhisperSpeechTrainingStage, AdamWOptimizer<T, Tensor<T>, Tensor<T>>> _optimizers = new();
    private readonly Dictionary<WhisperSpeechTrainingStage, int> _stageSteps = new();
    private WhisperSpeechTrainingStage _trainedStage;
    private bool _semanticTablesStale;

    /// <inheritdoc />
    protected override TtsSupervision RequiredSupervision => TtsSupervision.Recording;

    /// <inheritdoc />
    public override bool TrainsOnSampleObjective => true;

    /// <inheritdoc />
    protected internal override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? TrainingOptimizer => HasPaperLayers ? Optimizer(_options.Stage) : null;

    /// <inheritdoc />
    protected override bool TryApplyLearningRule(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        Train(new TtsTrainingSample<T> { Tokens = input, Audio = expectedOutput });
        return true;
    }

    private void RequireNative()
    {
        ThrowIfDisposed();
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        if (!HasPaperLayers) throw new NotSupportedException("The objective needs the release's networks; this model was built from caller-supplied layers.");
    }

    // ---------------------------------------------------------------- parameter groups

    // The release trainer's groups (init_transformer's attributes): (layer, learning-rate multiple, decayed). Linear
    // layers step at base width / fan-in (muP), embeddings at the embedding multiple without decay, layer norms at 1
    // without decay, the T2S embedding projectors at their own multiple; the frozen semantic tables are not stepped.
    private List<(LayerBase<T> Layer, double Rate, bool Decay)> Groups(WhisperSpeechTrainingStage stage)
    {
        var o = _options;
        var groups = new List<(LayerBase<T>, double, bool)>();
        void Blocks(IEnumerable<WhisperSpeechBlock<T>> blocks, int width, int ffn, double baseWidth)
        {
            foreach (var b in blocks)
            {
                foreach (var a in new[] { b.Attention, b.CrossAttention }.Where(a => a is not null))
                {
                    groups.Add((a!.Query, baseWidth / width, true));
                    groups.Add((a.Key, baseWidth / width, true));
                    groups.Add((a.Value, baseWidth / width, true));
                    groups.Add((a.Output, baseWidth / width, true));
                }
                foreach (var norm in new[] { b.AttentionNorm, b.CrossNorm, b.MlpNorm }.Where(n => n is not null)) groups.Add((norm!, 1, false));
                groups.Add((b.Mlp1, baseWidth / width, true));
                groups.Add((b.Mlp2, baseWidth / (width * ffn), true));
            }
        }
        void Embeddings(WhisperSpeechEmbeddings<T> e, double rate, double projector, bool frozenMain)
        {
            if (!frozenMain) groups.Add((e.Main, rate, false));
            if (e.Special is not null) groups.Add((e.Special, rate, false));
            if (e.EmbedToHidden is not null) groups.Add((e.EmbedToHidden, projector, true));
            if (e.HiddenToEmbed is not null) groups.Add((e.HiddenToEmbed, projector, true));
        }
        switch (stage)
        {
            case WhisperSpeechTrainingStage.TextToSemantic:
            {
                var m = Core.TextToSemantic;
                var c = m.Configuration;
                groups.Add((m.LanguageEmbeddings, o.T2SEmbeddingsLrScale, false));
                if (m.CpsEmbeddings is not null) groups.Add((m.CpsEmbeddings, o.T2SEmbeddingsLrScale, false));
                Embeddings(m.TextEmbeddings, o.T2SEmbeddingsLrScale, o.T2SProjectorLrScale, false);
                Embeddings(m.SemanticEmbeddings, o.T2SEmbeddingsLrScale, o.T2SProjectorLrScale, true);
                Blocks(m.EncoderBlocks.Concat(m.DecoderBlocks), c.Width, c.FfnMult, c.BaseWidth);
                groups.Add((m.EncoderNorm, 1, false));
                groups.Add((m.DecoderNorm, 1, false));
                break;
            }
            case WhisperSpeechTrainingStage.SemanticToAcoustic:
            {
                var m = Core.SemanticToAcoustic;
                var c = m.Configuration;
                double b = c.BaseWidth;
                if (m.EmbedToHidden is not null) groups.Add((m.EmbedToHidden, b / c.SemanticWidth, true));
                if (m.HiddenToEmbed is not null) groups.Add((m.HiddenToEmbed, b / c.Width, true));
                foreach (var table in new[] { m.LanguageEmbedding, m.SnrEmbedding, m.C50Embedding }.Where(t => t is not null))
                    groups.Add((table!, o.S2AEmbeddingsLrScale, false));
                if (m.SpeakerToHidden is not null) groups.Add((m.SpeakerToHidden, b / c.SpeakerWidth, true));
                foreach (var e in m.AcousticEmbeddings)
                {
                    groups.Add((e.Main, o.S2AEmbeddingsLrScale, false));
                    if (e.Special is not null) groups.Add((e.Special, o.S2AEmbeddingsLrScale, false));
                    if (e.EmbedToHidden is not null) groups.Add((e.EmbedToHidden, b / e.FrozenWidth, true));
                    if (e.HiddenToEmbed is not null) groups.Add((e.HiddenToEmbed, b / c.Width, true));
                }
                Blocks(m.EncoderBlocks.Concat(m.DecoderBlocks), c.Width, c.FfnMult, b);
                groups.Add((m.EncoderNorm, 1, false));
                groups.Add((m.DecoderNorm, 1, false));
                groups.Add((m.Splitter, b / c.Width, true));
                break;
            }
            default:
            {
                var m = Core.Tokenizer;
                var c = m.Configuration;
                double b = c.BaseWidth;
                groups.Add((m.MlpNorm, 1, false));
                groups.Add((m.Mlp1, b / c.Width, true));
                groups.Add((m.Mlp2, b / (c.Width * c.FfnMult), true));
                foreach (var l in m.ProjectIn) groups.Add((l, b / c.Width, true));
                foreach (var l in m.ProjectOut) groups.Add((l, b / c.CodebookWidth, true));
                groups.Add((m.Positions, 1, false));
                Blocks(m.Blocks, c.Width, c.FfnMult, b);
                groups.Add((m.PostNorm, 1, false));
                break;
            }
        }
        return groups;
    }

    private IEnumerable<Tensor<T>> Parameters(LayerBase<T> layer) => Training.TapeTrainingStep<T>.CollectParameters(new ILayer<T>[] { layer }, -1);

    // ---------------------------------------------------------------- optimizer and schedule

    private (double Rate, double Clip, double Decay, int Warmup) Tunables(WhisperSpeechTrainingStage stage) => stage switch
    {
        WhisperSpeechTrainingStage.TextToSemantic => (_options.T2SLearningRate, _options.T2SMaxGradientNorm, _options.T2SWeightDecay, _options.T2SWarmupSteps),
        WhisperSpeechTrainingStage.SemanticToAcoustic => (_options.S2ALearningRate, _options.S2AMaxGradientNorm, _options.S2AWeightDecay, _options.S2AWarmupSteps),
        _ => (_options.VqLearningRate, _options.VqMaxGradientNorm, _options.VqWeightDecay, _options.VqWarmupSteps),
    };

    // torch's OneCycleLR (two phases, cosine annealing, div_factor = final_div_factor = 25, momentum 0.95 → 0.85 → 0.95):
    // the learning-rate factor and β1 at a step. Phase one ends at pct_start · total − 1, phase two at total − 1.
    internal static (double Factor, double Beta1) OneCycle(int step, int total, int warmup)
    {
        double pct = Math.Min(0.3, (double)warmup / total), end1 = pct * total - 1, end2 = total - 1;
        const double initial = 1.0 / 25, minimum = initial / 25;
        static double Anneal(double start, double end, double p) => end + (start - end) / 2 * (Math.Cos(Math.PI * p) + 1);
        double s = Math.Min(step, end2);
        if (s <= end1)
        {
            double p = end1 > 0 ? s / end1 : 1;
            return (Anneal(initial, 1, p), Anneal(0.95, 0.85, p));
        }
        double q = end2 > end1 ? (s - end1) / (end2 - end1) : 1;
        return (Anneal(1, minimum, q), Anneal(0.85, 0.95, q));
    }

    // AdamW β (0.9, 0.95) — β1 then cycled by the schedule — ε 1e-8, the stage's decay masked per group and its rate
    // multiplied per group, gradient-norm clipping at the stage's clip, the one-cycle schedule over TotalSteps.
    private AdamWOptimizer<T, Tensor<T>, Tensor<T>> Optimizer(WhisperSpeechTrainingStage stage)
    {
        if (_optimizers.TryGetValue(stage, out var existing)) return existing;
        var (rate, clip, decay, warmup) = Tunables(stage);
        int total = Math.Max(1, _options.TotalSteps);
        var optimizer = new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this, new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
        {
            InitialLearningRate = rate, Beta1 = OneCycle(0, total, warmup).Beta1, Beta2 = 0.95, Epsilon = 1e-8, WeightDecay = decay,
            UseAdaptiveLearningRate = false, UseAdaptiveBetas = false,
            EnableGradientClipping = clip > 0, MaxGradientNorm = clip > 0 ? clip : 1.0,
            LearningRateScheduler = new LambdaLRScheduler(rate, step => OneCycle(step, total, warmup).Factor),
            SchedulerStepMode = SchedulerStepMode.StepPerBatch,
        });
        var parameters = new List<Tensor<T>>();
        var rates = new List<T>();
        var decays = new List<T>();
        foreach (var (layer, scale, decayed) in Groups(stage))
            foreach (var p in Parameters(layer))
            {
                parameters.Add(p);
                for (int i = 0; i < p.Length; i++)
                {
                    rates.Add(Ops.FromDouble(scale));
                    decays.Add(decayed ? Ops.One : Ops.Zero);
                }
            }
        var options = (AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>)optimizer.GetOptions();
        options.LearningRateMask = new Vector<T>(rates.ToArray());
        options.WeightDecayMask = new Vector<T>(decays.ToArray());
        options.MaskParameters = parameters;
        return _optimizers[stage] = optimizer;
    }

    private T Step(WhisperSpeechTrainingStage stage, Func<Tensor<T>> objective)
    {
        var optimizer = Optimizer(stage);
        _stageSteps.TryGetValue(stage, out int step);
        var (_, _, _, warmup) = Tunables(stage);
        ((AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>)optimizer.GetOptions()).Beta1 = OneCycle(step, Math.Max(1, _options.TotalSteps), warmup).Beta1;
        _trainedStage = stage;
        var marker = IndexTtsOps.Ids<T>(new[] { 0 });
        var loss = TrainWithCustomObjective(marker, marker, (_, _) => objective(), optimizer);
        _stageSteps[stage] = step + 1;
        if (stage == WhisperSpeechTrainingStage.SemanticTokenizer) _semanticTablesStale = true;
        return loss;
    }

    // ---------------------------------------------------------------- objectives

    // Python's len(str): code points.
    private static int CodePoints(string text) => text.Count(ch => !char.IsLowSurrogate(ch));

    // The stage's loss of a transcript and its recording at the model's rate, its targets computed in evaluation mode
    // (the release prepares them offline: semantic tokens by the VQ tokenizer, acoustic tokens by EnCodec, speaker
    // embeddings by ECAPA-TDNN).
    private Func<Tensor<T>> Objective(string text, Tensor<T> recording, bool training)
    {
        var core = Core;
        var stage = _options.Stage;
        if (stage != WhisperSpeechTrainingStage.SemanticTokenizer && _semanticTablesStale)
        {
            core.SyncSemanticTables();
            _semanticTablesStale = false;
        }
        int language = WhisperSpeechCore<T>.LanguageId(_options.Language);
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            var wave = core.Resampled(recording, _options.SampleRate, 16000);
            switch (stage)
            {
                case WhisperSpeechTrainingStage.TextToSemantic:
                {
                    var semantic = core.SemanticTokens(wave);
                    var tokens = semantic.Take(core.TextToSemantic.Configuration.SemanticLength - 1).ToArray();
                    // char_per_seconder: characters over the semantic tokens' seconds.
                    double cps = CodePoints(text) / Math.Max(semantic.Length / core.SemanticRate, 1e-9);
                    return () => core.TextToSemantic.Loss(text, tokens, language, cps);
                }
                case WhisperSpeechTrainingStage.SemanticToAcoustic:
                {
                    var s2a = core.SemanticToAcoustic;
                    var semantic = core.SemanticTokens(wave).Take(s2a.Configuration.SemanticLength - 1).ToArray();
                    var acoustic = s2a.Delayed(core.AcousticTokens(recording), s2a.Configuration.Context);
                    Tensor<T> speaker;
                    using (new NoGradScope<T>()) speaker = core.Speaker.Embed(wave.Length > 30 * 16000 ? wave.Take(30 * 16000).ToArray() : wave);
                    var input = s2a.SemanticInput(semantic);
                    var conditions = CurrentConditions();
                    return () => s2a.Loss(input, acoustic, speaker, conditions);
                }
                default:
                {
                    var (mel, mask) = core.TrainingWindow(wave);
                    // tokenize_text: the start-of-transcript sequence with no timestamps, the transcript, end-of-text
                    // padding to 200 for the input and −100 for the shifted targets.
                    var transcriber = core.Transcriber;
                    var tokens = transcriber.SotSequence(_options.Language).Concat(new[] { transcriber.NoTimestamps }).Concat(transcriber.Encode(text)).ToList();
                    int size = Math.Min(200, _options.WhisperTextContext);
                    if (tokens.Count > size) throw new ArgumentException($"The transcript is longer than {size} Whisper tokens.");
                    var inputTokens = tokens.Concat(Enumerable.Repeat(transcriber.Eot, size - tokens.Count)).ToArray();
                    var outputTokens = tokens.Skip(1).Concat(new[] { transcriber.Eot }).Concat(Enumerable.Repeat(-100, size - tokens.Count)).ToArray();
                    return () => core.Tokenizer.Loss(mel, mask, inputTokens, outputTokens, updateCodebook: training);
                }
            }
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    private (string Text, Tensor<T> Audio) Unpack(TtsTrainingSample<T> sample)
    {
        if (sample is null) throw new ArgumentNullException(nameof(sample));
        var audio = sample.Audio ?? throw new ArgumentException("WhisperSpeech trains on a recording (Audio) of the text.", nameof(sample));
        return (TextOf(sample.Tokens), audio);
    }

    /// <inheritdoc />
    /// <remarks>A step of <see cref="WhisperSpeechOptions.Stage"/> on the sample's text (UTF-8 bytes) and its recording.</remarks>
    protected override T TrainOnSample(TtsTrainingSample<T> sample)
    {
        RequireNative();
        var (text, audio) = Unpack(sample);
        return Step(_options.Stage, Objective(text, audio, training: true));
    }

    /// <summary>A step of <see cref="WhisperSpeechOptions.Stage"/> on a transcript and its recording at the model's rate.</summary>
    public T Train(string text, Tensor<T> recording)
    {
        RequireNative();
        if (text is null) throw new ArgumentNullException(nameof(text));
        if (recording is null) throw new ArgumentNullException(nameof(recording));
        return Step(_options.Stage, Objective(text, recording, training: true));
    }

    /// <inheritdoc />
    public override T EvaluateTrainingObjective(TtsTrainingSample<T> sample)
    {
        RequireNative();
        var (text, audio) = Unpack(sample);
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            var objective = Objective(text, audio, training: false);
            using var scope = new NoGradScope<T>();
            return objective().GetFlat(0);
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    /// <inheritdoc />
    /// <remarks>Only the trained stage's parameters, without the frozen semantic tables.</remarks>
    protected override IReadOnlyList<Tensor<T>> SelectTrainableParametersForTraining(IReadOnlyList<Tensor<T>> parameters)
    {
        if (!HasPaperLayers) return parameters;
        var trained = new HashSet<Tensor<T>>(Groups(_trainedStage).SelectMany(g => Parameters(g.Layer)), TensorReferenceComparer<Tensor<T>>.Instance);
        return parameters.Where(trained.Contains).ToList();
    }
}
