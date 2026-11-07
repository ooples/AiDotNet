using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.TextToSpeech.VoiceCloning;

internal sealed record ChatterboxT3Condition<T>(
    Tensor<T> SpeakerEmbedding,
    IReadOnlyList<int> PromptSpeechTokens,
    double Exaggeration);

/// <summary>The release's T3 token-to-token model: custom condition/text/speech embeddings around a bare Llama decoder.</summary>
internal sealed class ChatterboxT3<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly ChatterboxOptions _options;
    private readonly LlamaNetwork<T> _transformer;
    private readonly ChatterboxPerceiver<T> _perceiver;

    public ChatterboxT3(IEngine engine, List<LayerBase<T>> layers, ChatterboxOptions options, Random random)
    {
        _engine = engine;
        _options = options;

        TextEmbedding = Own(layers, new TiedEmbeddingLayer<T>(options.VocabSize, options.LLMDim));
        SpeechEmbedding = Own(layers, new TiedEmbeddingLayer<T>(options.SpeechVocabularySize, options.LLMDim));
        TextPositionEmbedding = Own(layers, new TiedEmbeddingLayer<T>(options.MaxTextLength + 2, options.LLMDim));
        SpeechPositionEmbedding = Own(layers, new TiedEmbeddingLayer<T>(options.MaxSpeechTokens + 4, options.LLMDim));
        SpeakerProjection = Own(layers, Dense(options.LLMDim));
        EmotionProjection = Own(layers, new BiasFreeLinearLayer<T>(1, options.LLMDim));
        _perceiver = new ChatterboxPerceiver<T>(engine, layers, options, random);
        TextHead = Own(layers, new BiasFreeLinearLayer<T>(options.LLMDim, options.VocabSize));
        SpeechHead = Own(layers, new BiasFreeLinearLayer<T>(options.LLMDim, options.SpeechVocabularySize));

        _transformer = new LlamaNetwork<T>(engine, layers, new LlamaConfiguration
        {
            VocabSize = 0,
            Hidden = options.LLMDim,
            Layers = options.NumLLMLayers,
            Heads = options.NumHeads,
            KeyValueHeads = options.KeyValueHeads,
            HeadDim = options.LLMDim / options.NumHeads,
            Intermediate = options.LmFeedForwardDim,
            RmsEpsilon = options.RmsEpsilon,
            RopeTheta = options.RopeTheta,
            RopeScalingFactor = options.RopeScalingFactor,
            RopeLowFrequencyFactor = options.RopeLowFrequencyFactor,
            RopeHighFrequencyFactor = options.RopeHighFrequencyFactor,
            RopeOriginalMaxPositions = options.RopeOriginalMaxPositions,
            IncludeHead = false,
        });
        _transformer.InitializeLikeReference(random);

        double Gaussian(double std)
        {
            double radius = std * Math.Sqrt(-2.0 * Math.Log(1.0 - random.NextDouble()));
            return radius * Math.Cos(2.0 * Math.PI * random.NextDouble());
        }
        TextEmbedding.Reinitialize(() => Gaussian(1.0));
        SpeechEmbedding.Reinitialize(() => Gaussian(1.0));
        TextPositionEmbedding.Reinitialize(() => Gaussian(0.02));
        SpeechPositionEmbedding.Reinitialize(() => Gaussian(0.02));
    }

    public TiedEmbeddingLayer<T> TextEmbedding { get; }
    public TiedEmbeddingLayer<T> SpeechEmbedding { get; }
    public TiedEmbeddingLayer<T> TextPositionEmbedding { get; }
    public TiedEmbeddingLayer<T> SpeechPositionEmbedding { get; }
    public DenseLayer<T> SpeakerProjection { get; }
    public BiasFreeLinearLayer<T> EmotionProjection { get; }
    public BiasFreeLinearLayer<T> TextHead { get; }
    public BiasFreeLinearLayer<T> SpeechHead { get; }
    internal LlamaConfiguration TransformerConfiguration => _transformer.Configuration;
    internal ChatterboxPerceiver<T> Perceiver => _perceiver;

    public Tensor<T> Loss(
        IReadOnlyList<int> text,
        IReadOnlyList<int> speech,
        ChatterboxT3Condition<T> condition)
    {
        var textIds = WithBoundaries(text, _options.TextStartToken, _options.TextStopToken, _options.VocabSize, _options.MaxTextLength, "text");
        var speechIds = WithBoundaries(speech, _options.SpeechStartToken, _options.SpeechStopToken,
            _options.SpeechVocabularySize, _options.MaxSpeechTokens, "speech");

        var logits = TeacherForcedLogits(textIds, speechIds, condition);
        var textLoss = NextTokenCrossEntropy(logits.Text, textIds);
        var speechLoss = NextTokenCrossEntropy(logits.Speech, speechIds);
        return _engine.TensorAdd(
            _engine.TensorMultiplyScalar(textLoss, NumOps.FromDouble(_options.TextLossWeight)),
            _engine.TensorMultiplyScalar(speechLoss, NumOps.FromDouble(_options.SpeechLossWeight)));
    }

    internal (Tensor<T> Text, Tensor<T> Speech) TeacherForcedLogits(IReadOnlyList<int> textIds,
        IReadOnlyList<int> speechIds, ChatterboxT3Condition<T> condition)
    {
        var textTensor = IdTensor(textIds, batch: 1);
        var speechTensor = IdTensor(speechIds, batch: 1);
        var embeds = PrepareEmbeddings(condition, textTensor, speechTensor, classifierFreeGuidance: false);
        var hidden = _transformer.Forward(embeds).Hidden;
        int conditions = ConditionLength;
        var textHidden = _engine.TensorSlice(hidden, new[] { 0, conditions, 0 }, new[] { 1, textIds.Count, _options.LLMDim });
        var speechHidden = _engine.TensorSlice(hidden, new[] { 0, conditions + textIds.Count, 0 },
            new[] { 1, speechIds.Count, _options.LLMDim });
        return (TextHead.Forward(textHidden), SpeechHead.Forward(speechHidden));
    }

    public int[] Generate(IReadOnlyList<int> text, ChatterboxT3Condition<T> condition, Random random)
    {
        var textIds = WithBoundaries(text, _options.TextStartToken, _options.TextStopToken, _options.VocabSize, _options.MaxTextLength, "text");
        bool cfg = _options.ClassifierFreeGuidanceWeight > 0;
        int batch = cfg ? 2 : 1;
        var textTensor = IdTensor(textIds, batch);
        var initialSpeech = IdTensor(new[] { _options.SpeechStartToken }, batch);
        var context = PrepareEmbeddings(condition, textTensor, initialSpeech, cfg);

        // The released inference path appends a second BOS after prepare_input_embeds, which already includes one.
        var bos = AddPositions(SpeechEmbedding.Forward(initialSpeech), SpeechPositionEmbedding, start: 0);
        context = _engine.TensorConcatenate(new[] { context, bos }, 1);

        var cache = new LlamaCache<T>();
        var output = _transformer.Forward(context, null, null, cache, lastLogitsOnly: true).Hidden;
        var generated = new List<int> { _options.SpeechStartToken };
        var predicted = new List<int>();

        for (int step = 0; step < _options.MaxGeneratedSpeechTokens; step++)
        {
            var last = _engine.TensorSlice(output, new[] { 0, output.Shape[1] - 1, 0 },
                new[] { batch, 1, _options.LLMDim });
            var logits = SpeechHead.Forward(last);
            int next = Sample(logits, generated, cfg, random);
            predicted.Add(next);
            generated.Add(next);
            if (next == _options.SpeechStopToken)
                break;

            var nextIds = IdTensor(new[] { next }, batch);
            var nextEmbedding = AddPositions(SpeechEmbedding.Forward(nextIds), SpeechPositionEmbedding, step + 1);
            output = _transformer.Forward(nextEmbedding, null, null, cache, lastLogitsOnly: true).Hidden;
        }

        return predicted.Where(token => token >= 0 && token < _options.CodebookSize).ToArray();
    }

    public void LoadTorchWeights(Func<string, int[], double[]> read)
    {
        int h = _options.LLMDim;
        TextEmbedding.LoadTable(read("text_emb.weight", new[] { _options.VocabSize, h }));
        SpeechEmbedding.LoadTable(read("speech_emb.weight", new[] { _options.SpeechVocabularySize, h }));
        TextPositionEmbedding.LoadTable(read("text_pos_emb.emb.weight", new[] { _options.MaxTextLength + 2, h }));
        SpeechPositionEmbedding.LoadTable(read("speech_pos_emb.emb.weight", new[] { _options.MaxSpeechTokens + 4, h }));
        TorchParameters.Linear(_engine, SpeakerProjection, _options.T3SpeakerEmbeddingDim, h,
            read("cond_enc.spkr_enc.weight", new[] { h, _options.T3SpeakerEmbeddingDim }),
            read("cond_enc.spkr_enc.bias", new[] { h }));
        TorchParameters.Linear(EmotionProjection, 1, h, read("cond_enc.emotion_adv_fc.weight", new[] { h, 1 }));
        _perceiver.LoadTorchWeights(read, "cond_enc.perceiver.");
        TorchParameters.Linear(TextHead, h, _options.VocabSize, read("text_head.weight", new[] { _options.VocabSize, h }));
        TorchParameters.Linear(SpeechHead, h, _options.SpeechVocabularySize,
            read("speech_head.weight", new[] { _options.SpeechVocabularySize, h }));
        _transformer.LoadTorchWeights(read, "tfmr.");
    }

    private int ConditionLength => 1 + _options.PerceiverQueries + 1;

    internal Tensor<T> PrepareEmbeddings(
        ChatterboxT3Condition<T> condition,
        Tensor<T> textIds,
        Tensor<T> speechIds,
        bool classifierFreeGuidance)
    {
        int batch = textIds.Shape[0];
        var speaker = SpeakerProjection.Forward(_engine.Reshape(condition.SpeakerEmbedding,
            new[] { 1, _options.T3SpeakerEmbeddingDim }));
        speaker = _engine.Reshape(speaker, new[] { 1, 1, _options.LLMDim });

        var promptIds = IdTensor(condition.PromptSpeechTokens, 1);
        var prompt = AddPositions(SpeechEmbedding.Forward(promptIds), SpeechPositionEmbedding, 0);
        prompt = _engine.Reshape(_perceiver.Forward(
            _engine.Reshape(prompt, new[] { prompt.Shape[1], _options.LLMDim })),
            new[] { 1, _options.PerceiverQueries, _options.LLMDim });

        var emotionInput = new Tensor<T>(new[] { 1, 1 });
        emotionInput[0, 0] = NumOps.FromDouble(condition.Exaggeration);
        var emotion = _engine.Reshape(EmotionProjection.Forward(emotionInput), new[] { 1, 1, _options.LLMDim });
        var conditions = _engine.TensorConcatenate(new[] { speaker, prompt, emotion }, 1);
        if (batch > 1) conditions = _engine.TensorTile(conditions, new[] { batch, 1, 1 });

        var text = TextEmbedding.Forward(textIds);
        if (classifierFreeGuidance && batch > 1)
            for (int i = 0; i < text.Shape[1]; i++)
                for (int j = 0; j < text.Shape[2]; j++)
                    text[1, i, j] = NumOps.Zero;
        // CFG drops lexical content before adding positions, as T3.prepare_input_embeds does.
        text = AddPositions(text, TextPositionEmbedding, 0);
        var speech = AddPositions(SpeechEmbedding.Forward(speechIds), SpeechPositionEmbedding, 0);
        return _engine.TensorConcatenate(new[] { conditions, text, speech }, 1);
    }

    private Tensor<T> AddPositions(Tensor<T> embeddings, TiedEmbeddingLayer<T> table, int start)
    {
        int batch = embeddings.Shape[0], length = embeddings.Shape[1];
        var positions = new Tensor<T>(new[] { length });
        for (int i = 0; i < length; i++) positions[i] = NumOps.FromDouble(start + i);
        var positionEmbeddings = _engine.Reshape(table.Forward(positions), new[] { 1, length, _options.LLMDim });
        if (batch > 1) positionEmbeddings = _engine.TensorTile(positionEmbeddings, new[] { batch, 1, 1 });
        return _engine.TensorAdd(embeddings, positionEmbeddings);
    }

    private Tensor<T> NextTokenCrossEntropy(Tensor<T> logits, IReadOnlyList<int> labels)
    {
        int batch = logits.Shape[0], sourceLength = logits.Shape[1], vocabulary = logits.Shape[2];
        int length = sourceLength - 1;
        if (length < 1 || labels.Count != sourceLength) throw new ArgumentException("Next-token training needs an aligned sequence with start and stop tokens.", nameof(labels));
        var picks = new int[batch * length];
        int n = 0;
        for (int b = 0; b < batch; b++)
            for (int i = 0; i < length; i++)
                picks[n++] = (b * sourceLength + i) * vocabulary + labels[i + 1];

        var logProbabilities = _engine.Reshape(_engine.TensorLogSoftmax(logits, axis: 2),
            new[] { batch * sourceLength * vocabulary, 1 });
        var index = new Tensor<int>(new[] { picks.Length }, new Vector<int>(picks));
        var selected = _engine.TensorIndexSelect(logProbabilities, index, 0);
        return _engine.TensorMultiplyScalar(_engine.ReduceSum(selected, null), NumOps.FromDouble(-1.0 / picks.Length));
    }

    private int Sample(Tensor<T> logits, IReadOnlyList<int> history, bool cfg, Random random)
    {
        int vocabulary = logits.Shape[2];
        var values = new double[vocabulary];
        for (int token = 0; token < vocabulary; token++)
        {
            double conditional = NumOps.ToDouble(logits[0, 0, token]);
            values[token] = cfg
                ? conditional + _options.ClassifierFreeGuidanceWeight
                    * (conditional - NumOps.ToDouble(logits[1, 0, token]))
                : conditional;
        }

        if (_options.RepetitionPenalty <= 0 || _options.Temperature <= 0)
            throw new InvalidOperationException("RepetitionPenalty and Temperature must be positive.");
        foreach (int token in history.Distinct())
            if (token >= 0 && token < vocabulary)
                values[token] = values[token] < 0
                    ? values[token] * _options.RepetitionPenalty
                    : values[token] / _options.RepetitionPenalty;
        if (_options.Temperature != 1.0)
            for (int i = 0; i < values.Length; i++) values[i] /= _options.Temperature;

        double maximum = values.Max();
        var candidates = Enumerable.Range(0, vocabulary)
            .Select(token => (Token: token, Weight: Math.Exp(values[token] - maximum)))
            .Where(item => _options.MinP <= 0 || item.Weight >= _options.MinP)
            .OrderByDescending(item => item.Weight)
            .ToList();
        if (candidates.Count == 0)
            return Array.IndexOf(values, maximum);

        HuggingFaceNucleusSampling.Retain(candidates, _options.GenerationTopP);

        double sum = candidates.Sum(item => item.Weight);
        double draw = random.NextDouble() * sum;
        foreach (var item in candidates)
        {
            draw -= item.Weight;
            if (draw <= 0) return item.Token;
        }
        return candidates[candidates.Count - 1].Token;
    }

    private static int[] WithBoundaries(
        IReadOnlyList<int> ids,
        int start,
        int stop,
        int vocabulary,
        int maximum,
        string name)
    {
        if (ids is null) throw new ArgumentNullException(name);
        if (ids.Count > maximum)
            throw new ArgumentException($"{name} has {ids.Count} tokens; the release supports at most {maximum}.", name);
        if (ids.Any(id => id < 0 || id >= vocabulary))
            throw new ArgumentOutOfRangeException(name, $"{name} ids lie in [0, {vocabulary}).");
        var result = new int[ids.Count + 2];
        result[0] = start;
        for (int i = 0; i < ids.Count; i++) result[i + 1] = ids[i];
        result[result.Length - 1] = stop;
        return result;
    }

    private Tensor<T> IdTensor(IReadOnlyList<int> ids, int batch)
    {
        var tensor = new Tensor<T>(new[] { batch, ids.Count });
        for (int b = 0; b < batch; b++)
            for (int i = 0; i < ids.Count; i++)
                tensor[b, i] = NumOps.FromDouble(ids[i]);
        return tensor;
    }

    private static TLayer Own<TLayer>(List<LayerBase<T>> layers, TLayer layer) where TLayer : LayerBase<T>
    {
        layers.Add(layer);
        return layer;
    }

    private static DenseLayer<T> Dense(int outputs) =>
        new(outputs, new IdentityActivation<T>() as IActivationFunction<T>);
}
/// <summary>The shared pre-norm attention block used for both prompt cross-attention and query self-attention.</summary>
internal sealed class ChatterboxPerceiver<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly int _dim;
    private readonly int _heads;
    private readonly DropoutLayer<T> _dropout;

    public ChatterboxPerceiver(IEngine engine, List<LayerBase<T>> layers, ChatterboxOptions options, Random random)
    {
        _engine = engine;
        _dim = options.LLMDim;
        _heads = options.PerceiverHeads;
        if (_dim % _heads != 0) throw new ArgumentException("Perceiver heads must divide LLMDim.");
        Query = Own(layers, new TiedEmbeddingLayer<T>(options.PerceiverQueries, _dim));
        Norm = Own(layers, new LayerNormalizationLayer<T>(_dim));
        ToQuery = Own(layers, Dense(_dim));
        ToKey = Own(layers, Dense(_dim));
        ToValue = Own(layers, Dense(_dim));
        Output = Own(layers, Dense(_dim));
        _dropout = Own(layers, new DropoutLayer<T>(options.PerceiverDropout));

        double bound = Math.Sqrt(3.0) * Math.Sqrt(2.0 / (2.0 * options.PerceiverQueries));
        Query.Reinitialize(() => (2.0 * random.NextDouble() - 1.0) * bound);
    }

    public TiedEmbeddingLayer<T> Query { get; }
    public LayerNormalizationLayer<T> Norm { get; }
    public DenseLayer<T> ToQuery { get; }
    public DenseLayer<T> ToKey { get; }
    public DenseLayer<T> ToValue { get; }
    public DenseLayer<T> Output { get; }

    public Tensor<T> Forward(Tensor<T> prompt)
    {
        int queries = Query.VocabularySize;
        var ids = new Tensor<T>(new[] { queries });
        for (int i = 0; i < queries; i++) ids[i] = NumOps.FromDouble(i);
        var q = Query.Forward(ids);
        q = Attention(q, prompt);
        return Attention(q, q);
    }

    public void LoadTorchWeights(Func<string, int[], double[]> read, string prefix)
    {
        Query.LoadTable(read(prefix + "pre_attention_query", new[] { 1, Query.VocabularySize, _dim }));
        TorchParameters.LayerNorm(_engine, Norm, read(prefix + "attn.norm.weight", new[] { _dim }),
            read(prefix + "attn.norm.bias", new[] { _dim }));
        Load(ToQuery, "to_q");
        Load(ToKey, "to_k");
        Load(ToValue, "to_v");
        Load(Output, "proj_out");

        void Load(DenseLayer<T> layer, string name) => TorchParameters.Linear(_engine, layer, _dim, _dim,
            read(prefix + $"attn.{name}.weight", new[] { _dim, _dim }),
            read(prefix + $"attn.{name}.bias", new[] { _dim }));
    }

    private Tensor<T> Attention(Tensor<T> query, Tensor<T> memory)
    {
        int queryLength = query.Shape[0], memoryLength = memory.Shape[0], head = _dim / _heads;
        var q = Split(ToQuery.Forward(Norm.Forward(query)), queryLength);
        var normalizedMemory = Norm.Forward(memory);
        var k = Split(ToKey.Forward(normalizedMemory), memoryLength);
        var v = Split(ToValue.Forward(normalizedMemory), memoryLength);
        var scores = _engine.TensorMultiplyScalar(
            _engine.BatchMatMul(q, _engine.TensorPermute(k, new[] { 0, 2, 1 })),
            NumOps.FromDouble(1.0 / Math.Sqrt(head)));
        var weights = _dropout.Forward(_engine.Softmax(scores, axis: 2));
        var attended = _engine.BatchMatMul(weights, v);
        var rows = _engine.Reshape(_engine.TensorPermute(attended, new[] { 1, 0, 2 }),
            new[] { queryLength, _dim });
        return _engine.TensorAdd(query, Output.Forward(rows));

        Tensor<T> Split(Tensor<T> tensor, int length) =>
            _engine.TensorPermute(_engine.Reshape(tensor, new[] { length, _heads, head }), new[] { 1, 0, 2 });
    }

    private static TLayer Own<TLayer>(List<LayerBase<T>> layers, TLayer layer) where TLayer : LayerBase<T>
    {
        layers.Add(layer);
        return layer;
    }

    private static DenseLayer<T> Dense(int outputs) =>
        new(outputs, new IdentityActivation<T>() as IActivationFunction<T>);
}
