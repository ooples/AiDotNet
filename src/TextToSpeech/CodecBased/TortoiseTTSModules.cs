using AiDotNet.ActivationFunctions;
using AiDotNet.Audio;
using AiDotNet.Audio.Codecs;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.Vocoders;
using AiDotNet.TextToSpeech.VoiceCloning;
using AiDotNet.TextToSpeech.Latest;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.TextToSpeech.CodecBased;

internal static class TortoiseOps<T>
{
    internal static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    internal static TLayer Own<TLayer>(List<LayerBase<T>> owner, TLayer layer) where TLayer : LayerBase<T>
    {
        owner.Add(layer);
        return layer;
    }

    internal static Tensor<T> Ids(IReadOnlyList<int> ids)
    {
        var result = new Tensor<T>(new[] { ids.Count });
        for (int i = 0; i < ids.Count; i++) result[i] = NumOps.FromDouble(ids[i]);
        return result;
    }

    internal static int[] IdsOf(Tensor<T> ids)
    {
        var result = new int[ids.Length];
        for (int i = 0; i < result.Length; i++) result[i] = (int)Math.Round(NumOps.ToDouble(ids[i]));
        return result;
    }

    internal static Tensor<T> CausalMask(int length)
    {
        var mask = new Tensor<T>(new[] { length, length });
        for (int row = 0; row < length; row++)
            for (int column = row + 1; column < length; column++)
                mask[row, column] = NumOps.FromDouble(-1e9);
        return mask;
    }

    internal static Tensor<T> MeanRows(IEngine engine, Tensor<T> rows)
        => engine.ReduceMean(rows, new[] { 0 }, keepDims: true);

    internal static Tensor<T> NormalizeRows(IEngine engine, Tensor<T> rows)
    {
        var square = engine.TensorMultiply(rows, rows);
        var norm = engine.TensorPow(engine.TensorAddScalar(engine.ReduceSum(square, new[] { 1 }, true),
            NumOps.FromDouble(1e-12)), NumOps.FromDouble(0.5));
        return engine.TensorDivide(rows, engine.TensorBroadcastTo(norm, rows._shape));
    }

    internal static Tensor<T> CrossEntropy(IEngine engine, Tensor<T> logits, IReadOnlyList<int> labels)
    {
        int rows = logits.Shape[0], classes = logits.Shape[1], count = Math.Min(rows, labels.Count);
        if (count == 0) return new Tensor<T>(new[] { 1 });
        var log = engine.Reshape(engine.TensorLogSoftmax(logits, 1), new[] { rows * classes, 1 });
        var indices = new Tensor<int>(new[] { count });
        for (int i = 0; i < count; i++)
            indices[i] = i * classes + MathPolyfill.Clamp(labels[i], 0, classes - 1);
        return engine.TensorNegate(engine.ReduceMean(engine.TensorIndexSelect(log, indices, 0), new[] { 0, 1 }, false));
    }

    internal static Tensor<T> Mse(IEngine engine, Tensor<T> prediction, Tensor<T> target)
    {
        var delta = engine.TensorSubtract(prediction, target);
        return engine.ReduceMean(engine.TensorMultiply(delta, delta), Enumerable.Range(0, delta.Rank).ToArray(), false);
    }

    internal static Tensor<T> ResizeRows(IEngine engine, Tensor<T> rows, int length)
    {
        if (rows.Shape[0] == length) return rows;
        var index = new Tensor<int>(new[] { length });
        for (int i = 0; i < length; i++)
            index[i] = Math.Min(rows.Shape[0] - 1, (int)((long)i * rows.Shape[0] / Math.Max(1, length)));
        return engine.TensorIndexSelect(rows, index, 0);
    }

    internal static Tensor<T> Dropout(IEngine engine, Tensor<T> x, double probability, bool training, Random random)
        => training && probability > 0 ? T5Seq2Seq<T>.Dropout(engine, x, probability, random) : x;
}

/// <summary>GPT-2 block: pre-LayerNorm, biased multi-head attention, gelu_new MLP and residual dropout.</summary>
internal sealed class TortoiseTransformerLayer<T>
{
    private readonly IEngine _engine;
    private readonly int _dim;
    private readonly int _heads;
    private readonly double _dropout;

    internal TortoiseTransformerLayer(IEngine engine, List<LayerBase<T>> owner, int dim, int heads, int feedForward, double dropout)
    {
        _engine = engine;
        _dim = dim;
        _heads = heads;
        _dropout = dropout;
        var identity = new IdentityActivation<T>() as IActivationFunction<T>;
        Norm1 = TortoiseOps<T>.Own(owner, new LayerNormalizationLayer<T>(dim));
        Qkv = TortoiseOps<T>.Own(owner, new DenseLayer<T>(3 * dim, identity));
        AttentionOut = TortoiseOps<T>.Own(owner, new DenseLayer<T>(dim, identity));
        Norm2 = TortoiseOps<T>.Own(owner, new LayerNormalizationLayer<T>(dim));
        FeedForwardIn = TortoiseOps<T>.Own(owner, new DenseLayer<T>(feedForward, identity));
        FeedForwardOut = TortoiseOps<T>.Own(owner, new DenseLayer<T>(dim, identity));
    }

    internal LayerNormalizationLayer<T> Norm1 { get; }
    internal DenseLayer<T> Qkv { get; }
    internal DenseLayer<T> AttentionOut { get; }
    internal LayerNormalizationLayer<T> Norm2 { get; }
    internal DenseLayer<T> FeedForwardIn { get; }
    internal DenseLayer<T> FeedForwardOut { get; }

    internal void LoadTorchWeights(Func<string, int[], double[]> read, string prefix, int feedForward)
    {
        IndexTtsOps.LoadNorm(_engine, read, Norm1, prefix + "ln_1", _dim);
        IndexTtsOps.LoadNorm(_engine, read, Norm2, prefix + "ln_2", _dim);
        void ConvLinear(DenseLayer<T> layer, string name, int input, int output)
        {
            var values = read(prefix + name + ".weight", new[] { input, output });
            var transposed = new double[values.Length];
            for (int i = 0; i < input; i++) for (int o = 0; o < output; o++) transposed[o * input + i] = values[i * output + o];
            TorchParameters.Linear(_engine, layer, input, output, transposed, read(prefix + name + ".bias", new[] { output }));
        }
        ConvLinear(Qkv, "attn.c_attn", _dim, 3 * _dim);
        ConvLinear(AttentionOut, "attn.c_proj", _dim, _dim);
        ConvLinear(FeedForwardIn, "mlp.c_fc", _dim, feedForward);
        ConvLinear(FeedForwardOut, "mlp.c_proj", feedForward, _dim);
    }

    internal Tensor<T> Forward(Tensor<T> x, Tensor<T>? mask, bool training, Random random)
    {
        var attention = Attention(Norm1.Forward(x), mask, training, random);
        x = _engine.TensorAdd(x, TortoiseOps<T>.Dropout(_engine, AttentionOut.Forward(attention), _dropout, training, random));
        var hidden = _engine.GELU(FeedForwardIn.Forward(Norm2.Forward(x)));
        return _engine.TensorAdd(x, TortoiseOps<T>.Dropout(_engine, FeedForwardOut.Forward(hidden), _dropout, training, random));
    }

    private Tensor<T> Attention(Tensor<T> x, Tensor<T>? mask, bool training, Random random)
    {
        int length = x.Shape[0], headDim = _dim / _heads;
        var projected = Qkv.Forward(x);
        Tensor<T> Heads(int part) => _engine.TensorPermute(_engine.Reshape(
            _engine.TensorSlice(projected, new[] { 0, part * _dim }, new[] { length, _dim }),
            new[] { length, _heads, headDim }), new[] { 1, 0, 2 });
        var q = _engine.TensorMultiplyScalar(Heads(0), TortoiseOps<T>.NumOps.FromDouble(1.0 / Math.Sqrt(headDim)));
        var scores = _engine.BatchMatMul(q, _engine.TensorPermute(Heads(1), new[] { 0, 2, 1 }));
        if (mask is not null)
            scores = _engine.TensorAdd(scores, _engine.TensorTile(_engine.Reshape(mask,
                new[] { 1, length, length }), new[] { _heads, 1, 1 }));
        var weights = TortoiseOps<T>.Dropout(_engine, _engine.Softmax(scores, 2), _dropout, training, random);
        var context = _engine.BatchMatMul(weights, Heads(2));
        return _engine.Reshape(_engine.TensorPermute(context, new[] { 1, 0, 2 }), new[] { length, _dim });
    }
}

internal sealed class TortoiseTransformer<T>
{
    private readonly IEngine _engine;
    private readonly List<TortoiseTransformerLayer<T>> _layers = new();

    internal TortoiseTransformer(IEngine engine, List<LayerBase<T>> owner, int dim, int heads, int depth,
        int feedForward, double dropout)
    {
        _engine = engine;
        for (int i = 0; i < depth; i++)
            _layers.Add(new TortoiseTransformerLayer<T>(engine, owner, dim, heads, feedForward, dropout));
        FinalNorm = TortoiseOps<T>.Own(owner, new LayerNormalizationLayer<T>(dim));
    }

    internal LayerNormalizationLayer<T> FinalNorm { get; }
    internal IReadOnlyList<TortoiseTransformerLayer<T>> Blocks => _layers;

    internal Tensor<T> Forward(Tensor<T> x, Tensor<T>? mask, bool training, Random random)
    {
        foreach (var layer in _layers) x = layer.Forward(x, mask, training, random);
        return FinalNorm.Forward(x);
    }
}

/// <summary>The paper's four-times-compressed, single-codebook VQ-VAE tokenizer.</summary>
internal sealed class TortoiseSpeechTokenizer<T>
{
    private readonly IEngine _engine;
    private readonly XttsDiscreteVae<T> _codec;

    internal TortoiseSpeechTokenizer(IEngine engine, List<LayerBase<T>> owner, TortoiseTTSOptions options)
    {
        _engine = engine;
        _codec = new XttsDiscreteVae<T>(engine, owner, new XTTSv2Options
        {
            MelChannels = options.AutoregressiveMelChannels, CodebookSize = options.CodebookSize,
            DvaeHiddenDim = options.TokenizerLatentDim, DvaeCodebookDim = options.TokenizerCodeDim,
            DvaeResidualBlocks = options.TokenizerResidualBlocks
        });
    }

    internal ResidualVectorQuantizerLayer<T> Quantizer => _codec.Quantizer;
    internal void LoadTorchWeights(Func<string, int[], double[]> read) => _codec.LoadTorchWeights(read);
    internal Tensor<T> EncodeLatent(Tensor<T> melRows) => _codec.EncodeFeatures(_engine.TensorTranspose(melRows));
    internal Tensor<T> QuantizedLatent(Tensor<T> melRows) => Quantizer.Forward(EncodeLatent(melRows));
    internal int[] Encode(Tensor<T> melRows) => XttsOps<T>.IdsOf(_codec.Encode(_engine.TensorTranspose(melRows)));
    internal Tensor<T> Decode(IReadOnlyList<int> codes) => _codec.Decode(XttsOps<T>.Ids(codes));
    internal Tensor<T> Reconstruct(Tensor<T> melRows) => _codec.Reconstruct(_engine.TensorTranspose(melRows));
}



internal sealed record TortoiseAutoregressiveResult<T>(Tensor<T> TextLogits, Tensor<T> SpeechLogits, Tensor<T> SpeechLatents);

/// <summary>The released UnifiedVoice GPT-2 prior with separate text/speech embeddings and learned positions.</summary>
internal sealed class TortoiseAutoregressive<T>
{
    private readonly IEngine _engine;
    private readonly TortoiseTTSOptions _options;
    private readonly Conv1DLayer<T> _conditioningInput;
    private readonly XttsConditioningAttention<T>[] _conditioningEncoder;
    private readonly EmbeddingLayer<T> _textEmbedding, _speechEmbedding;
    private readonly LearnedPositionalEmbeddingLayer<T> _textPosition, _speechPosition;
    private readonly TortoiseTransformer<T> _gpt;
    private readonly LayerNormalizationLayer<T> _finalNorm;
    private readonly DenseLayer<T> _textHead, _speechHead;

    internal void LoadTorchWeights(Func<string, int[], double[]> read)
    {
        int d = _options.LLMDim;
        XttsOps<T>.LoadConv(read, _conditioningInput, "conditioning_encoder.init", _options.AutoregressiveMelChannels, d, 1);
        for (int i = 0; i < _conditioningEncoder.Length; i++)
            _conditioningEncoder[i].LoadTorchWeights(read, $"conditioning_encoder.attn.{i}.");
        void Table(LayerBase<T> layer, string name, int rows) => layer.SetParameters(new Vector<T>(read(name,
            new[] { rows, d }).Select(TortoiseOps<T>.NumOps.FromDouble)));
        Table(_textEmbedding, "text_embedding.weight", _options.VocabSize);
        Table(_speechEmbedding, "mel_embedding.weight", _options.CodebookSize + 2);
        Table(_textPosition, "text_pos_embedding.emb.weight", _options.MaxTextLength + 2);
        Table(_speechPosition, "mel_pos_embedding.emb.weight", _options.MaxCodecFrames + 2 + _options.MaxConditioningInputs);
        for (int i = 0; i < _gpt.Blocks.Count; i++) _gpt.Blocks[i].LoadTorchWeights(read, $"gpt.h.{i}.", _options.AutoregressiveFeedForwardDim);
        IndexTtsOps.LoadNorm(_engine, read, _gpt.FinalNorm, "gpt.ln_f", d);
        IndexTtsOps.LoadNorm(_engine, read, _finalNorm, "final_norm", d);
        IndexTtsOps.LoadLinear(_engine, read, _textHead, "text_head", d, _options.VocabSize);
        IndexTtsOps.LoadLinear(_engine, read, _speechHead, "mel_head", d, _options.CodebookSize + 2);
    }

    internal TortoiseAutoregressive(IEngine engine, List<LayerBase<T>> owner, TortoiseTTSOptions options)
    {
        _engine = engine;
        _options = options;
        var identity = new IdentityActivation<T>() as IActivationFunction<T>;
        _conditioningInput = TortoiseOps<T>.Own(owner, new Conv1DLayer<T>(options.AutoregressiveMelChannels, options.LLMDim, 1, 1, 1, 0));
        _conditioningEncoder = Enumerable.Range(0, options.NumEncoderLayers).Select(_ =>
            new XttsConditioningAttention<T>(engine, owner, options.LLMDim, options.NumHeads, normalizeResidual: false)).ToArray();
        _textEmbedding = TortoiseOps<T>.Own(owner, new EmbeddingLayer<T>(options.VocabSize, options.LLMDim));
        _speechEmbedding = TortoiseOps<T>.Own(owner, new EmbeddingLayer<T>(options.CodebookSize + 2, options.LLMDim));
        _textPosition = TortoiseOps<T>.Own(owner, new LearnedPositionalEmbeddingLayer<T>(options.MaxTextLength + 2, options.LLMDim));
        _speechPosition = TortoiseOps<T>.Own(owner, new LearnedPositionalEmbeddingLayer<T>(options.MaxCodecFrames + 2 + options.MaxConditioningInputs, options.LLMDim));
        _gpt = new TortoiseTransformer<T>(engine, owner, options.LLMDim, options.NumHeads,
            options.NumLLMLayers, options.AutoregressiveFeedForwardDim, options.DropoutRate);
        _finalNorm = TortoiseOps<T>.Own(owner, new LayerNormalizationLayer<T>(options.LLMDim, 1e-5));
        _textHead = TortoiseOps<T>.Own(owner, new DenseLayer<T>(options.VocabSize, identity));
        _speechHead = TortoiseOps<T>.Own(owner, new DenseLayer<T>(options.CodebookSize + 2, identity));
    }

    internal Tensor<T> Conditioning(Tensor<T> melRows, bool training, Random random)
    {
        int frames = melRows.Shape[0];
        var encoded = _conditioningInput.Forward(_engine.Reshape(_engine.TensorTranspose(melRows),
            new[] { 1, _options.AutoregressiveMelChannels, frames }));
        foreach (var block in _conditioningEncoder) encoded = block.Forward(encoded);
        return _engine.Reshape(_engine.TensorNarrow(encoded, 2, 0, 1), new[] { 1, _options.LLMDim });
    }

    internal TortoiseAutoregressiveResult<T> TeacherForced(IReadOnlyList<int> text, IReadOnlyList<int> speech,
        Tensor<T> condition, bool training, Random random, bool terminalSpeech = false)
    {
        var textInput = new[] { _options.TextStartToken }.Concat(text.Take(_options.MaxTextLength))
            .Concat(new[] { _options.TextStopToken }).ToArray();
        var speechInput = new[] { _options.SpeechStartToken }.Concat(speech.Take(_options.MaxCodecFrames))
            .Concat(terminalSpeech ? new[] { _options.SpeechStopToken } : Array.Empty<int>()).ToArray();
        var textHidden = _textPosition.Forward(_textEmbedding.Forward(TortoiseOps<T>.Ids(textInput)));
        var speechHidden = _speechPosition.Forward(_speechEmbedding.Forward(TortoiseOps<T>.Ids(speechInput)));
        var sequence = _engine.TensorConcatenate(new[] { condition, textHidden, speechHidden }, 0);
        var hidden = _finalNorm.Forward(_gpt.Forward(sequence, TortoiseOps<T>.CausalMask(sequence.Shape[0]), training, random));
        int textStart = 1, speechStart = 1 + textInput.Length;
        var textStates = _engine.TensorSlice(hidden, new[] { textStart, 0 }, new[] { textInput.Length, _options.LLMDim });
        var speechStates = _engine.TensorSlice(hidden, new[] { speechStart, 0 }, new[] { speechInput.Length, _options.LLMDim });
        return new TortoiseAutoregressiveResult<T>(_textHead.Forward(textStates), _speechHead.Forward(speechStates), speechStates);
    }

    internal Tensor<T> Loss(IReadOnlyList<int> text, IReadOnlyList<int> speech, Tensor<T> condition, bool training, Random random)
    {
        var output = TeacherForced(text, speech, condition, training, random, terminalSpeech: true);
        var textLabels = text.Take(_options.MaxTextLength).Concat(new[] { _options.TextStopToken, _options.TextStopToken }).ToArray();
        var speechLabels = speech.Take(_options.MaxCodecFrames).Concat(new[] { _options.SpeechStopToken, _options.SpeechStopToken }).ToArray();
        var textLoss = TortoiseOps<T>.CrossEntropy(_engine, output.TextLogits, textLabels);
        var speechLoss = TortoiseOps<T>.CrossEntropy(_engine, output.SpeechLogits, speechLabels);
        return _engine.TensorAdd(speechLoss, _engine.TensorMultiplyScalar(textLoss,
            TortoiseOps<T>.NumOps.FromDouble(_options.TextLossWeight)));
    }

    internal int[] Generate(IReadOnlyList<int> text, Tensor<T> condition, Random random)
    {
        var generated = new List<int>();
        int maximum = Math.Min(_options.MaxCodecFrames, _options.MaximumAutoregressiveTokens);
        for (int step = 0; step < maximum; step++)
        {
            var result = TeacherForced(text, generated, condition, false, random);
            var row = _engine.TensorSlice(result.SpeechLogits,
                new[] { result.SpeechLogits.Shape[0] - 1, 0 }, new[] { 1, result.SpeechLogits.Shape[1] });
            int next = Sample(row, generated, random);
            generated.Add(next);
            if (next == _options.SpeechStopToken) break;
        }
        if (generated.Count == 0) generated.Add(83 % _options.CodebookSize);
        return generated.ToArray();
    }

    internal Tensor<T> Latents(IReadOnlyList<int> text, IReadOnlyList<int> speech, Tensor<T> condition)
    {
        var result = TeacherForced(text, speech, condition, false, new Random(_options.SamplingSeed));
        int keep = Math.Min(speech.Count, result.SpeechLatents.Shape[0]);
        return _engine.TensorSlice(result.SpeechLatents, new[] { 0, 0 }, new[] { Math.Max(1, keep), _options.LLMDim });
    }

    private int Sample(Tensor<T> logits, IReadOnlyCollection<int> previous, Random random)
    {
        int classes = logits.Shape[1];
        var candidates = new List<(int Id, double Weight)>(classes);
        double maximum = double.NegativeInfinity;
        for (int i = 0; i < classes; i++)
        {
            if (i > _options.CodebookSize + 1) continue;
            if (i == _options.SpeechStopToken && previous.Count < _options.MinimumAutoregressiveTokens) continue;
            double value = TortoiseOps<T>.NumOps.ToDouble(logits[0, i]) / _options.Temperature;
            // The original HF generation prefix contains placeholder ID 1 and
            // the speech BOS. Both participate in repetition-penalty processing.
            if (previous.Contains(i) || i == 1 || i == _options.SpeechStartToken)
                value = value < 0 ? value * _options.RepetitionPenalty : value / _options.RepetitionPenalty;
            maximum = Math.Max(maximum, value);
            candidates.Add((i, value));
        }
        var probabilities = candidates.Select(c => (c.Id, Weight: Math.Exp(c.Weight - maximum)))
            .OrderByDescending(c => c.Weight).ToList();
        double total = probabilities.Sum(c => c.Weight), cumulative = 0;
        var nucleus = new List<(int Id, double Weight)>();
        foreach (var candidate in probabilities)
        {
            nucleus.Add(candidate);
            cumulative += candidate.Weight / total;
            if (cumulative >= _options.TopP) break;
        }
        double draw = random.NextDouble() * nucleus.Sum(c => c.Weight);
        foreach (var candidate in nucleus)
        {
            draw -= candidate.Weight;
            if (draw <= 0) return candidate.Id;
        }
        return nucleus[^1].Id;
    }
}

/// <summary>The two-tower contrastive language/voice model used to rank TorToiSe candidates.</summary>
internal sealed class TortoiseClvp<T>
{
    private readonly IEngine _engine;
    private readonly TortoiseTTSOptions _options;
    private readonly TortoiseClvpTower<T> _text, _speech;
    private readonly TiedEmbeddingLayer<T> _temperature;

    internal TortoiseClvp(IEngine engine, List<LayerBase<T>> owner, TortoiseTTSOptions options)
    {
        _engine = engine;
        _options = options;
        _text = new TortoiseClvpTower<T>(engine, owner, options.VocabSize, options.ClvpDim, options.ClvpHeads, options.ClvpLayers);
        _speech = new TortoiseClvpTower<T>(engine, owner, options.CodebookSize, options.ClvpDim, options.ClvpHeads, options.ClvpLayers);
        _temperature = TortoiseOps<T>.Own(owner, new TiedEmbeddingLayer<T>(1, 1));
        _temperature.Reinitialize(() => 1.0);
    }

    internal Tensor<T> ScoreTensor(IReadOnlyList<int> text, IReadOnlyList<int> speech, bool training, Random random)
    {
        int[] textIds = text.Take(_options.ClvpTextLength).ToArray();
        int[] speechIds = speech.Take(_options.ClvpSpeechLength).Select(i => MathPolyfill.Clamp(i, 0, _options.CodebookSize - 1)).ToArray();
        var a = _text.Forward(textIds, training, _options.ClvpTokenDropout, random);
        var b = _speech.Forward(speechIds, training, _options.ClvpTokenDropout, random);
        var similarity = _engine.ReduceSum(_engine.TensorMultiply(a, b), new[] { 1 }, false);
        var zero = new Tensor<T>(new[] { 1 });
        var logTemperature = _temperature.Forward(zero);
        return _engine.TensorMultiply(similarity, _engine.TensorExp(logTemperature));
    }

    internal double Score(IReadOnlyList<int> text, IReadOnlyList<int> speech)
        => TortoiseOps<T>.NumOps.ToDouble(ScoreTensor(text, speech, false, new Random(_options.SamplingSeed))[0]);

    internal void LoadTorchWeights(Func<string, int[], double[]> read)
    {
        _text.LoadTorchWeights(read, "text_emb", "text_transformer.transformer", "to_text_latent", _options.VocabSize);
        _speech.LoadTorchWeights(read, "speech_emb", "speech_transformer.transformer", "to_speech_latent", _options.CodebookSize);
        _temperature.SetParameters(new Vector<T>(read("temperature", Array.Empty<int>()).Select(TortoiseOps<T>.NumOps.FromDouble)));
    }

    internal Tensor<T> ContrastiveLoss(IReadOnlyList<int> text, IReadOnlyList<int> speech, Random random)
        => ContrastiveBatchLoss(new[] { text }, new[] { speech }, random);

    internal Tensor<T> ContrastiveBatchLoss(IReadOnlyList<IReadOnlyList<int>> text, IReadOnlyList<IReadOnlyList<int>> speech, Random random, bool training = true)
    {
        if (text.Count == 0 || text.Count != speech.Count) throw new ArgumentException("CLVP requires equally sized nonempty text and speech batches.");
        var a = _engine.TensorConcatenate(text.Select(ids => _text.Forward(ids.Take(_options.ClvpTextLength).ToArray(),
            training, _options.ClvpTokenDropout, random)).ToArray(), 0);
        var b = _engine.TensorConcatenate(speech.Select(ids => _speech.Forward(ids.Take(_options.ClvpSpeechLength).ToArray(),
            training, _options.ClvpTokenDropout, random)).ToArray(), 0);
        var logits = _engine.TensorMultiply(_engine.TensorMatMul(a, _engine.TensorTranspose(b)),
            _engine.TensorExp(_temperature.Forward(new Tensor<T>(new[] { 1 }))));
        var labels = Enumerable.Range(0, text.Count).ToArray();
        var forward = UniAudioOps<T>.CrossEntropy(_engine, logits, labels);
        var backward = UniAudioOps<T>.CrossEntropy(_engine, _engine.TensorTranspose(logits), labels);
        return _engine.TensorMultiplyScalar(_engine.TensorAdd(forward, backward), TortoiseOps<T>.NumOps.FromDouble(0.5));
    }
}

/// <summary>The released latent-conditioned TorToiSe diffusion decoder and sampler.</summary>
internal sealed class TortoiseDiffusion<T>
{
    private const double MelMinimum = -11.512925148010254, MelMaximum = 2.3143386840820312;
    private readonly IEngine _engine;
    private readonly TortoiseTTSOptions _options;
    private readonly TortoiseDiffusionCore<T> _core;
    private readonly TortoiseDiffusionSchedule<T> _schedule;

    internal TortoiseDiffusion(IEngine engine, List<LayerBase<T>> owner, TortoiseTTSOptions options)
    {
        _engine = engine; _options = options;
        _core = new TortoiseDiffusionCore<T>(engine, owner, options);
        _schedule = new TortoiseDiffusionSchedule<T>(engine, options.DiffusionTrainingSteps, options.MelChannels);
    }

    internal void LoadTorchWeights(Func<string, int[], double[]> read) => _core.LoadTorchWeights(read);

    private Tensor<T> NormalizeMel(Tensor<T> mel) => _engine.TensorAddScalar(
        _engine.TensorMultiplyScalar(_engine.TensorAddScalar(mel, TortoiseOps<T>.NumOps.FromDouble(-MelMinimum)),
            TortoiseOps<T>.NumOps.FromDouble(2 / (MelMaximum - MelMinimum))), TortoiseOps<T>.NumOps.FromDouble(-1));

    internal Tensor<T> PredictNoise(Tensor<T> noisy, Tensor<T> timestep, Tensor<T> arLatents,
        Tensor<T>? speakerMel, bool conditioned)
    {
        var prediction = _core.Forward(noisy, TortoiseOps<T>.NumOps.ToDouble(timestep[0])
            * (_options.DiffusionTrainingSteps - 1), arLatents,
            speakerMel is null ? null : NormalizeMel(speakerMel), conditioningFree: !conditioned);
        return _engine.TensorNarrow(prediction, 1, 0, _options.MelChannels);
    }

    internal Tensor<T> TrainingLoss(Tensor<T> targetRows, Tensor<T> arLatents, Tensor<T> speakerMel, Random random)
    {
        int frames = targetRows.Shape[0], step = random.Next(_options.DiffusionTrainingSteps);
        var target = _engine.Reshape(_engine.TensorTranspose(NormalizeMel(targetRows)), new[] { 1, _options.MelChannels, frames });
        var noise = Gaussian(target.Shape.ToArray(), random);
        bool conditioned = random.NextDouble() >= _options.DiffusionConditioningDropout;
        var prediction = _core.Forward(_schedule.Noisy(target, noise, step), step, arLatents,
            NormalizeMel(speakerMel), conditioningFree: !conditioned);
        return _schedule.TrainingTerms(target, noise, prediction, step).Total;
    }

    internal Tensor<T> Sample(Tensor<T> arLatents, Tensor<T> speakerMel, Random random)
    {
        int frames = Math.Max(1, arLatents.Shape[0] * 4 * _options.SampleRate / _options.TokenizerSampleRate);
        var x = Gaussian(new[] { 1, _options.MelChannels, frames }, random);
        int steps = _options.DiffusionInferenceSteps;
        var selected = Enumerable.Range(0, steps).Select(i =>
            (int)Math.Round(i * (_options.DiffusionTrainingSteps - 1.0) / (steps - 1))).ToArray();
        var aligned = _core.Aligned(arLatents, _core.Conditioning(NormalizeMel(speakerMel)), frames);
        for (int i = steps - 1; i >= 0; i--)
        {
            int step = selected[i], previous = i == 0 ? -1 : selected[i - 1];
            double clipped = i == 0 ? _schedule.PosteriorVariance(selected[1], selected[0])
                : _schedule.PosteriorVariance(step, previous);
            var prediction = _core.Forward(x, step, arLatents, null, precomputedAligned: aligned);
            var free = _core.Forward(x, step, arLatents, null, conditioningFree: true);
            var statistics = _schedule.Statistics(x, prediction, step, previous, clipped, free,
                _options.ClassifierFreeGuidanceScale, guidanceIndex: i, guidanceSteps: steps);
            var noise = Gaussian(x.Shape.ToArray(), random);
            x = i == 0 ? statistics.Mean : _engine.TensorAdd(statistics.Mean, _engine.TensorMultiply(noise,
                _engine.TensorExp(_engine.TensorMultiplyScalar(statistics.LogVariance, TortoiseOps<T>.NumOps.FromDouble(0.5)))));
        }
        var rows = _engine.TensorTranspose(_engine.Reshape(x, new[] { _options.MelChannels, frames }));
        return _engine.TensorAddScalar(_engine.TensorMultiplyScalar(rows,
            TortoiseOps<T>.NumOps.FromDouble((MelMaximum - MelMinimum) / 2)),
            TortoiseOps<T>.NumOps.FromDouble((MelMaximum + MelMinimum) / 2));
    }


    private Tensor<T> Gaussian(int[] shape, Random random)
    {
        var tensor = new Tensor<T>(shape);
        for (int i = 0; i < tensor.Length; i += 2)
        {
            double radius = Math.Sqrt(-2 * Math.Log(Math.Max(1e-12, 1 - random.NextDouble())));
            double angle = 2 * Math.PI * random.NextDouble();
            tensor[i] = TortoiseOps<T>.NumOps.FromDouble(radius * Math.Cos(angle));
            if (i + 1 < tensor.Length) tensor[i + 1] = TortoiseOps<T>.NumOps.FromDouble(radius * Math.Sin(angle));
        }
        return tensor;
    }
}

/// <summary>The exact UnivNet-c32 generator used by the TorToiSe release.</summary>
internal sealed class TortoiseUnivNet<T>
{
    internal IReadOnlyList<LayerBase<T>> Layers => _generator.Layers;
    private readonly IEngine _engine;
    private readonly TortoiseTTSOptions _options;
    private readonly UnivNetGenerator<T> _generator;

    internal void LoadTorchWeights(Func<string, int[], double[]> read) => _generator.LoadTorchWeights(read);

    internal TortoiseUnivNet(IEngine engine, List<LayerBase<T>> owner, TortoiseTTSOptions options)
    {
        _engine = engine;
        _options = options;
        _generator = new UnivNetGenerator<T>(engine, options.MelChannels, options.UnivNetNoiseDim,
            options.UnivNetChannels, options.UnivNetUpsampleRates, options.UnivNetDilations, 0.2,
            options.UnivNetKernelPredictorHidden, options.UnivNetKernelPredictorConvSize);
        owner.AddRange(_generator.Layers);
    }

    internal Tensor<T> Generate(Tensor<T> melRows, Random random)
    {
        int frames = melRows.Shape[0], extra = _options.UnivNetPaddingFrames;
        var mel = _engine.Reshape(_engine.TensorTranspose(melRows), new[] { 1, _options.MelChannels, frames });
        if (extra > 0)
        {
            var silence = new Tensor<T>(new[] { 1, _options.MelChannels, extra });
            for (int i = 0; i < silence.Length; i++) silence[i] = TortoiseOps<T>.NumOps.FromDouble(Math.Log(1e-5));
            mel = _engine.TensorConcatenate(new[] { mel, silence }, 2);
        }
        var noise = new Tensor<T>(new[] { 1, _options.UnivNetNoiseDim, mel.Shape[2] });
        for (int i = 0; i < noise.Length; i += 2)
        {
            double radius = Math.Sqrt(-2 * Math.Log(Math.Max(1e-12, 1 - random.NextDouble())));
            double angle = 2 * Math.PI * random.NextDouble();
            noise[i] = TortoiseOps<T>.NumOps.FromDouble(radius * Math.Cos(angle));
            if (i + 1 < noise.Length) noise[i + 1] = TortoiseOps<T>.NumOps.FromDouble(radius * Math.Sin(angle));
        }
        var wave = _engine.Reshape(_generator.Forward(mel, noise), new[] { mel.Shape[2] * _options.HopSize });
        int keep = frames * _options.HopSize;
        return _engine.TensorSlice(wave, new[] { 0 }, new[] { keep });
    }
}
