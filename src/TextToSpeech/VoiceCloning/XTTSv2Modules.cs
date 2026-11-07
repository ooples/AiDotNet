using AiDotNet.ActivationFunctions;
using AiDotNet.Audio.Codecs;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.TextToSpeech.EndToEnd;
using AiDotNet.TextToSpeech.Latest;
using AiDotNet.TextToSpeech.Vocoders;

namespace AiDotNet.TextToSpeech.VoiceCloning;

internal static class XttsOps<T>
{
    internal static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    internal static TLayer Own<TLayer>(List<LayerBase<T>> owner, TLayer layer) where TLayer : LayerBase<T>
    {
        owner.Add(layer);
        return layer;
    }

    internal static Tensor<T> Ids(IEnumerable<int> values)
    {
        int[] ids = values.ToArray();
        var result = new Tensor<T>(new[] { ids.Length });
        for (int i = 0; i < ids.Length; i++) result[i] = NumOps.FromDouble(ids[i]);
        return result;
    }

    internal static int[] IdsOf(Tensor<T> values) => values.ToArray()
        .Select(value => (int)Math.Round(NumOps.ToDouble(value))).ToArray();

    internal static Tensor<T> CausalMask(int length)
    {
        var mask = new Tensor<T>(new[] { length, length });
        for (int row = 0; row < length; row++)
            for (int column = row + 1; column < length; column++)
                mask[row, column] = NumOps.FromDouble(-1e9);
        return mask;
    }

    internal static void LoadConv(Func<string, int[], double[]> read, Conv1DLayer<T> layer, string name, int input, int output, int kernel)
    {
        var values = read(name + ".weight", new[] { output, input, kernel }).Concat(read(name + ".bias", new[] { output }));
        layer.SetParameters(new Vector<T>(values.Select(NumOps.FromDouble)));
    }
}

/// <summary>A GPT-2 pre-norm block (biased projections, causal MHA, GELU MLP, residual dropout).</summary>
internal sealed class XttsGptBlock<T>
{
    private readonly IEngine _engine;
    private readonly int _dimension, _heads;
    private readonly LayerNormalizationLayer<T> _norm1, _norm2;
    private readonly DenseLayer<T> _query, _key, _value, _attentionOut, _mlpIn, _mlpOut;
    private readonly DropoutLayer<T>? _attentionDropout, _residualDropout1, _residualDropout2;

    internal XttsGptBlock(IEngine engine, List<LayerBase<T>> owner, int dimension, int heads, double dropout)
    {
        _engine = engine;
        _dimension = dimension;
        _heads = heads;
        var identity = new IdentityActivation<T>() as IActivationFunction<T>;
        _norm1 = XttsOps<T>.Own(owner, new LayerNormalizationLayer<T>(dimension, 1e-5));
        _query = XttsOps<T>.Own(owner, new DenseLayer<T>(dimension, identity, biasMode: BiasMode.Always));
        _key = XttsOps<T>.Own(owner, new DenseLayer<T>(dimension, identity, biasMode: BiasMode.Always));
        _value = XttsOps<T>.Own(owner, new DenseLayer<T>(dimension, identity, biasMode: BiasMode.Always));
        _attentionOut = XttsOps<T>.Own(owner, new DenseLayer<T>(dimension, identity, biasMode: BiasMode.Always));
        _norm2 = XttsOps<T>.Own(owner, new LayerNormalizationLayer<T>(dimension, 1e-5));
        _mlpIn = XttsOps<T>.Own(owner, new DenseLayer<T>(4 * dimension, identity, biasMode: BiasMode.Always));
        _mlpOut = XttsOps<T>.Own(owner, new DenseLayer<T>(dimension, identity, biasMode: BiasMode.Always));
        if (dropout > 0)
        {
            _attentionDropout = XttsOps<T>.Own(owner, new DropoutLayer<T>(dropout));
            _residualDropout1 = XttsOps<T>.Own(owner, new DropoutLayer<T>(dropout));
            _residualDropout2 = XttsOps<T>.Own(owner, new DropoutLayer<T>(dropout));
        }
    }

    internal Tensor<T> Forward(Tensor<T> input, Tensor<T> causalMask)
    {
        int length = input.Shape[0], headDimension = _dimension / _heads;
        Tensor<T> Heads(Tensor<T> x) => _engine.TensorPermute(
            _engine.Reshape(x, new[] { length, _heads, headDimension }), new[] { 1, 0, 2 });
        var normalized = _norm1.Forward(input);
        var query = _engine.TensorMultiplyScalar(Heads(_query.Forward(normalized)),
            XttsOps<T>.NumOps.FromDouble(1.0 / Math.Sqrt(headDimension)));
        var key = Heads(_key.Forward(normalized));
        var value = Heads(_value.Forward(normalized));
        var scores = _engine.BatchMatMul(query, _engine.TensorPermute(key, new[] { 0, 2, 1 }));
        scores = _engine.TensorAdd(scores, _engine.TensorTile(
            _engine.Reshape(causalMask, new[] { 1, length, length }), new[] { _heads, 1, 1 }));
        var weights = _engine.Softmax(scores, 2);
        if (_attentionDropout is not null) weights = _attentionDropout.Forward(weights);
        var context = _engine.Reshape(_engine.TensorPermute(_engine.BatchMatMul(weights, value),
            new[] { 1, 0, 2 }), new[] { length, _dimension });
        var attention = _attentionOut.Forward(context);
        if (_residualDropout1 is not null) attention = _residualDropout1.Forward(attention);
        var hidden = _engine.TensorAdd(input, attention);
        var feedForward = _engine.GELU(_mlpIn.Forward(_norm2.Forward(hidden)));
        feedForward = _mlpOut.Forward(feedForward);
        if (_residualDropout2 is not null) feedForward = _residualDropout2.Forward(feedForward);
        return _engine.TensorAdd(hidden, feedForward);
    }

    internal void LoadTorchWeights(Func<string, int[], double[]> read, string prefix)
    {
        int d = _dimension;
        IndexTtsOps.LoadNorm(_engine, read, _norm1, prefix + "ln_1", d);
        IndexTtsOps.LoadNorm(_engine, read, _norm2, prefix + "ln_2", d);
        var qkv = read(prefix + "attn.c_attn.weight", new[] { d, 3 * d });
        var bias = read(prefix + "attn.c_attn.bias", new[] { 3 * d });
        var projections = new[] { _query, _key, _value };
        for (int p = 0; p < 3; p++)
        {
            var weight = new double[d * d];
            for (int i = 0; i < d; i++) for (int o = 0; o < d; o++) weight[o * d + i] = qkv[i * 3 * d + p * d + o];
            TorchParameters.Linear(_engine, projections[p], d, d, weight, bias.Skip(p * d).Take(d).ToArray());
        }
        void Conv1D(DenseLayer<T> layer, string name, int input, int output)
        {
            var source = read(prefix + name + ".weight", new[] { input, output });
            var weight = new double[source.Length];
            for (int i = 0; i < input; i++) for (int o = 0; o < output; o++) weight[o * input + i] = source[i * output + o];
            TorchParameters.Linear(_engine, layer, input, output, weight, read(prefix + name + ".bias", new[] { output }));
        }
        Conv1D(_attentionOut, "attn.c_proj", d, d);
        Conv1D(_mlpIn, "mlp.c_fc", d, 4 * d);
        Conv1D(_mlpOut, "mlp.c_proj", 4 * d, d);
    }
}

/// <summary>The Tortoise conditioning attention block used by XTTS before its Perceiver resampler.</summary>
internal sealed class XttsConditioningAttention<T>
{
    private readonly bool _normalizeResidual;
    private readonly TiedEmbeddingLayer<T>? _relativeBias;
    private readonly IEngine _engine;
    private readonly int _dimension, _heads;
    private readonly GroupNormalizationLayer<T> _norm;
    private readonly Conv1DLayer<T> _qkv, _output;

    internal XttsConditioningAttention(IEngine engine, List<LayerBase<T>> owner, int dimension, int heads,
        bool normalizeResidual = true, bool relativePosition = false)
    {
        _normalizeResidual = normalizeResidual;
        _engine = engine;
        _dimension = dimension;
        _heads = heads;
        if (relativePosition) _relativeBias = XttsOps<T>.Own(owner, new TiedEmbeddingLayer<T>(32, heads));
        int groups = dimension <= 16 ? 8 : dimension <= 64 ? 16 : 32;
        while (dimension % groups != 0) groups /= 2;
        _norm = XttsOps<T>.Own(owner, new GroupNormalizationLayer<T>(groups, dimension));
        _qkv = XttsOps<T>.Own(owner, new Conv1DLayer<T>(dimension, 3 * dimension, 1, 1, 1, 0));
        _output = XttsOps<T>.Own(owner, new Conv1DLayer<T>(dimension, dimension, 1, 1, 1, 0, null,
            new AiDotNet.Initialization.ZeroInitializationStrategy<T>()));
    }

    internal Tensor<T> Forward(Tensor<T> input)
    {
        // Coqui's AttentionBlock residual is the normalized x (not the block input).
        int length = input.Shape[2];
        var x = _engine.Reshape(_norm.Forward(_engine.Reshape(input,
            new[] { _dimension, 1, length })), new[] { 1, _dimension, length });
        int headDimension = _dimension / _heads;
        var qkv = _qkv.Forward(x);
        var grouped = _engine.Reshape(qkv, new[] { _heads, 3, headDimension, length });
        Tensor<T> Part(int part) => _engine.TensorPermute(_engine.Reshape(
            _engine.TensorNarrow(grouped, 1, part, 1), new[] { _heads, headDimension, length }), new[] { 0, 2, 1 });
        double scale = 1.0 / Math.Sqrt(Math.Sqrt(headDimension));
        var query = _engine.TensorMultiplyScalar(Part(0), XttsOps<T>.NumOps.FromDouble(scale));
        var key = _engine.TensorMultiplyScalar(Part(1), XttsOps<T>.NumOps.FromDouble(scale));
        var value = Part(2);
        var scores = _engine.BatchMatMul(query, _engine.TensorPermute(key, new[] { 0, 2, 1 }));
        if (_relativeBias is not null)
        {
            var buckets = new int[length * length];
            for (int q = 0; q < length; q++) for (int k = 0; k < length; k++)
            {
                int distance = Math.Abs(q - k);
                int bucket = distance < 8 ? distance : Math.Min(15,
                    8 + (int)(Math.Log(distance / 8.0) / Math.Log(64.0 / 8) * 8));
                buckets[q * length + k] = bucket + (k > q ? 16 : 0);
            }
            var bias = _engine.TensorPermute(_engine.Reshape(_relativeBias.Forward(XttsOps<T>.Ids(buckets)),
                new[] { length, length, _heads }), new[] { 2, 0, 1 });
            scores = _engine.TensorAdd(scores, _engine.TensorMultiplyScalar(bias, XttsOps<T>.NumOps.FromDouble(Math.Sqrt(headDimension))));
        }
        var attended = _engine.BatchMatMul(_engine.Softmax(scores, 2), value);
        var channelsFirst = _engine.Reshape(_engine.TensorPermute(attended, new[] { 0, 2, 1 }),
            new[] { 1, _dimension, length });
        return _engine.TensorAdd(_normalizeResidual ? x : input, _output.Forward(channelsFirst));
    }

    internal void LoadTorchWeights(Func<string, int[], double[]> read, string prefix)
    {
        _norm.LoadAffine(read(prefix + "norm.weight", new[] { _dimension }), read(prefix + "norm.bias", new[] { _dimension }));
        XttsOps<T>.LoadConv(read, _qkv, prefix + "qkv", _dimension, 3 * _dimension, 1);
        XttsOps<T>.LoadConv(read, _output, prefix + "proj_out", _dimension, _dimension, 1);
        _relativeBias?.LoadTable(read(prefix + "relative_pos_embeddings.relative_attention_bias.weight", new[] { 32, _heads }));
    }
}

/// <summary>Six attention blocks and the two-layer 32-latent Perceiver from the XTTS-v2 release.</summary>
internal sealed class XttsConditioningEncoder<T>
{
    private readonly IEngine _engine;
    private readonly int _dimension, _melChannels;
    private readonly Conv1DLayer<T> _input;
    private readonly XttsConditioningAttention<T>[] _attention;
    private readonly IndexTtsPerceiver<T> _perceiver;

    internal XttsConditioningEncoder(IEngine engine, List<LayerBase<T>> owner, XTTSv2Options options, Random random)
    {
        _engine = engine;
        _dimension = options.LLMDim;
        _melChannels = options.MelChannels;
        _input = XttsOps<T>.Own(owner, new Conv1DLayer<T>(options.MelChannels, _dimension, 1, 1, 1, 0));
        _attention = Enumerable.Range(0, options.ConditioningBlocks)
            .Select(_ => new XttsConditioningAttention<T>(engine, owner, _dimension, options.ConditioningHeads)).ToArray();
        _perceiver = new IndexTtsPerceiver<T>(engine, owner, random, _dimension, _dimension,
            options.PerceiverLatents, options.PerceiverHeads, options.PerceiverFeedForwardMultiple, options.PerceiverDepth);
    }

    internal int AttentionBlocks => _attention.Length;
    internal int Latents => _perceiver.Latents.GetParameters().Length / _dimension;

    internal void LoadTorchWeights(Func<string, int[], double[]> read, string prefix)
    {
        XttsOps<T>.LoadConv(read, _input, prefix + "conditioning_encoder.init", _melChannels, _dimension, 1);
        for (int i = 0; i < _attention.Length; i++) _attention[i].LoadTorchWeights(read, $"{prefix}conditioning_encoder.attn.{i}.");
        _perceiver.Load(_engine, read, prefix + "conditioning_perceiver.");
    }

    /// <param name="mel">Normalized log-power mel, <c>[80, frames]</c>.</param>
    internal Tensor<T> Forward(Tensor<T> mel)
    {
        int frames = mel.Shape[1];
        var hidden = _input.Forward(_engine.Reshape(mel, new[] { 1, mel.Shape[0], frames }));
        foreach (var block in _attention) hidden = block.Forward(hidden);
        return _perceiver.Forward(_engine.Reshape(_engine.TensorPermute(hidden, new[] { 0, 2, 1 }),
            new[] { frames, _dimension }));
    }
}

/// <summary>XTTS-v2's separate text/audio embeddings, learned positions, GPT-2 and prediction heads.</summary>
internal sealed class XttsLanguageModel<T>
{
    private readonly IEngine _engine;
    private readonly XTTSv2Options _options;
    private readonly EmbeddingLayer<T> _textEmbedding, _audioEmbedding, _textPosition, _audioPosition;
    private readonly XttsGptBlock<T>[] _blocks;
    private readonly LayerNormalizationLayer<T> _gptNorm, _finalNorm;
    private readonly DenseLayer<T> _textHead, _audioHead;
    private readonly XttsConditioningEncoder<T> _conditioning;

    internal XttsLanguageModel(IEngine engine, List<LayerBase<T>> owner, XTTSv2Options options, Random random)
    {
        _engine = engine;
        _options = options;
        _conditioning = new XttsConditioningEncoder<T>(engine, owner, options, random);
        _textEmbedding = XttsOps<T>.Own(owner, new EmbeddingLayer<T>(options.TextVocabularySize, options.LLMDim));
        _audioEmbedding = XttsOps<T>.Own(owner, new EmbeddingLayer<T>(options.AudioVocabularySize, options.LLMDim));
        _textPosition = XttsOps<T>.Own(owner, new EmbeddingLayer<T>(options.MaxTextLength + 2, options.LLMDim));
        _audioPosition = XttsOps<T>.Own(owner, new EmbeddingLayer<T>(options.MaxCodecFrames + 3, options.LLMDim));
        _blocks = Enumerable.Range(0, options.NumLLMLayers)
            .Select(_ => new XttsGptBlock<T>(engine, owner, options.LLMDim, options.NumHeads, options.DropoutRate)).ToArray();
        _gptNorm = XttsOps<T>.Own(owner, new LayerNormalizationLayer<T>(options.LLMDim, 1e-5));
        _finalNorm = XttsOps<T>.Own(owner, new LayerNormalizationLayer<T>(options.LLMDim, 1e-5));
        var identity = new IdentityActivation<T>() as IActivationFunction<T>;
        _textHead = XttsOps<T>.Own(owner, new DenseLayer<T>(options.TextVocabularySize, identity, biasMode: BiasMode.Always));
        _audioHead = XttsOps<T>.Own(owner, new DenseLayer<T>(options.AudioVocabularySize, identity, biasMode: BiasMode.Always));
    }

    internal int LayerCount => _blocks.Length;
    internal int ConditioningBlocks => _conditioning.AttentionBlocks;
    internal int ConditioningLatents => _conditioning.Latents;
    internal Tensor<T> Condition(Tensor<T> mel) => _conditioning.Forward(mel);

    internal void LoadTorchWeights(Func<string, int[], double[]> read, string prefix = "gpt.", bool includeConditioning = true)
    {
        int d = _options.LLMDim;
        void Embedding(EmbeddingLayer<T> layer, string name, int rows) => layer.SetParameters(new Vector<T>(
            read(prefix + name + ".weight", new[] { rows, d }).Select(XttsOps<T>.NumOps.FromDouble)));
        Embedding(_textEmbedding, "text_embedding", _options.TextVocabularySize);
        Embedding(_audioEmbedding, "mel_embedding", _options.AudioVocabularySize);
        Embedding(_textPosition, "text_pos_embedding.emb", _options.MaxTextLength + 2);
        Embedding(_audioPosition, "mel_pos_embedding.emb", _options.MaxCodecFrames + 3);
        for (int i = 0; i < _blocks.Length; i++) _blocks[i].LoadTorchWeights(read, $"{prefix}gpt.h.{i}.");
        IndexTtsOps.LoadNorm(_engine, read, _gptNorm, prefix + "gpt.ln_f", d);
        IndexTtsOps.LoadNorm(_engine, read, _finalNorm, prefix + "final_norm", d);
        IndexTtsOps.LoadLinear(_engine, read, _textHead, prefix + "text_head", d, _options.TextVocabularySize);
        IndexTtsOps.LoadLinear(_engine, read, _audioHead, prefix + "mel_head", d, _options.AudioVocabularySize);
        if (includeConditioning) _conditioning.LoadTorchWeights(read, prefix);
    }

    internal (Tensor<T> Text, Tensor<T> Audio) Forward(
        IReadOnlyList<int> textInput, IReadOnlyList<int> audioInput, Tensor<T> condition)
    {
        if (textInput.Count == 0 || audioInput.Count == 0) throw new ArgumentException("XTTS GPT streams must not be empty.");
        var text = _engine.TensorAdd(_textEmbedding.Forward(XttsOps<T>.Ids(textInput)),
            _textPosition.Forward(XttsOps<T>.Ids(Enumerable.Range(0, textInput.Count))));
        var audio = _engine.TensorAdd(_audioEmbedding.Forward(XttsOps<T>.Ids(audioInput)),
            _audioPosition.Forward(XttsOps<T>.Ids(Enumerable.Range(0, audioInput.Count))));
        var hidden = _engine.TensorConcatenate(new[] { condition, text, audio }, 0);
        var mask = XttsOps<T>.CausalMask(hidden.Shape[0]);
        foreach (var block in _blocks) hidden = block.Forward(hidden, mask);
        hidden = _finalNorm.Forward(_gptNorm.Forward(hidden));
        int offset = condition.Shape[0];
        var textHidden = _engine.TensorSlice(hidden, new[] { offset, 0 }, new[] { textInput.Count, _options.LLMDim });
        var audioHidden = _engine.TensorSlice(hidden, new[] { offset + textInput.Count, 0 }, new[] { audioInput.Count, _options.LLMDim });
        return (textHidden, audioHidden);
    }

    internal Tensor<T> Objective(IReadOnlyList<int> text, IReadOnlyList<int> codes, Tensor<T> condition)
    {
        int startText = _options.StartTextToken >= 0 ? _options.StartTextToken
            : MathPolyfill.Clamp(text.Count == 0 ? 0 : text[0], 0, _options.TextVocabularySize - 1);
        int stopText = _options.StopTextToken >= 0 ? _options.StopTextToken
            : MathPolyfill.Clamp(text.Count == 0 ? 0 : text[^1], 0, _options.TextVocabularySize - 1);
        // The tokenizer supplies the release's [START]/[STOP] ids. For token-only training rows, using the boundary
        // ids preserves the two shifted streams without inventing a vocabulary-dependent constant.
        var textInput = new[] { startText }.Concat(text).Take(_options.MaxTextLength + 1).ToArray();
        var textTargets = text.Concat(new[] { stopText }).Take(textInput.Length).ToArray();
        var clippedCodes = codes.Take(_options.MaxCodecFrames).Select(x => MathPolyfill.Clamp(x, 0, _options.CodebookSize - 1)).ToArray();
        var audioInput = new[] { _options.StartAudioToken }.Concat(clippedCodes).ToArray();
        var audioTargets = clippedCodes.Concat(new[] { _options.StopAudioToken }).ToArray();
        var hidden = Forward(textInput, audioInput, condition);
        var textLoss = AiDotNet.TextToSpeech.CodecBased.UniAudioOps<T>.CrossEntropy(_engine,
            _textHead.Forward(hidden.Text), textTargets);
        var audioLoss = AiDotNet.TextToSpeech.CodecBased.UniAudioOps<T>.CrossEntropy(_engine,
            _audioHead.Forward(hidden.Audio), audioTargets);
        return _engine.TensorAdd(
            _engine.TensorMultiplyScalar(textLoss, XttsOps<T>.NumOps.FromDouble(_options.TextLossWeight)),
            _engine.TensorMultiplyScalar(audioLoss, XttsOps<T>.NumOps.FromDouble(_options.AudioLossWeight)));
    }

    internal (int[] Codes, Tensor<T> Latents) Generate(IReadOnlyList<int> text, Tensor<T> condition, Random random)
    {
        int[] textStream = text.Take(_options.MaxTextLength).ToArray();
        if (textStream.Length == 0) throw new ArgumentException("XTTS text produced no tokenizer ids.", nameof(text));
        int startText = MathPolyfill.Clamp(_options.StartTextToken, 0, _options.TextVocabularySize - 1);
        int stopText = MathPolyfill.Clamp(_options.StopTextToken, 0, _options.TextVocabularySize - 1);
        var textInput = new[] { startText }.Concat(textStream).Concat(new[] { stopText }).ToArray();
        var audio = new List<int> { _options.StartAudioToken };
        // HF's fake condition/text inputs contain ID 1 followed by audio BOS.
        // Repetition penalty is presence-based, so one instance of each is sufficient.
        var history = new List<int> { 1, _options.StartAudioToken };
        var generated = new List<int>();
        for (int step = 0; step < _options.MaxCodecFrames; step++)
        {
            var hidden = Forward(textInput, audio, condition).Audio;
            var logits = _audioHead.Forward(hidden);
            int token = Sample(logits, logits.Shape[0] - 1, history, random);
            generated.Add(token);
            audio.Add(token);
            history.Add(token);
            if (token == _options.StopAudioToken) break;
        }
        if (generated.Count == 0) return (Array.Empty<int>(), new Tensor<T>(new[] { 0, _options.LLMDim }));
        // The author's generated sequence retains EOS, and forward(..., return_latent=True)
        // retains its prediction state after padding and the final five-state trim.
        var teacher = Forward(textInput, audio, condition).Audio;
        var latents = _engine.TensorSlice(teacher, new[] { 0, 0 }, new[] { generated.Count, _options.LLMDim });
        return (generated.ToArray(), latents);
    }

    private int Sample(Tensor<T> logits, int row, IReadOnlyCollection<int> history, Random random)
    {
        var candidates = new List<(int Token, double Value)>(_options.AudioVocabularySize);
        for (int id = 0; id < _options.AudioVocabularySize; id++)
        {
            double value = XttsOps<T>.NumOps.ToDouble(logits[row, id]);
            if (history.Contains(id)) value = value >= 0 ? value / _options.RepetitionPenalty : value * _options.RepetitionPenalty;
            candidates.Add((id, value / _options.Temperature));
        }
        candidates = candidates.OrderByDescending(item => item.Value).ThenBy(item => item.Token).ToList();
        if (_options.TopK > 0 && candidates.Count > _options.TopK)
        {
            double threshold = candidates[_options.TopK - 1].Value;
            candidates.RemoveAll(item => item.Value < threshold);
        }
        double maximum = candidates[0].Value;
        var weighted = candidates.Select(item => (item.Token, Weight: Math.Exp(item.Value - maximum))).ToList();
        HuggingFaceNucleusSampling.Retain(weighted, _options.TopP);
        double draw = random.NextDouble() * weighted.Sum(item => item.Weight);
        foreach (var item in weighted)
        {
            draw -= item.Weight;
            if (draw <= 0) return item.Token;
        }
        return weighted[^1].Token;
    }
}

internal sealed class XttsDvaeResidual<T>
{
    private readonly IEngine _engine;
    private readonly Conv1DLayer<T> _first, _second, _projection;

    internal XttsDvaeResidual(IEngine engine, List<LayerBase<T>> owner, int channels)
    {
        _engine = engine;
        _first = XttsOps<T>.Own(owner, new Conv1DLayer<T>(channels, channels, 3, 1, 1, 1));
        _second = XttsOps<T>.Own(owner, new Conv1DLayer<T>(channels, channels, 3, 1, 1, 1));
        _projection = XttsOps<T>.Own(owner, new Conv1DLayer<T>(channels, channels, 1, 1, 1, 0));
    }

    internal Tensor<T> Forward(Tensor<T> input) => _engine.TensorAdd(input,
        _projection.Forward(_engine.ReLU(_second.Forward(_engine.ReLU(_first.Forward(input))))));

    internal void LoadTorchWeights(Func<string, int[], double[]> read, string prefix, int channels)
    {
        XttsOps<T>.LoadConv(read, _first, prefix + ".net.0", channels, channels, 3);
        XttsOps<T>.LoadConv(read, _second, prefix + ".net.2", channels, channels, 3);
        XttsOps<T>.LoadConv(read, _projection, prefix + ".net.4", channels, channels, 1);
    }
}

/// <summary>The released 13M two-stride dVAE used only to tokenize/reconstruct mel spectrograms.</summary>
internal sealed class XttsDiscreteVae<T>
{
    private readonly IEngine _engine;
    private readonly XTTSv2Options _options;
    private readonly Conv1DLayer<T> _encoder1, _encoder2, _encoderProjection;
    private readonly XttsDvaeResidual<T>[] _encoderResiduals, _decoderResiduals;
    private readonly ResidualVectorQuantizerLayer<T> _codebook;
    private readonly Conv1DLayer<T> _decoderInput, _decoderUp1, _decoderUp2, _decoderOutput;

    internal XttsDiscreteVae(IEngine engine, List<LayerBase<T>> owner, XTTSv2Options options)
    {
        _engine = engine;
        _options = options;
        int h = options.DvaeHiddenDim, inner = 2 * h;
        _encoder1 = XttsOps<T>.Own(owner, new Conv1DLayer<T>(options.MelChannels, h, 3, 1, 2, 1));
        _encoder2 = XttsOps<T>.Own(owner, new Conv1DLayer<T>(h, inner, 3, 1, 2, 1));
        _encoderResiduals = Enumerable.Range(0, options.DvaeResidualBlocks)
            .Select(_ => new XttsDvaeResidual<T>(engine, owner, inner)).ToArray();
        _encoderProjection = XttsOps<T>.Own(owner, new Conv1DLayer<T>(inner, options.DvaeCodebookDim, 1, 1, 1, 0));
        _codebook = XttsOps<T>.Own(owner, new ResidualVectorQuantizerLayer<T>(options.DvaeCodebookDim,
            1, options.CodebookSize, decay: 0.99, kmeansIterations: 0, deadCodeThreshold: 0, commitmentAsMean: true));
        _decoderInput = XttsOps<T>.Own(owner, new Conv1DLayer<T>(options.DvaeCodebookDim, inner, 1, 1, 1, 0));
        _decoderResiduals = Enumerable.Range(0, options.DvaeResidualBlocks)
            .Select(_ => new XttsDvaeResidual<T>(engine, owner, inner)).ToArray();
        _decoderUp1 = XttsOps<T>.Own(owner, new Conv1DLayer<T>(inner, inner, 3, 1, 1, 1));
        _decoderUp2 = XttsOps<T>.Own(owner, new Conv1DLayer<T>(inner, h, 3, 1, 1, 1));
        _decoderOutput = XttsOps<T>.Own(owner, new Conv1DLayer<T>(h, options.MelChannels, 1, 1, 1, 0));
    }

    internal int DownsampleFactor => 4;
    internal ResidualVectorQuantizerLayer<T> Quantizer => _codebook;

    internal void LoadTorchWeights(Func<string, int[], double[]> read)
    {
        int h = _options.DvaeHiddenDim, inner = 2 * h, r = _options.DvaeResidualBlocks, dim = _options.DvaeCodebookDim;
        XttsOps<T>.LoadConv(read, _encoder1, "encoder.0.0", _options.MelChannels, h, 3);
        XttsOps<T>.LoadConv(read, _encoder2, "encoder.1.0", h, inner, 3);
        for (int i = 0; i < r; i++) _encoderResiduals[i].LoadTorchWeights(read, $"encoder.{i + 2}", inner);
        XttsOps<T>.LoadConv(read, _encoderProjection, $"encoder.{r + 2}", inner, dim, 1);
        double[] TransposeCodebook(string name)
        {
            var source = read(name, new[] { dim, _options.CodebookSize });
            var table = new double[source.Length];
            for (int c = 0; c < _options.CodebookSize; c++) for (int d = 0; d < dim; d++) table[c * dim + d] = source[d * _options.CodebookSize + c];
            return table;
        }
        _codebook.LoadCodebook(0, TransposeCodebook("codebook.embed"), TransposeCodebook("codebook.embed_avg"),
            read("codebook.cluster_size", new[] { _options.CodebookSize }), initialized: true);
        XttsOps<T>.LoadConv(read, _decoderInput, "decoder.0", dim, inner, 1);
        for (int i = 0; i < r; i++) _decoderResiduals[i].LoadTorchWeights(read, $"decoder.{i + 1}", inner);
        XttsOps<T>.LoadConv(read, _decoderUp1, $"decoder.{r + 1}.0.conv", inner, inner, 3);
        XttsOps<T>.LoadConv(read, _decoderUp2, $"decoder.{r + 2}.0.conv", inner, h, 3);
        XttsOps<T>.LoadConv(read, _decoderOutput, $"decoder.{r + 3}", h, _options.MelChannels, 1);
    }

    internal Tensor<T> EncodeFeatures(Tensor<T> mel)
    {
        int frames = mel.Shape[1];
        var x = _engine.Reshape(mel, new[] { 1, _options.MelChannels, frames });
        x = _engine.ReLU(_encoder1.Forward(x));
        x = _engine.ReLU(_encoder2.Forward(x));
        foreach (var residual in _encoderResiduals) x = residual.Forward(x);
        return _encoderProjection.Forward(x);
    }

    internal Tensor<T> Encode(Tensor<T> mel)
    {
        var codes = _codebook.Encode(EncodeFeatures(mel), 1);
        return XttsOps<T>.Ids(Enumerable.Range(0, codes.GetLength(1)).Select(i => codes[0, i]));
    }

    internal Tensor<T> Reconstruct(Tensor<T> mel) => DecodeQuantized(_codebook.Forward(EncodeFeatures(mel)));

    internal Tensor<T> Decode(Tensor<T> tokens)
    {
        int[] ids = XttsOps<T>.IdsOf(tokens);
        if (ids.Any(id => id < 0 || id >= _options.CodebookSize))
            throw new ArgumentOutOfRangeException(nameof(tokens), "dVAE tokens must be codebook indices.");
        var codes = new int[1, ids.Length];
        for (int i = 0; i < ids.Length; i++) codes[0, i] = ids[i];
        return DecodeQuantized(_codebook.Decode(codes));
    }

    internal Tensor<T> DecodeQuantized(Tensor<T> quantized)
    {
        var x = _decoderInput.Forward(quantized);
        foreach (var residual in _decoderResiduals) x = residual.Forward(x);
        x = _engine.ReLU(_decoderUp1.Forward(Nearest(x, 2)));
        x = _engine.ReLU(_decoderUp2.Forward(Nearest(x, 2)));
        x = _decoderOutput.Forward(x);
        return _engine.Reshape(_engine.TensorPermute(x, new[] { 0, 2, 1 }), new[] { x.Shape[2], _options.MelChannels });
    }


    private Tensor<T> Nearest(Tensor<T> input, int factor)
    {
        var indices = new Tensor<int>(new[] { input.Shape[2] * factor });
        for (int t = 0; t < indices.Length; t++) indices[t] = t / factor;
        return _engine.Reshape(_engine.TensorIndexSelect(_engine.Reshape(input,
            new[] { input.Shape[0] * input.Shape[1], input.Shape[2] }), indices, 1),
            new[] { input.Shape[0], input.Shape[1], indices.Length });
    }
}

/// <summary>The 24 kHz XTTS decoder: GPT-latent interpolation and speaker-conditioned HiFi-GAN.</summary>
internal sealed class XttsWaveformDecoder<T>
{
    private readonly IEngine _engine;
    private readonly XTTSv2Options _options;
    private readonly HiFiGanGenerator<T> _generator;

    internal XttsWaveformDecoder(IEngine engine, List<LayerBase<T>> owner, XTTSv2Options options)
    {
        _engine = engine;
        _options = options;
        _generator = new HiFiGanGenerator<T>(engine, options.LLMDim, options.DecoderInitialChannels,
            options.DecoderUpsampleRates, options.DecoderUpsampleKernels, options.DecoderResblockKernels,
            options.DecoderResblockDilations, resblockType1: true, conditionChannels: options.SpeakerEmbeddingDim,
            plainEnds: true, plainOutputBias: false, conditionInEachUpsamplingLayer: true);
        owner.AddRange(_generator.Layers);
    }

    internal HiFiGanGenerator<T> Generator => _generator;

    /// <summary>Loads Coqui's global cond_layer and per-upsampling conds projections.</summary>
    internal void LoadTorchWeights(Func<string, double[]?> read) => _generator.LoadTorch(name =>
        read(name.StartsWith("cond.", StringComparison.Ordinal) ? "cond_layer." + name.Substring(5) : name));

    internal Tensor<T> Forward(Tensor<T> latents, Tensor<T> speaker)
    {
        if (_options.Speed != 1) latents = LinearScale(latents, 1.0 / Math.Max(0.05, _options.Speed));
        var interpolated = LinearScale(latents, (double)_options.GptCodeStride / _options.HopSize);
        if (_options.SampleRate != _options.InputSampleRate)
            interpolated = LinearScale(interpolated, (double)_options.SampleRate / _options.InputSampleRate);
        int frames = interpolated.Shape[0];
        var features = _engine.Reshape(_engine.TensorTranspose(interpolated), new[] { 1, _options.LLMDim, frames });
        var condition = _engine.Reshape(speaker, new[] { 1, _options.SpeakerEmbeddingDim, 1 });
        var waveform = _generator.Forward(features, condition);
        return _engine.Reshape(waveform, new[] { waveform.Length });
    }

    internal Tensor<T> LinearScale(Tensor<T> input, double scale)
    {
        int sourceFrames = input.Shape[0], channels = input.Shape[1];
        int frames = Math.Max(1, (int)Math.Floor(sourceFrames * scale));
        var leftIndices = new Tensor<int>(new[] { frames });
        var rightIndices = new Tensor<int>(new[] { frames });
        var weights = new Tensor<T>(new[] { frames, channels });
        for (int t = 0; t < frames; t++)
        {
            double position = MathPolyfill.Clamp((t + 0.5) / scale - 0.5, 0, sourceFrames - 1);
            int left = (int)Math.Floor(position), right = Math.Min(sourceFrames - 1, left + 1);
            double fraction = position - left;
            leftIndices[t] = left; rightIndices[t] = right;
            for (int c = 0; c < channels; c++) weights[t, c] = XttsOps<T>.NumOps.FromDouble(fraction);
        }
        var leftRows = _engine.TensorIndexSelect(input, leftIndices, 0);
        var rightRows = _engine.TensorIndexSelect(input, rightIndices, 0);
        return _engine.TensorAdd(leftRows, _engine.TensorMultiply(_engine.TensorSubtract(rightRows, leftRows), weights));
    }
}
