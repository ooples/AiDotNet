using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.ModelLoading;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.FrontEnd;
using AiDotNet.TextToSpeech.Latest;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.TextToSpeech.CodecBased;

internal static class VoiceCraftOps<T>
{
    internal static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    internal static TLayer Own<TLayer>(List<LayerBase<T>> owner, TLayer layer) where TLayer : LayerBase<T>
    {
        owner.Add(layer);
        return layer;
    }

    internal static Tensor<T> Ids(IReadOnlyList<int> values)
    {
        var result = new Tensor<T>(new[] { values.Count });
        for (int i = 0; i < values.Count; i++) result[i] = NumOps.FromDouble(values[i]);
        return result;
    }

    internal static int[] IdsOf(Tensor<T> values)
    {
        var result = new int[values.Length];
        for (int i = 0; i < result.Length; i++) result[i] = (int)Math.Round(NumOps.ToDouble(values[i]));
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
}

/// <summary>The exact 80-symbol <c>phn2num</c> table shipped with the GigaSpeech VoiceCraft checkpoints.</summary>
internal static class VoiceCraftPhoneTokenizer
{
    private static readonly Dictionary<string, int> Table = new(StringComparer.Ordinal)
    {
        ["ɑː"] = 0, ["u"] = 1, ["aɪɚ"] = 2, ["ɔ"] = 3, ["x"] = 4, ["ɹ"] = 5, ["eɪ"] = 6,
        ["ð"] = 7, ["nʲ"] = 8, ["m"] = 9, ["ç"] = 10, ["ɛɹ"] = 11, ["̩"] = 12, ["ɔɪ"] = 13,
        ["h"] = 14, ["_"] = 15, ["əl"] = 16, ["!"] = 17, ["ɛ"] = 18, ["w"] = 19, ["b"] = 20,
        ["ɜː"] = 21, ["z"] = 22, ["n"] = 23, ["ɾ"] = 24, ["oʊ"] = 25, ["ʃ"] = 26, ["i"] = 27,
        ["ʊɹ"] = 28, ["ɔː"] = 29, ["θ"] = 30, ["v"] = 31, ["æ"] = 32, ["ɔːɹ"] = 33,
        ["p"] = 34, ["ɚ"] = 35, ["aʊ"] = 36, ["ɡʲ"] = 37, ["ʌ"] = 38, ["<MUSIC>"] = 39,
        ["oːɹ"] = 40, ["k"] = 41, ["iə"] = 42, ["ʊ"] = 43, ["ɑːɹ"] = 44, ["̃"] = 45,
        ["ɬ"] = 46, ["uː"] = 47, ["aɪ"] = 48, ["ɡ"] = 49, ["ææ"] = 50, ["iːː"] = 51,
        ["<NOISE>"] = 52, ["<SIL>"] = 53, ["ə"] = 54, ["ʒ"] = 55, ["aɪə"] = 56, ["dʒ"] = 57,
        ["ŋ"] = 58, ["ʔ"] = 59, ["<OTHER>"] = 60, ["ɑ"] = 61, ["ᵻ"] = 62, ["l"] = 63,
        [","] = 64, ["ɪ"] = 65, ["s"] = 66, ["j"] = 67, ["iː"] = 68, ["f"] = 69, ["ɐ"] = 70,
        ["ɐɐ"] = 71, ["d"] = 72, ["t"] = 73, ["oː"] = 74, ["tʃ"] = 75, ["ɪɹ"] = 76,
        ["."] = 77, ["?"] = 78, ["r"] = 79
    };

    internal static int Count => Table.Count;

    internal static int[] Tokenize(string text, int maximum, IReadOnlyDictionary<string, int>? vocabulary = null,
        Func<string, IReadOnlyList<string>>? phonemizer = null)
    {
        if (string.IsNullOrWhiteSpace(text)) throw new ArgumentException("Text must not be empty.", nameof(text));
        var table = vocabulary ?? Table;
        return (phonemizer is null ? Phonemize(text) : phonemizer(text))
            .Where(table.ContainsKey).Select(phone => table[phone]).Take(maximum).ToArray();
    }

    internal static IReadOnlyList<string> Phonemize(string text, Func<string, string, string>? espeak = null)
    {
        // data/tokenizer.py: phonemizer EspeakBackend(en-us), no stress, punctuation preserved,
        // language_switch=keep-flags, phone='|', word='_', then TextTokenizer.to_list's Python regex.
        const string marks = ";:,.!?¡¿—…\"«»“”(){}[]";
        var phonemizer = new EspeakPhonemizer("en-us", marks, espeak, keepLanguageFlags: true);
        var serialized = phonemizer.Phonemize(text, "|", "_");
        var phones = new List<string>();
        var words = serialized.Split('_');
        for (int i = 0; i < words.Length; i++)
        {
            // Python \w excludes combining marks, unlike .NET \w. Keep IPA tie/length/stress punctuation separate.
            foreach (System.Text.RegularExpressions.Match match in System.Text.RegularExpressions.Regex.Matches(words[i],
                @"[\p{L}\p{N}_]+|[^\p{L}\p{N}_\s]"))
                if (match.Value != "|") phones.Add(match.Value);
            if (i + 1 < words.Length) phones.Add("_");
        }
        return phones;
    }

    internal static IReadOnlyDictionary<string, int> Vocabulary => Table;
}

/// <summary>VoiceCraft's released pre-norm Transformer encoder layer used as a causal decoder block.</summary>
internal sealed class VoiceCraftTransformerBlock<T>
{
    private readonly IEngine _engine;
    private readonly int _dimension, _heads;
    private readonly LayerNormalizationLayer<T> _norm1, _norm2;
    private readonly DenseLayer<T> _query, _key, _value, _attentionOut, _ffIn, _ffOut;
    private readonly DropoutLayer<T>? _attentionWeightsDropout, _attentionOutDropout, _activationDropout, _ffOutDropout;

    internal VoiceCraftTransformerBlock(IEngine engine, List<LayerBase<T>> owner, int dimension, int heads,
        int feedForward, double dropout)
    {
        _engine = engine;
        _dimension = dimension;
        _heads = heads;
        var identity = new IdentityActivation<T>() as IActivationFunction<T>;
        _norm1 = VoiceCraftOps<T>.Own(owner, new LayerNormalizationLayer<T>(dimension));
        _query = VoiceCraftOps<T>.Own(owner, new DenseLayer<T>(dimension, identity, biasMode: BiasMode.Always));
        _key = VoiceCraftOps<T>.Own(owner, new DenseLayer<T>(dimension, identity, biasMode: BiasMode.Always));
        _value = VoiceCraftOps<T>.Own(owner, new DenseLayer<T>(dimension, identity, biasMode: BiasMode.Always));
        _attentionOut = VoiceCraftOps<T>.Own(owner, new DenseLayer<T>(dimension, identity, biasMode: BiasMode.Always));
        _norm2 = VoiceCraftOps<T>.Own(owner, new LayerNormalizationLayer<T>(dimension));
        _ffIn = VoiceCraftOps<T>.Own(owner, new DenseLayer<T>(feedForward, identity, biasMode: BiasMode.Always));
        _ffOut = VoiceCraftOps<T>.Own(owner, new DenseLayer<T>(dimension, identity, biasMode: BiasMode.Always));
        if (dropout > 0)
        {
            _attentionWeightsDropout = VoiceCraftOps<T>.Own(owner, new DropoutLayer<T>(dropout));
            _attentionOutDropout = VoiceCraftOps<T>.Own(owner, new DropoutLayer<T>(dropout));
            _activationDropout = VoiceCraftOps<T>.Own(owner, new DropoutLayer<T>(dropout));
            _ffOutDropout = VoiceCraftOps<T>.Own(owner, new DropoutLayer<T>(dropout));
        }
    }

    internal Tensor<T> Forward(Tensor<T> input, Tensor<T> mask)
    {
        var x = _norm1.Forward(input);
        int length = x.Shape[0], headDimension = _dimension / _heads;
        Tensor<T> Heads(Tensor<T> projected) => _engine.TensorPermute(
            _engine.Reshape(projected, new[] { length, _heads, headDimension }), new[] { 1, 0, 2 });
        var query = _engine.TensorMultiplyScalar(Heads(_query.Forward(x)),
            VoiceCraftOps<T>.NumOps.FromDouble(1.0 / Math.Sqrt(headDimension)));
        var key = Heads(_key.Forward(x));
        var value = Heads(_value.Forward(x));
        var scores = _engine.BatchMatMul(query, _engine.TensorPermute(key, new[] { 0, 2, 1 }));
        scores = _engine.TensorAdd(scores, _engine.TensorTile(
            _engine.Reshape(mask, new[] { 1, length, length }), new[] { _heads, 1, 1 }));
        var weights = _engine.Softmax(scores, 2);
        if (_attentionWeightsDropout is not null) weights = _attentionWeightsDropout.Forward(weights);
        var context = _engine.Reshape(_engine.TensorPermute(_engine.BatchMatMul(weights, value),
            new[] { 1, 0, 2 }), new[] { length, _dimension });
        var attention = _attentionOut.Forward(context);
        if (_attentionOutDropout is not null) attention = _attentionOutDropout.Forward(attention);
        input = _engine.TensorAdd(input, attention);
        var hidden = _engine.ReLU(_ffIn.Forward(_norm2.Forward(input)));
        if (_activationDropout is not null) hidden = _activationDropout.Forward(hidden);
        var output = _ffOut.Forward(hidden);
        if (_ffOutDropout is not null) output = _ffOutDropout.Forward(output);
        return _engine.TensorAdd(input, output);
    }

    internal void LoadTorchWeights(TorchTensorSource source, string prefix, int feedForward)
    {
        TorchParameters.LayerNorm(_engine, _norm1, source, prefix + "norm1.weight", prefix + "norm1.bias");
        TorchParameters.LayerNorm(_engine, _norm2, source, prefix + "norm2.weight", prefix + "norm2.bias");
        // The fused query/key/value projection is split three ways.
        var weight = source.Read(prefix + "self_attn.in_proj_weight", new[] { 3 * _dimension, _dimension });
        var bias = source.Read(prefix + "self_attn.in_proj_bias", new[] { 3 * _dimension });
        var projections = new[] { _query, _key, _value };
        for (int i = 0; i < projections.Length; i++)
            TorchParameters.Linear(_engine, projections[i], _dimension, _dimension,
                weight.Skip(i * _dimension * _dimension).Take(_dimension * _dimension).ToArray(),
                bias.Skip(i * _dimension).Take(_dimension).ToArray());
        TorchParameters.Linear(_engine, _attentionOut, _dimension, _dimension, source, prefix + "self_attn.out_proj.weight", prefix + "self_attn.out_proj.bias");
        TorchParameters.Linear(_engine, _ffIn, _dimension, feedForward, source, prefix + "linear1.weight", prefix + "linear1.bias");
        TorchParameters.Linear(_engine, _ffOut, feedForward, _dimension, source, prefix + "linear2.weight", prefix + "linear2.bias");
    }
}

/// <summary>The released decoder-only Transformer and its four independent two-layer prediction heads.</summary>
internal sealed class VoiceCraftNetwork<T>
{
    private readonly IEngine _engine;
    private readonly VoiceCraftOptions _options;
    private readonly EmbeddingLayer<T> _textEmbedding;
    private readonly EmbeddingLayer<T>[] _audioEmbeddings;
    private readonly EmbeddingLayer<T> _maskEmbedding;
    private readonly ScaledPositionalEncodingLayer<T> _textPosition, _audioPosition;
    private readonly DropoutLayer<T>? _textDropout, _audioDropout, _textPositionDropout, _audioPositionDropout;
    private readonly VoiceCraftTransformerBlock<T>[] _blocks;
    private readonly LayerNormalizationLayer<T> _finalNorm;
    private readonly (DenseLayer<T> Hidden, DenseLayer<T> Output)[] _predictionHeads;

    internal VoiceCraftNetwork(IEngine engine, List<LayerBase<T>> owner, VoiceCraftOptions options)
    {
        _engine = engine;
        _options = options;
        var identity = new IdentityActivation<T>() as IActivationFunction<T>;
        _textEmbedding = VoiceCraftOps<T>.Own(owner,
            new EmbeddingLayer<T>(options.TextVocabularySize + 1, options.LLMDim));
        _audioEmbeddings = Enumerable.Range(0, options.NumCodebooks).Select(_ =>
            VoiceCraftOps<T>.Own(owner, new EmbeddingLayer<T>(options.AudioVocabularySize, options.LLMDim))).ToArray();
        _maskEmbedding = VoiceCraftOps<T>.Own(owner,
            new EmbeddingLayer<T>(options.MaxMaskSpans, options.LLMDim));
        _textPosition = VoiceCraftOps<T>.Own(owner, new ScaledPositionalEncodingLayer<T>(options.LLMDim));
        _audioPosition = VoiceCraftOps<T>.Own(owner, new ScaledPositionalEncodingLayer<T>(options.LLMDim));
        if (options.TextEmbeddingDropout > 0) _textDropout = VoiceCraftOps<T>.Own(owner, new DropoutLayer<T>(options.TextEmbeddingDropout));
        if (options.AudioEmbeddingDropout > 0) _audioDropout = VoiceCraftOps<T>.Own(owner, new DropoutLayer<T>(options.AudioEmbeddingDropout));
        if (options.TextPositionDropout > 0) _textPositionDropout = VoiceCraftOps<T>.Own(owner, new DropoutLayer<T>(options.TextPositionDropout));
        if (options.AudioPositionDropout > 0) _audioPositionDropout = VoiceCraftOps<T>.Own(owner, new DropoutLayer<T>(options.AudioPositionDropout));
        _blocks = Enumerable.Range(0, options.NumLLMLayers).Select(_ =>
            new VoiceCraftTransformerBlock<T>(engine, owner, options.LLMDim, options.NumHeads,
                options.FeedForwardDim, options.TransformerDropout)).ToArray();
        _finalNorm = VoiceCraftOps<T>.Own(owner, new LayerNormalizationLayer<T>(options.LLMDim));
        _predictionHeads = Enumerable.Range(0, options.NumCodebooks).Select(_ => (
            VoiceCraftOps<T>.Own(owner, new DenseLayer<T>(options.CodebookSize / 2,
                identity, biasMode: BiasMode.Always)),
            VoiceCraftOps<T>.Own(owner, new DenseLayer<T>(options.AudioVocabularySize, identity,
                biasMode: BiasMode.Always)))).ToArray();
    }

    internal int LayerCount => _blocks.Length;

    internal void LoadTorchWeights(Func<string, int[], double[]> read) => LoadTorchWeights(TorchTensorSource.From(read));

    internal void LoadTorchWeights(TorchTensorSource source)
    {
        void Embedding(EmbeddingLayer<T> layer, string name, int rows) => TorchParameters.Parameters(layer, source, name, new[] { rows, _options.LLMDim });
        Embedding(_textEmbedding, "text_embedding.word_embeddings.weight", _options.TextVocabularySize + 1);
        Embedding(_maskEmbedding, "mask_embedding", _options.MaxMaskSpans);
        for (int q = 0; q < _audioEmbeddings.Length; q++)
            Embedding(_audioEmbeddings[q], $"audio_embedding.{q}.word_embeddings.weight", _options.AudioVocabularySize);
        _textPosition.SetParameters(new Vector<T>(source.Read("text_positional_embedding.alpha", new[] { 1 }).Select(VoiceCraftOps<T>.NumOps.FromDouble)));
        _audioPosition.SetParameters(new Vector<T>(source.Read("audio_positional_embedding.alpha", new[] { 1 }).Select(VoiceCraftOps<T>.NumOps.FromDouble)));
        for (int i = 0; i < _blocks.Length; i++) _blocks[i].LoadTorchWeights(source, $"decoder.layers.{i}.", _options.FeedForwardDim);
        TorchParameters.LayerNorm(_engine, _finalNorm, source, "decoder.norm.weight", "decoder.norm.bias");
        for (int q = 0; q < _predictionHeads.Length; q++)
        {
            var head = _predictionHeads[q];
            TorchParameters.Linear(_engine, head.Hidden, _options.LLMDim, _options.CodebookSize / 2, source,
                $"predict_layer.{q}.0.weight", $"predict_layer.{q}.0.bias");
            TorchParameters.Linear(_engine, head.Output, _options.CodebookSize / 2, _options.AudioVocabularySize, source,
                $"predict_layer.{q}.2.weight", $"predict_layer.{q}.2.bias");
        }
    }

    /// <summary>Returns one <c>[audioRows, audioVocabulary]</c> tensor per codebook.</summary>
    internal Tensor<T>[] Forward(IReadOnlyList<int> text, IReadOnlyList<int[]> audioRows,
        IReadOnlyList<int>? maskIds = null)
    {
        if (text.Count == 0) throw new ArgumentException("VoiceCraft needs at least one phone.", nameof(text));
        if (audioRows.Count == 0) throw new ArgumentException("VoiceCraft needs at least one audio row.", nameof(audioRows));
        var textHidden = _textEmbedding.Forward(VoiceCraftOps<T>.Ids(text));
        if (_textDropout is not null) textHidden = _textDropout.Forward(textHidden);
        textHidden = _textPosition.Forward(textHidden);
        if (_textPositionDropout is not null) textHidden = _textPositionDropout.Forward(textHidden);

        var rows = new List<Tensor<T>>(audioRows.Count);
        for (int row = 0; row < audioRows.Count; row++)
        {
            int mask = maskIds is not null && row < maskIds.Count ? maskIds[row] : -1;
            Tensor<T> embedded;
            if (mask >= 0)
                embedded = _maskEmbedding.Forward(VoiceCraftOps<T>.Ids(new[] { MathPolyfill.Clamp(mask, 0, _options.MaxMaskSpans - 1) }));
            else
            {
                if (audioRows[row].Length != _options.NumCodebooks)
                    throw new ArgumentException("Every delayed row must contain one token per codebook.", nameof(audioRows));
                embedded = _audioEmbeddings[0].Forward(VoiceCraftOps<T>.Ids(new[] { audioRows[row][0] }));
                for (int q = 1; q < _options.NumCodebooks; q++)
                    embedded = _engine.TensorAdd(embedded,
                        _audioEmbeddings[q].Forward(VoiceCraftOps<T>.Ids(new[] { audioRows[row][q] })));
            }
            rows.Add(embedded);
        }
        var audioHidden = rows.Count == 1 ? rows[0] : _engine.TensorConcatenate(rows.ToArray(), 0);
        if (_audioDropout is not null) audioHidden = _audioDropout.Forward(audioHidden);
        audioHidden = _audioPosition.Forward(audioHidden);
        if (_audioPositionDropout is not null) audioHidden = _audioPositionDropout.Forward(audioHidden);

        var hidden = _engine.TensorConcatenate(new[] { textHidden, audioHidden }, 0);
        var causal = VoiceCraftOps<T>.CausalMask(hidden.Shape[0]);
        foreach (var block in _blocks) hidden = block.Forward(hidden, causal);
        hidden = _finalNorm.Forward(hidden);
        hidden = _engine.TensorSlice(hidden,
            new[] { text.Count, 0 }, new[] { audioRows.Count, _options.LLMDim });
        return _predictionHeads.Select(head => head.Output.Forward(ActivationHelper.ExactGelu(head.Hidden.Forward(hidden), _engine))).ToArray();
    }

    internal int[,] Generate(IReadOnlyList<int> text, IReadOnlyList<int[]> prefixRows,
        IReadOnlyList<int>? prefixMasks, int minimumFrames, int maximumFrames, Random random,
        bool endOfUtterance = false)
    {
        var sequence = GenerateDelayed(text, prefixRows, prefixMasks, minimumFrames, maximumFrames, random, endOfUtterance);
        var result = new int[_options.NumCodebooks, sequence.Frames];
        for (int frame = 0; frame < sequence.Frames; frame++)
            for (int q = 0; q < _options.NumCodebooks; q++)
                result[q, frame] = sequence.Rows[frame + q][q];
        return result;
    }

    /// <summary>Generates the complete delayed stream, including end markers and the codebook flush.</summary>
    internal (IReadOnlyList<int[]> Rows, int Frames) GenerateDelayed(IReadOnlyList<int> text,
        IReadOnlyList<int[]> prefixRows, IReadOnlyList<int>? prefixMasks,
        int minimumFrames, int maximumFrames, Random random, bool endOfUtterance = false)
    {
        int nq = _options.NumCodebooks;
        int endToken = endOfUtterance ? _options.EndOfUtteranceToken : _options.EndOfGenerationToken;
        var context = prefixRows.Select(row => (int[])row.Clone()).ToList();
        var masks = prefixMasks?.ToList() ?? Enumerable.Repeat(-1, context.Count).ToList();
        if (context.Count == 0)
        {
            context.Add(Enumerable.Repeat(_options.EmptyToken, nq).ToArray());
            masks.Add(-1);
        }
        var generated = new List<int[]>();
        int? endFrame = null;
        int silenceCount = 0, previous = -1;
        for (int step = 0; step < maximumFrames + nq; step++)
        {
            var logits = Forward(text, context, masks);
            var row = new int[nq];
            bool flushing = endFrame.HasValue;
            for (int q = 0; q < nq; q++)
            {
                if (!flushing && step < q) { row[q] = _options.EmptyToken; continue; }
                int frame = step - q;
                if (endFrame.HasValue && frame >= endFrame.Value)
                {
                    // Each codebook emits its end marker once. Completed
                    // codebooks feed EMPTY while later codebooks finish.
                    row[q] = frame == endFrame.Value ? endToken : _options.EmptyToken;
                    continue;
                }
                // The author's TTS gate suppresses EOS at cur_num_gen <= sr/5;
                // editing has no corresponding initial-duration gate.
                bool allowEnd = q == 0 && (endOfUtterance ? frame > minimumFrames : frame >= minimumFrames);
                int token = Sample(logits[q], logits[q].Shape[0] - 1, allowEnd, random,
                    q == 0 ? previous : -1, q == 0 ? silenceCount : 0, endToken);
                if (q == 0 && frame >= maximumFrames) token = endToken;
                if (q == 0)
                {
                    silenceCount = token == previous && _options.SilenceTokens.Contains(token)
                        ? silenceCount + 1 : 0;
                    previous = token;
                    if (token == endToken) endFrame = frame;
                }
                row[q] = token;
            }
            generated.Add(row);
            context.Add(row);
            masks.Add(-1);
            if (endFrame.HasValue && step >= endFrame.Value + nq - 1) break;
        }
        return (generated, MathPolyfill.Clamp(endFrame ?? maximumFrames, 0, maximumFrames));
    }

    private int Sample(Tensor<T> logits, int row, bool allowEnd, Random random,
        int previous, int silenceCount, int endToken)
    {
        double Adjust(int id)
        {
            double value = VoiceCraftOps<T>.NumOps.ToDouble(logits[row, id]);
            if (_options.StopSilenceRepetition > 0 && id == previous
                && _options.SilenceTokens.Contains(previous) && silenceCount > _options.StopSilenceRepetition)
            {
                double penalty = silenceCount - (_options.StopSilenceRepetition - 1);
                value = value < 0 ? value * penalty : value / penalty;
            }
            return value / _options.Temperature;
        }
        var candidates = new List<(int Id, double Logit)>(_options.CodebookSize + 1);
        for (int id = 0; id < _options.CodebookSize; id++) candidates.Add((id, Adjust(id)));
        double bestCodec = candidates.Max(x => x.Logit);
        double endLogit = allowEnd ? Adjust(endToken) : double.NegativeInfinity;
        if (allowEnd) candidates.Add((endToken, endLogit));
        // The original helper also stops when the adjusted end logit is the
        // argmax, even if the categorical draw selected another token.
        bool argmaxEnd = allowEnd && endLogit > bestCodec;
        candidates = candidates.OrderByDescending(x => x.Logit).ThenBy(x => x.Id).ToList();
        if (_options.TopK > 0 && candidates.Count > _options.TopK)
        {
            double threshold = candidates[_options.TopK - 1].Logit;
            candidates.RemoveAll(x => x.Logit < threshold);
        }
        double max = candidates[0].Logit, total = candidates.Sum(x => Math.Exp(x.Logit - max));
        if (_options.TopP < 1)
        {
            double cumulative = 0;
            int keep = candidates.Count;
            for (int i = 0; i < candidates.Count; i++)
            {
                cumulative += Math.Exp(candidates[i].Logit - max) / total;
                if (cumulative > _options.TopP) { keep = i + 1; break; }
            }
            if (keep < candidates.Count) candidates.RemoveRange(keep, candidates.Count - keep);
            total = candidates.Sum(x => Math.Exp(x.Logit - max));
        }
        double draw = random.NextDouble() * total, running = 0;
        int selected = candidates[^1].Id;
        foreach (var candidate in candidates)
        {
            running += Math.Exp(candidate.Logit - max);
            if (draw <= running) { selected = candidate.Id; break; }
        }
        return argmaxEnd ? endToken : selected;
    }
}
