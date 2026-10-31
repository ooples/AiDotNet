using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.Tokenization.Algorithms;

namespace AiDotNet.TextToSpeech.FrontEnd;

/// <summary>The shape of a G2PW model and its conditional-weighting terms (g2pW's <c>config.py</c>).</summary>
internal sealed record G2pwConfiguration
{
    /// <summary>The BERT encoder (bert-base-chinese for the released model).</summary>
    public BertConfiguration Bert { get; init; } = new() { VocabSize = 21128 };

    /// <summary>The phoneme labels (1,305 bopomofo readings in G2PW 1.1).</summary>
    public int Labels { get; init; } = 1305;

    /// <summary>The polyphonic characters (3,582 in G2PW 1.1).</summary>
    public int Chars { get; init; } = 3582;

    /// <summary>The part-of-speech tags (11).</summary>
    public int PosTags { get; init; } = 11;

    /// <summary><c>param_conditional['bias']</c>.</summary>
    public bool Bias { get; init; } = true;

    /// <summary><c>param_conditional['char-linear']</c>.</summary>
    public bool CharLinear { get; init; } = true;

    /// <summary><c>param_conditional['pos-linear']</c>.</summary>
    public bool PosLinear { get; init; }

    /// <summary><c>param_conditional['char+pos-second']</c>.</summary>
    public bool SecondOrder { get; init; } = true;

    /// <summary><c>affect_location</c>: true weights the softmax (<c>'softmax'</c>), false scales the hidden state
    /// (<c>'emb'</c>).</summary>
    public bool AffectSoftmax { get; init; } = true;
}

/// <summary>
/// G2PW (Chen et al. 2022, "g2pW: A Conditional Weighted Softmax BERT for Polyphone Disambiguation in Mandarin";
/// GitYCC/g2pW <c>g2pw/module.py</c>, Apache-2.0) in inference: BERT's hidden state at the polyphonic character, its
/// part of speech predicted by the joint POS classifier, the conditional weights <c>σ(bias + char + char×POS)</c> over
/// the character's candidate readings, and the weighted softmax of the classifier's logits.
/// </summary>
internal sealed class G2pwModel<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    public G2pwModel(IEngine engine, List<LayerBase<T>> layers, G2pwConfiguration configuration)
    {
        _engine = engine;
        Configuration = configuration;
        var c = configuration;
        Bert = new BertNetwork<T>(engine, layers, c.Bert);
        int target = c.AffectSoftmax ? c.Labels : c.Bert.Hidden;
        DenseLayer<T> Linear(int outputs)
        {
            var layer = new DenseLayer<T>(outputs, (IActivationFunction<T>)new IdentityActivation<T>());
            layers.Add(layer);
            return layer;
        }
        BiasFreeLinearLayer<T> Table(int rows)
        {
            var holder = new BiasFreeLinearLayer<T>(1, rows * target);
            layers.Add(holder);
            return holder;
        }
        Classifier = Linear(c.Labels);
        PosClassifier = Linear(c.PosTags);
        if (c.Bias) DescriptorBias = Table(1);
        if (c.CharLinear) CharDescriptor = Table(c.Chars);
        if (c.PosLinear) PosDescriptor = Table(c.PosTags);
        if (c.SecondOrder) SecondOrderDescriptor = Table(c.Chars * c.PosTags);
    }

    public G2pwConfiguration Configuration { get; }

    public BertNetwork<T> Bert { get; }

    public DenseLayer<T> Classifier { get; }

    /// <summary>The joint POS classifier (<c>param_pos['pos_joint_training']</c>), whose prediction indexes the
    /// POS-dependent terms in inference.</summary>
    public DenseLayer<T> PosClassifier { get; }

    public BiasFreeLinearLayer<T>? DescriptorBias { get; }

    public BiasFreeLinearLayer<T>? CharDescriptor { get; }

    public BiasFreeLinearLayer<T>? PosDescriptor { get; }

    public BiasFreeLinearLayer<T>? SecondOrderDescriptor { get; }

    private int Target => Configuration.AffectSoftmax ? Configuration.Labels : Configuration.Bert.Hidden;

    private Tensor<T> Rows(BiasFreeLinearLayer<T> holder, int rows, int[] ids)
    {
        var index = new Tensor<int>(new[] { ids.Length });
        for (int i = 0; i < ids.Length; i++)
        {
            if (ids[i] < 0 || ids[i] >= rows) throw new ArgumentOutOfRangeException(nameof(ids), $"Id {ids[i]} is outside [0, {rows}).");
            index[i] = ids[i];
        }
        var table = _engine.Reshape(Lfm2Network<T>.Paged(holder.Weights), new[] { rows, Target });
        return _engine.TensorIndexSelect(table, index, 0);
    }

    /// <summary>
    /// <c>G2PW.forward</c> in eval mode for queries into one tokenized text: the probabilities <c>[N, Labels]</c> of each
    /// query's readings.
    /// </summary>
    /// <param name="inputIds">[CLS] tokens [SEP].</param>
    /// <param name="positions">Each query's token position.</param>
    /// <param name="charIds">Each query character's index among the polyphonic characters.</param>
    /// <param name="phonemeMasks">Each query's candidate readings (1) among the labels.</param>
    public Tensor<T> Probabilities(int[] inputIds, int[] positions, int[] charIds, double[][] phonemeMasks, double eps = 1e-6)
    {
        var c = Configuration;
        int n = positions.Length, length = inputIds.Length;
        var ids = new int[1, length];
        for (int j = 0; j < length; j++) ids[0, j] = inputIds[j];
        var hidden = _engine.Reshape(Bert.Encode(ids), new[] { length, c.Bert.Hidden });
        var positionIndex = new Tensor<int>(new[] { n });
        for (int i = 0; i < n; i++) positionIndex[i] = positions[i];
        var selected = _engine.TensorIndexSelect(hidden, positionIndex, 0);

        var mask = new Tensor<T>(new[] { n, c.Labels });
        for (int i = 0; i < n; i++)
        {
            if (phonemeMasks[i].Length != c.Labels) throw new ArgumentException($"Each phoneme mask has {c.Labels} entries.", nameof(phonemeMasks));
            for (int j = 0; j < c.Labels; j++) mask[i, j] = NumOps.FromDouble(phonemeMasks[i][j]);
        }

        // The POS-dependent terms index by the joint classifier's prediction (teacher POS only in training).
        int[] predictedPos = new int[n];
        if (c.PosLinear || c.SecondOrder)
        {
            var posLogits = PosClassifier.Forward(selected);
            for (int i = 0; i < n; i++)
            {
                int best = 0;
                for (int j = 1; j < c.PosTags; j++)
                    if (NumOps.GreaterThan(posLogits[i, j], posLogits[i, best])) best = j;
                predictedPos[i] = best;
            }
        }
        Tensor<T>? affect = null;
        void AddTerm(Tensor<T> term) => affect = affect is null ? term : _engine.TensorAdd(affect, term);
        if (DescriptorBias is not null) AddTerm(Rows(DescriptorBias, 1, new int[n]));
        if (CharDescriptor is not null) AddTerm(Rows(CharDescriptor, c.Chars, charIds));
        if (PosDescriptor is not null) AddTerm(Rows(PosDescriptor, c.PosTags, predictedPos));
        if (SecondOrderDescriptor is not null)
            AddTerm(Rows(SecondOrderDescriptor, c.Chars * c.PosTags, charIds.Select((id, i) => id * c.PosTags + predictedPos[i]).ToArray()));

        // sum() of no terms is 0.
        affect ??= new Tensor<T>(new[] { n, Target });
        Tensor<T> weights = mask;
        if (c.AffectSoftmax) weights = _engine.TensorMultiply(mask, _engine.Sigmoid(affect!));
        else selected = _engine.TensorMultiply(selected, affect!);
        var logits = Classifier.Forward(selected);

        // _weighted_softmax: exp(logits − max) · weights, normalized, clamped to [eps, 1 − eps].
        var max = _engine.ReduceMax(logits, new[] { 1 }, keepDims: true, out _);
        var exp = _engine.TensorMultiply(_engine.TensorExp(_engine.TensorSubtract(logits, _engine.TensorBroadcastTo(max, new[] { n, c.Labels }))), weights);
        var norm = _engine.ReduceSum(exp, new[] { 1 }, keepDims: true);
        var probabilities = _engine.TensorDivide(exp, _engine.TensorBroadcastTo(norm, new[] { n, c.Labels }));
        return _engine.TensorClamp(probabilities, NumOps.FromDouble(eps), NumOps.FromDouble(1 - eps));
    }

    /// <summary>
    /// Loads a G2PW state dict: <c>bert.*</c>, <c>classifier</c>, <c>pos_classifier</c> and the descriptors (from the
    /// released ONNX model, read with the weights PyTorch's exporter left unnamed named after their biases).
    /// </summary>
    public void Load(Func<string, int[], double[]> read)
    {
        var c = Configuration;
        int h = c.Bert.Hidden;
        Bert.LoadTorchWeights(read, "bert.");
        TorchParameters.Linear(_engine, Classifier, h, c.Labels, read("classifier.weight", new[] { c.Labels, h }), read("classifier.bias", new[] { c.Labels }));
        TorchParameters.Linear(_engine, PosClassifier, h, c.PosTags, read("pos_classifier.weight", new[] { c.PosTags, h }),
            read("pos_classifier.bias", new[] { c.PosTags }));
        void Table(BiasFreeLinearLayer<T>? holder, string name, int rows)
        {
            if (holder is not null) TorchParameters.Linear(holder, 1, rows * Target, read($"{name}.weight", new[] { rows, Target }));
        }
        Table(DescriptorBias, "descriptor_bias", 1);
        Table(CharDescriptor, "char_descriptor", c.Chars);
        Table(PosDescriptor, "pos_descriptor", c.PosTags);
        Table(SecondOrderDescriptor, "second_order_descriptor", c.Chars * c.PosTags);
    }
}

/// <summary>
/// GPT-SoVITS's G2PW converter (<c>text/g2pw/onnx_api.py</c>, from PaddleSpeech and g2pW; Apache-2.0) with
/// <c>enable_non_tradional_chinese</c> and <c>style="pinyin"</c>: each sentence converted to Taiwan Traditional by OpenCC,
/// every polyphonic character predicted by G2PW in the sentence trimmed to 16 characters around the polyphonic ones,
/// monophonic characters read from G2PW's table, and every other character read by pypinyin (TONE3, neutral tone 5) from
/// the sentence converted back to Simplified.
/// </summary>
internal sealed class G2pwConverter<T>
{
    // onnx_api's non_polyphonic and non_monophonic.
    private static readonly string[] NonPolyphonic =
    {
        "一", "不", "和", "咋", "嗲", "剖", "差", "攢", "倒", "難", "奔", "勁",
        "拗", "肖", "瘙", "誒", "泊", "听", "噢",
    };
    private static readonly string[] NonMonophonic = { "似", "攢" };

    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly G2pwModel<T> _model;
    private readonly WordPieceTokenizer _tokenizer = WordPieceTokenizer.BertBaseChinese();
    private readonly List<string> _labels;
    private readonly Dictionary<string, List<int>> _charToPhonemes = new(StringComparer.Ordinal);
    private readonly Dictionary<string, int> _charToId = new(StringComparer.Ordinal);
    private readonly HashSet<string> _polyphonic;
    private readonly Dictionary<string, string> _monophonic = new(StringComparer.Ordinal);
    private readonly Dictionary<string, string> _bopomofoToPinyin;

    /// <summary>The trim around the polyphonic characters (<c>g2pw_polyphonic_context_chars</c>; 0 keeps the whole
    /// sentence).</summary>
    public int ContextChars { get; set; } = 16;

    public G2pwConverter(G2pwModel<T> model)
    {
        _model = model;
        var data = GptSoVitsChineseData.Instance;
        // get_phoneme_labels (use_char_phoneme is off): the sorted distinct readings, each character's in file order.
        _labels = data.PolyphonicChars.Select(p => p.Bopomofo).Distinct().ToList();
        _labels.Sort(GptSoVitsChineseData.CompareCodepoints);
        var labelIndex = _labels.Select((l, i) => (l, i)).ToDictionary(p => p.l, p => p.i, StringComparer.Ordinal);
        foreach (var (ch, bopomofo) in data.PolyphonicChars)
        {
            if (!_charToPhonemes.TryGetValue(ch, out var list)) _charToPhonemes[ch] = list = new List<int>();
            list.Add(labelIndex[bopomofo]);
        }
        var chars = _charToPhonemes.Keys.ToList();
        chars.Sort(GptSoVitsChineseData.CompareCodepoints);
        for (int i = 0; i < chars.Count; i++) _charToId[chars[i]] = i;
        if (_labels.Count != model.Configuration.Labels || chars.Count != model.Configuration.Chars)
            throw new ArgumentException($"G2PW's tables have {_labels.Count} labels and {chars.Count} characters; the model has " +
                $"{model.Configuration.Labels} and {model.Configuration.Chars}.", nameof(model));
        _polyphonic = new HashSet<string>(chars, StringComparer.Ordinal);
        foreach (var c in NonPolyphonic) _polyphonic.Remove(c);
        foreach (var (ch, bopomofo) in data.MonophonicChars) _monophonic[ch] = bopomofo;
        foreach (var c in NonMonophonic) _monophonic.Remove(c);
        _bopomofoToPinyin = data.BopomofoToPinyin;
    }

    // _convert_bopomofo_to_pinyin
    private string? ToPinyin(string bopomofo)
    {
        char tone = bopomofo[bopomofo.Length - 1];
        if ("12345".IndexOf(tone) < 0) throw new InvalidDataException($"The reading {bopomofo} has no tone.");
        return _bopomofoToPinyin.TryGetValue(bopomofo.Substring(0, bopomofo.Length - 1), out var component) ? component + tone : null;
    }

    /// <summary><c>G2PWOnnxConverter(sentences)</c>: each sentence's per-character TONE3 readings.</summary>
    public List<List<string?>> Predict(List<string> sentences)
    {
        var data = GptSoVitsChineseData.Instance;
        var results = new List<List<string?>>();
        var queries = new List<(List<string> Text, int ModelQuery, int ResultQuery, int Sentence)>();
        for (int s = 0; s < sentences.Count; s++)
        {
            string original = sentences[s];
            string converted = OpenCc.S2Tw.Convert(original);
            if (ChineseText.Codepoints(converted).Count != ChineseText.Codepoints(original).Count)
                throw new InvalidOperationException($"OpenCC changed the length of '{original}'.");
            var sentence = ChineseText.Codepoints(converted);
            var readings = GptSoVitsPinyin.Default.Read(data.ToSimplified(converted));
            var partial = new List<string?>(new string?[sentence.Count]);
            var polyphonic = new List<int>();
            for (int i = 0; i < sentence.Count; i++)
            {
                string ch = sentence[i];
                if (_polyphonic.Contains(ch)) polyphonic.Add(i);
                else if (_monophonic.TryGetValue(ch, out var bopomofo)) partial[i] = ToPinyin(bopomofo);
                else
                {
                    if (i >= readings.Count) throw new ArgumentOutOfRangeException(nameof(sentences), $"pypinyin read fewer items than '{original}' has characters.");
                    partial[i] = readings[i].Tone3;
                }
            }
            if (polyphonic.Count > 0)
            {
                int left = 0, right = sentence.Count;
                if (ContextChars > 0)
                {
                    left = Math.Max(0, polyphonic[0] - ContextChars);
                    right = Math.Min(sentence.Count, polyphonic[polyphonic.Count - 1] + ContextChars + 1);
                }
                var window = sentence.GetRange(left, right - left);
                foreach (int index in polyphonic) queries.Add((window, index - left, index, s));
            }
            results.Add(partial);
        }
        if (queries.Count == 0) return results;

        // One encoding per distinct text (the reference batches them; the predictions are the same).
        foreach (var group in queries.GroupBy(q => string.Concat(q.Text)))
        {
            var items = group.ToList();
            var prepared = items.Select(q => Prepare(q.Text, q.ModelQuery)).ToList();
            foreach (var batch in prepared.Select((p, i) => (p, i)).GroupBy(x => string.Join(",", x.p.InputIds)))
            {
                var members = batch.ToList();
                var first = members[0].p;
                var probabilities = _model.Probabilities(first.InputIds, members.Select(m => m.p.Position).ToArray(),
                    members.Select(m => m.p.CharId).ToArray(), members.Select(m => m.p.Mask).ToArray());
                for (int k = 0; k < members.Count; k++)
                {
                    int best = 0;
                    for (int j = 1; j < _labels.Count; j++)
                        if (NumOps.GreaterThan(probabilities[k, j], probabilities[k, best])) best = j;
                    var query = items[members[k].i];
                    results[query.Sentence][query.ResultQuery] = ToPinyin(_labels[best]);
                }
            }
        }
        return results;
    }

    private const int MaxLength = 512;

    // prepare_onnx_input for one query: the tokens (truncated around the query past 510), its token position, its
    // character's id and candidate readings.
    private (int[] InputIds, int Position, int CharId, double[] Mask) Prepare(List<string> window, int queryId)
    {
        var text = ChineseText.Codepoints(PythonText.Lower(string.Concat(window)));
        var (tokens, textToToken, tokenToText) = TokenizeAndMap(text);
        if (tokens.Count > MaxLength - 2) (text, queryId, tokens, textToToken) = Truncate(text, queryId, tokens, textToToken, tokenToText);
        var processed = new List<string> { "[CLS]" };
        processed.AddRange(tokens);
        processed.Add("[SEP]");
        var ids = processed.Select(t => _tokenizer.Vocabulary.GetTokenId(t)).ToArray();
        string queryChar = text[queryId];
        if (!_charToPhonemes.TryGetValue(queryChar, out var phonemes)) throw new KeyNotFoundException($"'{queryChar}' is not a G2PW character.");
        var mask = new double[_labels.Count];
        foreach (int p in phonemes) mask[p] = 1;
        int? token = textToToken[queryId];
        if (token is null) throw new InvalidOperationException($"'{queryChar}' maps to no token.");
        return (ids, token.Value + 1, _charToId[queryChar], mask);
    }

    // utils.wordize_and_map then tokenize_and_map.
    private (List<string> Tokens, List<int?> TextToToken, List<(int Start, int End)> TokenToText) TokenizeAndMap(List<string> text)
    {
        var words = new List<string>();
        var textToWord = new List<int?>();
        var wordToText = new List<(int, int)>();
        int at = 0;
        static bool IsAsciiAlnum(string c) => c.Length == 1 && (c[0] is >= 'a' and <= 'z' or >= 'A' and <= 'Z' or >= '0' and <= '9');
        while (at < text.Count)
        {
            if (text[at] == " ")
            {
                while (at < text.Count && text[at] == " ") { textToWord.Add(null); at++; }
                continue;
            }
            int end = at;
            while (end < text.Count && IsAsciiAlnum(text[end])) end++;
            if (end == at) end = at + 1;
            int start = textToWord.Count;
            wordToText.Add((start, start + end - at));
            for (int i = at; i < end; i++) textToWord.Add(words.Count);
            words.Add(string.Concat(text.GetRange(at, end - at)));
            at = end;
        }
        var tokens = new List<string>();
        var tokenToText = new List<(int Start, int End)>();
        for (int w = 0; w < words.Count; w++)
        {
            var (wordStart, wordEnd) = wordToText[w];
            var pieces = _tokenizer.Tokenize(words[w]);
            if (pieces.Count == 0 || (pieces.Count == 1 && pieces[0] == "[UNK]"))
            {
                tokenToText.Add((wordStart, wordEnd));
                tokens.Add("[UNK]");
                continue;
            }
            int current = wordStart;
            foreach (var piece in pieces)
            {
                int length = ChineseText.Codepoints(piece.StartsWith("##", StringComparison.Ordinal) ? piece.Substring(2) : piece).Count;
                tokenToText.Add((current, current + length));
                current += length;
                tokens.Add(piece);
            }
        }
        var textToToken = textToWord;
        for (int i = 0; i < tokenToText.Count; i++)
            for (int p = tokenToText[i].Start; p < tokenToText[i].End; p++) textToToken[p] = i;
        return (tokens, textToToken, tokenToText);
    }

    // dataset._truncate: 510 tokens centred on the query's, shifted inside the text.
    private static (List<string>, int, List<string>, List<int?>) Truncate(List<string> text, int queryId, List<string> tokens,
        List<int?> textToToken, List<(int Start, int End)> tokenToText)
    {
        int truncateLength = MaxLength - 2;
        int tokenPosition = textToToken[queryId] ?? throw new InvalidOperationException("The query maps to no token.");
        int tokenStart = tokenPosition - truncateLength / 2;
        int tokenEnd = tokenStart + truncateLength;
        int frontExceed = -tokenStart, backExceed = tokenEnd - tokens.Count;
        if (frontExceed > 0) { tokenStart += frontExceed; tokenEnd += frontExceed; }
        else if (backExceed > 0) { tokenStart -= backExceed; tokenEnd -= backExceed; }
        int start = tokenToText[tokenStart].Start, end = tokenToText[tokenEnd - 1].End;
        var map = textToToken.GetRange(start, end - start).Select(i => i is null ? (int?)null : i - tokenStart).ToList();
        return (text.GetRange(start, end - start), queryId - start, tokens.GetRange(tokenStart, tokenEnd - tokenStart), map);
    }
}
