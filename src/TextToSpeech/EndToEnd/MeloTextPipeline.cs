using System.Linq;
using System.Text.RegularExpressions;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tokenization.Algorithms;
using AiDotNet.TextToSpeech.FrontEnd;

namespace AiDotNet.TextToSpeech.EndToEnd;

/// <summary>The languages MeloTTS's checkpoints are trained for (one per checkpoint).</summary>
public enum MeloLanguage
{
    /// <summary>English (MeloTTS-English; BERT: bert-base-uncased).</summary>
    English,

    /// <summary>Mandarin Chinese (BERT: hfl/chinese-roberta-wwm-ext-large).</summary>
    Chinese,

    /// <summary>Mandarin with English words (MeloTTS-Chinese; BERT: bert-base-multilingual-uncased).</summary>
    ChineseMixEnglish,

    /// <summary>Japanese (BERT: tohoku-nlp/bert-base-japanese-v3).</summary>
    Japanese,

    /// <summary>Korean (BERT: kykim/bert-kor-base).</summary>
    Korean,

    /// <summary>Spanish (BERT: dccuchile/bert-base-spanish-wwm-uncased).</summary>
    Spanish,

    /// <summary>French (BERT: dbmdz/bert-base-french-europeana-cased).</summary>
    French,
}

/// <summary>One utterance as MeloTTS's model reads it (<c>get_text_for_tts_infer</c> before the BERT features).</summary>
internal sealed record MeloText(string Normalized, int[] Symbols, int[] Tones, int[] Languages, int[] PhonesPerPiece);

/// <summary>
/// MeloTTS's text side: <c>clean_text</c> (the language's normalization and G2P), <c>cleaned_text_to_sequence</c>,
/// the blanks <c>add_blank</c> intersperses (with the phones-per-piece doubled and the first piece given the leading
/// blank), the BERT token ids, and <c>split_utils.split_sentence</c>.
/// </summary>
internal static class MeloTextPipeline
{
    public static string Code(MeloLanguage language) => language switch
    {
        MeloLanguage.English => "EN",
        MeloLanguage.Chinese => "ZH",
        MeloLanguage.ChineseMixEnglish => "ZH_MIX_EN",
        MeloLanguage.Japanese => "JP",
        MeloLanguage.Korean => "KR",
        MeloLanguage.Spanish => "ES",
        MeloLanguage.French => "FR",
        _ => throw new ArgumentOutOfRangeException(nameof(language)),
    };

    /// <summary>Whether the language's BERT features fill the 1024-wide slot (Chinese) or the 768-wide one.</summary>
    public static bool UsesWideBert(MeloLanguage language) => language == MeloLanguage.Chinese;

    private static (string Normalized, List<string> Phones, List<int> Tones, List<int> PhonesPerPiece) Clean(MeloLanguage language, string text)
    {
        switch (language)
        {
            case MeloLanguage.English:
            {
                string n = MeloEnglishFrontEnd.Default.Normalize(text);
                var (p, t, w) = MeloEnglishFrontEnd.Default.G2P(n);
                return (n, p, t, w);
            }
            case MeloLanguage.Chinese:
            {
                string n = MeloChineseFrontEnd.Default.Normalize(text);
                var (p, t, w) = MeloChineseFrontEnd.Default.G2P(n);
                return (n, p, t, w);
            }
            case MeloLanguage.ChineseMixEnglish:
            {
                string n = MeloChineseMixFrontEnd.Default.Normalize(text);
                var (p, t, w) = MeloChineseMixFrontEnd.Default.G2P(n);
                return (n, p, t, w);
            }
            case MeloLanguage.Japanese:
            {
                string n = MeloJapaneseFrontEnd.Default.Normalize(text);
                var (p, t, w) = MeloJapaneseFrontEnd.Default.G2P(n);
                return (n, p, t, w);
            }
            case MeloLanguage.Korean:
            {
                string n = MeloKoreanFrontEnd.Default.Normalize(text);
                var (p, t, w) = MeloKoreanFrontEnd.Default.G2P(n);
                return (n, p, t, w);
            }
            case MeloLanguage.Spanish:
            {
                string n = MeloSpanishFrontEnd.Default.Normalize(text);
                var (p, t, w) = MeloSpanishFrontEnd.Default.G2P(n);
                return (n, p, t, w);
            }
            case MeloLanguage.French:
            {
                string n = MeloFrenchFrontEnd.Default.Normalize(text);
                var (p, t, w) = MeloFrenchFrontEnd.Default.G2P(n);
                return (n, p, t, w);
            }
            default:
                throw new ArgumentOutOfRangeException(nameof(language));
        }
    }

    /// <summary>The model's input for one sentence, blanks interspersed (MeloTTS's configs set add_blank).</summary>
    public static MeloText Prepare(MeloLanguage language, string text, bool addBlank = true)
    {
        var (normalized, phones, tones, perPiece) = Clean(language, text);
        var (symbols, toneIds, languages) = MeloSymbols.ToSequence(phones, tones, Code(language));
        var word2ph = perPiece.ToArray();
        if (addBlank)
        {
            symbols = MeloSymbols.Intersperse(symbols);
            toneIds = MeloSymbols.Intersperse(toneIds);
            languages = MeloSymbols.Intersperse(languages);
            for (int i = 0; i < word2ph.Length; i++) word2ph[i] *= 2;
            if (word2ph.Length > 0) word2ph[0] += 1;
        }
        return new MeloText(normalized, symbols, toneIds, languages, word2ph);
    }

    private static readonly Lazy<WordPieceTokenizer> EnglishTokenizer = new(WordPieceTokenizer.BertBaseUncased, isThreadSafe: true);
    private static readonly Lazy<WordPieceTokenizer> ChineseTokenizer = new(() => WordPieceTokenizer.BertBaseChinese(), isThreadSafe: true);
    private static readonly Lazy<WordPieceTokenizer> MultilingualTokenizer = new(WordPieceTokenizer.BertBaseMultilingualUncased, isThreadSafe: true);
    private static readonly Lazy<WordPieceTokenizer> JapaneseTokenizer = new(WordPieceTokenizer.BertBaseJapaneseV3, isThreadSafe: true);
    private static readonly Lazy<WordPieceTokenizer> KoreanTokenizer = new(WordPieceTokenizer.BertKorBase, isThreadSafe: true);
    private static readonly Lazy<WordPieceTokenizer> SpanishTokenizer = new(WordPieceTokenizer.BertBaseSpanishWwmUncased, isThreadSafe: true);
    private static readonly Lazy<WordPieceTokenizer> FrenchTokenizer = new(WordPieceTokenizer.BertBaseFrenchEuropeanaCased, isThreadSafe: true);

    /// <summary>
    /// <c>tokenizer(norm_text)["input_ids"]</c> of the language's BERT: [CLS], the word pieces' ids, [SEP]
    /// (Japanese splits words with MeCab first, as BertJapaneseTokenizer does).
    /// </summary>
    public static int[] BertTokenIds(MeloLanguage language, string normalized)
    {
        WordPieceTokenizer tokenizer = language switch
        {
            MeloLanguage.English => EnglishTokenizer.Value,
            MeloLanguage.Chinese => ChineseTokenizer.Value,
            MeloLanguage.ChineseMixEnglish => MultilingualTokenizer.Value,
            MeloLanguage.Japanese => JapaneseTokenizer.Value,
            MeloLanguage.Korean => KoreanTokenizer.Value,
            MeloLanguage.Spanish => SpanishTokenizer.Value,
            MeloLanguage.French => FrenchTokenizer.Value,
            _ => throw new ArgumentOutOfRangeException(nameof(language)),
        };
        var pieces = language == MeloLanguage.Japanese ? MeloJapaneseFrontEnd.Default.WordPieces(normalized) : tokenizer.Tokenize(normalized);
        var tokens = new List<string> { "[CLS]" };
        tokens.AddRange(pieces);
        tokens.Add("[SEP]");
        return tokenizer.ConvertTokensToIds(tokens).ToArray();
    }

    /// <summary>The shape of the language's BERT (each released checkpoint's config.json).</summary>
    public static BertConfiguration BertShape(MeloLanguage language) => language switch
    {
        MeloLanguage.English => new BertConfiguration { VocabSize = 30522 },
        MeloLanguage.Chinese => new BertConfiguration { VocabSize = 21128, Hidden = 1024, Layers = 24, Heads = 16, Intermediate = 4096 },
        MeloLanguage.ChineseMixEnglish => new BertConfiguration { VocabSize = 105879 },
        MeloLanguage.Japanese => new BertConfiguration { VocabSize = 32768 },
        MeloLanguage.Korean => new BertConfiguration { VocabSize = 42000 },
        MeloLanguage.Spanish => new BertConfiguration { VocabSize = 31002 },
        MeloLanguage.French => new BertConfiguration { VocabSize = 32000 },
        _ => throw new ArgumentOutOfRangeException(nameof(language)),
    };

    // ------------------------------------------------------------------ split_utils.py

    /// <summary><c>split_sentence(text, language_str)</c>: Latin-script languages through txtsplit, the rest through
    /// the punctuation splitter that merges short pieces.</summary>
    public static List<string> SplitSentence(string text, MeloLanguage language, int minLength = 10) =>
        language is MeloLanguage.English or MeloLanguage.French or MeloLanguage.Spanish
            ? SplitLatin(text)
            : SplitOther(text, minLength);

    private static readonly Regex LatinAux = new("[<>()\\[\\]\"«»]+", RegexOptions.CultureInvariant);

    private static List<string> SplitLatin(string text)
    {
        text = Regex.Replace(text, "[。！？；]", ".");
        text = Regex.Replace(text, "[，]", ",");
        text = Regex.Replace(text, "[“”]", "\"");
        text = Regex.Replace(text, "[‘’]", "'");
        text = LatinAux.Replace(text, "");
        return TxtSplit(text, 256, 512).Select(PythonText.Strip).Where(s => s.Length > 0).ToList();
    }

    private static readonly string Ws = "[" + PythonText.WhitespaceClass + "]";

    // txtsplit(text, desired_length, max_length): chunks of about the desired length that keep sentences whole.
    private static List<string> TxtSplit(string input, int desiredLength, int maxLength)
    {
        input = Regex.Replace(input, "\n\n+", "\n");
        input = Regex.Replace(input, Ws + "+", " ");
        input = Regex.Replace(input, "([,.?!])", "$1 ");
        input = Regex.Replace(input, Ws + "+", " ");
        var text = ChineseText.Codepoints(input);
        var rv = new List<string>();
        bool inQuote = false;
        var current = new List<string>();
        var splitPositions = new List<int>();
        int pos = -1, endPos = text.Count - 1;

        string Seek(int delta)
        {
            for (int i = 0; i < Math.Abs(delta); i++)
            {
                if (delta < 0)
                {
                    pos--;
                    if (current.Count > 0) current.RemoveAt(current.Count - 1);
                }
                else
                {
                    pos++;
                    current.Add(text[pos]);
                }
                if (text[pos] == "\"") inQuote = !inQuote;
            }
            return text[pos];
        }
        string Peek(int delta)
        {
            int p = pos + delta;
            return p < endPos && p >= 0 ? text[p] : "";
        }
        void Commit()
        {
            rv.Add(string.Concat(current));
            current.Clear();
            splitPositions.Clear();
        }
        static bool In(string c, string set) => set.Contains(c);   // Python's `c in set` ("" is in every string)

        while (pos < endPos)
        {
            string c = Seek(1);
            if (current.Count >= maxLength)
            {
                if (splitPositions.Count > 0 && current.Count > desiredLength / 2.0)
                {
                    int d = pos - splitPositions[splitPositions.Count - 1];
                    Seek(-d);
                }
                else
                {
                    while (!In(c, "!?.\n ") && pos > 0 && current.Count > desiredLength) c = Seek(-1);
                }
                Commit();
            }
            else if (!inQuote && (In(c, "!?\n") || (In(c, ".,") && In(Peek(1), "\n "))))
            {
                while (pos < text.Count - 1 && current.Count < maxLength && In(Peek(1), "!?.")) c = Seek(1);
                splitPositions.Add(pos);
                if (current.Count >= desiredLength) Commit();
            }
            else if (inQuote && Peek(1) == "\"" && In(Peek(2), "\n "))
            {
                Seek(2);
                splitPositions.Add(pos);
            }
        }
        rv.Add(string.Concat(current));
        var punctuationOnly = new Regex("^[" + PythonText.WhitespaceClass + "\\.,;:!?]*$", RegexOptions.CultureInvariant);
        return rv.Select(PythonText.Strip).Where(s => s.Length > 0 && !punctuationOnly.IsMatch(s)).ToList();
    }

    private static List<string> SplitOther(string text, int minLength)
    {
        text = Regex.Replace(text, "[。！？；]", ".");
        text = Regex.Replace(text, "[，]", ",");
        text = Regex.Replace(text, "[\n\t ]+", " ");
        text = Regex.Replace(text, "([,.!?;])", "$1 $$#!");
        var sentences = text.Split(new[] { "$#!" }, StringSplitOptions.None).Select(PythonText.Strip).ToList();
        if (sentences[sentences.Count - 1].Length == 0) sentences.RemoveAt(sentences.Count - 1);
        var merged = new List<string>();
        var pending = new List<string>();
        int count = 0;
        for (int i = 0; i < sentences.Count; i++)
        {
            pending.Add(sentences[i]);
            count += ChineseText.Codepoints(sentences[i]).Count;
            if (count > minLength || i == sentences.Count - 1)
            {
                count = 0;
                merged.Add(string.Join(" ", pending));
                pending.Clear();
            }
        }
        // merge_short_sentences_zh
        var output = new List<string>();
        foreach (var s in merged)
        {
            if (output.Count > 0 && ChineseText.Codepoints(output[output.Count - 1]).Count <= 2) output[output.Count - 1] = output[output.Count - 1] + " " + s;
            else output.Add(s);
        }
        if (output.Count >= 2 && ChineseText.Codepoints(output[output.Count - 1]).Count <= 2)
        {
            output[output.Count - 2] = output[output.Count - 2] + " " + output[output.Count - 1];
            output.RemoveAt(output.Count - 1);
        }
        return output;
    }
}

/// <summary>
/// The BERT features MeloTTS conditions on: the language's BERT (frozen, as MeloTTS only ever runs it under no_grad),
/// its <c>hidden_states[-3]</c> for the sentence, each piece's row repeated for its phones.
/// </summary>
internal sealed class MeloBertFeatures<T>
{
    private readonly MeloLanguage _language;

    public MeloBertFeatures(IEngine engine, List<LayerBase<T>> layers, MeloLanguage language, BertConfiguration shape)
    {
        _language = language;
        Network = new BertNetwork<T>(engine, layers, shape);
    }

    public BertNetwork<T> Network { get; }

    /// <summary>The features <c>[1, hidden, phones]</c> of a sentence for its (blank-adjusted) phones per piece.</summary>
    public Tensor<T> Features(string normalized, IReadOnlyList<int> phonesPerPiece)
    {
        var ids = MeloTextPipeline.BertTokenIds(_language, normalized);
        // Chinese's get_bert_feature does not assert the lengths; it reads the first rows.
        if (phonesPerPiece.Count > ids.Length || (_language != MeloLanguage.Chinese && phonesPerPiece.Count != ids.Length))
            throw new ArgumentException($"BERT reads {ids.Length} tokens but the front end gave {phonesPerPiece.Count} pieces.", nameof(phonesPerPiece));
        var input = new int[1, ids.Length];
        for (int i = 0; i < ids.Length; i++) input[0, i] = ids[i];
        bool wasTraining = Network.Training;
        Network.Training = false;
        Tensor<T> hidden;
        try
        {
            using var _ = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
            hidden = Network.Encode(input, layerCount: Network.Configuration.Layers - 2);   // hidden_states[-3]
        }
        finally
        {
            Network.Training = wasTraining;
        }
        int width = Network.Configuration.Hidden, phones = phonesPerPiece.Sum();
        var features = new Tensor<T>(new[] { 1, width, phones });
        int column = 0;
        for (int i = 0; i < phonesPerPiece.Count; i++)
            for (int r = 0; r < phonesPerPiece[i]; r++, column++)
                for (int h = 0; h < width; h++) features[0, h, column] = hidden[0, i, h];
        return features;
    }
}
