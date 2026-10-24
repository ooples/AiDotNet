using System.Globalization;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using AiDotNet.Tokenization.Algorithms;

namespace AiDotNet.TextToSpeech.FrontEnd;

/// <summary>
/// pykakasi 2.2.1's legacy converter as MeloTTS configures it (<c>setMode("J", "K")</c>, <c>setMode("H", "K")</c>,
/// <c>getConverter().do(text)</c>): kanji runs read through KAKASI's kanwa dictionary (the first longest entry, its
/// first reading, after variant kanji are normalized) and written in katakana, hiragana written in katakana, katakana
/// and everything else kept. Its quirks are kept: a reading's characters outside the hiragana block are dropped, a
/// long-vowel mark after hiragana repeats the previous kana, and one opening a katakana run becomes "-".
/// </summary>
internal sealed class PykakasiKatakana
{
    private static readonly Lazy<PykakasiKatakana> SharedInstance = new(() => new PykakasiKatakana(), isThreadSafe: true);
    private readonly Dictionary<int, List<(int[] Key, string Reading)>> _kanwa = new();
    private readonly Dictionary<int, int[]> _itaiji = new();

    public static PykakasiKatakana Default => SharedInstance.Value;

    private const int MaxLength = 32;
    private const string LongSymbols = "ー―−ｰ";

    private static int[] CodePoints(string s)
    {
        var result = new List<int>(s.Length);
        for (int i = 0; i < s.Length; i++)
        {
            int c = char.ConvertToUtf32(s, i);
            if (char.IsHighSurrogate(s[i])) i++;
            result.Add(c);
        }
        return result.ToArray();
    }

    private static string Text(IEnumerable<int> codePoints) => string.Concat(codePoints.Select(char.ConvertFromUtf32));

    private PykakasiKatakana()
    {
        foreach (var line in ChineseText.ReadLines("pykakasi221_kanwa.tsv.gz"))
        {
            int tab = line.IndexOf('\t');
            var key = CodePoints(line.Substring(0, tab));
            string reading = line.Substring(tab + 1).Split('\x1f')[0];
            if (!_kanwa.TryGetValue(key[0], out var entries)) _kanwa[key[0]] = entries = new List<(int[], string)>();
            entries.Add((key, reading));
        }
        foreach (var line in ChineseText.ReadLines("pykakasi221_itaiji.tsv.gz"))
        {
            int tab = line.IndexOf('\t');
            _itaiji[int.Parse(line.Substring(0, tab), CultureInfo.InvariantCulture)] = CodePoints(line.Substring(tab + 1));
        }
    }

    private bool IsKanji(int c) => (c >= 0x3400 && c < 0xE000) || _itaiji.ContainsKey(c);
    private static bool IsHiragana(int c) => (c > 0x3040 && c < 0x3097) || (c >= 0x1B150 && c <= 0x1B152);
    private static bool IsKatakana(int c) => (c > 0x30A0 && c < 0x30FD) || (c > 0xFF65 && c < 0xFF9F) || (c >= 0x1B164 && c <= 0x1B167);

    // H2("K").convert_K: the leading run of hiragana, in katakana.
    private static (List<int> Kana, int Length) HiraganaToKatakana(IReadOnlyList<int> text, int start, int end)
    {
        var kana = new List<int>();
        int i = start;
        for (; i < end; i++)
        {
            int c = text[i];
            if (c > 0x3040 && c < 0x3097) kana.Add(c + (0x30A1 - 0x3041));
            else if (c >= 0x1B150 && c <= 0x1B152) kana.Add(c + (0x1B164 - 0x1B150));
            else break;
        }
        return (kana, i - start);
    }

    // JConv.convert: the reading of the longest kanwa key at the start of the text.
    private (int[] Reading, int Length) KanjiReading(IReadOnlyList<int> itext)
    {
        var text = new List<int>(itext.Count);
        foreach (int c in itext)
        {
            if (_itaiji.TryGetValue(c, out var replacement)) text.AddRange(replacement);
            else text.Add(c);
        }
        int variationSelectors = itext.Count - text.Count;
        if (!_kanwa.TryGetValue(text[0], out var table)) return (Array.Empty<int>(), 0);
        int maxLength = 0;
        string reading = "";
        foreach (var (key, yomi) in table)
        {
            if (text.Count < key.Length || maxLength >= key.Length) continue;
            bool prefix = true;
            for (int k = 0; k < key.Length && prefix; k++) prefix = text[k] == key[k];
            if (!prefix) continue;
            reading = yomi;
            maxLength = key.Length;
        }
        // Python indexing: [-1] is the last element; past the end raises.
        static int At(IReadOnlyList<int> list, int index) => list[index < 0 ? list.Count + index : index];
        for (int v = 0; v < variationSelectors; v++)
        {
            if (maxLength > itext.Count) break;
            if (At(text, maxLength - 1) != At(itext, maxLength - 1)) maxLength++;
            else if (maxLength < variationSelectors + text.Count && maxLength <= itext.Count && IsVariationSelector(At(itext, maxLength))) maxLength++;
        }
        return (CodePoints(reading), maxLength);
    }

    private static bool IsVariationSelector(int c) => (c >= 0x0E0100 && c <= 0x0E1EF) || (c >= 0xFE00 && c <= 0xFE02);

    /// <summary><c>conv.do(text)</c>.</summary>
    public string Convert(string input)
    {
        var text = CodePoints(input);
        var output = new StringBuilder();
        int i = 0;
        while (i < text.Length)
        {
            int c = text[i];
            if (IsKanji(c))
            {
                int w = Math.Min(i + MaxLength, text.Length);
                var (reading, length) = KanjiReading(new ArraySegment<int>(text, i, w - i));
                if (length > 0)
                {
                    // J2.convert_nonh: the reading through H2("K"), characters it cannot convert skipped.
                    int m = 0;
                    while (m < reading.Length)
                    {
                        var (kana, n) = HiraganaToKatakana(reading, m, reading.Length);
                        if (n <= 0) m++;
                        else
                        {
                            m += n;
                            output.Append(Text(kana));
                        }
                    }
                    i += length;
                }
                else
                {
                    output.Append(char.ConvertFromUtf32(c));
                    i++;
                }
            }
            else if (IsHiragana(c))
            {
                var chunk = new List<int>();
                while (i < text.Length)
                {
                    int d = text[i];
                    if (d <= 0xFFFF && LongSymbols.IndexOf((char)d) >= 0)
                    {
                        if (chunk.Count > 0)
                        {
                            chunk.Add(chunk[chunk.Count - 1]);
                            i++;
                        }
                        else
                        {
                            chunk.Add('-');
                            i++;
                            break;
                        }
                    }
                    else if (IsHiragana(d))
                    {
                        var (kana, n) = HiraganaToKatakana(text, i, Math.Min(i + MaxLength, text.Length));
                        chunk.AddRange(kana);
                        i += n;
                    }
                    else break;
                }
                output.Append(Text(chunk));
            }
            else if (IsKatakana(c))
            {
                var chunk = new List<int>();
                while (i < text.Length)
                {
                    int d = text[i];
                    if (d <= 0xFFFF && LongSymbols.IndexOf((char)d) >= 0)
                    {
                        chunk.Add(chunk.Count == 0 ? '-' : d);
                        i++;
                        break;
                    }
                    if (!IsKatakana(d)) break;
                    chunk.Add(d);
                    i++;
                }
                output.Append(Text(chunk));
            }
            else
            {
                output.Append(char.ConvertFromUtf32(c));
                i++;
            }
        }
        return output.ToString();
    }
}

/// <summary>
/// MeloTTS's Japanese front end (myshell-ai/MeloTTS <c>melo/text/japanese.py</c>): NFKC, numbers read by num2words,
/// everything but kana and kanji dropped, pykakasi's katakana; then tohoku-nlp/bert-base-japanese-v3's word pieces
/// (MeCab with unidic-lite, then WordPiece), each word's katakana read by MeloTTS's kana-to-phoneme table.
/// </summary>
internal sealed class MeloJapaneseFrontEnd
{
    private static readonly Lazy<MeloJapaneseFrontEnd> SharedInstance = new(() => new MeloJapaneseFrontEnd(), isThreadSafe: true);
    private readonly WordPieceTokenizer _wordPieces = WordPieceTokenizer.BertBaseJapaneseV3();
    private readonly Dictionary<string, string[]> _rules1 = new(StringComparer.Ordinal);
    private readonly Dictionary<string, string[]> _rules2 = new(StringComparer.Ordinal);

    public static MeloJapaneseFrontEnd Default => SharedInstance.Value;

    private static readonly string[] Punctuation = { "!", "?", "…", ",", ".", "'", "-" };

    private MeloJapaneseFrontEnd()
    {
        foreach (var rule in ConversionRules)
        {
            var parts = rule.Split('/');
            var phonemes = parts[1].Split(' ').Skip(1).ToArray();
            int length = ChineseText.Codepoints(parts[0]).Count;
            if (length == 1) _rules1[parts[0]] = phonemes;
            else if (length == 2) _rules2[parts[0]] = phonemes;
        }
    }

    // ------------------------------------------------------------------ normalization

    private static readonly Regex NumberWithSeparator = new("[0-9]{1,3}(,[0-9]{3})+", RegexOptions.CultureInvariant);
    private static readonly Regex Currency = new(@"([$¥£€])([0-9.]*[0-9])", RegexOptions.CultureInvariant);
    private static readonly Regex Number = new(@"[0-9]+(\.[0-9]+)?", RegexOptions.CultureInvariant);
    private static readonly Dictionary<string, string> CurrencyNames = new(StringComparer.Ordinal) { ["$"] = "ドル", ["¥"] = "円", ["£"] = "ポンド", ["€"] = "ユーロ" };
    private static readonly (string From, string To)[] PunctuationMap =
        { ("：", ","), ("；", ","), ("，", ","), ("。", "."), ("！", "!"), ("？", "?"), ("\n", "."), ("·", ","), ("、", ","), ("...", "…") };
    private static readonly Regex NonJapanese = new("[^぀-ゟ゠-ヿ一-鿿㐀-䶿!?…,.'-]+", RegexOptions.CultureInvariant);

    /// <summary><c>japanese_convert_numbers_to_words</c>.</summary>
    internal static string ConvertNumbers(string text)
    {
        text = NumberWithSeparator.Replace(text, m => m.Value.Replace(",", ""));
        text = Currency.Replace(text, m => m.Groups[2].Value + (CurrencyNames.TryGetValue(m.Groups[1].Value, out var name) ? name : m.Groups[1].Value));
        return Number.Replace(text, m => Num2WordsJapanese.Default.Convert(m.Value));
    }

    private static bool IsJapaneseCharacter(int c) =>
        (c >= 0x3040 && c <= 0x309F) || (c >= 0x30A0 && c <= 0x30FF) || (c >= 0x4E00 && c <= 0x9FFF)
        || (c >= 0x3400 && c <= 0x4DBF) || (c >= 0x20000 && c <= 0x2A6DF);

    /// <summary><c>japanese.text_normalize</c>.</summary>
    public string Normalize(string text)
    {
        string res = ConvertNumbers(text.Normalize(NormalizationForm.FormKC));
        var kept = new StringBuilder(res.Length);
        foreach (var c in ChineseText.Codepoints(res))
            if (IsJapaneseCharacter(char.ConvertToUtf32(c, 0))) kept.Append(c);
        res = kept.ToString();
        var pattern = new Regex(string.Join("|", PunctuationMap.Select(p => Regex.Escape(p.From))), RegexOptions.CultureInvariant);
        res = pattern.Replace(res, m => PunctuationMap.First(p => p.From == m.Value).To);
        res = NonJapanese.Replace(res, "");
        return PykakasiKatakana.Default.Convert(res);
    }

    // ------------------------------------------------------------------ G2P

    /// <summary>tohoku-nlp/bert-base-japanese-v3's <c>tokenize</c>: MeCab words (after NFKC), then WordPiece.</summary>
    public List<string> WordPieces(string text) =>
        _wordPieces.TokenizeWords(MeCab.UnidicLite.Segment(text.Normalize(NormalizationForm.FormKC)));

    /// <summary><c>kata2phoneme</c>: two-character rules first, then one-character rules, other characters kept.</summary>
    public List<string> KatakanaToPhonemes(string text)
    {
        var chars = ChineseText.Codepoints(PythonText.Strip(text));
        var result = new List<string>();
        int i = 0;
        while (i < chars.Count)
        {
            if (chars.Count - i >= 2 && _rules2.TryGetValue(chars[i] + chars[i + 1], out var pair))
            {
                result.AddRange(pair);
                i += 2;
                continue;
            }
            if (_rules1.TryGetValue(chars[i], out var single))
            {
                result.AddRange(single);
                i++;
                continue;
            }
            result.Add(chars[i]);
            i++;
        }
        return result;
    }

    /// <summary>
    /// <c>japanese.g2p(norm_text)</c>: per word, its phonemes (asserted to be MeloTTS symbols), "_" for an unknown
    /// piece and punctuation as itself, spread over the word's pieces, with the silence symbol at each end.
    /// </summary>
    public (List<string> Phones, List<int> Tones, List<int> PhonesPerPiece) G2P(string text)
    {
        var tokens = WordPieces(text);
        var groups = new List<List<string>>();
        foreach (var token in tokens)
        {
            if (!token.StartsWith("#", StringComparison.Ordinal)) groups.Add(new List<string> { token });
            else if (groups.Count == 0) throw new ArgumentException("The text starts with a word piece continuation.", nameof(text));
            else groups[groups.Count - 1].Add(token.Replace("#", ""));
        }
        var phones = new List<string>();
        var perPiece = new List<int>();
        foreach (var group in groups)
        {
            string word = string.Concat(group);
            if (word == "[UNK]")
            {
                phones.Add("_");
                perPiece.Add(1);
                continue;
            }
            if (Punctuation.Contains(word))
            {
                phones.Add(word);
                perPiece.Add(1);
                continue;
            }
            var phonemes = KatakanaToPhonemes(word);
            foreach (var p in phonemes)
                if (!AiDotNet.TextToSpeech.EndToEnd.MeloSymbols.Contains(p))
                    throw new ArgumentException($"'{p}' (from '{word}') is not a MeloTTS symbol.", nameof(text));
            phones.AddRange(phonemes);
            perPiece.AddRange(MeloEnglishFrontEnd.DistributePhones(phonemes.Count, group.Count));
        }
        if (perPiece.Count != tokens.Count)
            throw new ArgumentException("The phones do not align with the word pieces.", nameof(text));
        phones.Insert(0, "_");
        phones.Add("_");
        perPiece.Insert(0, 1);
        perPiece.Add(1);
        return (phones, Enumerable.Repeat(0, phones.Count).ToList(), perPiece);
    }

    // japanese._CONVRULES, verbatim (later rules for the same kana replace earlier ones, as the dict comprehension does).
    private static readonly string[] ConversionRules =
    {
        "アァ/ a a", "イィ/ i i", "イェ/ i e", "イャ/ y a", "ウゥ/ u:", "エェ/ e e", "オォ/ o:", "カァ/ k a:", "キィ/ k i:",
        "クゥ/ k u:", "クャ/ ky a", "クュ/ ky u", "クョ/ ky o", "ケェ/ k e:", "コォ/ k o:", "ガァ/ g a:", "ギィ/ g i:",
        "グゥ/ g u:", "グャ/ gy a", "グュ/ gy u", "グョ/ gy o", "ゲェ/ g e:", "ゴォ/ g o:", "サァ/ s a:", "シィ/ sh i:",
        "スゥ/ s u:", "スャ/ sh a", "スュ/ sh u", "スョ/ sh o", "セェ/ s e:", "ソォ/ s o:", "ザァ/ z a:", "ジィ/ j i:",
        "ズゥ/ z u:", "ズャ/ zy a", "ズュ/ zy u", "ズョ/ zy o", "ゼェ/ z e:", "ゾォ/ z o:", "タァ/ t a:", "チィ/ ch i:",
        "ツァ/ ts a", "ツィ/ ts i", "ツゥ/ ts u:", "ツャ/ ch a", "ツュ/ ch u", "ツョ/ ch o", "ツェ/ ts e", "ツォ/ ts o",
        "テェ/ t e:", "トォ/ t o:", "ダァ/ d a:", "ヂィ/ j i:", "ヅゥ/ d u:", "ヅャ/ zy a", "ヅュ/ zy u", "ヅョ/ zy o",
        "デェ/ d e:", "ドォ/ d o:", "ナァ/ n a:", "ニィ/ n i:", "ヌゥ/ n u:", "ヌャ/ ny a", "ヌュ/ ny u", "ヌョ/ ny o",
        "ネェ/ n e:", "ノォ/ n o:", "ハァ/ h a:", "ヒィ/ h i:", "フゥ/ f u:", "フャ/ hy a", "フュ/ hy u", "フョ/ hy o",
        "ヘェ/ h e:", "ホォ/ h o:", "バァ/ b a:", "ビィ/ b i:", "ブゥ/ b u:", "フャ/ hy a", "ブュ/ by u", "フョ/ hy o",
        "ベェ/ b e:", "ボォ/ b o:", "パァ/ p a:", "ピィ/ p i:", "プゥ/ p u:", "プャ/ py a", "プュ/ py u", "プョ/ py o",
        "ペェ/ p e:", "ポォ/ p o:", "マァ/ m a:", "ミィ/ m i:", "ムゥ/ m u:", "ムャ/ my a", "ムュ/ my u", "ムョ/ my o",
        "メェ/ m e:", "モォ/ m o:", "ヤァ/ y a:", "ユゥ/ y u:", "ユャ/ y a:", "ユュ/ y u:", "ユョ/ y o:", "ヨォ/ y o:",
        "ラァ/ r a:", "リィ/ r i:", "ルゥ/ r u:", "ルャ/ ry a", "ルュ/ ry u", "ルョ/ ry o", "レェ/ r e:", "ロォ/ r o:",
        "ワァ/ w a:", "ヲォ/ o:", "ディ/ d i", "デェ/ d e:", "デャ/ dy a", "デュ/ dy u", "デョ/ dy o", "ティ/ t i",
        "テェ/ t e:", "テャ/ ty a", "テュ/ ty u", "テョ/ ty o", "スィ/ s i", "ズァ/ z u a", "ズィ/ z i", "ズゥ/ z u",
        "ズャ/ zy a", "ズュ/ zy u", "ズョ/ zy o", "ズェ/ z e", "ズォ/ z o", "キャ/ ky a", "キュ/ ky u", "キョ/ ky o",
        "シャ/ sh a", "シュ/ sh u", "シェ/ sh e", "ショ/ sh o", "チャ/ ch a", "チュ/ ch u", "チェ/ ch e", "チョ/ ch o",
        "トゥ/ t u", "トャ/ ty a", "トュ/ ty u", "トョ/ ty o", "ドァ/ d o a", "ドゥ/ d u", "ドャ/ dy a", "ドュ/ dy u",
        "ドョ/ dy o", "ドォ/ d o:", "ニャ/ ny a", "ニュ/ ny u", "ニョ/ ny o", "ヒャ/ hy a", "ヒュ/ hy u", "ヒョ/ hy o",
        "ミャ/ my a", "ミュ/ my u", "ミョ/ my o", "リャ/ ry a", "リュ/ ry u", "リョ/ ry o", "ギャ/ gy a", "ギュ/ gy u",
        "ギョ/ gy o", "ヂェ/ j e", "ヂャ/ j a", "ヂュ/ j u", "ヂョ/ j o", "ジェ/ j e", "ジャ/ j a", "ジュ/ j u", "ジョ/ j o",
        "ビャ/ by a", "ビュ/ by u", "ビョ/ by o", "ピャ/ py a", "ピュ/ py u", "ピョ/ py o", "ウァ/ u a", "ウィ/ w i",
        "ウェ/ w e", "ウォ/ w o", "ファ/ f a", "フィ/ f i", "フゥ/ f u", "フャ/ hy a", "フュ/ hy u", "フョ/ hy o",
        "フェ/ f e", "フォ/ f o", "ヴァ/ b a", "ヴィ/ b i", "ヴェ/ b e", "ヴォ/ b o", "ヴュ/ by u", "ア/ a", "イ/ i",
        "ウ/ u", "エ/ e", "オ/ o", "カ/ k a", "キ/ k i", "ク/ k u", "ケ/ k e", "コ/ k o", "サ/ s a", "シ/ sh i", "ス/ s u",
        "セ/ s e", "ソ/ s o", "タ/ t a", "チ/ ch i", "ツ/ ts u", "テ/ t e", "ト/ t o", "ナ/ n a", "ニ/ n i", "ヌ/ n u",
        "ネ/ n e", "ノ/ n o", "ハ/ h a", "ヒ/ h i", "フ/ f u", "ヘ/ h e", "ホ/ h o", "マ/ m a", "ミ/ m i", "ム/ m u",
        "メ/ m e", "モ/ m o", "ラ/ r a", "リ/ r i", "ル/ r u", "レ/ r e", "ロ/ r o", "ガ/ g a", "ギ/ g i", "グ/ g u",
        "ゲ/ g e", "ゴ/ g o", "ザ/ z a", "ジ/ j i", "ズ/ z u", "ゼ/ z e", "ゾ/ z o", "ダ/ d a", "ヂ/ j i", "ヅ/ z u",
        "デ/ d e", "ド/ d o", "バ/ b a", "ビ/ b i", "ブ/ b u", "ベ/ b e", "ボ/ b o", "パ/ p a", "ピ/ p i", "プ/ p u",
        "ペ/ p e", "ポ/ p o", "ヤ/ y a", "ユ/ y u", "ヨ/ y o", "ワ/ w a", "ヰ/ i", "ヱ/ e", "ヲ/ o", "ン/ N", "ッ/ q",
        "ヴ/ b u", "ー/:", "ァ/ a", "ィ/ i", "ゥ/ u", "ェ/ e", "ォ/ o", "ヮ/ w a", "ォ/ o", "ャ/ y a", "ョ/ y o", "ュ/ y u",
        "琦/ ch i", "ヶ/ k e", "髙/ t a k a", "煞/ sh y a", "、/ ,", "。/ .", "！/ !", "？/ ?", "・/ ,",
    };
}
