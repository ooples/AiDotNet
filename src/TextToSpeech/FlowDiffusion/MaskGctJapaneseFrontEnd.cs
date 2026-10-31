using System.Text;
using System.Text.RegularExpressions;
using AiDotNet.TextToSpeech.FrontEnd;

namespace AiDotNet.TextToSpeech.FlowDiffusion;

/// <summary>
/// MaskGCT's Japanese front end (Amphion <c>models/tts/maskgct/g2p/g2p/japanese.py</c> at 26f6883,
/// <c>JapanesePhoneConverter.getSentencePhone</c>, the phone string <c>japanese_to_ipa</c> returns): Open JTalk's words
/// (pyopenjtalk 0.3.4's <c>run_frontend</c>; the release's <c>estimate_accent</c> needs marine, which its environment
/// lacks, so the bare <c>except</c> leaves the words as they are), each word's katakana read into phonemes, the accents
/// read off Open JTalk's full-context labels and aligned with them, and the phonemes written in IPA separated by "|".
/// A word Open JTalk cannot read is read by pykakasi 2.3.0 instead, as the release does.
/// </summary>
/// <remarks>
/// The release's checks are kept as exceptions: a reading that is not katakana, an accent sequence it does not expect
/// and a phone the accents do not account for each raise there and throw here.
/// </remarks>
internal sealed class MaskGctJapaneseFrontEnd
{
    private readonly OpenJTalk _openJTalk;

    /// <summary>A front end over Open JTalk (its dictionary loaded by the caller).</summary>
    public MaskGctJapaneseFrontEnd(OpenJTalk openJTalk)
    {
        _openJTalk = openJTalk ?? throw new ArgumentNullException(nameof(openJTalk));
    }

    private static readonly string[] Punctuation = { ",", ".", "!", "?", ":", ";", "'", "…" };

    // jp_xphone2ipa; "ç" is written as the release writes it, c + U+0327.
    private static readonly string[] XPhoneToIpa =
    {
        " a a", " i i", " u ɯ", " e e", " o o", " a: aː", " i: iː", " u: ɯː", " e: eː", " o: oː", " k k", " s s", " t t",
        " n n", " h c\u0327", " f ɸ", " m m", " y j", " r ɾ", " w ɰᵝ", " N ɴ", " g g", " j d ʑ", " z z", " d d", " b b", " p p",
        " q q", " v v", " : :", " by b j", " ch t ɕ", " dy d e j", " ty t e j", " gy g j", " gw g ɯ", " hy c\u0327 j", " ky k j",
        " kw k ɯ", " my m j", " ny n j", " py p j", " ry ɾ j", " sh ɕ", " ts t s ɯ",
    };

    private static readonly Dictionary<string, string> IpaDictionary = XPhoneToIpa.ToDictionary(
        line => PythonText.Strip(line).Split(new[] { ' ' }, 2)[0],
        line => Regex.Replace(PythonText.Strip(line).Split(new[] { ' ' }, 2)[1], @"\s", ""), StringComparer.Ordinal);

    // _mora_list_minimum + _mora_list_additional: katakana mora, consonant (null for none), vowel.
    private static readonly (string Kana, string? Consonant, string Vowel)[] Moras =
    {
        ("ヴォ", "v", "o"), ("ヴェ", "v", "e"), ("ヴィ", "v", "i"), ("ヴァ", "v", "a"), ("ヴ", "v", "u"), ("ン", null, "N"),
        ("ワ", "w", "a"), ("ロ", "r", "o"), ("レ", "r", "e"), ("ル", "r", "u"), ("リョ", "ry", "o"), ("リュ", "ry", "u"),
        ("リャ", "ry", "a"), ("リェ", "ry", "e"), ("リ", "r", "i"), ("ラ", "r", "a"), ("ヨ", "y", "o"), ("ユ", "y", "u"),
        ("ヤ", "y", "a"), ("モ", "m", "o"), ("メ", "m", "e"), ("ム", "m", "u"), ("ミョ", "my", "o"), ("ミュ", "my", "u"),
        ("ミャ", "my", "a"), ("ミェ", "my", "e"), ("ミ", "m", "i"), ("マ", "m", "a"), ("ポ", "p", "o"), ("ボ", "b", "o"),
        ("ホ", "h", "o"), ("ペ", "p", "e"), ("ベ", "b", "e"), ("ヘ", "h", "e"), ("プ", "p", "u"), ("ブ", "b", "u"),
        ("フォ", "f", "o"), ("フェ", "f", "e"), ("フィ", "f", "i"), ("ファ", "f", "a"), ("フ", "f", "u"), ("ピョ", "py", "o"),
        ("ピュ", "py", "u"), ("ピャ", "py", "a"), ("ピェ", "py", "e"), ("ピ", "p", "i"), ("ビョ", "by", "o"), ("ビュ", "by", "u"),
        ("ビャ", "by", "a"), ("ビェ", "by", "e"), ("ビ", "b", "i"), ("ヒョ", "hy", "o"), ("ヒュ", "hy", "u"), ("ヒャ", "hy", "a"),
        ("ヒェ", "hy", "e"), ("ヒ", "h", "i"), ("パ", "p", "a"), ("バ", "b", "a"), ("ハ", "h", "a"), ("ノ", "n", "o"),
        ("ネ", "n", "e"), ("ヌ", "n", "u"), ("ニョ", "ny", "o"), ("ニュ", "ny", "u"), ("ニャ", "ny", "a"), ("ニェ", "ny", "e"),
        ("ニ", "n", "i"), ("ナ", "n", "a"), ("ドゥ", "d", "u"), ("ド", "d", "o"), ("トゥ", "t", "u"), ("ト", "t", "o"),
        ("デョ", "dy", "o"), ("デュ", "dy", "u"), ("デャ", "dy", "a"), ("ディ", "d", "i"), ("デ", "d", "e"), ("テョ", "ty", "o"),
        ("テュ", "ty", "u"), ("テャ", "ty", "a"), ("ティ", "t", "i"), ("テ", "t", "e"), ("ツォ", "ts", "o"), ("ツェ", "ts", "e"),
        ("ツィ", "ts", "i"), ("ツァ", "ts", "a"), ("ツ", "ts", "u"), ("ッ", null, "q"), ("チョ", "ch", "o"), ("チュ", "ch", "u"),
        ("チャ", "ch", "a"), ("チェ", "ch", "e"), ("チ", "ch", "i"), ("ダ", "d", "a"), ("タ", "t", "a"), ("ゾ", "z", "o"),
        ("ソ", "s", "o"), ("ゼ", "z", "e"), ("セ", "s", "e"), ("ズィ", "z", "i"), ("ズ", "z", "u"), ("スィ", "s", "i"),
        ("ス", "s", "u"), ("ジョ", "j", "o"), ("ジュ", "j", "u"), ("ジャ", "j", "a"), ("ジェ", "j", "e"), ("ジ", "j", "i"),
        ("ショ", "sh", "o"), ("シュ", "sh", "u"), ("シャ", "sh", "a"), ("シェ", "sh", "e"), ("シ", "sh", "i"), ("ザ", "z", "a"),
        ("サ", "s", "a"), ("ゴ", "g", "o"), ("コ", "k", "o"), ("ゲ", "g", "e"), ("ケ", "k", "e"), ("グヮ", "gw", "a"),
        ("グ", "g", "u"), ("クヮ", "kw", "a"), ("ク", "k", "u"), ("ギョ", "gy", "o"), ("ギュ", "gy", "u"), ("ギャ", "gy", "a"),
        ("ギェ", "gy", "e"), ("ギ", "g", "i"), ("キョ", "ky", "o"), ("キュ", "ky", "u"), ("キャ", "ky", "a"), ("キェ", "ky", "e"),
        ("キ", "k", "i"), ("ガ", "g", "a"), ("カ", "k", "a"), ("オ", null, "o"), ("エ", null, "e"), ("ウォ", "w", "o"),
        ("ウェ", "w", "e"), ("ウィ", "w", "i"), ("ウ", null, "u"), ("イェ", "y", "e"), ("イ", null, "i"), ("ア", null, "a"),
        // _mora_list_additional
        ("ヴョ", "by", "o"), ("ヴュ", "by", "u"), ("ヴャ", "by", "a"), ("ヲ", null, "o"), ("ヱ", null, "e"), ("ヰ", null, "i"),
        ("ヮ", "w", "a"), ("ョ", "y", "o"), ("ュ", "y", "u"), ("ヅ", "z", "u"), ("ヂ", "j", "i"), ("ヶ", "k", "e"),
        ("ャ", "y", "a"), ("ォ", null, "o"), ("ェ", null, "e"), ("ゥ", null, "u"), ("ィ", null, "i"), ("ァ", null, "a"),
    };

    private static readonly Dictionary<string, (string? Consonant, string Vowel)> MoraPhonemes = BuildMoraPhonemes();

    private static Dictionary<string, (string?, string)> BuildMoraPhonemes()
    {
        var table = new Dictionary<string, (string?, string)>(StringComparer.Ordinal);
        foreach (var (kana, consonant, vowel) in Moras) table[kana] = (consonant, vowel);
        return table;
    }

    // rep_map, in its order (the alternation tries the keys in turn at each position).
    private static readonly (string From, string To)[] ReplacementMap =
    {
        ("：", ":"), ("；", ";"), ("，", ","), ("。", "."), ("！", "!"), ("？", "?"), ("\n", "."), ("．", "."), ("⋯", "…"),
        ("···", "…"), ("・・・", "…"), ("·", ","), ("・", ","), ("•", ","), ("、", ","), ("$", "."), ("‘", "'"), ("’", "'"),
    };

    private static readonly Regex ReplacementPattern = new(string.Join("|", ReplacementMap.Select(p => Regex.Escape(p.From))), RegexOptions.CultureInvariant);

    private static readonly Regex Unreadable = new(
        "[^぀-ゟ゠-ヿ一-鿿㐀-䶿々A-Za-zＡ-Ｚａ-ｚ"
        + "Ͱ-Ͽἀ-῿" + Regex.Escape(string.Concat(Punctuation)).Replace("]", "\\]") + "]+",
        RegexOptions.CultureInvariant);

    /// <summary><c>replace_punctuation</c>: punctuation normalized, then everything Open JTalk cannot read removed.</summary>
    internal static string ReplacePunctuation(string text)
    {
        string replaced = ReplacementPattern.Replace(text, m => ReplacementMap.First(p => p.From == m.Value).To);
        return Unreadable.Replace(replaced, "");
    }

    private static readonly Regex Katakana = new("^[゠-ヿ]+$", RegexOptions.CultureInvariant);

    // Python's \w on str: alphanumeric characters and the underscore.
    private static bool IsWordCharacter(char c) => char.IsLetterOrDigit(c) || c == '_';

    /// <summary><c>kata2phoneme_list</c>: punctuation as itself; katakana mora by mora (two-character moras first),
    /// a long-vowel mark repeating the phoneme before it.</summary>
    internal static List<string> KatakanaToPhonemes(string text)
    {
        if (Punctuation.Contains(text)) return new List<string> { text };
        // re.fullmatch: "$" in .NET also matches before a final newline, which the character class already excludes.
        if (!Katakana.IsMatch(text) || text.EndsWith("\n", StringComparison.Ordinal))
            throw new ArgumentException($"Input must be katakana only: {text}", nameof(text));

        var spaced = new StringBuilder();
        for (int i = 0; i < text.Length;)
        {
            string? mora = i + 2 <= text.Length && MoraPhonemes.ContainsKey(text.Substring(i, 2)) ? text.Substring(i, 2)
                : MoraPhonemes.ContainsKey(text.Substring(i, 1)) ? text.Substring(i, 1) : null;
            if (mora is null)
            {
                spaced.Append(text[i]);
                i++;
                continue;
            }
            var (consonant, vowel) = MoraPhonemes[mora];
            spaced.Append(consonant is null ? " " + vowel : " " + consonant + " " + vowel);
            i += mora.Length;
        }

        // (\w)(ー*): each word character followed by long-vowel marks repeats itself once per mark
        string s = spaced.ToString();
        var withLong = new StringBuilder();
        for (int i = 0; i < s.Length;)
        {
            if (!IsWordCharacter(s[i]))
            {
                withLong.Append(s[i]);
                i++;
                continue;
            }
            char c = s[i];
            int j = i + 1;
            while (j < s.Length && s[j] == 'ー') j++;
            withLong.Append(c);
            for (int k = i + 1; k < j; k++) withLong.Append(' ').Append(c);
            i = j;
        }
        return PythonText.Strip(withLong.ToString()).Split(' ').ToList();
    }

    // handle_long_word: a leading "ー" takes the last phoneme of the word before (Python's [-1] wrapping to the last
    // word for the first), any other the last character of the phoneme before it; "▁" after each word.
    private static List<List<string>> HandleLongWord(List<List<string>> words)
    {
        var result = new List<List<string>>();
        for (int i = 0; i < words.Count; i++)
        {
            if (words[i][0] == "ー")
            {
                var before = words[i == 0 ? words.Count - 1 : i - 1];
                words[i][0] = before[before.Count - 1];
            }
            if (words[i].Contains("ー"))
                for (int j = 0; j < words[i].Count; j++)
                    if (words[i][j] == "ー")
                    {
                        string previous = words[i][j == 0 ? words[i].Count - 1 : j - 1];
                        words[i][j] = previous.Substring(previous.Length - 1);
                    }
            result.Add(words[i]);
            result.Add(new List<string> { "▁" });
        }
        return result;
    }

    private static readonly Regex CurrentPhoneme = new(@"\-(.*?)\+", RegexOptions.CultureInvariant);
    private static readonly Regex A1 = new(@"/A:([0-9\-]+)\+", RegexOptions.CultureInvariant);
    private static readonly Regex A2 = new(@"\+(\d+)\+", RegexOptions.CultureInvariant);
    private static readonly Regex A3 = new(@"\+(\d+)/", RegexOptions.CultureInvariant);
    private static readonly Regex F1 = new(@"/F:(\d+)_", RegexOptions.CultureInvariant);

    private static int Numeric(Regex regex, string s)
    {
        var match = regex.Match(s);
        return match.Success ? int.Parse(match.Groups[1].Value, System.Globalization.CultureInfo.InvariantCulture) : -50;
    }

    /// <summary><c>frontend2phoneme(labels, drop_unvoiced_vowels=True)</c>: phonemes with "_" for pauses, "#" at accent
    /// phrase borders, "]" where the pitch falls and "[" where it rises.</summary>
    internal static List<string> FrontendToPhoneme(IReadOnlyList<string> labels)
    {
        var phones = new List<string>();
        for (int n = 0; n < labels.Count; n++)
        {
            string label = labels[n];
            string p3 = CurrentPhoneme.Match(label).Groups[1].Value;
            if ("AEIOU".Contains(p3)) p3 = p3.ToLowerInvariant();
            if (p3 == "sil") continue;
            if (p3 == "pau")
            {
                phones.Add("_");
                continue;
            }
            phones.Add(p3);
            int a1 = Numeric(A1, label), a2 = Numeric(A2, label), a3 = Numeric(A3, label), f1 = Numeric(F1, label);
            int a2Next = Numeric(A2, labels[n + 1]);
            if (a3 == 1 && a2Next == 1 && "aeiouAEIOUNcl".Contains(p3)) phones.Add("#");
            else if (a1 == 0 && a2Next == a2 + 1 && a2 != f1) phones.Add("]");
            else if (a2 == 1 && a2Next == 2) phones.Add("[");
        }
        return phones;
    }

    // fix_phone_tone: a phrase's tones moved into 0-1.
    private static IEnumerable<(string, int)> FixPhoneTone(List<(string Phone, int Tone)> phrase)
    {
        var values = new HashSet<int>(phrase.Select(p => p.Tone));
        if (values.Count == 1)
        {
            if (!values.Contains(0)) throw new InvalidOperationException($"Unexpected tone values: {{{string.Join(", ", values)}}}");
            return phrase;
        }
        if (values.Count == 2)
        {
            if (values.SetEquals(new[] { 0, 1 })) return phrase;
            if (values.SetEquals(new[] { -1, 0 })) return phrase.Select(p => (p.Phone, p.Tone == -1 ? 0 : 1));
        }
        throw new InvalidOperationException($"Unexpected tone values: {{{string.Join(", ", values)}}}");
    }

    /// <summary><c>g2phone_tone_wo_punct</c>: each phoneme with its tone; the phrase still open at the end is dropped,
    /// as the release drops it (its end marks never reach here).</summary>
    internal static List<(string Phone, int Tone)> PhoneTones(IReadOnlyList<string> prosodies)
    {
        var result = new List<(string, int)>();
        var phrase = new List<(string Phone, int Tone)>();
        int tone = 0;
        string lastAccent = "";
        for (int i = 0; i < prosodies.Count; i++)
        {
            string letter = prosodies[i];
            if (letter == "^")
            {
                if (i != 0) throw new InvalidOperationException("Unexpected ^");
            }
            else if (letter is "$" or "?" or "_" or "#")
            {
                result.AddRange(FixPhoneTone(phrase));
                if ((letter is "$" or "?") && i != prosodies.Count - 1) throw new InvalidOperationException($"Unexpected {letter}");
                phrase = new List<(string, int)>();
                tone = 0;
                lastAccent = "";
            }
            else if (letter == "[")
            {
                if (lastAccent != letter) tone++;
                lastAccent = letter;
            }
            else if (letter == "]")
            {
                if (lastAccent != letter) tone--;
                lastAccent = letter;
            }
            else
            {
                phrase.Add((letter == "cl" ? "q" : letter, tone));
            }
        }
        return result;
    }

    /// <summary><c>align_tones</c>: the tones carried onto the phonemes with punctuation.</summary>
    private static List<(string Phone, int Tone)> AlignTones(List<string> phonesWithPunctuation, List<(string Phone, int Tone)> phoneTones)
    {
        var result = new List<(string, int)>();
        int index = 0;
        foreach (var phone in phonesWithPunctuation)
        {
            if (index >= phoneTones.Count) result.Add((phone, 0));
            else if (phone == phoneTones[index].Phone)
            {
                result.Add((phone, phoneTones[index].Tone));
                index++;
            }
            else if (Punctuation.Contains(phone) || phone == "▁") result.Add((phone, 0));
            else throw new InvalidOperationException($"Unexpected phone: {phone}");
        }
        return result;
    }

    private static readonly string[] IterationMarks = { "々", "ゝ", "ヽ", "ゞ", "ヾ", "゛" };
    private static readonly string[] Readable = { ".", ",", "!", "'", "-", "?", ":", ";", "…", "" };

    /// <summary><c>text2sep_kata</c>: each word's normalized text and katakana reading, and the words kept (an
    /// iteration mark joining the word before it, words reduced to nothing dropped).</summary>
    private static (List<string> Words, List<string> Readings, List<NjdFeature> Kept) SeparateKatakana(List<NjdFeature> parsed)
    {
        var words = new List<string>();
        var readings = new List<string>();
        var kept = new List<NjdFeature>();
        for (int i = 0; i <= parsed.Count - 1;)
        {
            string yomi = parsed[i].Pron;
            var current = parsed[i];
            string word;
            if (i != parsed.Count - 1 && IterationMarks.Contains(parsed[i + 1].String))
            {
                word = parsed[i].String + parsed[i + 1].String;
                i++;
            }
            else
            {
                word = parsed[i].String;
            }
            word = ReplacePunctuation(word);
            yomi = yomi.Replace("’", "");
            if (yomi.Length == 0) throw new InvalidOperationException($"Empty yomi: {word}");
            if (yomi == "、")
            {
                if (!Readable.Contains(word))
                {
                    // Open JTalk cannot read the word: pykakasi's reading of its first segment
                    word = Pykakasi.Default.Convert(word)[0].Kana;
                    current.Pron = word;
                }
                yomi = word;
            }
            else if (yomi == "？")
            {
                if (word != "?") throw new InvalidOperationException($"yomi `？` comes from: {word}");
                yomi = "?";
            }
            if (word.Length == 0)
            {
                i++;
                continue;
            }
            words.Add(word);
            readings.Add(yomi);
            kept.Add(current);
            i++;
        }
        return (words, readings, kept);
    }

    /// <summary><c>japanese_to_ipa(text)</c>: <c>getSentencePhone(text)["jp_p"]</c>.</summary>
    public string JapaneseToIpa(string sentence)
    {
        if (sentence is null) throw new ArgumentNullException(nameof(sentence));
        sentence = PythonText.Strip(sentence).Trim('\'');
        sentence = new string(sentence.Where(c => !PythonText.IsSpace(c)).ToArray());
        var frontend = _openJTalk.RunFrontend(sentence);
        var (_, readings, kept) = SeparateKatakana(frontend);
        var separated = HandleLongWord(readings.Select(KatakanaToPhonemes).ToList());
        var prosodies = FrontendToPhoneme(OpenJTalk.MakeLabel(kept));
        var phoneTones = PhoneTones(prosodies);

        var phonesWithPunctuation = separated.SelectMany(p => p).ToList();
        if (phonesWithPunctuation.Count > 0) phonesWithPunctuation.RemoveAt(phonesWithPunctuation.Count - 1);   // [:-1]
        var aligned = AlignTones(phonesWithPunctuation, phoneTones);

        var jp = new StringBuilder();
        foreach (var (phone, _) in aligned)
        {
            if (IpaDictionary.TryGetValue(phone, out var ipa)) jp.Append(ipa);
            else if (Punctuation.Contains(phone) || phone == "▁") jp.Append(phone);
            jp.Append('|');
        }
        string result = jp.ToString().Replace("▁", " ").Replace("| |", "|");
        return result.TrimEnd('|');
    }
}
