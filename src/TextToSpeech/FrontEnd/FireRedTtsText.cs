using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;
using AiDotNet.Tokenization.Algorithms;

namespace AiDotNet.TextToSpeech.FrontEnd;

/// <summary>
/// The rule stage of lingua 2.0.2 (lingua-rs 1.6.2, <c>detect_language_with_rules</c> then
/// <c>filter_languages_by_rules</c>), restricted to English and Chinese as FireRedTTS's text normalizer builds it.
/// </summary>
/// <remarks>
/// <para>The text is lowercased and split into words: each Han, Hangul, Hiragana or Katakana character alone, every
/// other run of letters as one. With only English and Chinese, a word votes Chinese when one of its characters is Han and
/// counts as unknown otherwise (Latin has no single-language alphabet among the two); unknown words are dropped when
/// fewer than half, and a clear majority decides. Otherwise the alphabet with the most characters (over words wholly in
/// one alphabet) keeps the languages written in it.</para>
/// <para>When that leaves both languages (alphabets tied in character count), lingua scores character n-grams with its
/// language models, which are not ported: the port returns English when a word is in the Latin alphabet and no language
/// otherwise. Measured against lingua, ties of Han symbols with Latin letters give English as lingua does; a tie of a
/// symbol the Chinese model knows (U+3005) with Greek or Hangul gives Chinese in lingua and none here. Ties among the
/// top alphabets when a third differs follow lingua's hash order there; here Latin wins.</para>
/// </remarks>
internal static class FireRedTtsLanguageDetector
{
    private static readonly (string Name, HashSet<int> Set)[] Alphabets = FireRedTtsTextData.LinguaAlphabets
        .Select(a => (a.Name, Expand(a.Ranges))).ToArray();
    private static readonly HashSet<int> Han = Alphabets.First(a => a.Name == "Han").Set;
    private static readonly HashSet<int> SingleCharacterWords = new(new[] { "Han", "Hangul", "Hiragana", "Katakana" }
        .SelectMany(n => Alphabets.First(a => a.Name == n).Set));
    private static readonly HashSet<int> Japanese = new(new[] { "Hiragana", "Katakana", "Han" }.SelectMany(n => Alphabets.First(a => a.Name == n).Set));

    private static HashSet<int> Expand(int[] ranges)
    {
        var set = new HashSet<int>();
        for (int i = 0; i < ranges.Length; i += 2)
            for (int c = ranges[i]; c <= ranges[i + 1]; c++) set.Add(c);
        return set;
    }

    private enum Vote { Unknown, Chinese, Japanese }

    /// <summary>"en", "zh", or null where lingua detects no language.</summary>
    public static string? Detect(string text)
    {
        var words = Words(text);
        if (words.Count == 0) return null;
        double half = words.Count * 0.5;

        // detect_language_with_rules
        var totals = new Dictionary<Vote, int>();
        foreach (var word in words)
        {
            int chinese = 0, japanese = 0;
            foreach (int c in word)
            {
                if (Han.Contains(c)) chinese++;
                else if (Japanese.Contains(c)) japanese++;
            }
            Vote vote;
            if (chinese == 0 && japanese == 0) vote = Vote.Unknown;
            else if (chinese > 0 && japanese > 0) vote = Vote.Japanese;      // both counted: Japanese, outside the two
            else vote = chinese > 0 ? Vote.Chinese : Vote.Japanese;
            // Japanese is not among the detector's languages: the word is unknown.
            if (vote == Vote.Japanese) vote = Vote.Unknown;
            totals[vote] = (totals.TryGetValue(vote, out int n) ? n : 0) + 1;
        }
        if (totals.TryGetValue(Vote.Unknown, out int unknown) && unknown < half) totals.Remove(Vote.Unknown);
        if (totals.Count == 1)
        {
            if (totals.ContainsKey(Vote.Chinese)) return "zh";
        }
        else if (totals.Count == 2 && totals[Vote.Chinese] > totals[Vote.Unknown])
        {
            return "zh";
        }

        // filter_languages_by_rules
        var counts = new List<(string Name, int Count)>();
        foreach (var word in words)
        {
            foreach (var (name, set) in Alphabets)
            {
                if (word.All(set.Contains))
                {
                    int i = counts.FindIndex(c => c.Name == name);
                    if (i < 0) counts.Add((name, word.Length));
                    else counts[i] = (name, counts[i].Count + word.Length);
                    break;
                }
            }
        }
        bool latinWord = counts.Any(c => c.Name == "Latin");
        if (counts.Count == 0 || (counts.Count > 1 && counts.Select(c => c.Count).Distinct().Count() == 1))
            return latinWord ? "en" : null;                                           // both languages remain: n-gram stage
        int best = counts.Max(c => c.Count);
        string alphabet = counts.Any(c => c.Count == best && c.Name == "Latin") ? "Latin" : counts.First(c => c.Count == best).Name;
        return alphabet == "Latin" ? "en" : alphabet == "Han" ? "zh" : null;
    }

    // split_text_into_words: \p{Han}|\p{Hangul}|\p{Hiragana}|\p{Katakana}|\p{L}+ over the trimmed, lowercased text, as code
    // points.
    private static List<int[]> Words(string text)
    {
        var points = CodePoints(PythonText.Lower(PythonText.Strip(text)));
        var words = new List<int[]>();
        var run = new List<int>();
        foreach (int c in points)
        {
            if (SingleCharacterWords.Contains(c))
            {
                if (run.Count > 0) { words.Add(run.ToArray()); run.Clear(); }
                words.Add(new[] { c });
            }
            else if (IsLetter(c))
            {
                run.Add(c);
            }
            else if (run.Count > 0)
            {
                words.Add(run.ToArray());
                run.Clear();
            }
        }
        if (run.Count > 0) words.Add(run.ToArray());
        return words;
    }

    private static bool IsLetter(int c)
    {
        var category = CharUnicodeInfo.GetUnicodeCategory(char.ConvertFromUtf32(c), 0);
        return category is UnicodeCategory.UppercaseLetter or UnicodeCategory.LowercaseLetter or UnicodeCategory.TitlecaseLetter
            or UnicodeCategory.ModifierLetter or UnicodeCategory.OtherLetter;
    }

    internal static List<int> CodePoints(string text)
    {
        var points = new List<int>(text.Length);
        for (int i = 0; i < text.Length; i++)
        {
            if (char.IsHighSurrogate(text[i]) && i + 1 < text.Length && char.IsLowSurrogate(text[i + 1]))
            {
                points.Add(char.ConvertToUtf32(text[i], text[i + 1]));
                i++;
            }
            else
            {
                points.Add(text[i]);
            }
        }
        return points;
    }
}

/// <summary>
/// FireRedTTS's text normalizer (<c>fireredtts/modules/text_normalizer/normalize.py</c>, <c>TextNormalizer.tn</c>):
/// cleaning (<c>preprocess_text</c>), English abbreviations (<c>rettt</c>), English/Chinese detection, WeTextProcessing's
/// Chinese or English normalizer (with inflect's numbers after the English one), capitals spelled out, and the model's
/// punctuation limits; it returns the text and its language token ("en" or "zh").
/// </summary>
internal static class FireRedTtsTextNormalizer
{
    private static readonly TimeSpan Timeout = TimeSpan.FromSeconds(1);
    private static readonly Regex Kaomoji = new(FireRedTtsTextData.KaomojiPattern, RegexOptions.CultureInvariant, Timeout);
    private static readonly Regex ValidChar = new(FireRedTtsTextData.ValidCharPattern, RegexOptions.CultureInvariant, Timeout);
    private static readonly Dictionary<string, string> SymbolReduction =
        FireRedTtsTextData.SymbolReduction.ToDictionary(p => p[0], p => p[1], StringComparer.Ordinal);
    private static readonly Regex ThreeStops = new("[\u3002.]{3,}", RegexOptions.CultureInvariant, Timeout);
    private static readonly Regex Ellipses = new("[\u2026\u22ef]{1,}", RegexOptions.CultureInvariant, Timeout);
    private static readonly Regex Spaces = new("[ ]+", RegexOptions.CultureInvariant, Timeout);
    private static readonly Regex NotEnglish = new(@"[^ 0-9A-Za-z\[\]'.,:?!_\-]", RegexOptions.CultureInvariant, Timeout);
    private static readonly Regex Digits = new(@"(\d+)", RegexOptions.CultureInvariant, Timeout);
    private static readonly Regex Capitals = new("([A-Z]{2,4}|[ ])", RegexOptions.CultureInvariant, Timeout);
    private static readonly Regex CapitalsStart = new("^[A-Z]{2,4}", RegexOptions.CultureInvariant, Timeout);
    private static readonly Regex TrailingCommas = new("[\uff0c,]+$", RegexOptions.CultureInvariant, Timeout);
    private static readonly Regex Pauses = new("[\u2026~\uff01\uff0c&*%$#^\uff1a\uff1b!:;]+", RegexOptions.CultureInvariant, Timeout);
    private static readonly Regex Commas = new("[,]+", RegexOptions.CultureInvariant, Timeout);
    private static readonly Regex TrailingStops = new("[,. ]+$", RegexOptions.CultureInvariant, Timeout);
    private static readonly Regex UpperEnglishAndDigits = new("^[ 0-9A-Z\"'.,:?!\\-]+$", RegexOptions.CultureInvariant, Timeout);
    private static readonly Regex SaintBefore = new(@" St\. ([A-Z])", RegexOptions.CultureInvariant, Timeout);
    private static readonly Regex Street = new(@" St\.", RegexOptions.CultureInvariant, Timeout);
    private static readonly Regex Road = new(@" Rd\.", RegexOptions.CultureInvariant, Timeout);
    private static readonly Regex Am = new(@"[Aa]\.[Mm]\.", RegexOptions.CultureInvariant, Timeout);
    private static readonly Regex Pm = new(@"[Pp]\.[Mm]\.", RegexOptions.CultureInvariant, Timeout);
    private static readonly Regex Bc = new(@"[Bb]\.[Cc]\.", RegexOptions.CultureInvariant, Timeout);
    private static readonly Regex Ad = new(@"[Ad]\.[Dd]\.", RegexOptions.CultureInvariant, Timeout);

    /// <summary><c>tn</c>: the normalized text and its language token ("en" or "zh").</summary>
    public static (string Text, string Language) Normalize(string text)
    {
        if (text is null) throw new ArgumentNullException(nameof(text));
        text = Preprocess(text);
        text = Abbreviations(text);
        string? detected = FireRedTtsLanguageDetector.Detect(text);
        bool chinese = detected == "zh";
        // enforce chinese if text contains any chinese character
        if (ContainsChinese(text))
        {
            chinese = true;
            detected = "zh";
        }
        string language = detected ?? "zh";
        if (UpperEnglishAndDigits.IsMatch(text)) chinese = true;

        if (chinese)
        {
            text = WeTextNormalizer.Chinese.Normalize(text);
            text = text.Replace("\n", "");
            text = TrailingCommas.Replace(text, "\u3002");
        }
        else
        {
            text = NotEnglish.Replace(text, "");
            text = WeTextNormalizer.English.Normalize(text);
            // fallback number normalization
            var pieces = Digits.Split(text);
            text = string.Concat(pieces.Where(p => p.Length > 0).Select(p => IsNumeric(p) ? InflectNumberWords.NumberToWords(p) : p));
        }

        // cleanup
        text = text.Replace("_", " ");
        text = Spaces.Replace(text, " ");

        // spell capital words
        var parts = Capitals.Split(text);
        for (int i = 0; i < parts.Length; i++)
            if (CapitalsStart.IsMatch(parts[i])) parts[i] = string.Join(" ", parts[i].Select(c => c.ToString()));
        text = string.Join(" ", parts.Where(p => p != " "));

        // post TN full to half
        text = text.Replace("\u3002", ".").Replace("\uff0c", ",").Replace("\uff1a", ":");

        // model limitations
        text = PythonText.Strip(PythonText.Lower(text));
        text = text.Replace("\"", "");
        text = text.Replace("\u00b7", " ");
        text = Pauses.Replace(text, ",");
        text = Commas.Replace(text, ",");
        text = TrailingStops.Replace(text, ".");
        if (text.Length > 0 && text[text.Length - 1] != '.' && text[text.Length - 1] != '?') text += ".";
        return (text, language);
    }

    /// <summary><c>contains_chinese</c>: a character in U+4E00 \u2026 U+9FA5.</summary>
    public static bool ContainsChinese(string text) => text.Any(IsChinese);

    private static bool IsChinese(char c) => c >= '\u4e00' && c <= '\u9fa5';

    // str.isnumeric for the digit runs the split leaves.
    private static bool IsNumeric(string s) => s.Length > 0 && s.All(char.IsDigit);

    /// <summary><c>preprocess_text</c>.</summary>
    internal static string Preprocess(string sentence)
    {
        // bytes(sentence, "utf-8").decode("utf-8", "ignore") drops unpaired surrogates.
        var builder = new StringBuilder(sentence.Length);
        for (int i = 0; i < sentence.Length; i++)
        {
            char c = sentence[i];
            if (char.IsHighSurrogate(c) && i + 1 < sentence.Length && char.IsLowSurrogate(sentence[i + 1]))
            {
                builder.Append(c).Append(sentence[++i]);
            }
            else if (!char.IsSurrogate(c))
            {
                builder.Append(c);
            }
        }
        sentence = builder.ToString();
        // [\p{Cf}--[\u200d]] and \p{Co} removed; NBSP to space; U+FFFD removed; \p{Zl} and \p{Zp} to newlines.
        builder.Clear();
        for (int i = 0; i < sentence.Length; i++)
        {
            var category = CharUnicodeInfo.GetUnicodeCategory(sentence, i);
            int width = char.IsHighSurrogate(sentence[i]) && i + 1 < sentence.Length ? 2 : 1;
            string piece = sentence.Substring(i, width);
            i += width - 1;
            if (category == UnicodeCategory.Format && piece != "\u200d") continue;
            if (category == UnicodeCategory.PrivateUse) continue;
            if (piece == "\u00a0") piece = " ";
            else if (piece == "\ufffd") piece = "";
            else if (category is UnicodeCategory.LineSeparator or UnicodeCategory.ParagraphSeparator) piece = "\n";
            builder.Append(piece);
        }
        sentence = builder.ToString();

        // Strip accents: NFD, then every non-spacing mark dropped.
        builder.Clear();
        string decomposed = sentence.Normalize(NormalizationForm.FormD);
        for (int i = 0; i < decomposed.Length; i++)
            if (CharUnicodeInfo.GetUnicodeCategory(decomposed, i) != UnicodeCategory.NonSpacingMark) builder.Append(decomposed[i]);
        sentence = builder.ToString();

        sentence = Kaomoji.Replace(sentence, " ");
        // full to half with exemption (to be converted after number TN): \u3002\uff0c\uff1a
        sentence = FullToHalf(sentence, "\u3002\uff0c\uff1a");

        // clean spaces
        sentence = sentence.Replace("\n", "\uff0c").Replace("\t", "\uff0c").Replace("\r", "\uff0c");
        sentence = ThreeStops.Replace(sentence, "\u2026");
        sentence = Ellipses.Replace(sentence, "\u2026");
        sentence = Spaces.Replace(sentence, " ");
        sentence = PythonText.Strip(sentence);

        // punctuation reduction
        builder.Clear();
        for (int index = 0; index < sentence.Length; index++)
        {
            string character = sentence[index].ToString();
            if (SymbolReduction.TryGetValue(character, out var reduced)) character = reduced;
            if (character == " ")
            {
                if (index == 0) continue;
                bool nextChinese = index + 1 < sentence.Length && IsChinese(sentence[index + 1]);
                if (nextChinese && (IsChinese(sentence[index - 1]) || "\") ".IndexOf(sentence[index - 1]) >= 0)) builder.Append('\uff0c');
                else builder.Append(' ');
                continue;
            }
            var valid = ValidChar.Match(character);
            if (valid.Success && valid.Index == 0) builder.Append(character);
        }
        return Spaces.Replace(builder.ToString(), " ");
    }

    /// <summary><c>f2b</c>: U+3000 to a space, U+FF01 \u2026 U+FF5E to ASCII, except the exempted characters.</summary>
    internal static string FullToHalf(string text, string exemption)
    {
        var builder = new StringBuilder(text.Length);
        foreach (char u in text)
        {
            if (u == '\u3000') builder.Append(' ');
            else if (exemption.IndexOf(u) >= 0) builder.Append(u);
            else if (u >= '\uff01' && u <= '\uff5e') builder.Append((char)(u - 0xFEE0));
            else builder.Append(u);
        }
        return builder.ToString();
    }

    /// <summary><c>rettt</c>: English abbreviations, in the release's order.</summary>
    internal static string Abbreviations(string sentence)
    {
        sentence = sentence.Replace("&nd", "and");
        foreach (var (from, to) in new[]
        {
            ("Jan.", "january"), ("Feb.", "febrary"), ("Mar.", "march"), ("Apr.", "april"), ("May.", "may"), ("Jun.", "june"),
            ("Jul.", "july"), ("Aug.", "august"), ("Sept.", "september"), ("Sep.", "september"), ("Oct.", "october"),
            ("Nov.", "november"), ("Dec.", "december"), ("Mon.", "monday"), ("Tues.", "tuesday"), ("Wed.", "wednesday"),
            ("Thur.", "thursday"), ("Fri.", "friday"), ("Sat.", "saturday"),
        })
            sentence = sentence.Replace(from, to);
        if (sentence != "Sun.") sentence = sentence.Replace("Sun.", "sunday");
        sentence = SaintBefore.Replace(sentence, " saint $1");
        sentence = Street.Replace(sentence, " street");
        sentence = Road.Replace(sentence, " road");
        sentence = Am.Replace(sentence, "A_M");
        sentence = Pm.Replace(sentence, "P_M");
        sentence = Bc.Replace(sentence, "B_C");
        sentence = Ad.Replace(sentence, "A_D");
        foreach (var (from, to) in new[]
        {
            ("Mr.", "mister"), ("Ms.", "miss"), ("Mrs.", "misses"), ("Ph.D", "P_H_D"), ("i.e.", "that is"), ("e.g.", "for example"),
            ("btw.", "by the way"), ("btw", "by the way"), ("b.t.w.", "by the way"), ("@", " at "),
        })
            sentence = sentence.Replace(from, to);
        return sentence;
    }
}

/// <summary>
/// sentencex 0.6.1's sentence segmenter (<c>sentencex.segment</c>), English rules: FireRedTTS's <c>text_split</c> segments
/// with it for English and, by sentencex's language fallback (zh \u2192 zh-hans \u2192 en), for Chinese too.
/// </summary>
internal static class SentencexSegmenter
{
    private static readonly TimeSpan Timeout = TimeSpan.FromSeconds(1);
    private static readonly HashSet<int> Terminators = new(FireRedTtsTextData.SentenceTerminators);
    private static readonly HashSet<string> Abbreviations = new(FireRedTtsTextData.EnglishAbbreviations, StringComparer.Ordinal);
    private static readonly HashSet<string> ExclamationWords = new(FireRedTtsTextData.ExclamationWords, StringComparer.Ordinal);
    private static readonly Regex Quotes = new(string.Join("|", FireRedTtsTextData.QuotePairs.Select(p => Regex.Escape(p[0]) + @"(\n|.)*?" + Regex.Escape(p[1]))) + "+",
        RegexOptions.CultureInvariant, Timeout);
    private static readonly Regex Parens = new(@"([\(\uff08<{\[])(?:\\\1|.)*?[\)\]}\uff09]", RegexOptions.CultureInvariant, Timeout);
    private static readonly Regex Email = new(@"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}", RegexOptions.CultureInvariant, Timeout);
    private static readonly Regex Paragraphs = new(@"(\n{2,})", RegexOptions.CultureInvariant, Timeout);
    private static readonly Regex NumberedReference = new(@"^(\[\d+])+", RegexOptions.CultureInvariant, Timeout);
    private static readonly Regex WordSeparators = new(@"[" + PythonText.WhitespaceClass + @"\.]+", RegexOptions.CultureInvariant, Timeout);
    private static readonly Regex ContinuesInNextWord = new("^[0-9a-z]", RegexOptions.CultureInvariant, Timeout);

    /// <summary>The sentences of <paramref name="text"/>.</summary>
    public static List<string> Segment(string text)
    {
        var sentences = new List<string>();
        foreach (var paragraph in Paragraphs.Split(text))
        {
            var boundaries = new List<int> { 0 };
            var skippable = Quotes.Matches(paragraph).Cast<Match>().Concat(Parens.Matches(paragraph).Cast<Match>())
                .Concat(Email.Matches(paragraph).Cast<Match>()).Select(m => (Start: m.Index, End: m.Index + m.Length)).ToList();
            foreach (var (start, length) in BreakRuns(paragraph))
            {
                int? found = FindBoundary(paragraph, start, length);
                if (found is null) continue;
                int boundary = found.Value;
                bool inRange = false;
                foreach (var (qstart, qend) in skippable)
                {
                    if (boundary > qstart && boundary < qend)
                    {
                        // is_punctuation_between_quotes() is false for English: inside a quote is never a boundary.
                        inRange = true;
                        break;
                    }
                }
                if (inRange) continue;
                boundaries.Add(boundary);
            }
            for (int b = 0; b < boundaries.Count; b++)
            {
                int i = boundaries[b];
                int j = b + 1 < boundaries.Count ? boundaries[b + 1] : paragraph.Length;
                // paragraph[i:j]: Python slicing clamps, and a later boundary before an earlier one yields "".
                if (j <= i) continue;
                string sentence = paragraph.Substring(i, Math.Min(j, paragraph.Length) - i);
                if (sentence.Length > 0) sentences.Add(sentence.Trim(' '));
            }
        }
        return sentences;
    }

    // Runs of sentence terminators (the GLOBAL_SENTENCE_BOUNDARY_REGEX's matches), as UTF-16 (start, length).
    private static IEnumerable<(int Start, int Length)> BreakRuns(string text)
    {
        int i = 0;
        while (i < text.Length)
        {
            if (!Terminators.Contains(PointAt(text, i)))
            {
                i += Width(text, i);
                continue;
            }
            int start = i;
            while (i < text.Length && Terminators.Contains(PointAt(text, i))) i += Width(text, i);
            yield return (start, i - start);
        }
    }

    private static int PointAt(string text, int i) => Width(text, i) == 2 ? char.ConvertToUtf32(text[i], text[i + 1]) : text[i];

    private static int Width(string text, int i) =>
        char.IsHighSurrogate(text[i]) && i + 1 < text.Length && char.IsLowSurrogate(text[i + 1]) ? 2 : 1;

    private static int? FindBoundary(string text, int start, int length)
    {
        // tail = text[match.start() + 1:] (after the run's first code point), head = text[:match.start()].
        int first = Width(text, start);
        string tail = text.Substring(start + first);
        string head = text.Substring(0, start);
        var reference = NumberedReference.Match(tail);
        if (reference.Success) return start + first + reference.Length;
        if (ContinuesInNextWord.IsMatch(tail)) return null;
        string separator = text.Substring(start, length);
        if (IsAbbreviation(head, separator)) return null;
        if (ExclamationWords.Contains(LastWord(head) + "!")) return null;
        return start + length;
    }

    private static bool IsAbbreviation(string head, string separator)
    {
        if (separator != ".") return false;
        string last = LastWord(head);
        if (last.Length == 0) return false;
        return Abbreviations.Contains(last) || Abbreviations.Contains(char.ToLowerInvariant(last[0]) + last.Substring(1))
            || Abbreviations.Contains(last.ToUpperInvariant());
    }

    private static string LastWord(string text)
    {
        var parts = WordSeparators.Split(text);
        return parts[parts.Length - 1];
    }
}

/// <summary>
/// FireRedTTS's long-text splitter (<c>text_normalizer/utils.py</c>, <c>text_split</c>): sentences (sentencex), Chinese
/// sentences over 30 characters cut at the last strong or weak break (<c>zh_text_split</c>), then pieces merged until
/// each passes 30 characters, a short last piece (under 15) joined to the one before.
/// </summary>
internal static class FireRedTtsTextSplitter
{
    private static readonly TimeSpan Timeout = TimeSpan.FromSeconds(1);
    private static readonly Regex StrongBreak = new("([\u3002\u201d;\uff1b!\uff01\uff1a\u2026?\uff1f\uff09\\)\\]\u300f\u3011\u300d}~\r\n]| \\.)",
        RegexOptions.CultureInvariant, Timeout);

    /// <summary>The text's pieces.</summary>
    /// <param name="dropLastPiece">The release's behaviour: with two or more pieces the last one is removed even when it
    /// was not merged into the one before (15 characters or more), losing its text.</param>
    public static List<string> Split(string text, bool dropLastPiece = false)
    {
        List<string> substrings;
        if (FireRedTtsTextNormalizer.ContainsChinese(text))
        {
            substrings = new List<string>();
            foreach (var s in SentencexSegmenter.Segment(text))
            {
                if (PythonLength(s) > 30) substrings.AddRange(ChineseSplit(s, 30));
                else substrings.Add(s);
            }
        }
        else
        {
            substrings = SentencexSegmenter.Segment(text);
        }

        // merge substrings
        var final = new List<string>();
        string temp = "";
        for (int i = 0; i < substrings.Count; i++)
        {
            string s = substrings[i].Replace("\u3002", ",");
            temp = i == 0 ? s : temp + s;
            if (PythonLength(temp) > 30)
            {
                final.Add(temp);
                temp = "";
            }
            if (i == substrings.Count - 1)
            {
                if (temp.Length > 0) final.Add(temp);
                break;
            }
        }
        if (final.Count >= 2)
        {
            if (PythonLength(final[final.Count - 1]) < 15)
            {
                final[final.Count - 2] += final[final.Count - 1];
                final.RemoveAt(final.Count - 1);
            }
            else if (dropLastPiece)
            {
                final.RemoveAt(final.Count - 1);
            }
        }
        return final;
    }

    /// <summary><c>zh_text_split</c>: the text cut into pieces of at most <paramref name="length"/> characters at the
    /// last strong break, else the last weak one, in the window (the recursion continues at the default 80, as the
    /// release's calls omit the length).</summary>
    internal static List<string> ChineseSplit(string text, int length = 80)
    {
        var result = new List<string>();
        var points = FireRedTtsLanguageDetector.CodePoints(text);
        while (true)
        {
            if (length == 0) return result;
            if (points.Count <= length)
            {
                result.Add(FromPoints(points, 0, points.Count));
                return result;
            }
            var reversed = Enumerable.Reverse(points.GetRange(0, length)).ToList();
            string window = FromPoints(reversed, 0, reversed.Count);
            int strong = FirstMatch(window, StrongBreak);
            int weak = FirstIndex(reversed, IsWeakBreak);
            int endStrong = strong >= 0 ? length - strong : 0;
            int endWeak = weak >= 0 ? length - weak : 0;
            int cut;
            if (endStrong < length / 3)
            {
                if (endWeak < length / 3)
                {
                    int validMax = Math.Max(endStrong, endWeak);
                    cut = validMax >= 3 ? validMax : length;
                }
                else
                {
                    cut = endWeak;
                }
            }
            else
            {
                cut = endStrong;
            }
            result.Add(FromPoints(points, 0, cut));
            points = points.GetRange(cut, points.Count - cut);
            length = 80;
        }
    }

    // The code-point index of the first regex match in a string.
    private static int FirstMatch(string text, Regex regex)
    {
        var m = regex.Match(text);
        if (!m.Success) return -1;
        return FireRedTtsLanguageDetector.CodePoints(text.Substring(0, m.Index)).Count;
    }

    private static int FirstIndex(List<int> points, Func<int, bool> predicate)
    {
        for (int i = 0; i < points.Count; i++)
            if (predicate(points[i])) return i;
        return -1;
    }

    // weak_break: [\U00002702-\U000027B0\U0001f926-\U0001f937\U00010000-\U0001fbff\U00030000-\U0010ffff
    //              \u2640-\u2642\u2600-\u2b55\u23cf\u23e9\u231a\ufe0f\u3030\t\uff0c,. ]
    private static bool IsWeakBreak(int c) =>
        (c >= 0x2702 && c <= 0x27B0) || (c >= 0x1F926 && c <= 0x1F937) || (c >= 0x10000 && c <= 0x1FBFF) || (c >= 0x30000 && c <= 0x10FFFF)
        || (c >= 0x2640 && c <= 0x2642) || (c >= 0x2600 && c <= 0x2B55) || c == 0x23CF || c == 0x23E9 || c == 0x231A || c == 0xFE0F
        || c == 0x3030 || c == '\t' || c == 0xFF0C || c == ',' || c == '.' || c == ' ';

    private static string FromPoints(List<int> points, int start, int count)
    {
        var builder = new StringBuilder(count);
        for (int i = start; i < start + count; i++) builder.Append(char.ConvertFromUtf32(points[i]));
        return builder.ToString();
    }

    private static int PythonLength(string s) => FireRedTtsLanguageDetector.CodePoints(s).Count;
}

/// <summary>
/// FireRedTTS's text tokenizer (<c>VoiceBpeTokenizer</c>): the normalized text with its language tag, <c>[lang]text</c>,
/// encoded by the release's tiktoken vocabulary (Whisper's multilingual ranks plus CJK characters, 57,184 in all) with
/// its special tokens after them and the GPT-2 pattern extended for bracketed tags and CJK characters.
/// </summary>
internal sealed class FireRedTtsTokenizer
{
    private const string Resource = "AiDotNet.TextToSpeech.FrontEnd.fireredtts_multilingual.tiktoken.gz";

    /// <summary>The release's special tokens (<c>whisper_tokenizer.get_encoding</c>), in id order after the ranks.</summary>
    public static readonly IReadOnlyList<string> ReleaseSpecialTokens = BuildSpecials();

    /// <summary>The paper's token-insertion paralinguistic tags (Table 1) the release does not define, appended after its
    /// special tokens for instruction tuning.</summary>
    public static readonly IReadOnlyList<string> InstructionTokens = new[] { "[hic]", "[rep]", "[elong]", "[sss]", "[tsk]" };

    private static readonly Lazy<Dictionary<byte[], int>> Ranks = new(LoadRanks);
    private static readonly Lazy<FireRedTtsTokenizer> ReleaseInstance = new(() => new FireRedTtsTokenizer(false));
    private static readonly Lazy<FireRedTtsTokenizer> InstructionInstance = new(() => new FireRedTtsTokenizer(true));

    private FireRedTtsTokenizer(bool instructionTokens)
    {
        var specials = ReleaseSpecialTokens.ToList();
        if (instructionTokens) specials.AddRange(InstructionTokens.Where(t => !specials.Contains(t)));
        Encoding = new TiktokenEncoding(Ranks.Value, specials, TiktokenPattern.FireRed);
    }

    /// <summary>The release's tokenizer, or with the paper's instruction tags added.</summary>
    public static FireRedTtsTokenizer Get(bool instructionTokens) => instructionTokens ? InstructionInstance.Value : ReleaseInstance.Value;

    public TiktokenEncoding Encoding { get; }

    /// <summary>The number of ranks before the special tokens.</summary>
    public static int RankCount => Ranks.Value.Count;

    /// <summary><c>encode(text, lang)</c>: normalized, tagged and encoded.</summary>
    public int[] Encode(string text, string language = "auto")
    {
        if (language is not ("zh" or "en" or "auto")) throw new ArgumentException("The language is zh, en or auto.", nameof(language));
        var (normalized, detected) = FireRedTtsTextNormalizer.Normalize(PythonText.Strip(text));
        if (language == "auto") language = detected;
        return Encoding.Encode("[" + language + "]" + normalized);
    }

    /// <summary>The id of a special token.</summary>
    public int Special(string token) => Encoding.SpecialTokenId(token);

    private static IReadOnlyList<string> BuildSpecials()
    {
        var specials = new List<string>
        {
            "[STOP]", "[UNK]", "[SPACE]", "[START]", "[nospk]", "[spkemb]", "[emotionemb]", "[contextemb]", "[sbreak]", "[pbreak]",
            "[uvbreak]", "[bsing]", "[esing]", "[sing]", "[hum]", "[laugh]", "[break]", "[breath]", "[oralsii]", "[oralze]", "[prolong]",
            "[stress]", "[bstrong]", "[estrong]", "[hiccup]", "[inhale]", "[exhale]", "[emounknown]", "[happy]", "[neutral]", "[sad]",
            "[surprise]", "[angry]", "[disgust]", "[emo]", "[laugha]", "[laughb]", "[laughc]", "[orala]", "[oralb]", "[oralc]", "[orald]",
            "[orale]", "[breaka]", "[breakb]", "[breakc]", "[breakd]", "[breake]", "[breakf]", "[endoftext]", "[startoftranscript]",
        };
        specials.AddRange(WhisperTiktoken.LanguageCodes.Take(100).Select(code => "[" + code + "]"));
        specials.AddRange(new[] { "[translate]", "[transcribe]", "[startoflm]", "[startofprev]", "[nospeech]" });
        return specials;
    }

    private static Dictionary<byte[], int> LoadRanks()
    {
        using var stream = typeof(FireRedTtsTokenizer).Assembly.GetManifestResourceStream(Resource)
            ?? throw new System.IO.InvalidDataException($"The embedded resource {Resource} is missing.");
        using var gzip = new System.IO.Compression.GZipStream(stream, System.IO.Compression.CompressionMode.Decompress);
        using var reader = new System.IO.StreamReader(gzip, System.Text.Encoding.ASCII);
        return TiktokenEncoding.ReadRanks(reader);
    }
}
