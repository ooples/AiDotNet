using System.Linq;
using System.Numerics;
using System.Text.RegularExpressions;

namespace AiDotNet.TextToSpeech.FrontEnd;

/// <summary>
/// gruut 2.2.3's <c>TextProcessor</c> for plain text in one language, as MeloTTS's Gruut wrapper calls it
/// (<c>gruut.sentences(text, lang, espeak=True)</c>): the text tree split into words, up to five passes of
/// replacements, punctuation and break splitting, sentence breaking, date/currency/number detection and
/// verbalization and word breaking; then per sentence the part-of-speech tags (French), lexicon look-up or CRF guess,
/// and French liaison. <see cref="Phonemize"/> returns what the wrapper's <c>phonemize_gruut(text, separator="")</c>
/// builds from the sentences.
/// </summary>
internal sealed class GruutTextProcessor
{
    private enum NodeKind { Container, Sentence, Word, Break, Punctuation }

    private sealed class Node
    {
        public NodeKind Kind;
        public Node? Parent;
        public readonly List<Node> Children = new();
        public string Text = "";
        public string TextWithWs = "";
        public string InterpretAs = "";
        public PythonDecimal? Number;
        public DateTime? Date;
        public string? CurrencySymbol;
        public string? CurrencyName;
        public string Format = "";
        public string Role = "";
        public string? Pos;
        public List<string>? Phonemes;
        public bool? InLexicon;
        public bool Implicit = true;
        public bool IsMaybeNumber = true, IsMaybeDate = true, IsMaybeCurrency = true;
        public bool MajorBreak;

        public void Add(Node child)
        {
            child.Parent = this;
            Children.Add(child);
        }
    }

    private readonly GruutLanguage _language;

    public GruutTextProcessor(GruutLanguage language) => _language = language;

    // ------------------------------------------------------------------ whitespace helpers (gruut.const)

    private static readonly Regex WordPattern = new("([" + PythonText.WhitespaceClass + @"]*[^" + PythonText.WhitespaceClass + "]+(?:[" + PythonText.WhitespaceClass + "]+|$))", RegexOptions.CultureInvariant);
    private static readonly Regex Surrounding = new("^([" + PythonText.WhitespaceClass + "]*)[^" + PythonText.WhitespaceClass + "]+([" + PythonText.WhitespaceClass + "]*)$", RegexOptions.CultureInvariant);
    private static readonly Regex Spaces = new("[" + PythonText.WhitespaceClass + "]+", RegexOptions.CultureInvariant);
    private static readonly Regex HasDigit = new("[0-9]", RegexOptions.CultureInvariant);
    private static readonly Regex AllDigits = new(@"^\d+$", RegexOptions.CultureInvariant);

    private static IEnumerable<string> SplitWords(string s) => WordPattern.Matches(s).Cast<Match>().Select(m => m.Value).Where(v => v.Length > 0);
    private static string Normalize(string s) => Spaces.Replace(PythonText.Strip(s), " ");

    private static (string First, string Last) GetWhitespace(string s)
    {
        var m = Surrounding.Match(s);
        return m.Success ? (m.Groups[1].Value, m.Groups[2].Value) : ("", "");
    }

    // re.split(pattern, s, maxsplit=1) for a pattern with one group, empty parts dropped.
    private static List<string> SplitOnce(Regex pattern, string s)
    {
        var m = pattern.Match(s);
        var parts = m.Success ? new List<string> { s.Substring(0, m.Index), m.Groups[1].Value, s.Substring(m.Index + m.Length) } : new List<string> { s };
        return parts.Where(p => p.Length > 0).ToList();
    }

    // ------------------------------------------------------------------ the pipeline

    /// <summary>The phonemized text of one wrapper segment, with the session's lexicon cache.</summary>
    public string Phonemize(string text, GruutLexiconSession session)
    {
        var root = new Node { Kind = NodeKind.Container };
        var sentence = new Node { Kind = NodeKind.Sentence };
        root.Add(sentence);
        foreach (var wordText in SplitWords(text))
        {
            string norm = Normalize(wordText);
            if (norm.Length == 0) continue;
            sentence.Add(new Node { Kind = NodeKind.Word, Text = norm, TextWithWs = wordText, InLexicon = InLexicon(norm, session) });
        }

        for (int pass = 0; pass < 5; pass++)
        {
            bool changed = false;
            changed |= Split(root, n => SplitReplacements(n, session));
            changed |= Split(root, n => SplitPunctuations(n, session));
            changed |= Split(root, n => SplitBreak(n, session, major: false));
            changed |= Split(root, n => SplitBreak(n, session, major: true));
            changed |= BreakSentences(root);
            changed |= Transform(root, TransformDate);
            changed |= Transform(root, TransformCurrency);
            changed |= Transform(root, TransformNumber);
            changed |= Split(root, VerbalizeDate);
            changed |= Split(root, VerbalizeNumber);
            changed |= Split(root, VerbalizeCurrency);
            changed |= Split(root, n => BreakWords(n, session));
            if (!changed) break;
        }

        // Every sentence is phonemized, then post-processed, before anything is read: liaison can lengthen a phoneme
        // list an earlier sentence shares.
        var sentences = root.Children.Select(s => Leaves(s).ToList()).ToList();
        foreach (var leaves in sentences) ProcessSentence(leaves.Where(l => l.Kind == NodeKind.Word).ToList(), session);
        if (_language.IsFrench)
            foreach (var leaves in sentences) Liaison(leaves.Where(l => l.Kind == NodeKind.Word).ToList());

        var output = new List<List<string>>();
        foreach (var leaves in sentences)
        {
            foreach (var leaf in leaves)
            {
                if (leaf.Kind == NodeKind.Break)
                {
                    if (output.Count > 0) output[output.Count - 1].Add(leaf.Text);
                    else output.Add(new List<string> { leaf.Text });
                }
                else if (leaf.Kind == NodeKind.Word && leaf.Phonemes is { Count: > 0 })
                {
                    var characters = new List<string>();
                    foreach (var phoneme in leaf.Phonemes)
                        characters.AddRange(ChineseText.Codepoints(phoneme.Replace('g', 'ɡ')));
                    if (characters.Count > 0) output.Add(characters);
                }
            }
        }
        return string.Join(" ", output.Select(chars => string.Concat(chars)));
    }

    private bool InLexicon(string word, GruutLexiconSession session) => session.Lookup(word, null, doTransforms: false) is { Count: > 0 };

    private static IEnumerable<Node> Leaves(Node node)
    {
        if (node.Children.Count == 0)
        {
            yield return node;
            yield break;
        }
        foreach (var child in node.Children)
            foreach (var leaf in Leaves(child)) yield return leaf;
    }

    private static bool Split(Node root, Func<Node, IEnumerable<Node>> split)
    {
        bool changed = false;
        foreach (var leaf in Leaves(root).ToList())
            foreach (var child in split(leaf))
            {
                leaf.Add(child);
                changed = true;
            }
        return changed;
    }

    private static bool Transform(Node root, Func<Node, bool> transform)
    {
        bool changed = false;
        foreach (var leaf in Leaves(root).ToList())
            if (transform(leaf)) changed = true;
        return changed;
    }

    private Node NewWord(string text, string textWithWs, GruutLexiconSession? session) =>
        new() { Kind = NodeKind.Word, Text = text, TextWithWs = textWithWs, InLexicon = session is null ? null : InLexicon(text, session) };

    private static bool Spoken(Node word) => word.InterpretAs.Length > 0 || word.InLexicon == true;

    // _split_replacements: ’ -> '
    private IEnumerable<Node> SplitReplacements(Node word, GruutLexiconSession session)
    {
        if (word.Kind != NodeKind.Word || Spoken(word) || !word.TextWithWs.Contains("’")) yield break;
        foreach (var part in SplitWords(word.TextWithWs.Replace("’", "'")))
        {
            string norm = Normalize(part);
            if (norm.Length == 0) continue;
            yield return NewWord(norm, part, session);
        }
    }

    private IEnumerable<Node> SplitPunctuations(Node word, GruutLexiconSession session)
    {
        if (word.Kind != NodeKind.Word || Spoken(word)) yield break;
        string wordText = word.Text;
        var (firstWs, lastWs) = GetWhitespace(word.TextWithWs);
        bool hasPunctuation = false;
        var produced = new List<Node>();

        var parts = SplitOnce(_language.BeginPattern, wordText);
        bool firstWord = true;
        while (wordText.Length > 0 && parts.Count == 2)
        {
            string punct = parts[0];
            wordText = parts[1];
            if (firstWord)
            {
                punct = firstWs + punct;
                firstWord = false;
            }
            hasPunctuation = true;
            produced.Add(new Node { Kind = NodeKind.Punctuation, Text = Normalize(punct), TextWithWs = punct });
            parts = SplitOnce(_language.BeginPattern, wordText);
        }

        var endPunctuations = new List<string>();
        parts = SplitOnce(_language.EndPattern, wordText);
        while (wordText.Length > 0 && parts.Count == 2)
        {
            wordText = parts[0];
            hasPunctuation = true;
            endPunctuations.Add(parts[1]);
            parts = SplitOnce(_language.EndPattern, wordText);
        }

        if (!hasPunctuation) yield break;
        if (endPunctuations.Count == 0) wordText += lastWs;
        foreach (var node in produced) yield return node;
        if (wordText.Length > 0) yield return NewWord(Normalize(wordText), wordText, session);
        endPunctuations.Reverse();
        for (int i = 0; i < endPunctuations.Count; i++)
        {
            string punct = endPunctuations[i];
            if (i == endPunctuations.Count - 1) punct += lastWs;
            yield return new Node { Kind = NodeKind.Punctuation, Text = PythonText.Strip(punct), TextWithWs = punct };
        }
    }

    // _split_minor_breaks / _split_major_breaks: only the text before the first break and that break are kept.
    private IEnumerable<Node> SplitBreak(Node word, GruutLexiconSession session, bool major)
    {
        if (word.Kind != NodeKind.Word || Spoken(word)) yield break;
        var m = (major ? _language.MajorBreakPattern : _language.MinorBreakPattern).Match(word.TextWithWs);
        if (!m.Success) yield break;
        string wordPart = word.TextWithWs.Substring(0, m.Index);
        string breakPart = m.Groups[1].Value;
        if (PythonText.Strip(wordPart).Length > 0) yield return NewWord(Normalize(wordPart), wordPart, session);
        else if (major) breakPart = wordPart + breakPart;
        yield return new Node { Kind = NodeKind.Break, MajorBreak = major, Text = Normalize(breakPart), TextWithWs = breakPart };
    }

    private static bool BreakSentences(Node root)
    {
        bool changed = false;
        foreach (var leaf in Leaves(root).ToList())
        {
            if (leaf.Kind != NodeKind.Break || !leaf.MajorBreak) continue;
            var below = leaf;
            while (below.Parent!.Kind != NodeKind.Sentence) below = below.Parent;
            var sentence = below.Parent;
            int index = sentence.Children.IndexOf(below);
            if (index == sentence.Children.Count - 1) continue;
            var moved = sentence.Children.GetRange(index + 1, sentence.Children.Count - index - 1);
            sentence.Children.RemoveRange(index + 1, moved.Count);
            var next = new Node { Kind = NodeKind.Sentence, Parent = root };
            root.Children.Insert(root.Children.IndexOf(sentence) + 1, next);
            foreach (var node in moved) next.Add(node);
            changed = true;
        }
        return changed;
    }

    // ------------------------------------------------------------------ transforms

    private bool TransformDate(Node word)
    {
        if (word.Kind != NodeKind.Word || !word.IsMaybeDate || (word.InterpretAs.Length > 0 && word.InterpretAs != "date")) return false;
        if (!HasDigit.IsMatch(word.Text))
        {
            word.IsMaybeDate = false;
            return false;
        }
        if (_language.TryParseDate(word.Text, out var date))
        {
            word.InterpretAs = "date";
            word.Date = date;
        }
        return true;
    }

    private bool TransformCurrency(Node word)
    {
        if (word.Kind != NodeKind.Word || !word.IsMaybeCurrency || (word.InterpretAs.Length > 0 && word.InterpretAs != "currency")) return false;
        if (!HasDigit.IsMatch(word.Text))
        {
            word.IsMaybeCurrency = false;
            return false;
        }
        foreach (var (symbol, _) in _language.Currencies)
        {
            if (!word.Text.StartsWith(symbol, StringComparison.Ordinal)) continue;
            if (_language.TryParseDecimal(word.Text.Substring(symbol.Length), out var number))
            {
                word.InterpretAs = "currency";
                word.CurrencySymbol = symbol;
                word.Number = number;
                break;
            }
        }
        return true;
    }

    private bool TransformNumber(Node word)
    {
        if (word.Kind != NodeKind.Word || !word.IsMaybeNumber || (word.InterpretAs.Length > 0 && word.InterpretAs != "number")) return false;
        if (_language.TryParseDecimal(word.Text, out var number) && number.IsFinite)
        {
            word.InterpretAs = "number";
            if (word.Format.Length == 0) word.Format = "cardinal";
            word.Number = number;
            if (number.Between(1000, 3000) && AllDigits.IsMatch(word.Text)) word.Format = "year";
        }
        else
        {
            word.IsMaybeNumber = false;
        }
        return true;
    }

    // ------------------------------------------------------------------ verbalization

    private IEnumerable<Node> WordsOf(string verbalized, Node word)
    {
        var (firstWs, lastWs) = GetWhitespace(word.TextWithWs);
        foreach (var part in SplitWords(firstWs + verbalized + lastWs))
        {
            string norm = Normalize(part);
            if (norm.Length == 0) continue;
            yield return NewWord(norm, part, null);
        }
    }

    private IEnumerable<Node> VerbalizeNumber(Node word)
    {
        if (word.Kind != NodeKind.Word || word.InterpretAs != "number" || word.Number is null || !HasDigit.IsMatch(word.Text))
            return Enumerable.Empty<Node>();
        var number = word.Number.Value;
        string text = number.HasFraction
            ? _language.Numbers.ToCardinal(number.ToDouble())
            : word.Format == "year" ? _language.Numbers.ToYear(number.Truncate()) : _language.Numbers.ToCardinal(number.Truncate());
        return WordsOf(text, word).ToList();
    }

    private IEnumerable<Node> VerbalizeDate(Node word)
    {
        if (word.Kind != NodeKind.Word || word.InterpretAs != "date" || word.Date is null) return Enumerable.Empty<Node>();
        var date = word.Date.Value;
        string text;
        try
        {
            string month = _language.Months[date.Month - 1];
            string day = _language.IsFrench ? Num2WordsFrench.Default.ToOrdinal(new BigInteger(date.Day)) : _language.Numbers.ToCardinal(new BigInteger(date.Day));
            string year = _language.Numbers.ToYear(new BigInteger(date.Year));
            text = string.Join(" ", _language.DateFormat.Select(c => c switch { 'M' => month, 'Y' => year, _ => day }));
        }
        catch (Exception ex) when (ex is not OutOfMemoryException)
        {
            // gruut logs the failure and leaves the word as written.
            return Enumerable.Empty<Node>();
        }
        return WordsOf(text, word).ToList();
    }

    private IEnumerable<Node> VerbalizeCurrency(Node word)
    {
        if (word.Kind != NodeKind.Word || word.InterpretAs != "currency" || (word.CurrencySymbol is null && word.CurrencyName is null) || word.Number is null)
            return Enumerable.Empty<Node>();
        var number = word.Number.Value;
        if (number.Kind is PythonDecimal.DecimalKind.Infinity or PythonDecimal.DecimalKind.SignalingNaN)
            throw new ArithmeticException("InvalidOperation: the amount has no fractional part to test.");
        bool hasFraction = number.Kind == PythonDecimal.DecimalKind.QuietNaN || number.HasFraction;
        if (string.IsNullOrEmpty(word.CurrencyName))
            word.CurrencyName = _language.CurrencyCodes.TryGetValue(word.CurrencySymbol ?? "", out var code) ? code : _language.DefaultCurrency;
        string text;
        try
        {
            text = _language.Numbers.ToCurrency(number.ToDouble(), word.CurrencyName!, "|");
        }
        catch (Exception ex) when (ex is not OutOfMemoryException)
        {
            // gruut logs the failure and leaves the word as written.
            return Enumerable.Empty<Node>();
        }
        int bar = text.IndexOf('|');
        text = hasFraction ? text.Replace("|", "") : bar >= 0 ? text.Substring(0, bar) : text;
        return WordsOf(text, word).ToList();
    }

    private IEnumerable<Node> BreakWords(Node word, GruutLexiconSession session)
    {
        if (word.Kind != NodeKind.Word || word.InterpretAs.Length > 0 || word.InLexicon == true || !word.Implicit) yield break;
        var parts = _language.WordBreakPattern.Split(word.Text);
        if (parts.Length < 2) yield break;
        var (firstWs, lastWs) = GetWhitespace(word.TextWithWs);
        for (int i = 0; i < parts.Length; i++)
        {
            string norm = Normalize(parts[i]);
            if (norm.Length == 0) continue;
            string part = parts[i];
            if (i == 0) part = firstWs + part;
            part += i == parts.Length - 1 ? lastWs : " ";
            yield return NewWord(norm, part, session);
        }
    }

    // ------------------------------------------------------------------ per sentence

    private void ProcessSentence(List<Node> words, GruutLexiconSession session)
    {
        if (words.Count == 0) return;
        if (_language.IsFrench)
        {
            var tags = _language.TagPartsOfSpeech(words.Select(w => w.Text).ToList());
            for (int i = 0; i < words.Count; i++)
            {
                words[i].Pos = tags[i];
                if (words[i].Role.Length == 0) words[i].Role = "gruut:" + tags[i];
            }
        }
        foreach (var word in words)
        {
            if (word.Phonemes is { Count: > 0 }) continue;
            word.Phonemes = session.Lookup(word.Text, word.Role);
            if (word.Phonemes is not { Count: > 0 }) word.Phonemes = _language.GuessPhonemes(word.Text);
        }
    }

    private static readonly HashSet<string> FrenchVowels = new(StringComparer.Ordinal)
        { "i", "y", "u", "e", "ø", "o", "ə", "ɛ", "œ", "ɔ", "a", "ɔ̃", "ɛ̃", "ɑ̃", "œ̃" };

    private static bool HasSilentConsonant(string lastChar, string lastPhoneme) => lastChar switch
    {
        "d" or "p" or "t" => lastPhoneme != lastChar,
        "r" => lastPhoneme != "ʁ",
        "s" or "x" or "z" => lastPhoneme != "s" && lastPhoneme != "z",
        "n" => lastPhoneme != "n" && lastPhoneme != "ŋ",
        _ => false,
    };

    // fr_post_process_sentence: liaison appends to word 1's phoneme list (shared with its other occurrences).
    private static void Liaison(List<Node> words)
    {
        for (int i = 0; i + 1 < words.Count; i++)
        {
            var word1 = words[i];
            var word2 = words[i + 1];
            if (word1.Text.Length == 0 || word1.Phonemes is not { Count: > 0 } || word2.Text.Length == 0 || word2.Phonemes is not { Count: > 0 })
                continue;
            var chars = ChineseText.Codepoints(word1.Text);
            string lastChar = chars[chars.Count - 1];
            if (!HasSilentConsonant(lastChar, word1.Phonemes[word1.Phonemes.Count - 1]) || !FrenchVowels.Contains(word2.Phonemes[0]))
                continue;
            bool liaison;
            if (word1.Text == "et") liaison = false;
            else if (word1.Pos is "DET" or "NUM") liaison = true;
            else if (word1.Pos == "PRON" && word2.Pos is "AUX" or "VERB") liaison = true;
            else if (word1.Pos == "ADP" || word1.Text == "très") liaison = true;
            else if (word1.Pos == "ADJ" && word2.Pos is "NOUN" or "PROPN") liaison = true;
            else liaison = word1.Pos is "AUX" or "VERB";
            if (!liaison) continue;
            if (lastChar is "s" or "x" or "z") word1.Phonemes.Add("z");
            else if (lastChar == "d") word1.Phonemes.Add("t");
            else if (lastChar is "t" or "p" or "n") word1.Phonemes.Add(lastChar);
        }
    }
}
