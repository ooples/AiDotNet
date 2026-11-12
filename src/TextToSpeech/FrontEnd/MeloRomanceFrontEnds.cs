using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using AiDotNet.Tokenization.Algorithms;

namespace AiDotNet.TextToSpeech.FrontEnd;

/// <summary>
/// MeloTTS's Gruut wrapper (<c>es_phonemizer/gruut_wrapper.py</c> and <c>fr_phonemizer/gruut_wrapper.py</c> over
/// <c>BasePhonemizer</c> and <c>Punctuation</c>, keep_puncs and keep_stress on, separator ""): the punctuation marks
/// ;:,.!?¡¿—…"«»“” cut out of the text, each remaining piece phonemized by gruut, the marks put back. The two copies
/// differ in one line: Spanish drops leading empty pieces before phonemizing, French does not.
/// </summary>
internal sealed class MeloGruutPhonemizer
{
    private enum Position { Begin, End, Middle, Alone }

    private static readonly Regex Punctuation = new("([" + PythonText.WhitespaceClass + "]*[;:,.!?¡¿—…\"«»“”]+[" + PythonText.WhitespaceClass + "]*)+", RegexOptions.CultureInvariant);

    private readonly GruutLanguage _language;
    private readonly GruutTextProcessor _processor;
    private readonly bool _dropLeadingEmpty;

    public MeloGruutPhonemizer(GruutLanguage language, bool dropLeadingEmpty)
    {
        _language = language;
        _processor = new GruutTextProcessor(language);
        _dropLeadingEmpty = dropLeadingEmpty;
    }

    /// <summary><c>Gruut.phonemize(text, separator="")</c>, with a fresh lexicon cache (a fresh MeloTTS process).</summary>
    public string Phonemize(string text)
    {
        var session = new GruutLexiconSession(_language);
        var (pieces, marks) = StripToRestore(text);
        var phonemized = pieces.Select(p => _processor.Phonemize(p, session)).ToList();
        return Restore(phonemized, marks, 0)[0];
    }

    private (List<string> Pieces, List<(string Mark, Position Position)> Marks) StripToRestore(string text)
    {
        var matches = Punctuation.Matches(text).Cast<Match>().ToList();
        if (matches.Count == 0) return (new List<string> { text }, new List<(string, Position)>());
        if (matches.Count == 1 && matches[0].Value == text) return (new List<string>(), new List<(string, Position)> { (text, Position.Alone) });
        var marks = new List<(string Mark, Position Position)>();
        for (int i = 0; i < matches.Count; i++)
        {
            var position = Position.Middle;
            if (i == 0 && text.StartsWith(matches[i].Value, StringComparison.Ordinal)) position = Position.Begin;
            else if (i == matches.Count - 1 && text.EndsWith(matches[i].Value, StringComparison.Ordinal)) position = Position.End;
            marks.Add((matches[i].Value, position));
        }
        var pieces = new List<string>();
        for (int i = 0; i < marks.Count; i++)
        {
            int at = text.IndexOf(marks[i].Mark, StringComparison.Ordinal);
            string prefix = at < 0 ? text : text.Substring(0, at);
            string suffix = at < 0 ? "" : text.Substring(at + marks[i].Mark.Length);
            pieces.Add(prefix);
            if (i == marks.Count - 1 && suffix.Length > 0) pieces.Add(suffix);
            text = suffix;
        }
        if (_dropLeadingEmpty)
            while (pieces[0].Length == 0) pieces.RemoveAt(0);
        return (pieces, marks);
    }

    private static List<string> Restore(List<string> text, List<(string Mark, Position Position)> marks, int num)
    {
        if (marks.Count == 0) return text;
        if (text.Count == 0) return new List<string> { string.Concat(marks.Select(m => m.Mark)) };
        var (mark, position) = marks[0];
        var rest = marks.GetRange(1, marks.Count - 1);
        switch (position)
        {
            case Position.Begin:
                return Restore(new[] { mark + text[0] }.Concat(text.Skip(1)).ToList(), rest, num);
            case Position.End:
                return new[] { text[0] + mark }.Concat(Restore(text.Skip(1).ToList(), rest, num + 1)).ToList();
            case Position.Alone:
                throw new InvalidOperationException("A lone punctuation mark among phonemized text has no restore rule ('_punc_index' has no 'mark').");
            default:
                if (text.Count == 1) return Restore(new List<string> { text[0] + mark }, rest, num);
                return Restore(new[] { text[0] + mark + text[1] }.Concat(text.Skip(2)).ToList(), rest, num);
        }
    }
}

/// <summary>What MeloTTS's Spanish and French front ends share: BERT word pieces grouped into words, each word phonemized
/// alone and its phones spread over its pieces.</summary>
internal abstract class MeloRomanceFrontEnd
{
    private readonly WordPieceTokenizer _tokenizer;

    protected MeloRomanceFrontEnd(WordPieceTokenizer tokenizer) => _tokenizer = tokenizer;

    /// <summary>The word pieces of the language's BERT tokenizer.</summary>
    public List<string> WordPieces(string text) => _tokenizer.Tokenize(text);

    /// <summary>The language's <c>text_normalize</c>.</summary>
    public abstract string Normalize(string text);

    /// <summary>The phonemes of one word (<c>es2ipa</c> / <c>fr2ipa</c>).</summary>
    public abstract string WordToIpa(string word);

    /// <summary>
    /// The language's <c>g2p(text)</c>: per word, its phonemes' characters (spaces dropped) with tone 0, and the
    /// phones of each word piece, with the silence symbol at each end. An unknown piece reads "UNK".
    /// </summary>
    public (List<string> Phones, List<int> Tones, List<int> PhonesPerPiece) G2P(string text)
    {
        var groups = new List<List<string>>();
        foreach (var token in WordPieces(text))
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
            var wordPhones = word == "[UNK]"
                ? new List<string> { "UNK" }
                : ChineseText.Codepoints(WordToIpa(word)).Where(p => p != " ").ToList();
            phones.AddRange(wordPhones);
            perPiece.AddRange(MeloEnglishFrontEnd.DistributePhones(wordPhones.Count, group.Count));
        }
        var tones = Enumerable.Repeat(0, phones.Count).ToList();
        phones.Insert(0, "_");
        phones.Add("_");
        tones.Insert(0, 0);
        tones.Add(0);
        perPiece.Insert(0, 1);
        perPiece.Add(1);
        return (phones, tones, perPiece);
    }

    // Shared cleaner steps (es_phonemizer/cleaner.py, fr_phonemizer/cleaner.py).
    protected static string ReplaceMap(string text, (string From, string To)[] map)
    {
        var pattern = new Regex(string.Join("|", map.Select(m => Regex.Escape(m.From))), RegexOptions.CultureInvariant);
        return pattern.Replace(text, m => map.First(r => r.From == m.Value).To);
    }

    private static readonly Regex BeginPunctuation = new("^[,.!?]+", RegexOptions.CultureInvariant);
    private static readonly Regex Whitespace = new("[" + PythonText.WhitespaceClass + "]+", RegexOptions.CultureInvariant);
    private static readonly Regex FinalMark = new(@"([^\.,!\?\-…])$", RegexOptions.CultureInvariant);

    protected static string Finish(string text)
    {
        text = BeginPunctuation.Replace(text, "");
        text = PythonText.Strip(Whitespace.Replace(text, " "));
        return FinalMark.Replace(text, "$1.");
    }
}

/// <summary>
/// MeloTTS's Spanish front end (myshell-ai/MeloTTS <c>melo/text/spanish.py</c>): <c>spanish_cleaners</c>, BETO's
/// word pieces, each word phonemized by gruut (es-es, eSpeak lexicon).
/// </summary>
internal sealed class MeloSpanishFrontEnd : MeloRomanceFrontEnd
{
    private static readonly Lazy<MeloSpanishFrontEnd> SharedInstance = new(() => new MeloSpanishFrontEnd(), isThreadSafe: true);
    private readonly MeloGruutPhonemizer _phonemizer = new(GruutLanguage.Spanish, dropLeadingEmpty: true);

    public static MeloSpanishFrontEnd Default => SharedInstance.Value;

    private MeloSpanishFrontEnd() : base(WordPieceTokenizer.BertBaseSpanishWwmUncased())
    {
    }

    private static readonly (string, string)[] Map =
    {
        ("：", ","), ("；", ","), ("，", ","), ("。", "."), ("！", "!"), ("？", "?"), ("\n", "."), ("·", ","), ("、", ","),
        ("...", "."), ("…", "."), ("$", "."), ("“", "'"), ("”", "'"), ("‘", "'"), ("’", "'"), ("（", "'"), ("）", "'"),
        ("(", "'"), (")", "'"), ("《", "'"), ("》", "'"), ("【", "'"), ("】", "'"), ("[", "'"), ("]", "'"), ("—", ""),
        ("～", "-"), ("~", "-"), ("「", "'"), ("」", "'"),
    };

    private static readonly Regex AuxSymbols = new("[<>()\\[\\]\"«»']+", RegexOptions.CultureInvariant);

    /// <summary><c>spanish_cleaners</c>.</summary>
    public override string Normalize(string text)
    {
        text = PythonText.Lower(text);
        text = text.Replace(";", ",").Replace("-", " ").Replace(":", ",").Replace("&", "y").Replace("'", "");
        text = ReplaceMap(text, Map);
        text = AuxSymbols.Replace(text, "");
        return Finish(text);
    }

    /// <summary><c>es_to_ipa.es2ipa(word)</c>.</summary>
    public override string WordToIpa(string word) => _phonemizer.Phonemize(word);
}

/// <summary>
/// MeloTTS's French front end (myshell-ai/MeloTTS <c>melo/text/french.py</c>): <c>french_cleaners</c> (abbreviations
/// expanded, case kept), dbmdz's cased French BERT word pieces, each word phonemized by gruut (fr-fr, eSpeak lexicon,
/// part-of-speech tags and liaison), runs of three or more t removed.
/// </summary>
internal sealed class MeloFrenchFrontEnd : MeloRomanceFrontEnd
{
    private static readonly Lazy<MeloFrenchFrontEnd> SharedInstance = new(() => new MeloFrenchFrontEnd(), isThreadSafe: true);
    private readonly MeloGruutPhonemizer _phonemizer = new(GruutLanguage.French, dropLeadingEmpty: false);

    public static MeloFrenchFrontEnd Default => SharedInstance.Value;

    private MeloFrenchFrontEnd() : base(WordPieceTokenizer.BertBaseFrenchEuropeanaCased())
    {
    }

    // french_abbreviations.abbreviations_fr: the patterns are written unescaped ("N.B" lets "." match any character),
    // \b before each. The first group ignores case and needs a trailing period; the second does neither.
    private static readonly (Regex Pattern, string Replacement)[] Abbreviations = BuildAbbreviations();

    private static (Regex, string)[] BuildAbbreviations()
    {
        const string Boundary = "(?<![" + PythonText.WordClass + "])";
        var dotted = new[]
        {
            ("M", "monsieur"), ("Mlle", "mademoiselle"), ("Mlles", "mesdemoiselles"), ("Mme", "Madame"), ("Mmes", "Mesdames"),
            ("N.B", "nota bene"), ("M", "monsieur"), ("p.c.q", "parce que"), ("Pr", "professeur"), ("qqch", "quelque chose"),
            ("rdv", "rendez-vous"), ("max", "maximum"), ("min", "minimum"), ("no", "numéro"), ("adr", "adresse"),
            ("dr", "docteur"), ("st", "saint"), ("co", "companie"), ("jr", "junior"), ("sgt", "sergent"),
            ("capt", "capitain"), ("col", "colonel"), ("av", "avenue"), ("av. J.-C", "avant Jésus-Christ"),
            ("apr. J.-C", "après Jésus-Christ"), ("art", "article"), ("boul", "boulevard"), ("c.-à-d", "c’est-à-dire"),
            ("etc", "et cetera"), ("ex", "exemple"), ("excl", "exclusivement"), ("boul", "boulevard"),
        };
        var bare = new[] { ("Mlle", "mademoiselle"), ("Mlles", "mesdemoiselles"), ("Mme", "Madame"), ("Mmes", "Mesdames") };
        return dotted.Select(a => (new Regex(Boundary + a.Item1 + @"\.", RegexOptions.CultureInvariant | RegexOptions.IgnoreCase), a.Item2))
            .Concat(bare.Select(a => (new Regex(Boundary + a.Item1, RegexOptions.CultureInvariant), a.Item2)))
            .ToArray();
    }

    private static readonly (string, string)[] Map =
    {
        ("：", ","), ("；", ","), ("，", ","), ("。", "."), ("！", "!"), ("？", "?"), ("\n", "."), ("·", ","), ("、", ","),
        ("...", "."), ("…", "."), ("$", "."), ("“", ""), ("”", ""), ("‘", ""), ("’", ""), ("（", ""), ("）", ""),
        ("(", ""), (")", ""), ("《", ""), ("》", ""), ("【", ""), ("】", ""), ("[", ""), ("]", ""), ("—", ""),
        ("～", "-"), ("~", "-"), ("「", ""), ("」", ""), ("¿", ""), ("¡", ""),
    };

    private static readonly Regex AuxSymbols = new("[<>()\\[\\]\"«»]+", RegexOptions.CultureInvariant);

    /// <summary><c>french_cleaners</c>.</summary>
    public override string Normalize(string text)
    {
        foreach (var (pattern, replacement) in Abbreviations) text = pattern.Replace(text, replacement.Replace("$", "$$"));
        text = ReplaceMap(text, Map);
        text = text.Replace(";", ",").Replace("-", " ").Replace(":", ",").Replace("&", " et ");
        text = AuxSymbols.Replace(text, "");
        return Finish(text);
    }

    /// <summary><c>fr_to_ipa.fr2ipa(word)</c>: gruut's phonemes with runs of three or more t removed.</summary>
    public override string WordToIpa(string word) => RemoveConsecutiveT(_phonemizer.Phonemize(word));

    private static string RemoveConsecutiveT(string input)
    {
        var result = new StringBuilder(input.Length);
        int count = 0;
        foreach (char c in input)
        {
            if (c == 't')
            {
                count++;
                continue;
            }
            if (count < 3) result.Append('t', count);
            count = 0;
            result.Append(c);
        }
        if (count < 3) result.Append('t', count);
        return result.ToString();
    }
}
