using System.Linq;
using System.Text;

namespace AiDotNet.Safety.Watermarking;

/// <summary>
/// Base class for post-hoc text watermarks that rewrite substitution sites to a keyed choice: at every occurrence of
/// a phrase from a group of interchangeable phrases, the key and the preceding word select which member of the group
/// the text uses.
/// </summary>
/// <remarks>
/// <para>
/// <b>Embedding</b> scans the text left to right. At each site, a phrase from one of the groups, the key and the word
/// before the site (as it reads after any earlier substitution) are hashed with HMAC-SHA256 to choose a member of the
/// group, and the phrase is replaced by that member with its capitalisation kept. Text without sites is unchanged,
/// and embedding twice gives the same text.
/// </para>
/// <para>
/// <b>Detection</b> finds the same sites and counts how many use the keyed choice. Without the watermark each site
/// matches with probability 1/|group|, so the count gives z = (hits - sum p_i) / sqrt(sum p_i (1 - p_i)). Each
/// (group, preceding word) pair is scored once, so a repeated phrase cannot inflate z. With two-member groups a
/// fully marked text reaches z = sqrt(n) for n distinct sites: 36 sites give z = 6, full confidence.
/// </para>
/// <para>
/// Every site carries the keyed choice, so <c>watermarkStrength</c> does not change the embedding; it is validated
/// and kept for the shared watermarker API.
/// </para>
/// <para>
/// <b>For Beginners:</b> some words and phrases can be swapped without changing the meaning ("big" and "large",
/// "do not" and "don't"). The watermark picks one of them at every such place using the secret key. Ordinary text
/// agrees with the key's choice about half the time; marked text agrees every time.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public abstract class KeyedSubstitutionWatermarker<T> : TextWatermarkerBase<T>, ITextWatermarkEmbedder<T>
{
    private readonly string[][] _groups;
    private readonly Dictionary<string, (int Group, int Variant)> _index = new(StringComparer.Ordinal);
    private readonly int _maxPhraseWords;

    /// <summary>
    /// Initializes the substitution watermarker.
    /// </summary>
    /// <param name="groups">
    /// The groups of interchangeable phrases, each with at least two members. Members are matched case-insensitively
    /// on whole words; a member may span several words separated by single spaces. A phrase may belong to one group only.
    /// </param>
    /// <param name="watermarkStrength">Validated and kept for the shared API; every site is embedded.</param>
    /// <param name="key">The secret key; <c>null</c> uses the published <see cref="WatermarkKey.Public"/> key.</param>
    protected KeyedSubstitutionWatermarker(IReadOnlyList<IReadOnlyList<string>> groups, double watermarkStrength, string? key)
        : base(watermarkStrength, key)
    {
        if (groups is null) throw new ArgumentNullException(nameof(groups));
        if (groups.Count == 0) throw new ArgumentException("At least one substitution group is required.", nameof(groups));
        _groups = new string[groups.Count][];
        for (int g = 0; g < groups.Count; g++)
        {
            var group = groups[g] ?? throw new ArgumentException($"Substitution group {g} is null.", nameof(groups));
            if (group.Count < 2) throw new ArgumentException($"Substitution group {g} needs at least two members.", nameof(groups));
            _groups[g] = new string[group.Count];
            for (int v = 0; v < group.Count; v++)
            {
                string phrase = Normalize(group[v] ?? string.Empty);
                if (phrase.Length == 0 || phrase.Split(' ').Any(w => !IsWord(w)))
                    throw new ArgumentException($"'{group[v]}' in group {g} is not a sequence of words.", nameof(groups));
                if (_index.ContainsKey(phrase))
                    throw new ArgumentException($"'{group[v]}' appears in more than one place among the substitution groups.", nameof(groups));
                _index[phrase] = (g, v);
                _groups[g][v] = phrase;
                _maxPhraseWords = Math.Max(_maxPhraseWords, phrase.Split(' ').Length);
            }
        }
    }

    /// <summary>Separates this watermarker's choices from those of other watermarkers sharing the key.</summary>
    protected abstract int SeedSalt { get; }

    /// <summary>The number of substitution groups.</summary>
    public int GroupCount => _groups.Length;

    /// <inheritdoc />
    public string EmbedWatermark(string text)
    {
        if (text is null) throw new ArgumentNullException(nameof(text));
        var words = Words(text);
        var output = new StringBuilder(text.Length + 16);
        int copied = 0;
        string previous = string.Empty;
        int i = 0;
        while (i < words.Count)
        {
            if (TryMatch(text, words, i, out var site))
            {
                string chosen = _groups[site.Group][Choice(site.Group, previous)];
                output.Append(text, copied, words[i].Start - copied);
                output.Append(MatchCase(text.Substring(words[i].Start, site.End - words[i].Start), chosen));
                copied = site.End;
                previous = chosen.Substring(chosen.LastIndexOf(' ') + 1);
                i += site.WordCount;
                continue;
            }

            previous = Normalize(text.Substring(words[i].Start, words[i].Length));
            i++;
        }

        output.Append(text, copied, text.Length - copied);
        return output.ToString();
    }

    /// <inheritdoc />
    public override WatermarkResult Detect(string text)
    {
        if (string.IsNullOrWhiteSpace(text)) return TextResult(0, 0);
        var words = Words(text);
        var seen = new HashSet<(int, string)>();
        double hits = 0, expected = 0, variance = 0;
        int scored = 0;
        string previous = string.Empty;
        int i = 0;
        while (i < words.Count)
        {
            if (TryMatch(text, words, i, out var site))
            {
                if (seen.Add((site.Group, previous)))
                {
                    double p = 1.0 / _groups[site.Group].Length;
                    if (site.Variant == Choice(site.Group, previous)) hits++;
                    expected += p;
                    variance += p * (1 - p);
                    scored++;
                }

                string matched = _groups[site.Group][site.Variant];
                previous = matched.Substring(matched.LastIndexOf(' ') + 1);
                i += site.WordCount;
                continue;
            }

            previous = Normalize(text.Substring(words[i].Start, words[i].Length));
            i++;
        }

        double z = variance > 0 ? (hits - expected) / Math.Sqrt(variance) : 0;
        return TextResult(z, scored);
    }

    // The keyed member of a group after a given preceding word.
    private int Choice(int group, string previousWord)
    {
        ulong seed = Key.Seed(SeedSalt, new[] { group, StableHash(previousWord) });
        int choice = (int)(WatermarkKey.Uniform(seed, 0) * _groups[group].Length);
        return Math.Min(choice, _groups[group].Length - 1);
    }

    // The longest group phrase starting at word i whose words are separated by whitespace only.
    private bool TryMatch(string text, List<(int Start, int Length)> words, int i, out (int Group, int Variant, int WordCount, int End) site)
    {
        for (int n = Math.Min(_maxPhraseWords, words.Count - i); n >= 1; n--)
        {
            var phrase = new StringBuilder();
            bool contiguous = true;
            for (int k = 0; k < n && contiguous; k++)
            {
                if (k > 0)
                {
                    int gapStart = words[i + k - 1].Start + words[i + k - 1].Length;
                    for (int c = gapStart; c < words[i + k].Start; c++)
                        if (!char.IsWhiteSpace(text[c])) { contiguous = false; break; }
                    phrase.Append(' ');
                }

                phrase.Append(text, words[i + k].Start, words[i + k].Length);
            }

            if (contiguous && _index.TryGetValue(Normalize(phrase.ToString()), out var entry))
            {
                var last = words[i + n - 1];
                site = (entry.Group, entry.Variant, n, last.Start + last.Length);
                return true;
            }
        }

        site = default;
        return false;
    }

    // Words: maximal runs of letters, with apostrophes inside a word (don't, o'clock).
    private static List<(int Start, int Length)> Words(string text)
    {
        var words = new List<(int, int)>();
        int i = 0;
        while (i < text.Length)
        {
            if (!char.IsLetter(text[i])) { i++; continue; }
            int start = i;
            while (i < text.Length && (char.IsLetter(text[i])
                || (IsApostrophe(text[i]) && i + 1 < text.Length && char.IsLetter(text[i + 1]) && i > start)))
                i++;
            words.Add((start, i - start));
        }

        return words;
    }

    private static bool IsApostrophe(char c) => c == '\'' || c == '’';

    private static bool IsWord(string w) => w.Length > 0 && char.IsLetter(w[0]) && w.All(c => char.IsLetter(c) || c == '\'');

    // Lower case, typographic apostrophes as ASCII, single spaces.
    private static string Normalize(string phrase) =>
        string.Join(" ", phrase.Replace('’', '\'').ToLowerInvariant()
            .Split(new[] { ' ', '\t', '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries));

    // The replacement in the original's case pattern: ALL CAPS, Capitalised, or lower case.
    private static string MatchCase(string original, string replacement)
    {
        bool hasLetter = original.Any(char.IsLetter);
        if (hasLetter && original.Where(char.IsLetter).All(char.IsUpper) && original.Count(char.IsLetter) > 1)
            return replacement.ToUpperInvariant();
        if (original.Length > 0 && char.IsUpper(original[0]))
            return char.ToUpperInvariant(replacement[0]) + replacement.Substring(1);
        return replacement;
    }

    // FNV-1a over the UTF-16 code units: stable across processes and runtimes.
    private static int StableHash(string s)
    {
        unchecked
        {
            uint hash = 2166136261;
            foreach (char c in s)
            {
                hash ^= c;
                hash *= 16777619;
            }

            return (int)hash;
        }
    }
}
