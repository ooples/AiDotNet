using System.Text.RegularExpressions;
using AiDotNet.Models.Options;

namespace AiDotNet.FederatedLearning.PSI;

/// <summary>
/// Implements approximate entity matching using multiple similarity strategies.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> In real-world data, the same entity may have slightly different
/// identifiers across organizations. "John Smith" at one hospital might be "Jon Smith" or
/// "SMITH, JOHN" at another. Fuzzy matching finds these approximate matches.</para>
///
/// <para>This class provides a unified PSI wrapper around fuzzy matching that first applies
/// approximate matching to find candidate pairs, then runs the underlying PSI protocol
/// on the matched identifiers.</para>
/// </remarks>
public class FuzzyPsi : PsiBase
{
    private readonly IPrivateSetIntersection _innerProtocol;

    /// <summary>
    /// Initializes a new instance of <see cref="FuzzyPsi"/> wrapping an inner PSI protocol.
    /// </summary>
    /// <param name="innerProtocol">The exact PSI protocol to use after fuzzy matching.</param>
    public FuzzyPsi(IPrivateSetIntersection innerProtocol)
    {
        _innerProtocol = innerProtocol ?? throw new ArgumentNullException(nameof(innerProtocol));
    }

    /// <inheritdoc/>
    public override string ProtocolName => $"Fuzzy-{_innerProtocol.ProtocolName}";

    /// <inheritdoc/>
    protected override PsiResult ComputeExactIntersection(
        IReadOnlyList<string> localIds, IReadOnlyList<string> remoteIds, PsiOptions options)
    {
        return _innerProtocol.ComputeIntersection(localIds, remoteIds, options);
    }
}

/// <summary>
/// Exact string equality matcher (no fuzzy matching).
/// </summary>
internal class ExactMatcher : IFuzzyMatcher
{
    public string StrategyName => "Exact";

    public double ComputeSimilarity(string id1, string id2, FuzzyMatchOptions options)
    {
        return string.Equals(id1, id2, GetComparison(options)) ? 1.0 : 0.0;
    }

    public bool IsMatch(string id1, string id2, FuzzyMatchOptions options)
    {
        return string.Equals(id1, id2, GetComparison(options));
    }

    public string Normalize(string id, FuzzyMatchOptions options)
    {
        if (string.IsNullOrEmpty(id))
        {
            return string.Empty;
        }

        string result = id;
        if (!options.CaseSensitive)
        {
            result = result.ToUpperInvariant();
        }

        if (options.NormalizeWhitespace)
        {
            result = PsiBase.NormalizeWhitespace(result);
        }

        return result;
    }

    public IReadOnlyList<(int CandidateIndex, double Similarity)> FindMatches(
        string id, IReadOnlyList<string> candidates, FuzzyMatchOptions options)
    {
        var matches = new List<(int, double)>();
        for (int i = 0; i < candidates.Count; i++)
        {
            if (string.Equals(id, candidates[i], GetComparison(options)))
            {
                matches.Add((i, 1.0));
            }
        }

        return matches;
    }

    private static StringComparison GetComparison(FuzzyMatchOptions options)
    {
        return options.CaseSensitive ? StringComparison.Ordinal : StringComparison.OrdinalIgnoreCase;
    }
}

/// <summary>
/// Levenshtein edit distance based fuzzy matcher.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Edit distance counts the minimum number of single-character
/// changes (insertions, deletions, substitutions) needed to transform one string into another.
/// For example, "kitten" to "sitting" requires 3 edits (k->s, e->i, +g), so the distance is 3.</para>
/// </remarks>
internal class EditDistanceMatcher : IFuzzyMatcher
{
    public string StrategyName => "EditDistance";

    public double ComputeSimilarity(string id1, string id2, FuzzyMatchOptions options)
    {
        return ComputeEditDistance(id1, id2);
    }

    public bool IsMatch(string id1, string id2, FuzzyMatchOptions options)
    {
        int distance = ComputeEditDistance(id1, id2);
        return distance <= (int)options.Threshold;
    }

    public string Normalize(string id, FuzzyMatchOptions options)
    {
        if (string.IsNullOrEmpty(id))
        {
            return string.Empty;
        }

        string result = id;
        if (!options.CaseSensitive)
        {
            result = result.ToUpperInvariant();
        }

        if (options.NormalizeWhitespace)
        {
            result = PsiBase.NormalizeWhitespace(result);
        }

        return result;
    }

    public IReadOnlyList<(int CandidateIndex, double Similarity)> FindMatches(
        string id, IReadOnlyList<string> candidates, FuzzyMatchOptions options)
    {
        int threshold = (int)options.Threshold;
        var matches = new List<(int CandidateIndex, double Similarity)>();

        for (int i = 0; i < candidates.Count; i++)
        {
            int distance = ComputeEditDistance(id, candidates[i]);
            if (distance <= threshold)
            {
                int maxLen = Math.Max(id.Length, candidates[i].Length);
                double similarity = maxLen > 0 ? 1.0 - (double)distance / maxLen : 1.0;
                matches.Add((i, similarity));
            }
        }

        matches.Sort((a, b) => b.Similarity.CompareTo(a.Similarity));
        return matches;
    }

    /// <summary>
    /// Computes the Levenshtein edit distance between two strings using the Wagner-Fischer algorithm.
    /// Uses O(min(m,n)) space by only keeping two rows of the DP matrix.
    /// </summary>
    private static int ComputeEditDistance(string s, string t)
    {
        if (string.IsNullOrEmpty(s))
        {
            return string.IsNullOrEmpty(t) ? 0 : t.Length;
        }

        if (string.IsNullOrEmpty(t))
        {
            return s.Length;
        }

        if (s.Length > t.Length)
        {
            var temp = s;
            s = t;
            t = temp;
        }

        int m = s.Length;
        int n = t.Length;

        var previousRow = new int[m + 1];
        var currentRow = new int[m + 1];

        for (int i = 0; i <= m; i++)
        {
            previousRow[i] = i;
        }

        for (int j = 1; j <= n; j++)
        {
            currentRow[0] = j;
            for (int i = 1; i <= m; i++)
            {
                int cost = s[i - 1] == t[j - 1] ? 0 : 1;
                currentRow[i] = Math.Min(
                    Math.Min(currentRow[i - 1] + 1, previousRow[i] + 1),
                    previousRow[i - 1] + cost);
            }

            var swap = previousRow;
            previousRow = currentRow;
            currentRow = swap;
        }

        return previousRow[m];
    }
}

/// <summary>
/// Phonetic matching using Soundex algorithm.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Phonetic matching groups strings by how they sound.
/// "Smith" and "Smyth" produce the same Soundex code (S530), so they match.
/// This is particularly useful for matching names across different data sources.</para>
/// </remarks>
internal class PhoneticMatcher : IFuzzyMatcher
{
    public string StrategyName => "Phonetic";

    public double ComputeSimilarity(string id1, string id2, FuzzyMatchOptions options)
    {
        string code1 = ComputeSoundex(id1);
        string code2 = ComputeSoundex(id2);
        return string.Equals(code1, code2, StringComparison.Ordinal) ? 1.0 : 0.0;
    }

    public bool IsMatch(string id1, string id2, FuzzyMatchOptions options)
    {
        string code1 = ComputeSoundex(id1);
        string code2 = ComputeSoundex(id2);
        return string.Equals(code1, code2, StringComparison.Ordinal);
    }

    public string Normalize(string id, FuzzyMatchOptions options)
    {
        if (string.IsNullOrEmpty(id))
        {
            return string.Empty;
        }

        string result = id.ToUpperInvariant();
        if (options.NormalizeWhitespace)
        {
            result = PsiBase.NormalizeWhitespace(result);
        }

        return result;
    }

    public IReadOnlyList<(int CandidateIndex, double Similarity)> FindMatches(
        string id, IReadOnlyList<string> candidates, FuzzyMatchOptions options)
    {
        string targetCode = ComputeSoundex(id);
        var matches = new List<(int, double)>();

        for (int i = 0; i < candidates.Count; i++)
        {
            string candidateCode = ComputeSoundex(candidates[i]);
            if (string.Equals(targetCode, candidateCode, StringComparison.Ordinal))
            {
                matches.Add((i, 1.0));
            }
        }

        return matches;
    }

    /// <summary>
    /// Computes the Soundex phonetic code for a string.
    /// </summary>
    private static string ComputeSoundex(string input)
    {
        if (string.IsNullOrWhiteSpace(input))
        {
            return "0000";
        }

        string cleaned = input.ToUpperInvariant();
        var result = new char[4];
        result[0] = cleaned[0];
        int resultIndex = 1;
        char lastCode = GetSoundexCode(cleaned[0]);

        for (int i = 1; i < cleaned.Length && resultIndex < 4; i++)
        {
            char c = cleaned[i];
            if (!char.IsLetter(c))
            {
                continue;
            }

            char code = GetSoundexCode(c);
            if (code != '0' && code != lastCode)
            {
                result[resultIndex++] = code;
            }

            lastCode = code;
        }

        while (resultIndex < 4)
        {
            result[resultIndex++] = '0';
        }

        return new string(result);
    }

    private static char GetSoundexCode(char c)
    {
        return c switch
        {
            'B' or 'F' or 'P' or 'V' => '1',
            'C' or 'G' or 'J' or 'K' or 'Q' or 'S' or 'X' or 'Z' => '2',
            'D' or 'T' => '3',
            'L' => '4',
            'M' or 'N' => '5',
            'R' => '6',
            _ => '0'
        };
    }
}

/// <summary>
/// Character n-gram similarity matcher.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> N-grams split a string into overlapping pieces of length N.
/// For example, "hello" with N=2 produces: {"he", "el", "ll", "lo"}.
/// Two strings are similar if they share many n-grams. This handles typos well because
/// a single typo only affects N adjacent n-grams, leaving the rest identical.</para>
/// </remarks>
internal class NGramMatcher : IFuzzyMatcher
{
    public string StrategyName => "NGram";

    public double ComputeSimilarity(string id1, string id2, FuzzyMatchOptions options)
    {
        int n = options.NGramSize > 0 ? options.NGramSize : 2;
        var ngrams1 = GetNGrams(id1, n);
        var ngrams2 = GetNGrams(id2, n);

        if (ngrams1.Count == 0 && ngrams2.Count == 0)
        {
            return 1.0;
        }

        if (ngrams1.Count == 0 || ngrams2.Count == 0)
        {
            return 0.0;
        }

        int intersection = 0;
        var remaining = new Dictionary<string, int>(ngrams2, StringComparer.Ordinal);
        foreach (var ngram in ngrams1)
        {
            if (remaining.TryGetValue(ngram.Key, out int count) && count > 0)
            {
                int common = Math.Min(ngram.Value, count);
                intersection += common;
                remaining[ngram.Key] = count - common;
            }
        }

        int total1 = ngrams1.Values.Sum();
        int total2 = ngrams2.Values.Sum();
        int union = total1 + total2 - intersection;
        return union > 0 ? (double)intersection / union : 0.0;
    }

    public bool IsMatch(string id1, string id2, FuzzyMatchOptions options)
    {
        double similarity = ComputeSimilarity(id1, id2, options);
        return similarity >= options.Threshold;
    }

    public string Normalize(string id, FuzzyMatchOptions options)
    {
        if (string.IsNullOrEmpty(id))
        {
            return string.Empty;
        }

        string result = id;
        if (!options.CaseSensitive)
        {
            result = result.ToUpperInvariant();
        }

        if (options.NormalizeWhitespace)
        {
            result = PsiBase.NormalizeWhitespace(result);
        }

        return result;
    }

    public IReadOnlyList<(int CandidateIndex, double Similarity)> FindMatches(
        string id, IReadOnlyList<string> candidates, FuzzyMatchOptions options)
    {
        var matches = new List<(int, double)>();
        double threshold = options.Threshold;

        for (int i = 0; i < candidates.Count; i++)
        {
            double similarity = ComputeSimilarity(id, candidates[i], options);
            if (similarity >= threshold)
            {
                matches.Add((i, similarity));
            }
        }

        matches.Sort((a, b) => b.Item2.CompareTo(a.Item2));
        return matches;
    }

    private static Dictionary<string, int> GetNGrams(string input, int n)
    {
        var ngrams = new Dictionary<string, int>(StringComparer.Ordinal);
        if (string.IsNullOrEmpty(input) || input.Length < n)
        {
            if (!string.IsNullOrEmpty(input))
            {
                ngrams[input] = 1;
            }

            return ngrams;
        }

        for (int i = 0; i <= input.Length - n; i++)
        {
            string ngram = input.Substring(i, n);
            ngrams.TryGetValue(ngram, out int count);
            ngrams[ngram] = count + 1;
        }

        return ngrams;
    }
}

/// <summary>
/// Jaccard similarity coefficient matcher.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Jaccard similarity measures the overlap between two sets.
/// It computes |A intersect B| / |A union B|, producing a score from 0.0 (no overlap)
/// to 1.0 (identical sets). For strings, the "sets" are typically the sets of unique
/// characters or tokens in each string.</para>
/// </remarks>
internal class JaccardMatcher : IFuzzyMatcher
{
    public string StrategyName => "Jaccard";

    public double ComputeSimilarity(string id1, string id2, FuzzyMatchOptions options)
    {
        var set1 = GetTokenSet(id1);
        var set2 = GetTokenSet(id2);

        if (set1.Count == 0 && set2.Count == 0)
        {
            return 1.0;
        }

        if (set1.Count == 0 || set2.Count == 0)
        {
            return 0.0;
        }

        int intersectionCount = 0;
        foreach (string token in set1)
        {
            if (set2.Contains(token))
            {
                intersectionCount++;
            }
        }

        int unionCount = set1.Count + set2.Count - intersectionCount;
        return unionCount > 0 ? (double)intersectionCount / unionCount : 0.0;
    }

    public bool IsMatch(string id1, string id2, FuzzyMatchOptions options)
    {
        double similarity = ComputeSimilarity(id1, id2, options);
        return similarity >= options.Threshold;
    }

    public string Normalize(string id, FuzzyMatchOptions options)
    {
        if (string.IsNullOrEmpty(id))
        {
            return string.Empty;
        }

        string result = id;
        if (!options.CaseSensitive)
        {
            result = result.ToUpperInvariant();
        }

        if (options.NormalizeWhitespace)
        {
            result = PsiBase.NormalizeWhitespace(result);
        }

        return result;
    }

    public IReadOnlyList<(int CandidateIndex, double Similarity)> FindMatches(
        string id, IReadOnlyList<string> candidates, FuzzyMatchOptions options)
    {
        var matches = new List<(int, double)>();
        double threshold = options.Threshold;

        for (int i = 0; i < candidates.Count; i++)
        {
            double similarity = ComputeSimilarity(id, candidates[i], options);
            if (similarity >= threshold)
            {
                matches.Add((i, similarity));
            }
        }

        matches.Sort((a, b) => b.Item2.CompareTo(a.Item2));
        return matches;
    }

    private static HashSet<string> GetTokenSet(string input)
    {
        if (string.IsNullOrWhiteSpace(input))
        {
            return new HashSet<string>(StringComparer.Ordinal);
        }

        var tokens = new HashSet<string>(StringComparer.Ordinal);
        foreach (char c in input)
        {
            if (!char.IsWhiteSpace(c))
            {
                tokens.Add(c.ToString());
            }
        }

        return tokens;
    }
}
