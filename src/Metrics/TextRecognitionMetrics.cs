using System.Text;

namespace AiDotNet.Metrics;

/// <summary>
/// Evaluation metrics for text recognition (OCR): Character Error Rate, Word Error Rate,
/// normalized edit distance and exact-match accuracy.
/// </summary>
/// <remarks>
/// <para>
/// Recognition output is a string, so quality is measured by how many edits turn the prediction
/// into the reference. All the metrics here are built on Levenshtein distance - the smallest
/// number of single-character insertions, deletions and substitutions that transform one string
/// into another.
/// </para>
/// <para><b>Corpus versus sentence averaging.</b>
/// The overloads taking a whole list are <i>corpus-level</i>: they sum the edit distances and
/// divide by the summed reference length. That is the convention used by the ICDAR Robust Reading
/// competitions and by speech recognition. Averaging the per-sample rates instead would let a
/// single short reference dominate, so prefer the corpus overloads when reporting a benchmark
/// number.
/// </para>
/// <para><b>Which number to report:</b>
/// <list type="bullet">
/// <item><description><see cref="CharacterErrorRate(IReadOnlyList{string}, IReadOnlyList{string})"/> -
/// the standard number for line- and page-level OCR. Lower is better; 0 is perfect.</description></item>
/// <item><description><see cref="WordErrorRate(IReadOnlyList{string}, IReadOnlyList{string})"/> -
/// the same idea over whitespace-separated tokens, for document OCR.</description></item>
/// <item><description><see cref="NormalizedEditDistance(IReadOnlyList{string}, IReadOnlyList{string})"/> -
/// ICDAR 1-NED. Higher is better; 1 is perfect. This is the one scene-text papers quote.</description></item>
/// <item><description><see cref="ExactMatchAccuracy(IReadOnlyList{string}, IReadOnlyList{string}, bool, bool)"/> -
/// word-level accuracy for cropped-word benchmarks (IIIT5K, SVT, IC13, IC15). The field convention
/// is case-insensitive and alphanumeric-only, which is this method default.</description></item>
/// </list>
/// </para>
/// <para><b>For Beginners:</b> Error rates answer "what fraction of the text did the model get
/// wrong?" A CER of 0.05 means about 5 characters in every 100 needed fixing. Note that an error
/// rate can exceed 1.0: if the model emits far more text than the reference contains, the number
/// of edits can be larger than the reference length. Accuracy metrics run the other way - higher
/// is better - so always check which direction a reported number points.
/// </para>
/// <example>
/// <code>
/// string[] references = { "hello world", "aidotnet" };
/// string[] predictions = { "hell0 world", "aidotnet" };
/// double cer = TextRecognitionMetrics.CharacterErrorRate(references, predictions);   // 1 edit / 19 chars
/// double acc = TextRecognitionMetrics.ExactMatchAccuracy(references, predictions);   // 0.5
/// </code>
/// </example>
/// </remarks>
public static class TextRecognitionMetrics
{
    /// <summary>
    /// Computes the Levenshtein edit distance between two strings: the minimum number of
    /// single-character insertions, deletions and substitutions needed to turn
    /// <paramref name="source"/> into <paramref name="target"/>.
    /// </summary>
    /// <param name="source">The first string. Null is treated as empty.</param>
    /// <param name="target">The second string. Null is treated as empty.</param>
    /// <returns>The edit distance, always at least the difference in lengths.</returns>
    public static int LevenshteinDistance(string? source, string? target)
    {
        string a = source ?? string.Empty;
        string b = target ?? string.Empty;

        if (a.Length == 0)
        {
            return b.Length;
        }

        if (b.Length == 0)
        {
            return a.Length;
        }

        // Two rolling rows rather than the full matrix: the recurrence only ever reads the previous
        // row, so a page of text costs O(min(n, m)) memory instead of O(n * m).
        var previous = new int[b.Length + 1];
        var current = new int[b.Length + 1];

        for (int j = 0; j <= b.Length; j++)
        {
            previous[j] = j;
        }

        for (int i = 1; i <= a.Length; i++)
        {
            current[0] = i;
            for (int j = 1; j <= b.Length; j++)
            {
                int substitution = previous[j - 1] + (a[i - 1] == b[j - 1] ? 0 : 1);
                int deletion = previous[j] + 1;
                int insertion = current[j - 1] + 1;
                current[j] = Math.Min(substitution, Math.Min(deletion, insertion));
            }

            (previous, current) = (current, previous);
        }

        return previous[b.Length];
    }

    /// <summary>
    /// Computes the Levenshtein edit distance between two token sequences, used by
    /// <see cref="WordErrorRate(string, string)"/>.
    /// </summary>
    /// <param name="source">The first token sequence. Null is treated as empty.</param>
    /// <param name="target">The second token sequence. Null is treated as empty.</param>
    /// <returns>The edit distance in tokens.</returns>
    public static int TokenEditDistance(IReadOnlyList<string>? source, IReadOnlyList<string>? target)
    {
        int n = source is null ? 0 : source.Count;
        int m = target is null ? 0 : target.Count;

        if (n == 0)
        {
            return m;
        }

        if (m == 0)
        {
            return n;
        }

        var previous = new int[m + 1];
        var current = new int[m + 1];

        for (int j = 0; j <= m; j++)
        {
            previous[j] = j;
        }

        for (int i = 1; i <= n; i++)
        {
            current[0] = i;
            for (int j = 1; j <= m; j++)
            {
                bool equal = string.Equals(source![i - 1], target![j - 1], StringComparison.Ordinal);
                int substitution = previous[j - 1] + (equal ? 0 : 1);
                int deletion = previous[j] + 1;
                int insertion = current[j - 1] + 1;
                current[j] = Math.Min(substitution, Math.Min(deletion, insertion));
            }

            (previous, current) = (current, previous);
        }

        return previous[m];
    }

    /// <summary>
    /// Computes the Character Error Rate for one prediction: edit distance divided by the
    /// reference length.
    /// </summary>
    /// <param name="reference">The ground-truth text.</param>
    /// <param name="hypothesis">The recognised text.</param>
    /// <returns>CER, 0 when the strings match. Can exceed 1 when the hypothesis is much longer
    /// than the reference. Returns 0 for an empty reference matched by an empty hypothesis, and
    /// <see cref="double.NaN"/> for an empty reference with a non-empty hypothesis, where the rate
    /// is undefined.</returns>
    public static double CharacterErrorRate(string? reference, string? hypothesis)
    {
        string reference1 = reference ?? string.Empty;
        string hypothesis1 = hypothesis ?? string.Empty;

        if (reference1.Length == 0)
        {
            return hypothesis1.Length == 0 ? 0.0 : double.NaN;
        }

        return LevenshteinDistance(reference1, hypothesis1) / (double)reference1.Length;
    }

    /// <summary>
    /// Computes the corpus-level Character Error Rate: the summed edit distance over the summed
    /// reference length. This is the standard way to report CER over a dataset.
    /// </summary>
    /// <param name="references">The ground-truth texts.</param>
    /// <param name="hypotheses">The recognised texts, aligned with <paramref name="references"/>.</param>
    /// <returns>CER over the whole corpus. With no reference characters, returns 0 if no edits
    /// are needed, otherwise <see cref="double.NaN"/> because the rate is undefined.</returns>
    /// <exception cref="ArgumentNullException">A required argument is null.</exception>
    /// <exception cref="ArgumentException">The lists have different lengths.</exception>
    public static double CharacterErrorRate(IReadOnlyList<string> references, IReadOnlyList<string> hypotheses)
    {
        ValidateAligned(references, hypotheses);

        long distance = 0;
        long length = 0;
        for (int i = 0; i < references.Count; i++)
        {
            string reference = references[i] ?? string.Empty;
            distance += LevenshteinDistance(reference, hypotheses[i]);
            length += reference.Length;
        }

        return length > 0 ? distance / (double)length : distance == 0 ? 0.0 : double.NaN;
    }

    /// <summary>
    /// Computes the Word Error Rate for one prediction: token-level edit distance divided by the
    /// reference token count. Tokens are whitespace-separated.
    /// </summary>
    /// <param name="reference">The ground-truth text.</param>
    /// <param name="hypothesis">The recognised text.</param>
    /// <returns>WER, 0 when the token sequences match. Returns <see cref="double.NaN"/> when the
    /// reference has no tokens but the hypothesis does.</returns>
    public static double WordErrorRate(string? reference, string? hypothesis)
    {
        var referenceTokens = Tokenize(reference);
        var hypothesisTokens = Tokenize(hypothesis);

        if (referenceTokens.Count == 0)
        {
            return hypothesisTokens.Count == 0 ? 0.0 : double.NaN;
        }

        return TokenEditDistance(referenceTokens, hypothesisTokens) / (double)referenceTokens.Count;
    }

    /// <summary>
    /// Computes the corpus-level Word Error Rate: the summed token edit distance over the summed
    /// reference token count.
    /// </summary>
    /// <param name="references">The ground-truth texts.</param>
    /// <param name="hypotheses">The recognised texts, aligned with <paramref name="references"/>.</param>
    /// <returns>WER over the whole corpus. With no reference tokens, returns 0 if no edits
    /// are needed, otherwise <see cref="double.NaN"/> because the rate is undefined.</returns>
    /// <exception cref="ArgumentNullException">A required argument is null.</exception>
    /// <exception cref="ArgumentException">The lists have different lengths.</exception>
    public static double WordErrorRate(IReadOnlyList<string> references, IReadOnlyList<string> hypotheses)
    {
        ValidateAligned(references, hypotheses);

        long distance = 0;
        long count = 0;
        for (int i = 0; i < references.Count; i++)
        {
            var referenceTokens = Tokenize(references[i]);
            distance += TokenEditDistance(referenceTokens, Tokenize(hypotheses[i]));
            count += referenceTokens.Count;
        }

        return count > 0 ? distance / (double)count : distance == 0 ? 0.0 : double.NaN;
    }

    /// <summary>
    /// Computes ICDAR normalized edit distance (1-NED) for one prediction:
    /// <c>1 - distance / max(referenceLength, hypothesisLength)</c>.
    /// </summary>
    /// <param name="reference">The ground-truth text.</param>
    /// <param name="hypothesis">The recognised text.</param>
    /// <returns>A similarity in [0, 1] where 1 is an exact match. Two empty strings score 1.</returns>
    /// <remarks>
    /// Unlike <see cref="CharacterErrorRate(string, string)"/> this is bounded above by 1 and is a
    /// similarity rather than an error, because it divides by the longer of the two strings. That
    /// is what makes it safe to average across samples of very different lengths.
    /// </remarks>
    public static double NormalizedEditDistance(string? reference, string? hypothesis)
    {
        string reference1 = reference ?? string.Empty;
        string hypothesis1 = hypothesis ?? string.Empty;

        int longest = Math.Max(reference1.Length, hypothesis1.Length);
        if (longest == 0)
        {
            return 1.0;
        }

        return 1.0 - (LevenshteinDistance(reference1, hypothesis1) / (double)longest);
    }

    /// <summary>
    /// Computes mean ICDAR 1-NED over a dataset: the average of the per-sample
    /// <see cref="NormalizedEditDistance(string, string)"/>.
    /// </summary>
    /// <param name="references">The ground-truth texts.</param>
    /// <param name="hypotheses">The recognised texts, aligned with <paramref name="references"/>.</param>
    /// <returns>Mean 1-NED in [0, 1], or 1 for an empty dataset.</returns>
    /// <exception cref="ArgumentNullException">A required argument is null.</exception>
    /// <exception cref="ArgumentException">The lists have different lengths.</exception>
    public static double NormalizedEditDistance(IReadOnlyList<string> references, IReadOnlyList<string> hypotheses)
    {
        ValidateAligned(references, hypotheses);

        if (references.Count == 0)
        {
            return 1.0;
        }

        double sum = 0.0;
        for (int i = 0; i < references.Count; i++)
        {
            sum += NormalizedEditDistance(references[i], hypotheses[i]);
        }

        return sum / references.Count;
    }

    /// <summary>
    /// Computes exact-match accuracy: the fraction of predictions that equal their reference after
    /// normalization.
    /// </summary>
    /// <param name="references">The ground-truth texts.</param>
    /// <param name="hypotheses">The recognised texts, aligned with <paramref name="references"/>.</param>
    /// <param name="caseSensitive">When false (the default) both strings are lower-cased first.</param>
    /// <param name="alphanumericOnly">When true (the default) every character that is not a letter or
    /// digit is stripped first.</param>
    /// <returns>Accuracy in [0, 1], or 1 for an empty dataset.</returns>
    /// <remarks>
    /// The defaults reproduce the scene-text benchmark protocol (IIIT5K, SVT, IC13, IC15), which
    /// scores case-insensitively over the 36-character alphanumeric set. Pass
    /// <paramref name="caseSensitive"/> as true and <paramref name="alphanumericOnly"/> as false to
    /// score raw strings instead.
    /// </remarks>
    /// <exception cref="ArgumentNullException">A required argument is null.</exception>
    /// <exception cref="ArgumentException">The lists have different lengths.</exception>
    public static double ExactMatchAccuracy(
        IReadOnlyList<string> references,
        IReadOnlyList<string> hypotheses,
        bool caseSensitive = false,
        bool alphanumericOnly = true)
    {
        ValidateAligned(references, hypotheses);

        if (references.Count == 0)
        {
            return 1.0;
        }

        int matched = 0;
        for (int i = 0; i < references.Count; i++)
        {
            string reference = Normalize(references[i], caseSensitive, alphanumericOnly);
            string hypothesis = Normalize(hypotheses[i], caseSensitive, alphanumericOnly);
            if (string.Equals(reference, hypothesis, StringComparison.Ordinal))
            {
                matched++;
            }
        }

        return matched / (double)references.Count;
    }

    /// <summary>
    /// Applies the benchmark normalization used by <see cref="ExactMatchAccuracy"/>.
    /// </summary>
    /// <param name="value">The text to normalize. Null is treated as empty.</param>
    /// <param name="caseSensitive">When false the text is lower-cased.</param>
    /// <param name="alphanumericOnly">When true non-alphanumeric characters are removed.</param>
    /// <returns>The normalized text.</returns>
    public static string Normalize(string? value, bool caseSensitive = false, bool alphanumericOnly = true)
    {
        string text = value ?? string.Empty;
        if (!caseSensitive)
        {
            text = text.ToLowerInvariant();
        }

        if (!alphanumericOnly)
        {
            return text;
        }

        var builder = new StringBuilder(text.Length);
        foreach (char c in text)
        {
            if (char.IsLetterOrDigit(c))
            {
                builder.Append(c);
            }
        }

        return builder.ToString();
    }

    private static List<string> Tokenize(string? value)
    {
        var tokens = new List<string>();
        if (string.IsNullOrEmpty(value))
        {
            return tokens;
        }

        foreach (string token in value!.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries))
        {
            tokens.Add(token);
        }

        return tokens;
    }

    private static void ValidateAligned(IReadOnlyList<string> references, IReadOnlyList<string> hypotheses)
    {
        if (references is null)
        {
            throw new ArgumentNullException(nameof(references));
        }

        if (hypotheses is null)
        {
            throw new ArgumentNullException(nameof(hypotheses));
        }

        if (references.Count != hypotheses.Count)
        {
            throw new ArgumentException(
                $"There are {references.Count} references but {hypotheses.Count} hypotheses. "
                + "Both lists must be indexed by the same sample order.",
                nameof(references));
        }
    }
}
