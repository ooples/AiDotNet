namespace AiDotNet.Data.Quality;

/// <summary>
/// Filters documents based on detected language using character n-gram profiles.
/// </summary>
/// <remarks>
/// <para>
/// Uses a simple but effective character n-gram frequency approach for language detection.
/// Requires training with reference text for each target language.
/// Based on the Cavnar-Trenkle (1994) text categorization method.
/// </para>
/// </remarks>
public class LanguageIdFilter
{
    private readonly LanguageIdFilterOptions _options;
    private readonly Dictionary<string, Dictionary<string, int>> _languageProfiles;

    public LanguageIdFilter(LanguageIdFilterOptions? options = null)
    {
        _options = options ?? new LanguageIdFilterOptions();
        _languageProfiles = new Dictionary<string, Dictionary<string, int>>();
    }

    /// <summary>
    /// Adds a language profile built from reference text.
    /// </summary>
    /// <param name="languageCode">ISO 639-1 language code (e.g., "en", "fr").</param>
    /// <param name="referenceTexts">Representative text samples in the language.</param>
    public void AddLanguageProfile(string languageCode, IReadOnlyList<string> referenceTexts)
    {
        if (string.IsNullOrWhiteSpace(languageCode))
            throw new ArgumentException("Language code must not be null or empty.", nameof(languageCode));
        if (referenceTexts == null || referenceTexts.Count == 0)
            throw new ArgumentException("Reference texts must not be null or empty.", nameof(referenceTexts));

        var ngramFreqs = new Dictionary<string, int>();

        foreach (string text in referenceTexts)
        {
            string normalized = text.ToLowerInvariant();
            for (int i = 0; i <= normalized.Length - _options.ProfileNGramSize; i++)
            {
                string ngram = normalized.Substring(i, _options.ProfileNGramSize);
                ngramFreqs[ngram] = ngramFreqs.GetValueOrDefault(ngram, 0) + 1;
            }
        }

        // Keep only top-N n-grams by frequency
        var topNgrams = ngramFreqs
            .OrderByDescending(kv => kv.Value)
            .Take(_options.MaxProfileSize)
            .ToDictionary(kv => kv.Key, kv => kv.Value);

        _languageProfiles[languageCode] = topNgrams;
    }

    /// <summary>
    /// Detects the most likely language of a text.
    /// </summary>
    /// <param name="text">The text to classify.</param>
    /// <returns>Tuple of (language code, confidence score). Returns ("unknown", 0) if no profiles loaded.</returns>
    public (string Language, double Confidence) DetectLanguage(string text)
    {
        if (_languageProfiles.Count == 0)
            return ("unknown", 0.0);

        if (text.Length < _options.MinTextLength)
            return ("unknown", 0.0);

        // Build n-gram profile for input text
        var textProfile = new Dictionary<string, int>();
        string normalized = text.ToLowerInvariant();
        for (int i = 0; i <= normalized.Length - _options.ProfileNGramSize; i++)
        {
            string ngram = normalized.Substring(i, _options.ProfileNGramSize);
            textProfile[ngram] = textProfile.GetValueOrDefault(ngram, 0) + 1;
        }

        // Rank n-grams by frequency
        var textRanked = textProfile
            .OrderByDescending(kv => kv.Value)
            .Take(_options.MaxProfileSize)
            .Select((kv, idx) => (kv.Key, Rank: idx))
            .ToDictionary(x => x.Key, x => x.Rank);

        string bestLang = "unknown";
        double bestScore = double.MaxValue;

        foreach (var (langCode, profile) in _languageProfiles)
        {
            var langRanked = profile
                .OrderByDescending(kv => kv.Value)
                .Select((kv, idx) => (kv.Key, Rank: idx))
                .ToDictionary(x => x.Key, x => x.Rank);

            // Compute out-of-place distance
            double distance = 0;
            foreach (var (ngram, textRank) in textRanked)
            {
                if (langRanked.TryGetValue(ngram, out int langRank))
                    distance += Math.Abs(textRank - langRank);
                else
                    distance += _options.MaxProfileSize;
            }

            if (distance < bestScore)
            {
                bestScore = distance;
                bestLang = langCode;
            }
        }

        // Normalize confidence: 0 = max distance, 1 = perfect match
        double maxDistance = textRanked.Count * _options.MaxProfileSize;
        double confidence = maxDistance > 0 ? 1.0 - (bestScore / maxDistance) : 0.0;

        return (bestLang, confidence);
    }

    /// <summary>
    /// Filters documents by language, returning indices of documents that should be removed.
    /// </summary>
    /// <param name="documents">Documents to filter.</param>
    /// <returns>Set of indices that are not in the target language(s) (should be removed).</returns>
    public HashSet<int> Filter(IReadOnlyList<string> documents)
    {
        if (_languageProfiles.Count == 0)
            throw new InvalidOperationException("No language profiles loaded. Call AddLanguageProfile() before filtering.");

        var filtered = new HashSet<int>();
        var targetSet = new HashSet<string>(_options.TargetLanguages, StringComparer.OrdinalIgnoreCase);

        for (int i = 0; i < documents.Count; i++)
        {
            var (lang, confidence) = DetectLanguage(documents[i]);
            if (!targetSet.Contains(lang) || confidence < _options.MinConfidence)
            {
                filtered.Add(i);
            }
        }

        return filtered;
    }
}
