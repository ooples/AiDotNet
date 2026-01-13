using System.Text.RegularExpressions;

namespace AiDotNet.RetrievalAugmentedGeneration.QueryProcessors;

// ReSharper disable once UnusedMember.Local

/// <summary>
/// Reduces words to their base or dictionary form (lemma) for better matching.
/// </summary>
/// <remarks>
/// <para>
/// Lemmatization transforms words to their base form considering the word's meaning.
/// Unlike stemming, it produces valid dictionary words.
/// </para>
/// <para><b>For Beginners:</b> Converts words to their basic form.
/// 
/// Examples:
/// - "running" → "run"
/// - "better" → "good"
/// - "was" → "be"
/// - "cars" → "car"
/// 
/// This helps match documents that use different forms of the same word!
/// </para>
/// </remarks>
public class LemmatizationQueryProcessor : QueryProcessorBase
{
    /// <summary>
    /// Timeout for regex operations to prevent ReDoS attacks.
    /// </summary>

    private readonly Dictionary<string, string> _lemmaMap;

    /// <summary>
    /// Initializes a new instance of the LemmatizationQueryProcessor class.
    /// </summary>
    /// <param name="customLemmaMap">Optional custom lemmatization dictionary.</param>
    public LemmatizationQueryProcessor(Dictionary<string, string>? customLemmaMap = null)
    {
        _lemmaMap = customLemmaMap ?? GetDefaultLemmaMap();
    }

    protected override string ProcessQueryCore(string query)
    {
        if (string.IsNullOrWhiteSpace(query))
            return query;

        var words = RegexHelper.Split(query, @"(\s+)", RegexOptions.None);
        var lemmatized = new List<string>();

        foreach (var word in words)
        {
            if (string.IsNullOrWhiteSpace(word))
            {
                lemmatized.Add(word);
                continue;
            }

            var lowerWord = word.ToLowerInvariant();

            if (_lemmaMap.TryGetValue(lowerWord, out var lemma))
            {
                lemmatized.Add(PreserveCase(word, lemma));
            }
            else if (lowerWord.EndsWith("ing") && lowerWord.Length > 4)
            {
                var stem = lowerWord.Substring(0, lowerWord.Length - 3);
                if (_lemmaMap.TryGetValue(stem, out lemma))
                {
                    lemmatized.Add(PreserveCase(word, lemma));
                }
                else
                {
                    lemmatized.Add(PreserveCase(word, stem));
                }
            }
            else if (lowerWord.EndsWith("s") && lowerWord.Length > 2 && !lowerWord.EndsWith("ss"))
            {
                var singular = lowerWord.Substring(0, lowerWord.Length - 1);
                lemmatized.Add(PreserveCase(word, singular));
            }
            else
            {
                lemmatized.Add(word);
            }
        }

        return string.Join("", lemmatized);
    }

    private static string PreserveCase(string original, string lemma)
    {
        return QueryProcessorHelpers.PreserveCase(original, lemma);
    }

    private static Dictionary<string, string> GetDefaultLemmaMap()
    {
        return new Dictionary<string, string>
        {
            // Verbs
            { "running", "run" },
            { "ran", "run" },
            { "runs", "run" },
            { "was", "be" },
            { "were", "be" },
            { "been", "be" },
            { "being", "be" },
            { "is", "be" },
            { "are", "be" },
            { "had", "have" },
            { "has", "have" },
            { "having", "have" },
            
            // Adjectives
            { "better", "good" },
            { "best", "good" },
            { "worse", "bad" },
            { "worst", "bad" },
            
            // Common plural nouns
            { "children", "child" },
            { "people", "person" },
            { "men", "man" },
            { "women", "woman" },
            { "feet", "foot" },
            { "teeth", "tooth" },
            { "mice", "mouse" },
            
            // Technical terms
            { "algorithms", "algorithm" },
            { "models", "model" },
            { "networks", "network" },
            { "transformers", "transformer" }
        };
    }
}



