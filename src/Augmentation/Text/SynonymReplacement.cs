namespace AiDotNet.Augmentation.Text;

/// <summary>
/// Replaces random words with their synonyms.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Synonym replacement swaps words with similar-meaning words.
/// For example, "happy" might become "joyful" or "glad". This helps models understand
/// that different words can have the same meaning.</para>
/// <para><b>When to use:</b>
/// <list type="bullet">
/// <item>Text classification tasks</item>
/// <item>Sentiment analysis</item>
/// <item>When training data has limited vocabulary variation</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class SynonymReplacement<T> : TextAugmenterBase<T>
{
    private static readonly Dictionary<string, string[]> DefaultSynonyms = new(StringComparer.OrdinalIgnoreCase)
    {
        // Common adjectives
        { "good", new[] { "great", "excellent", "fine", "wonderful", "nice" } },
        { "bad", new[] { "poor", "terrible", "awful", "horrible", "dreadful" } },
        { "big", new[] { "large", "huge", "enormous", "massive", "giant" } },
        { "small", new[] { "tiny", "little", "miniature", "minute", "compact" } },
        { "happy", new[] { "joyful", "glad", "pleased", "delighted", "cheerful" } },
        { "sad", new[] { "unhappy", "sorrowful", "melancholy", "gloomy", "dejected" } },
        { "fast", new[] { "quick", "rapid", "swift", "speedy", "hasty" } },
        { "slow", new[] { "sluggish", "leisurely", "gradual", "unhurried" } },
        { "easy", new[] { "simple", "effortless", "straightforward", "uncomplicated" } },
        { "hard", new[] { "difficult", "challenging", "tough", "demanding" } },

        // Common verbs
        { "make", new[] { "create", "produce", "build", "construct", "form" } },
        { "get", new[] { "obtain", "acquire", "receive", "gain", "fetch" } },
        { "use", new[] { "utilize", "employ", "apply", "operate" } },
        { "say", new[] { "state", "mention", "declare", "express", "assert" } },
        { "go", new[] { "move", "travel", "proceed", "advance", "journey" } },
        { "see", new[] { "observe", "view", "notice", "perceive", "witness" } },
        { "think", new[] { "believe", "consider", "suppose", "assume", "reckon" } },
        { "know", new[] { "understand", "comprehend", "realize", "recognize" } },
        { "want", new[] { "desire", "wish", "need", "require", "crave" } },
        { "like", new[] { "enjoy", "appreciate", "prefer", "fancy", "favor" } },

        // Common nouns
        { "way", new[] { "method", "manner", "approach", "path", "route" } },
        { "thing", new[] { "object", "item", "article", "matter", "entity" } },
        { "problem", new[] { "issue", "difficulty", "challenge", "trouble", "dilemma" } },
        { "idea", new[] { "concept", "notion", "thought", "plan", "proposal" } },
        { "place", new[] { "location", "area", "spot", "site", "position" } }
    };

    /// <summary>
    /// Gets the fraction of words to replace.
    /// </summary>
    /// <remarks>
    /// <para>Default: 0.1 (10% of words)</para>
    /// </remarks>
    public double ReplacementFraction { get; }

    /// <summary>
    /// Gets or sets the custom synonym dictionary.
    /// </summary>
    /// <remarks>
    /// <para>If null, uses the default English synonyms.</para>
    /// </remarks>
    public Dictionary<string, string[]>? CustomSynonyms { get; set; }

    /// <summary>
    /// Creates a new synonym replacement augmentation.
    /// </summary>
    /// <param name="replacementFraction">Fraction of words to replace (default: 0.1).</param>
    /// <param name="probability">Probability of applying this augmentation (default: 0.3).</param>
    /// <param name="languageCode">Language code for language-specific operations.</param>
    public SynonymReplacement(
        double replacementFraction = 0.1,
        double probability = 0.3,
        string languageCode = "en") : base(probability, languageCode)
    {
        if (replacementFraction < 0 || replacementFraction > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(replacementFraction),
                "Replacement fraction must be between 0 and 1.");
        }

        ReplacementFraction = replacementFraction;
    }

    /// <inheritdoc />
    protected override string[] ApplyAugmentation(string[] data, AugmentationContext<T> context)
    {
        var result = new string[data.Length];

        for (int i = 0; i < data.Length; i++)
        {
            result[i] = AugmentText(data[i], context);
        }

        return result;
    }

    private string AugmentText(string text, AugmentationContext<T> context)
    {
        var tokens = Tokenize(text);
        if (tokens.Length == 0) return text;

        var synonymDict = CustomSynonyms ?? DefaultSynonyms;
        int numReplacements = (int)(tokens.Length * ReplacementFraction);
        if (numReplacements == 0 && ReplacementFraction > 0 && tokens.Length > 0)
        {
            // Ensure at least one replacement when fraction is non-zero
            numReplacements = 1;
        }

        // Get indices of words that have synonyms and are not stopwords
        var replaceableIndices = new List<int>();
        for (int i = 0; i < tokens.Length; i++)
        {
            string word = tokens[i].ToLowerInvariant();
            if (!IsStopword(word) && synonymDict.ContainsKey(word))
            {
                replaceableIndices.Add(i);
            }
        }

        // Randomly select words to replace
        var indicesToReplace = replaceableIndices
            .OrderBy(_ => context.Random.Next())
            .Take(numReplacements)
            .ToHashSet();

        // Replace selected words
        var resultTokens = (string[])tokens.Clone();
        foreach (int idx in indicesToReplace)
        {
            string originalWord = tokens[idx];
            string lowerWord = originalWord.ToLowerInvariant();

            if (synonymDict.TryGetValue(lowerWord, out var synonyms) && synonyms.Length > 0)
            {
                string synonym = synonyms[context.Random.Next(synonyms.Length)];

                // Preserve case if needed
                if (PreserveCase)
                {
                    synonym = MatchCase(originalWord, synonym);
                }

                resultTokens[idx] = synonym;
            }
        }

        return Detokenize(resultTokens);
    }

    private static string MatchCase(string original, string replacement)
    {
        if (string.IsNullOrEmpty(original) || string.IsNullOrEmpty(replacement))
            return replacement;

        // All uppercase
        if (original.All(char.IsUpper))
            return replacement.ToUpperInvariant();

        // Title case (first letter uppercase)
        if (char.IsUpper(original[0]))
        {
            if (replacement.Length == 1)
            {
                return char.ToUpperInvariant(replacement[0]).ToString();
            }

            return char.ToUpperInvariant(replacement[0]) + replacement.Substring(1).ToLowerInvariant();
        }

        // All lowercase
        return replacement.ToLowerInvariant();
    }

    /// <inheritdoc />
    public override IDictionary<string, object> GetParameters()
    {
        var parameters = base.GetParameters();
        parameters["replacementFraction"] = ReplacementFraction;
        return parameters;
    }
}
