namespace AiDotNet.Augmentation.Text;

/// <summary>
/// Randomly inserts synonyms of existing words into the text.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Random insertion adds new words to the text based on
/// synonyms of existing words. For example, "I love programming" might become
/// "I love really programming" if "really" is considered related to context.</para>
/// <para><b>When to use:</b>
/// <list type="bullet">
/// <item>Text classification tasks</item>
/// <item>Training data expansion</item>
/// <item>Making models robust to longer text variations</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class RandomInsertion<T> : TextAugmenterBase<T>
{
    private static readonly Dictionary<string, string[]> RelatedWords = new(StringComparer.OrdinalIgnoreCase)
    {
        // Words that can be inserted near adjectives
        { "good", new[] { "very", "quite", "really", "extremely" } },
        { "bad", new[] { "very", "quite", "really", "extremely" } },
        { "big", new[] { "very", "quite", "really", "extremely" } },
        { "small", new[] { "very", "quite", "really", "extremely" } },
        { "happy", new[] { "very", "quite", "really", "extremely" } },
        { "sad", new[] { "very", "quite", "really", "extremely" } },
        { "fast", new[] { "very", "quite", "really", "extremely" } },
        { "slow", new[] { "very", "quite", "really", "extremely" } },

        // Words that can be inserted near nouns
        { "problem", new[] { "significant", "major", "serious", "critical" } },
        { "solution", new[] { "effective", "optimal", "practical", "viable" } },
        { "result", new[] { "final", "expected", "positive", "negative" } },
        { "method", new[] { "new", "alternative", "standard", "common" } },
        { "idea", new[] { "new", "good", "interesting", "creative" } },

        // Words that can be inserted near verbs
        { "work", new[] { "hard", "well", "efficiently", "together" } },
        { "make", new[] { "quickly", "easily", "carefully", "successfully" } },
        { "think", new[] { "carefully", "deeply", "critically", "logically" } },
        { "learn", new[] { "quickly", "easily", "continuously", "effectively" } }
    };

    /// <summary>
    /// Gets the number of insertions to perform.
    /// </summary>
    /// <remarks>
    /// <para>Default: 1</para>
    /// </remarks>
    public int NumInsertions { get; }

    /// <summary>
    /// Gets or sets custom words for insertion.
    /// </summary>
    public Dictionary<string, string[]>? CustomInsertionWords { get; set; }

    /// <summary>
    /// Creates a new random insertion augmentation.
    /// </summary>
    /// <param name="numInsertions">Number of words to insert (default: 1).</param>
    /// <param name="probability">Probability of applying this augmentation (default: 0.2).</param>
    /// <param name="languageCode">Language code for language-specific operations.</param>
    public RandomInsertion(
        int numInsertions = 1,
        double probability = 0.2,
        string languageCode = "en") : base(probability, languageCode)
    {
        if (numInsertions < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(numInsertions),
                "Number of insertions must be non-negative.");
        }

        NumInsertions = numInsertions;
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

        var wordDict = CustomInsertionWords ?? RelatedWords;
        var result = new List<string>(tokens);

        for (int n = 0; n < NumInsertions; n++)
        {
            // Find a word that has related words we can insert
            var candidateIndices = new List<int>();
            for (int i = 0; i < result.Count; i++)
            {
                string word = result[i].ToLowerInvariant();
                if (wordDict.ContainsKey(word))
                {
                    candidateIndices.Add(i);
                }
            }

            if (candidateIndices.Count == 0)
            {
                // No suitable words found, insert a random common word
                int randomPos = context.Random.Next(result.Count + 1);
                string[] commonInsertions = { "the", "a", "very", "also", "then" };
                string insertWord = commonInsertions[context.Random.Next(commonInsertions.Length)];
                result.Insert(randomPos, insertWord);
            }
            else
            {
                // Pick a random candidate and insert a related word
                int candidateIdx = candidateIndices[context.Random.Next(candidateIndices.Count)];
                string baseWord = result[candidateIdx].ToLowerInvariant();

                if (wordDict.TryGetValue(baseWord, out var relatedWordsList) && relatedWordsList.Length > 0)
                {
                    string insertWord = relatedWordsList[context.Random.Next(relatedWordsList.Length)];

                    // Insert before or after the base word
                    int insertPos = context.Random.NextDouble() < 0.5 ? candidateIdx : candidateIdx + 1;

                    // Preserve case of surrounding text
                    if (PreserveCase && insertPos < result.Count && char.IsUpper(result[insertPos][0]))
                    {
                        insertWord = char.ToUpperInvariant(insertWord[0]) + insertWord.Substring(1);
                    }

                    result.Insert(insertPos, insertWord);
                }
            }
        }

        return Detokenize(result.ToArray());
    }

    /// <inheritdoc />
    public override IDictionary<string, object> GetParameters()
    {
        var parameters = base.GetParameters();
        parameters["numInsertions"] = NumInsertions;
        return parameters;
    }
}
