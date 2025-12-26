namespace AiDotNet.Augmentation.Text;

/// <summary>
/// Randomly deletes words from text.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Random deletion removes some words from text,
/// simulating how people often skip words when speaking quickly or how text
/// might have missing words in noisy transcriptions.</para>
/// <para><b>When to use:</b>
/// <list type="bullet">
/// <item>Text classification where exact wording isn't critical</item>
/// <item>Training robust models for noisy/incomplete text</item>
/// <item>Simulating transcription errors</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class RandomDeletion<T> : TextAugmenterBase<T>
{
    /// <summary>
    /// Gets the probability of deleting each word.
    /// </summary>
    /// <remarks>
    /// <para>Default: 0.1 (10% of words deleted)</para>
    /// </remarks>
    public double DeletionProbability { get; }

    /// <summary>
    /// Gets the minimum number of words to keep.
    /// </summary>
    /// <remarks>
    /// <para>Default: 1</para>
    /// <para>Ensures the text isn't completely deleted.</para>
    /// </remarks>
    public int MinWordsToKeep { get; }

    /// <summary>
    /// Creates a new random deletion augmentation.
    /// </summary>
    /// <param name="deletionProbability">Probability of deleting each word (default: 0.1).</param>
    /// <param name="minWordsToKeep">Minimum words to keep (default: 1).</param>
    /// <param name="probability">Probability of applying this augmentation (default: 0.2).</param>
    /// <param name="languageCode">Language code for language-specific operations.</param>
    public RandomDeletion(
        double deletionProbability = 0.1,
        int minWordsToKeep = 1,
        double probability = 0.2,
        string languageCode = "en") : base(probability, languageCode)
    {
        if (deletionProbability < 0 || deletionProbability > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(deletionProbability),
                "Deletion probability must be between 0 and 1.");
        }

        if (minWordsToKeep < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(minWordsToKeep),
                "Minimum words to keep must be non-negative.");
        }

        DeletionProbability = deletionProbability;
        MinWordsToKeep = minWordsToKeep;
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
        if (tokens.Length <= MinWordsToKeep) return text;

        var keptTokens = new List<string>();

        foreach (string token in tokens)
        {
            if (context.Random.NextDouble() >= DeletionProbability)
            {
                keptTokens.Add(token);
            }
        }

        // Ensure minimum words are kept
        if (keptTokens.Count < MinWordsToKeep)
        {
            // Randomly add back some deleted words
            var allIndices = Enumerable.Range(0, tokens.Length).ToList();
            var keptIndices = new HashSet<int>();

            // First, mark which indices we already kept
            for (int i = 0, j = 0; i < tokens.Length && j < keptTokens.Count; i++)
            {
                if (tokens[i] == keptTokens[j])
                {
                    keptIndices.Add(i);
                    j++;
                }
            }

            // Add random indices until we have minimum
            while (keptIndices.Count < MinWordsToKeep && keptIndices.Count < tokens.Length)
            {
                int idx = context.Random.Next(tokens.Length);
                keptIndices.Add(idx);
            }

            // Rebuild in original order
            keptTokens.Clear();
            foreach (int idx in keptIndices.OrderBy(x => x))
            {
                keptTokens.Add(tokens[idx]);
            }
        }

        return Detokenize(keptTokens.ToArray());
    }

    /// <inheritdoc />
    public override IDictionary<string, object> GetParameters()
    {
        var parameters = base.GetParameters();
        parameters["deletionProbability"] = DeletionProbability;
        parameters["minWordsToKeep"] = MinWordsToKeep;
        return parameters;
    }
}
