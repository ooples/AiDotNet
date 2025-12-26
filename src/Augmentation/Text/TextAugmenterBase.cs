namespace AiDotNet.Augmentation.Text;

/// <summary>
/// Base class for text data augmentations.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Text augmentation creates variations of text to improve
/// model robustness to different phrasings and writing styles. Common techniques include:
/// <list type="bullet">
/// <item>Synonym replacement (replacing words with similar meanings)</item>
/// <item>Random deletion (removing random words)</item>
/// <item>Random swap (swapping word positions)</item>
/// <item>Random insertion (adding synonyms of random words)</item>
/// <item>Back-translation (translate to another language and back)</item>
/// </list>
/// </para>
/// <para>Text data is represented as an array of strings (sentences/documents).</para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public abstract class TextAugmenterBase<T> : AugmentationBase<T, string[]>
{
    /// <summary>
    /// Gets or sets the language code for language-specific operations.
    /// </summary>
    /// <remarks>
    /// <para>Default: "en" (English)</para>
    /// <para>Used for synonym lookup, tokenization, etc.</para>
    /// </remarks>
    public string LanguageCode { get; set; } = "en";

    /// <summary>
    /// Gets or sets whether to preserve case when modifying text.
    /// </summary>
    /// <remarks>
    /// <para>Default: true</para>
    /// <para>When true, replaced words will match the case of the original word.</para>
    /// </remarks>
    public bool PreserveCase { get; set; } = true;

    /// <summary>
    /// Initializes a new text augmentation.
    /// </summary>
    /// <param name="probability">The probability of applying this augmentation (0.0 to 1.0).</param>
    /// <param name="languageCode">The language code for language-specific operations.</param>
    protected TextAugmenterBase(double probability = 1.0, string languageCode = "en") : base(probability)
    {
        LanguageCode = languageCode;
    }

    /// <summary>
    /// Tokenizes text into words.
    /// </summary>
    /// <param name="text">The text to tokenize.</param>
    /// <returns>An array of word tokens.</returns>
    protected virtual string[] Tokenize(string text)
    {
        // Simple whitespace tokenization - can be overridden for language-specific tokenization
        return text.Split(new[] { ' ', '\t', '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries);
    }

    /// <summary>
    /// Joins tokens back into text.
    /// </summary>
    /// <param name="tokens">The tokens to join.</param>
    /// <returns>The joined text.</returns>
    protected virtual string Detokenize(string[] tokens)
    {
        return string.Join(" ", tokens);
    }

    /// <summary>
    /// Checks if a word is a stopword (common word to skip during augmentation).
    /// </summary>
    /// <param name="word">The word to check.</param>
    /// <returns>True if the word is a stopword.</returns>
    protected virtual bool IsStopword(string word)
    {
        // Common English stopwords - override for other languages
        var stopwords = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
        {
            "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
            "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
            "do", "does", "did", "will", "would", "could", "should", "may", "might",
            "of", "with", "as", "by", "from", "this", "that", "these", "those",
            "it", "its", "i", "you", "he", "she", "we", "they", "my", "your", "his", "her"
        };

        return stopwords.Contains(word.ToLowerInvariant());
    }

    /// <inheritdoc />
    public override IDictionary<string, object> GetParameters()
    {
        var parameters = base.GetParameters();
        parameters["languageCode"] = LanguageCode;
        parameters["preserveCase"] = PreserveCase;
        return parameters;
    }
}
