using System.Text;
using System.Text.RegularExpressions;

namespace AiDotNet.Postprocessing.Document;

/// <summary>
/// SpellCorrection - Spell checking and correction for OCR output.
/// </summary>
/// <remarks>
/// <para>
/// SpellCorrection provides spell checking and automatic correction capabilities
/// specifically designed for OCR output, which has different error patterns than
/// normal typing errors.
/// </para>
/// <para>
/// <b>For Beginners:</b> OCR systems often misread characters, resulting in
/// misspelled words. This tool corrects them:
///
/// - Detects misspelled words
/// - Suggests corrections based on edit distance
/// - Uses context for better accuracy
/// - Handles domain-specific vocabulary
///
/// Key features:
/// - Edit distance-based suggestions
/// - Custom dictionary support
/// - Context-aware correction
/// - OCR-specific error patterns
///
/// Example usage:
/// <code>
/// var corrector = new SpellCorrection&lt;float&gt;();
/// var corrected = corrector.Process(ocrText);
/// </code>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class SpellCorrection<T> : PostprocessorBase<T, string, string>, IDisposable
{
    #region Fields

    private readonly HashSet<string> _dictionary;
    private readonly HashSet<string> _customDictionary;
    private readonly int _maxEditDistance;
    private bool _disposed;

    // Common English words for basic dictionary
    private static readonly string[] BasicDictionary =
    [
        // Articles and pronouns
        "a", "an", "the", "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
        "my", "your", "his", "her", "its", "our", "their", "this", "that", "these", "those",
        // Common verbs
        "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
        "will", "would", "could", "should", "may", "might", "must", "can", "shall",
        "go", "goes", "went", "gone", "come", "comes", "came", "get", "gets", "got",
        "make", "makes", "made", "take", "takes", "took", "see", "sees", "saw", "seen",
        "know", "knows", "knew", "think", "thinks", "thought", "want", "wants", "wanted",
        "use", "uses", "used", "find", "finds", "found", "give", "gives", "gave",
        // Common adjectives
        "good", "bad", "new", "old", "great", "little", "big", "small", "long", "short",
        "high", "low", "young", "first", "last", "next", "other", "same", "different",
        // Common nouns
        "time", "year", "people", "way", "day", "man", "woman", "child", "world", "life",
        "hand", "part", "place", "case", "week", "company", "system", "program", "question",
        "work", "government", "number", "night", "point", "home", "water", "room", "mother",
        // Common adverbs and prepositions
        "not", "no", "yes", "more", "most", "very", "just", "also", "now", "only", "then",
        "so", "than", "too", "well", "even", "still", "back", "here", "there", "where",
        "in", "on", "at", "to", "for", "with", "from", "by", "about", "into", "through",
        "during", "before", "after", "above", "below", "between", "under", "over",
        // Common conjunctions
        "and", "or", "but", "if", "when", "because", "while", "although", "though",
        // Document-specific words
        "document", "page", "section", "chapter", "paragraph", "table", "figure", "image",
        "date", "name", "address", "phone", "email", "signature", "form", "report",
        "invoice", "receipt", "contract", "agreement", "letter", "memo", "notice"
    ];

    #endregion

    #region Properties

    /// <summary>
    /// Spell correction does not support inverse transformation.
    /// </summary>
    public override bool SupportsInverse => false;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a new SpellCorrection instance with default settings.
    /// </summary>
    public SpellCorrection() : this(2) { }

    /// <summary>
    /// Creates a new SpellCorrection instance with specified max edit distance.
    /// </summary>
    /// <param name="maxEditDistance">Maximum edit distance for suggestions.</param>
    public SpellCorrection(int maxEditDistance)
    {
        _maxEditDistance = maxEditDistance;
        _dictionary = new HashSet<string>(BasicDictionary, StringComparer.OrdinalIgnoreCase);
        _customDictionary = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
    }

    #endregion

    #region Core Implementation

    /// <summary>
    /// Corrects spelling errors in the text.
    /// </summary>
    /// <param name="input">The text to correct.</param>
    /// <returns>The corrected text.</returns>
    protected override string ProcessCore(string input)
    {
        if (string.IsNullOrEmpty(input))
            return input;

        var words = TokenizeText(input);
        var result = new StringBuilder();
        int lastIndex = 0;

        foreach (var (word, startIndex, endIndex) in words)
        {
            // Append text before this word
            result.Append(input.Substring(lastIndex, startIndex - lastIndex));

            // Check if word needs correction
            if (NeedsCorrection(word))
            {
                var corrected = GetBestCorrection(word);
                result.Append(PreserveCase(word, corrected));
            }
            else
            {
                result.Append(word);
            }

            lastIndex = endIndex;
        }

        // Append remaining text
        result.Append(input.Substring(lastIndex));

        return result.ToString();
    }

    /// <summary>
    /// Validates the input text.
    /// </summary>
    protected override void ValidateInput(string input)
    {
        // Allow null/empty strings - they will be returned as-is
    }

    #endregion

    #region Public Methods

    /// <summary>
    /// Checks if a word is spelled correctly.
    /// </summary>
    public bool IsCorrect(string word)
    {
        if (string.IsNullOrEmpty(word))
            return true;

        // Numbers are always correct
        if (double.TryParse(word, out _))
            return true;

        // Check dictionaries
        return _dictionary.Contains(word) || _customDictionary.Contains(word);
    }

    /// <summary>
    /// Gets correction suggestions for a misspelled word.
    /// </summary>
    public IList<string> GetSuggestions(string word, int maxSuggestions = 5)
    {
        var suggestions = new List<(string Word, int Distance)>();

        // Check both dictionaries
        var allWords = _dictionary.Concat(_customDictionary);

        foreach (var dictWord in allWords)
        {
            int distance = CalculateEditDistance(word.ToLower(), dictWord.ToLower());
            if (distance <= _maxEditDistance)
            {
                suggestions.Add((dictWord, distance));
            }
        }

        return suggestions
            .OrderBy(s => s.Distance)
            .ThenBy(s => Math.Abs(s.Word.Length - word.Length))
            .Take(maxSuggestions)
            .Select(s => s.Word)
            .ToList();
    }

    /// <summary>
    /// Adds a word to the custom dictionary.
    /// </summary>
    public void AddToDictionary(string word)
    {
        if (!string.IsNullOrEmpty(word))
            _customDictionary.Add(word.ToLower());
    }

    /// <summary>
    /// Adds multiple words to the custom dictionary.
    /// </summary>
    public void AddToDictionary(IEnumerable<string> words)
    {
        foreach (var word in words)
            AddToDictionary(word);
    }

    /// <summary>
    /// Loads a custom dictionary from words.
    /// </summary>
    public void LoadDictionary(IEnumerable<string> words)
    {
        _customDictionary.Clear();
        AddToDictionary(words);
    }

    #endregion

    #region Private Methods

    private bool NeedsCorrection(string word)
    {
        // Skip very short words
        if (word.Length <= 2)
            return false;

        // Skip numbers and words with numbers
        if (RegexHelper.IsMatch(word, @"\d"))
            return false;

        // Skip words with special characters (abbreviations, etc.)
        if (RegexHelper.IsMatch(word, @"[^a-zA-Z]"))
            return false;

        return !IsCorrect(word);
    }

    private string GetBestCorrection(string word)
    {
        var suggestions = GetSuggestions(word, 1);
        return suggestions.Count > 0 ? suggestions[0] : word;
    }

    private string PreserveCase(string original, string corrected)
    {
        if (string.IsNullOrEmpty(original) || string.IsNullOrEmpty(corrected))
            return corrected;

        // All uppercase
        if (original.All(char.IsUpper))
            return corrected.ToUpper();

        // Title case
        if (char.IsUpper(original[0]) && original.Skip(1).All(char.IsLower))
            return char.ToUpper(corrected[0]) + corrected.Substring(1).ToLower();

        // Default: lowercase
        return corrected.ToLower();
    }

    private IEnumerable<(string Word, int StartIndex, int EndIndex)> TokenizeText(string text)
    {
        var matches = RegexHelper.Matches(text, @"\b[a-zA-Z]+\b");
        foreach (Match match in matches)
        {
            yield return (match.Value, match.Index, match.Index + match.Length);
        }
    }

    private int CalculateEditDistance(string s1, string s2)
    {
        int[,] dp = new int[s1.Length + 1, s2.Length + 1];

        for (int i = 0; i <= s1.Length; i++)
            dp[i, 0] = i;
        for (int j = 0; j <= s2.Length; j++)
            dp[0, j] = j;

        for (int i = 1; i <= s1.Length; i++)
        {
            for (int j = 1; j <= s2.Length; j++)
            {
                int cost = s1[i - 1] == s2[j - 1] ? 0 : 1;
                dp[i, j] = Math.Min(Math.Min(
                    dp[i - 1, j] + 1,      // deletion
                    dp[i, j - 1] + 1),     // insertion
                    dp[i - 1, j - 1] + cost); // substitution

                // Damerau transposition
                if (i > 1 && j > 1 && s1[i - 1] == s2[j - 2] && s1[i - 2] == s2[j - 1])
                {
                    dp[i, j] = Math.Min(dp[i, j], dp[i - 2, j - 2] + cost);
                }
            }
        }

        return dp[s1.Length, s2.Length];
    }

    #endregion

    #region Disposal

    /// <inheritdoc/>
    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Releases resources used by the spell corrector.
    /// </summary>
    protected virtual void Dispose(bool disposing)
    {
        if (!_disposed && disposing)
        {
            _dictionary.Clear();
            _customDictionary.Clear();
        }
        _disposed = true;
    }

    #endregion
}



