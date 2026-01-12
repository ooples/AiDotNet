using System.Text;
using System.Text.RegularExpressions;

namespace AiDotNet.Audio.TextToSpeech;

/// <summary>
/// Preprocesses text for text-to-speech synthesis.
/// </summary>
/// <remarks>
/// <para>
/// This class handles text normalization and grapheme-to-phoneme (G2P) conversion
/// to prepare text for acoustic model input.
/// </para>
/// <para><b>For Beginners:</b> Before TTS can synthesize speech, we need to convert
/// written text into phonemes (speech sounds). This involves:
/// <list type="bullet">
/// <item>Normalizing text (expanding abbreviations, numbers)</item>
/// <item>Converting graphemes (letters) to phonemes (sounds)</item>
/// </list>
/// For example: "Dr. Smith, 123 Main St." becomes phonemes like "D AH K T ER S M IH TH..."
/// </para>
/// </remarks>
public class TtsPreprocessor
{

    // Common abbreviations
    private static readonly Dictionary<string, string> Abbreviations = new(StringComparer.OrdinalIgnoreCase)
    {
        ["Mr."] = "Mister",
        ["Mrs."] = "Misses",
        ["Dr."] = "Doctor",
        ["Prof."] = "Professor",
        ["Jr."] = "Junior",
        ["Sr."] = "Senior",
        ["St."] = "Street",
        ["Ave."] = "Avenue",
        ["Blvd."] = "Boulevard",
        ["etc."] = "et cetera",
        ["vs."] = "versus",
        ["e.g."] = "for example",
        ["i.e."] = "that is",
    };

    // Simple phoneme mappings (CMU dict style, simplified)
    // In production, would use a full CMU dictionary or neural G2P
    private static readonly Dictionary<string, int[]> WordToPhonemes = new(StringComparer.OrdinalIgnoreCase)
    {
        ["hello"] = [1, 2, 3, 4],
        ["world"] = [5, 6, 7, 8],
        ["the"] = [9, 10],
        ["a"] = [11],
        ["is"] = [12, 13],
        ["this"] = [14, 15, 16],
        ["test"] = [17, 18, 19, 20],
    };

    // Phoneme IDs for basic characters (fallback)
    private static readonly Dictionary<char, int> CharToPhoneme = new()
    {
        ['a'] = 1,
        ['b'] = 2,
        ['c'] = 3,
        ['d'] = 4,
        ['e'] = 5,
        ['f'] = 6,
        ['g'] = 7,
        ['h'] = 8,
        ['i'] = 9,
        ['j'] = 10,
        ['k'] = 11,
        ['l'] = 12,
        ['m'] = 13,
        ['n'] = 14,
        ['o'] = 15,
        ['p'] = 16,
        ['q'] = 17,
        ['r'] = 18,
        ['s'] = 19,
        ['t'] = 20,
        ['u'] = 21,
        ['v'] = 22,
        ['w'] = 23,
        ['x'] = 24,
        ['y'] = 25,
        ['z'] = 26,
        [' '] = 0,
        ['.'] = 27,
        [','] = 28,
        ['!'] = 29,
        ['?'] = 30,
        ['\''] = 31,
    };

    /// <summary>
    /// Special phoneme IDs.
    /// </summary>
    public const int PadPhoneme = 0;
    public const int StartPhoneme = 100;
    public const int EndPhoneme = 101;
    public const int SilencePhoneme = 102;

    /// <summary>
    /// Converts text to phoneme IDs.
    /// </summary>
    /// <param name="text">The text to convert.</param>
    /// <returns>Array of phoneme IDs.</returns>
    public int[] TextToPhonemes(string text)
    {
        // Normalize text
        var normalized = NormalizeText(text);

        // Convert to phonemes
        var phonemes = new List<int> { StartPhoneme };

        var words = normalized.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);

        foreach (var word in words)
        {
            var wordPhonemes = WordToPhonemeIds(word);
            phonemes.AddRange(wordPhonemes);
            phonemes.Add(SilencePhoneme); // Add silence between words
        }

        phonemes.Add(EndPhoneme);

        return [.. phonemes];
    }

    /// <summary>
    /// Normalizes text for TTS processing.
    /// </summary>
    /// <param name="text">The text to normalize.</param>
    /// <returns>Normalized text.</returns>
    public string NormalizeText(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
            return string.Empty;

        var result = text;

        // Expand abbreviations
        foreach (var (abbrev, expansion) in Abbreviations)
        {
            result = RegexHelper.Replace(result, RegexHelper.Escape(abbrev), expansion, RegexOptions.IgnoreCase);
        }

        // Expand numbers
        result = ExpandNumbers(result);

        // Normalize whitespace
        result = RegexHelper.Replace(result, @"\s+", " ", RegexOptions.None).Trim();

        // Convert to lowercase for phoneme lookup
        result = result.ToLowerInvariant();

        // Remove unsupported characters
        var sb = new StringBuilder();
        foreach (char c in result)
        {
            if (char.IsLetterOrDigit(c) || c == ' ' || c == '.' || c == ',' || c == '!' || c == '?' || c == '\'')
            {
                sb.Append(c);
            }
        }

        return sb.ToString();
    }

    /// <summary>
    /// Expands numbers to words.
    /// </summary>
    private static string ExpandNumbers(string text)
    {
        // Simple number expansion (would be more comprehensive in production)
        return RegexHelper.Replace(text, @"\d+", match =>
        {
            if (int.TryParse(match.Value, out int number))
            {
                return NumberToWords(number);
            }
            return match.Value;
        }, RegexOptions.None);
    }

    /// <summary>
    /// Converts a number to words.
    /// </summary>
    private static string NumberToWords(int number)
    {
        if (number == 0) return "zero";
        if (number < 0) return "minus " + NumberToWords(Math.Abs(number));

        var parts = new List<string>();

        if (number / 1000000 > 0)
        {
            parts.Add(NumberToWords(number / 1000000) + " million");
            number %= 1000000;
        }

        if (number / 1000 > 0)
        {
            parts.Add(NumberToWords(number / 1000) + " thousand");
            number %= 1000;
        }

        if (number / 100 > 0)
        {
            parts.Add(NumberToWords(number / 100) + " hundred");
            number %= 100;
        }

        if (number > 0)
        {
            string[] ones = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
                            "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
                            "seventeen", "eighteen", "nineteen"];
            string[] tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"];

            if (number < 20)
            {
                parts.Add(ones[number]);
            }
            else
            {
                string tensWord = tens[number / 10];
                int onesDigit = number % 10;
                if (onesDigit > 0)
                {
                    parts.Add(tensWord + " " + ones[onesDigit]);
                }
                else
                {
                    parts.Add(tensWord);
                }
            }
        }

        return string.Join(" ", parts);
    }

    /// <summary>
    /// Converts a word to phoneme IDs.
    /// </summary>
    private static int[] WordToPhonemeIds(string word)
    {
        // Try dictionary lookup first
        if (WordToPhonemes.TryGetValue(word, out var phonemes))
        {
            return phonemes;
        }

        // Fallback to character-by-character
        var result = new List<int>();
        foreach (char c in word.ToLowerInvariant())
        {
            if (CharToPhoneme.TryGetValue(c, out int phoneme))
            {
                result.Add(phoneme);
            }
        }

        return [.. result];
    }

    /// <summary>
    /// Splits text into sentences for chunked synthesis.
    /// </summary>
    /// <param name="text">The text to split.</param>
    /// <returns>List of sentences.</returns>
    public List<string> SplitIntoSentences(string text)
    {
        // Split on sentence-ending punctuation
        var pattern = @"(?<=[.!?])\s+";
        var sentences = RegexHelper.Split(text, pattern, RegexOptions.None)
            .Where(s => !string.IsNullOrWhiteSpace(s))
            .ToList();

        return sentences;
    }
}



