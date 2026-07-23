using System.Text;
using System.Text.RegularExpressions;

namespace AiDotNet.Postprocessing.Document;

/// <summary>
/// TextPostprocessor - OCR text postprocessing utilities.
/// </summary>
/// <remarks>
/// <para>
/// TextPostprocessor provides a comprehensive pipeline for cleaning and correcting
/// text output from OCR systems, improving readability and accuracy.
/// </para>
/// <para>
/// <b>For Beginners:</b> OCR output often contains errors and formatting issues.
/// This tool cleans up the text:
///
/// - Remove unwanted characters
/// - Fix common OCR errors
/// - Normalize whitespace
/// - Correct formatting
///
/// Key features:
/// - Character normalization
/// - Whitespace handling
/// - Common OCR error correction
/// - Language-aware processing
///
/// Example usage:
/// <code>
/// var processor = new TextPostprocessor&lt;float&gt;();
/// var cleanText = processor.Process(rawOcrText);
/// </code>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class TextPostprocessor<T> : PostprocessorBase<T, string, string>, IDisposable
{
    #region Fields

    private readonly SpellCorrection<T> _spellCorrector;
    private readonly TextPostprocessorOptions _options;
    private bool _disposed;

    // Common OCR error patterns
    private static readonly Dictionary<string, string> CommonOcrErrors = new()
    {
        { "0", "O" },  // Zero often confused with O
        { "1", "l" },  // One often confused with l
        { "5", "S" },  // Five often confused with S
        { "8", "B" },  // Eight often confused with B
        { "rn", "m" }, // rn often appears as m
        { "vv", "w" }, // vv often appears as w
        { "cl", "d" }, // cl often appears as d
        { "li", "h" }, // li often appears as h
    };

    // Character replacements for normalization
    private static readonly Dictionary<char, char> CharReplacements = new()
    {
        { '\u2018', '\'' }, // Left single quote
        { '\u2019', '\'' }, // Right single quote
        { '\u201C', '"' },  // Left double quote
        { '\u201D', '"' },  // Right double quote
        { '\u2013', '-' },  // En dash
        { '\u2014', '-' },  // Em dash
        { '\u00A0', ' ' },  // Non-breaking space
        { '\u00AD', '-' },  // Soft hyphen
        { '\u2026', '.' },  // Ellipsis (simplify to dot)
    };

    #endregion

    #region Properties

    /// <summary>
    /// Text postprocessor supports inverse transformation (returns original).
    /// </summary>
    public override bool SupportsInverse => false;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a new TextPostprocessor with default options.
    /// </summary>
    public TextPostprocessor() : this(new TextPostprocessorOptions()) { }

    /// <summary>
    /// Creates a new TextPostprocessor with specified options.
    /// </summary>
    public TextPostprocessor(TextPostprocessorOptions options)
    {
        _options = options;
        _spellCorrector = new SpellCorrection<T>();
    }

    #endregion

    #region Core Implementation

    /// <summary>
    /// Processes OCR text through the full postprocessing pipeline.
    /// </summary>
    /// <param name="input">The raw OCR text.</param>
    /// <returns>The cleaned and corrected text.</returns>
    protected override string ProcessCore(string input)
    {
        if (string.IsNullOrEmpty(input))
            return input;

        var result = input;

        // Step 1: Basic cleaning
        if (_options.RemoveControlCharacters)
            result = RemoveControlCharacters(result);

        // Step 2: Character normalization
        if (_options.NormalizeCharacters)
            result = NormalizeCharacters(result);

        // Step 3: Whitespace normalization
        if (_options.NormalizeWhitespace)
            result = NormalizeWhitespace(result);

        // Step 4: Fix common OCR errors
        if (_options.FixCommonOcrErrors)
            result = FixCommonOcrErrors(result);

        // Step 5: Spell correction
        if (_options.ApplySpellCorrection)
            result = _spellCorrector.Process(result);

        // Step 6: Fix line breaks
        if (_options.MergeBrokenLines)
            result = MergeBrokenLines(result);

        // Step 7: Remove duplicate spaces
        if (_options.RemoveDuplicateSpaces)
            result = RemoveDuplicateSpaces(result);

        return result.Trim();
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
    /// Removes control characters from text.
    /// </summary>
    public string RemoveControlCharacters(string text)
    {
        var sb = new StringBuilder(text.Length);
        foreach (char c in text)
        {
            if (!char.IsControl(c) || c == '\n' || c == '\r' || c == '\t')
                sb.Append(c);
        }
        return sb.ToString();
    }

    /// <summary>
    /// Normalizes special characters to ASCII equivalents.
    /// </summary>
    public string NormalizeCharacters(string text)
    {
        var sb = new StringBuilder(text.Length);
        foreach (char c in text)
        {
            if (CharReplacements.TryGetValue(c, out char replacement))
                sb.Append(replacement);
            else
                sb.Append(c);
        }
        return sb.ToString();
    }

    /// <summary>
    /// Normalizes whitespace in the text.
    /// </summary>
    public string NormalizeWhitespace(string text)
    {
        // Replace tabs with spaces
        text = text.Replace('\t', ' ');

        // Normalize line endings
        text = text.Replace("\r\n", "\n").Replace("\r", "\n");

        return text;
    }

    /// <summary>
    /// Fixes common OCR recognition errors.
    /// </summary>
    public string FixCommonOcrErrors(string text)
    {
        // Fix common character substitutions based on context
        // These are conservative fixes that look at surrounding context

        // Fix "tbe" -> "the" (common OCR error)
        text = RegexHelper.Replace(text, @"\btbe\b", "the", RegexOptions.IgnoreCase);

        // Fix "witb" -> "with"
        text = RegexHelper.Replace(text, @"\bwitb\b", "with", RegexOptions.IgnoreCase);

        // Fix "tbat" -> "that"
        text = RegexHelper.Replace(text, @"\btbat\b", "that", RegexOptions.IgnoreCase);

        // Fix "bave" -> "have"
        text = RegexHelper.Replace(text, @"\bbave\b", "have", RegexOptions.IgnoreCase);

        // Fix common number/letter confusions in context
        // Only fix 0/O in word context, not in numbers
        text = RegexHelper.Replace(text, @"(?<=[a-zA-Z])0(?=[a-zA-Z])", "o");

        // Fix l/1 in word context
        text = RegexHelper.Replace(text, @"(?<=[a-zA-Z])1(?=[a-zA-Z])", "l");

        return text;
    }

    /// <summary>
    /// Merges lines that were incorrectly broken.
    /// </summary>
    public string MergeBrokenLines(string text)
    {
        // Merge lines that end with a hyphen
        text = RegexHelper.Replace(text, @"-\n(?=[a-z])", "");

        // Merge lines that don't end with sentence-ending punctuation
        // and the next line starts with a lowercase letter
        text = RegexHelper.Replace(text, @"(?<![.!?:])\n(?=[a-z])", " ");

        return text;
    }

    /// <summary>
    /// Removes duplicate consecutive spaces.
    /// </summary>
    public string RemoveDuplicateSpaces(string text)
    {
        return RegexHelper.Replace(text, @" {2,}", " ");
    }

    /// <summary>
    /// Extracts sentences from processed text.
    /// </summary>
    public IList<string> ExtractSentences(string text)
    {
        var sentences = new List<string>();

        // Split on sentence-ending punctuation
        var parts = RegexHelper.Split(text, @"(?<=[.!?])\s+");

        foreach (var part in parts)
        {
            var trimmed = part.Trim();
            if (!string.IsNullOrEmpty(trimmed))
                sentences.Add(trimmed);
        }

        return sentences;
    }

    /// <summary>
    /// Extracts paragraphs from processed text.
    /// </summary>
    public IList<string> ExtractParagraphs(string text)
    {
        var paragraphs = new List<string>();

        // Split on double newlines
        var parts = RegexHelper.Split(text, @"\n\s*\n");

        foreach (var part in parts)
        {
            var trimmed = part.Trim();
            if (!string.IsNullOrEmpty(trimmed))
                paragraphs.Add(trimmed);
        }

        return paragraphs;
    }

    /// <summary>
    /// Removes headers and footers from document text.
    /// </summary>
    public string RemoveHeadersFooters(string text, int headerLines = 2, int footerLines = 2)
    {
        var lines = text.Split('\n');
        if (lines.Length <= headerLines + footerLines)
            return text;

        var contentLines = lines.Skip(headerLines).Take(lines.Length - headerLines - footerLines);
        return string.Join("\n", contentLines);
    }

    /// <summary>
    /// Removes page numbers from text.
    /// </summary>
    public string RemovePageNumbers(string text)
    {
        // Remove standalone numbers that are likely page numbers
        // (at start or end of lines, optionally with "Page" prefix)
        text = RegexHelper.Replace(text, @"^\s*(?:Page\s*)?\d+\s*$", "", RegexOptions.Multiline | RegexOptions.IgnoreCase);
        text = RegexHelper.Replace(text, @"^-\s*\d+\s*-$", "", RegexOptions.Multiline);

        return text;
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
    /// Releases resources used by the text postprocessor.
    /// </summary>
    protected virtual void Dispose(bool disposing)
    {
        if (!_disposed && disposing)
        {
            _spellCorrector.Dispose();
        }
        _disposed = true;
    }

    #endregion
}



