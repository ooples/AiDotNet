using System.Globalization;
using System.Text;

namespace AiDotNet.TextToSpeech.FrontEnd;

/// <summary>Python 3's <c>str</c> predicates and case mapping, for front ends ported from Python whose output depends on
/// them (code point by code point, as Python iterates strings).</summary>
internal static class PythonString
{
    /// <summary><c>str.isspace</c> for one character: Unicode whitespace plus the information separators U+001C–U+001F.</summary>
    public static bool IsSpace(char c) => char.IsWhiteSpace(c) || (c >= (char)0x1C && c <= (char)0x1F);

    /// <summary><c>str.isspace</c>: non-empty and all whitespace.</summary>
    public static bool IsSpace(string s) => s.Length > 0 && s.All(IsSpace);

    /// <summary><c>str.strip()</c>.</summary>
    public static string Strip(string text)
    {
        int start = 0, end = text.Length;
        while (start < end && IsSpace(text[start])) start++;
        while (end > start && IsSpace(text[end - 1])) end--;
        return text.Substring(start, end - start);
    }

    private static UnicodeCategory Category(string glyph) => CharUnicodeInfo.GetUnicodeCategory(glyph, 0);

    /// <summary><c>c.isalpha()</c> for one code point: a letter (Lu, Ll, Lt, Lm, Lo).</summary>
    public static bool IsAlpha(string glyph) => Category(glyph) is UnicodeCategory.UppercaseLetter or UnicodeCategory.LowercaseLetter
        or UnicodeCategory.TitlecaseLetter or UnicodeCategory.ModifierLetter or UnicodeCategory.OtherLetter;

    /// <summary><c>c.isdigit()</c> for one code point: a decimal digit, or a character with a digit value (superscripts).</summary>
    public static bool IsDigit(string glyph) =>
        Category(glyph) == UnicodeCategory.DecimalDigitNumber || (Category(glyph) == UnicodeCategory.OtherNumber && CharUnicodeInfo.GetDigitValue(glyph, 0) >= 0);

    private static bool Cased(string glyph) => Category(glyph) is UnicodeCategory.UppercaseLetter or UnicodeCategory.LowercaseLetter
        or UnicodeCategory.TitlecaseLetter;

    /// <summary><c>c.isupper()</c> for one code point.</summary>
    public static bool IsUpper(string glyph) => Category(glyph) == UnicodeCategory.UppercaseLetter;

    /// <summary><c>str.islower()</c>: at least one cased character, and no upper- or title-case one.</summary>
    public static bool IsLower(string text)
    {
        bool cased = false;
        foreach (var glyph in ChineseText.Codepoints(text))
        {
            var category = Category(glyph);
            if (category is UnicodeCategory.UppercaseLetter or UnicodeCategory.TitlecaseLetter) return false;
            if (category == UnicodeCategory.LowercaseLetter) cased = true;
        }
        return cased;
    }

    // Unicode's Case_Ignorable: marks, format characters, modifier letters and symbols, and the word-internal
    // punctuation (Word_Break MidLetter, MidNumLet, Single_Quote).
    private static bool CaseIgnorable(string glyph)
    {
        if (Category(glyph) is UnicodeCategory.NonSpacingMark or UnicodeCategory.EnclosingMark or UnicodeCategory.Format
            or UnicodeCategory.ModifierLetter or UnicodeCategory.ModifierSymbol) return true;
        int c = char.ConvertToUtf32(glyph, 0);
        return c is 0x27 or 0x2E or 0x3A or 0xB7 or 0x387 or 0x55F or 0x5F4 or 0x2018 or 0x2019 or 0x2024 or 0x2027
            or 0xFE13 or 0xFE52 or 0xFE55 or 0xFF07 or 0xFF0E or 0xFF1A;
    }

    /// <summary><c>str.lower()</c>: the full lowercase mapping (U+0130 to <c>i̇</c>) and the Final_Sigma rule (a capital
    /// sigma ending a word becomes ς).</summary>
    public static string Lower(string text)
    {
        var glyphs = ChineseText.Codepoints(text);
        var output = new StringBuilder(text.Length);
        for (int i = 0; i < glyphs.Count; i++)
        {
            string g = glyphs[i];
            if (g == "\u0130")
            {
                output.Append('i').Append((char)0x0307);
            }
            else if (g == "\u03A3")
            {
                bool before = false, after = false;
                for (int j = i - 1; j >= 0; j--)
                {
                    if (CaseIgnorable(glyphs[j])) continue;
                    before = Cased(glyphs[j]);
                    break;
                }
                for (int j = i + 1; j < glyphs.Count; j++)
                {
                    if (CaseIgnorable(glyphs[j])) continue;
                    after = Cased(glyphs[j]);
                    break;
                }
                output.Append(before && !after ? (char)0x03C2 : (char)0x03C3);
            }
            else
            {
                output.Append(g.ToLowerInvariant());
            }
        }
        return output.ToString();
    }
}
