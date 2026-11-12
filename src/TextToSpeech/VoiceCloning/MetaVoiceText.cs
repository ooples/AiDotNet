using System.Text;

namespace AiDotNet.TextToSpeech.VoiceCloning;

/// <summary>The released MetaVoice text normalization and Latin-1 input restriction.</summary>
internal static class MetaVoiceText
{
    private static readonly Dictionary<char, string> Replacements = new()
    {
        [(char)8175] = "'", [(char)8189] = "'", [(char)8190] = "'",
        [(char)8208] = "-", [(char)8209] = "-", [(char)8210] = "-", [(char)8211] = "-", [(char)8212] = "-", [(char)8213] = "-",
        [(char)8214] = "||", [(char)8216] = "'", [(char)8217] = "'", [(char)8218] = ",", [(char)8219] = "`",
        [(char)8220] = "\"", [(char)8221] = "\"", [(char)8222] = ",,", [(char)8223] = "\"",
        [(char)8228] = ".", [(char)8229] = "..", [(char)8230] = "...", [(char)8242] = "'", [(char)8243] = "\"",
        [(char)8245] = "'", [(char)8246] = "\"", [(char)180] = "'",
        // Preserve the author's literal decimal key, rather than guessing from its comment.
        [(char)2122] = "TM"
    };

    private static bool Space(char value) => char.IsWhiteSpace(value) || (value >= 28 && value <= 31);

    internal static string Normalize(string text)
    {
        if (text is null) throw new ArgumentNullException(nameof(text));
        var translated = new StringBuilder();
        foreach (char value in text)
        {
            if (Replacements.TryGetValue(value, out var replacement)) translated.Append(replacement);
            else if (value >= 256) throw new ArgumentException("The released MetaVoice normalizer does not support this character.", nameof(text));
            else translated.Append(value);
        }
        string cleaned = translated.ToString().Replace('\t', ' ').Replace('\n', ' ').Replace('\r', ' ').Replace('*', ' ');
        int start = 0, end = cleaned.Length;
        while (start < end && Space(cleaned[start])) start++;
        while (end > start && Space(cleaned[end - 1])) end--;
        var output = new StringBuilder();
        for (int i = start; i < end;)
        {
            char value = cleaned[i++]; int after = i;
            if (Space(value)) while (after < end && Space(cleaned[after])) after++;
            output.Append(after > i ? ' ' : value); i = after;
        }
        return output.ToString();
    }
}
