using System.IO.Compression;
using System.Text;
using System.Text.Json;

namespace AiDotNet.TextToSpeech.FrontEnd;

/// <summary>Unidecode 1.3.8's exact ASCII replacements, exported by tortoise_text_resources.py.</summary>
internal static class AsciiTransliterator
{
    private static readonly Lazy<Dictionary<int, string>> Replacements = new(() =>
    {
        using var stream = typeof(AsciiTransliterator).Assembly.GetManifestResourceStream("AiDotNet.TextToSpeech.FrontEnd.unidecode.json.gz")
            ?? throw new InvalidDataException("Missing Unidecode replacement tables.");
        using var gzip = new GZipStream(stream, CompressionMode.Decompress);
        return System.Text.Json.JsonSerializer.Deserialize<Dictionary<int, string>>(gzip) ?? throw new InvalidDataException("Invalid Unidecode tables.");
    });

    internal static string Convert(string text)
    {
        var result = new StringBuilder();
        for (int i = 0; i < text.Length; i++)
        {
            int code = text[i];
            if (char.IsHighSurrogate(text[i]) && i + 1 < text.Length && char.IsLowSurrogate(text[i + 1]))
                code = char.ConvertToUtf32(text[i], text[++i]);
            if (code < 128) result.Append((char)code);
            else if (Replacements.Value.TryGetValue(code, out var replacement)) result.Append(replacement);
        }
        return result.ToString();
    }
}
