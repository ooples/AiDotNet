using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.TextToSpeech.FrontEnd;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// The ftfy port reproduces ftfy 6.3.1's fix_text (tools/reference-data/ftfy_reference.py) on clean text, HTML entities,
/// ligatures, full-width and curly characters, control characters, line breaks and mojibake through each codec ftfy
/// knows, nested and partial.
/// </summary>
public class FtfyTests
{
    private static JObject Reference()
    {
        const string fileName = "ftfy_reference.json";
        string output = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", fileName);
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); !File.Exists(output) && dir is not null; dir = dir.Parent)
            output = Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", fileName);
        return JObject.Parse(File.ReadAllText(output));
    }

    private static string Show(string s) => string.Concat(s.Select(c => c < 0x20 || (c >= 0x7f && c <= 0x9f) || char.IsSurrogate(c) ? $"\\u{(int)c:x4}" : c.ToString()));

    [Fact(Timeout = 120000)]
    public async Task FixText_MatchesFtfy()
    {
        await Task.Yield();
        var mismatches = Reference()["corpus"]!
            .Select(c => (Text: (string)c["text"]!, Expected: (string)c["fixed"]!))
            .Select(c => (c.Text, c.Expected, Actual: Ftfy.FixText(c.Text)))
            .Where(c => c.Expected != c.Actual)
            .Select(c => $"{Show(c.Text)}\n  ftfy {Show(c.Expected)}\n  port {Show(c.Actual)}")
            .ToList();
        Assert.True(mismatches.Count == 0, $"{mismatches.Count} mismatches:\n" + string.Join("\n", mismatches));
    }
}
