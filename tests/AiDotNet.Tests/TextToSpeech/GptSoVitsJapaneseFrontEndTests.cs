using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using AiDotNet.TextToSpeech.FrontEnd;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// GPT-SoVITS's Japanese front end against GPT-SoVITS's own japanese.py with pyopenjtalk 0.4.1 and its user dictionary
/// (tools/reference-data/gptsovits_japanese_reference.py), where the Open JTalk dictionary is present
/// (<c>OPENJTALK_DIC</c>: open_jtalk_dic_utf_8-1.11): the normalized text and the phones with prosody marks.
/// </summary>
public class GptSoVitsJapaneseFrontEndTests
{
    private static JsonElement Load()
    {
        const string name = "gptsovits_japanese_reference.json";
        string relative = Path.Combine("TextToSpeech", "ReferenceData", name);
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", relative))) dir = dir.Parent;
        string path = Path.Combine(dir?.FullName ?? throw new FileNotFoundException(name), "tests", "AiDotNet.Tests", relative);
        return JsonDocument.Parse(File.ReadAllText(path)).RootElement;
    }

    [Fact(Timeout = 600000)]
    public async Task G2P_MatchesJapanesePy_WhenTheDictionaryIsPresent()
    {
        await Task.Yield();
        string? dic = Environment.GetEnvironmentVariable("OPENJTALK_DIC");
        if (dic is null || !File.Exists(Path.Combine(dic, "sys.dic"))) return;
        var frontEnd = new GptSoVitsJapaneseFrontEnd(dic);
        var failures = new List<string>();
        foreach (var s in Load().GetProperty("sentences").EnumerateArray())
        {
            string text = s.GetProperty("text").GetString()!;
            string normalized = GptSoVitsJapaneseFrontEnd.TextNormalize(text);
            Assert.Equal(s.GetProperty("normalized").GetString(), normalized);
            string expected = string.Join(" ", s.GetProperty("phones").EnumerateArray().Select(v => v.GetString()));
            string actual = string.Join(" ", frontEnd.G2P(normalized));
            if (expected != actual) failures.Add($"{text}:\n expected {expected}\n actual   {actual}");
        }
        Assert.True(failures.Count == 0, string.Join(Environment.NewLine, failures));
    }
}
