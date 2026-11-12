using System;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using AiDotNet.TextToSpeech.FrontEnd;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// mecab-ko with mecab-ko-dic against python-mecab-ko's <c>MeCab().pos</c> (tools/reference-data/mecab_ko_reference.py):
/// words and part-of-speech tags of Korean, mixed, spaced and emoji text.
/// </summary>
public class MeCabKoTests
{
    [Fact(Timeout = 300000)]
    public async Task Pos_MatchesPythonMecabKo()
    {
        await Task.Yield();
        const string name = "mecab_ko_reference.json";
        string relative = Path.Combine("TextToSpeech", "ReferenceData", name);
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", relative))) dir = dir.Parent;
        string path = Path.Combine(dir?.FullName ?? throw new FileNotFoundException(name), "tests", "AiDotNet.Tests", relative);
        foreach (var entry in JsonDocument.Parse(File.ReadAllText(path)).RootElement.GetProperty("corpus").EnumerateArray())
        {
            string text = entry.GetProperty("text").GetString()!;
            string expected = string.Join(" ", entry.GetProperty("pos").EnumerateArray().Select(p => $"{p[0].GetString()}/{p[1].GetString()}"));
            string actual = string.Join(" ", MeCab.KoDic.Parse(text).Select(p => $"{p.Surface}/{p.Tag}"));
            Assert.Equal(expected, actual);
        }
    }
}
