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
/// MaskGCT's Japanese front end against the release's (tools/reference-data/maskgct_japanese_reference.py, in the
/// release's environment): Open JTalk's labels from the recorded words with no dictionary needed; and, where the Open
/// JTalk dictionary is present (<c>OPENJTALK_DIC</c>, a directory holding open_jtalk_dic_utf_8-1.11), pyopenjtalk's
/// <c>run_frontend</c> word for word.
/// </summary>
public class MaskGctJapaneseTests
{
    private static JsonElement Load(string name)
    {
        string relative = Path.Combine("TextToSpeech", "ReferenceData", name);
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", relative))) dir = dir.Parent;
        string path = Path.Combine(dir?.FullName ?? throw new FileNotFoundException(name), "tests", "AiDotNet.Tests", relative);
        return JsonDocument.Parse(File.ReadAllText(path)).RootElement;
    }

    private static NjdFeature Feature(JsonElement f) => new()
    {
        String = f.GetProperty("string").GetString()!, Pos = f.GetProperty("pos").GetString()!,
        PosGroup1 = f.GetProperty("pos_group1").GetString()!, PosGroup2 = f.GetProperty("pos_group2").GetString()!,
        PosGroup3 = f.GetProperty("pos_group3").GetString()!, CType = f.GetProperty("ctype").GetString()!,
        CForm = f.GetProperty("cform").GetString()!, Orig = f.GetProperty("orig").GetString()!,
        Read = f.GetProperty("read").GetString()!, Pron = f.GetProperty("pron").GetString()!,
        Acc = f.GetProperty("acc").GetInt32(), MoraSize = f.GetProperty("mora_size").GetInt32(),
        ChainRule = f.GetProperty("chain_rule").GetString()!, ChainFlag = f.GetProperty("chain_flag").GetInt32(),
    };

    private static string Describe(NjdFeature f) =>
        string.Join(",", f.String, f.Pos, f.PosGroup1, f.PosGroup2, f.PosGroup3, f.CType, f.CForm, f.Orig, f.Read, f.Pron,
            f.Acc, f.MoraSize, f.ChainRule, f.ChainFlag);

    [Fact(Timeout = 60000)]
    public async Task MakeLabel_MatchesPyOpenJTalk()
    {
        await Task.Yield();
        foreach (var c in Load("maskgct_japanese_reference.json").GetProperty("cases").EnumerateArray())
        {
            var features = c.GetProperty("features").EnumerateArray().Select(Feature).ToList();
            var expected = c.GetProperty("labels").EnumerateArray().Select(l => l.GetString()!).ToList();
            Assert.Equal(expected, OpenJTalk.MakeLabel(features));
        }
    }

    [Fact(Timeout = 300000)]
    public async Task RunFrontend_MatchesPyOpenJTalk_WhenTheDictionaryIsPresent()
    {
        await Task.Yield();
        string? dic = Environment.GetEnvironmentVariable("OPENJTALK_DIC");
        if (dic is null || !File.Exists(Path.Combine(dic, "sys.dic"))) return;
        var openJTalk = new OpenJTalk(dic);
        foreach (var c in Load("maskgct_japanese_reference.json").GetProperty("cases").EnumerateArray())
        {
            string text = c.GetProperty("text").GetString()!;
            var expected = c.GetProperty("features").EnumerateArray().Select(f => Describe(Feature(f))).ToList();
            var actual = openJTalk.RunFrontend(text).Select(Describe).ToList();
            Assert.True(expected.SequenceEqual(actual), $"{text}\nexpected:\n{string.Join("\n", expected)}\nactual:\n{string.Join("\n", actual)}");
        }
    }
}
