using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using AiDotNet.TextToSpeech.FlowDiffusion;
using AiDotNet.TextToSpeech.FrontEnd;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// MaskGCT's eSpeak-language front end against the release's (tools/reference-data/maskgct_g2p_reference.py, run in the
/// release's environment with eSpeak NG 1.49.2): each text's phone string and ids, with eSpeak's outputs replayed from the
/// recording, every call checked for its voice and text; and, where eSpeak NG 1.49.2 is installed, the native binding
/// against the same calls.
/// </summary>
public class MaskGctFrontEndTests
{
    private static JsonElement Load(string name)
    {
        string relative = Path.Combine("TextToSpeech", "ReferenceData", name);
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", relative))) dir = dir.Parent;
        string path = Path.Combine(dir?.FullName ?? throw new FileNotFoundException(name), "tests", "AiDotNet.Tests", relative);
        return JsonDocument.Parse(File.ReadAllText(path)).RootElement;
    }

    [Fact(Timeout = 120000)]
    public async Task G2P_MatchesTheRelease_WithEspeakReplayed()
    {
        await Task.Yield();
        var reference = Load("maskgct_g2p_reference.json");
        foreach (var c in reference.GetProperty("cases").EnumerateArray())
        {
            var calls = new Queue<JsonElement>(c.GetProperty("calls").EnumerateArray());
            var frontEnd = new MaskGctEspeakFrontEnd((language, text) =>
            {
                Assert.True(calls.Count > 0, $"an unexpected eSpeak call: {language} \"{text}\"");
                var call = calls.Dequeue();
                Assert.Equal(call.GetProperty("voice").GetString(), language);
                Assert.Equal(call.GetProperty("text").GetString(), text);
                return call.GetProperty("phonemes").GetString() ?? "";
            });
            string input = c.GetProperty("text").GetString()!;
            var (phonemes, tokens) = frontEnd.G2P(input, c.GetProperty("language").GetString()!);
            Assert.Empty(calls);
            Assert.Equal(c.GetProperty("phonemes").GetString(), phonemes);
            Assert.Equal(c.GetProperty("tokens").EnumerateArray().Select(t => t.GetInt32()).ToArray(), tokens);
        }
    }

    [Fact(Timeout = 120000)]
    public async Task Vocabulary_IsTheRelease()
    {
        await Task.Yield();
        Assert.Equal(363, MaskGctEspeakFrontEnd.Vocabulary.Count);
        Assert.Equal(new[] { MaskGctEspeakFrontEnd.Vocabulary["h"], MaskGctEspeakFrontEnd.Vocabulary["_"] }, MaskGctEspeakFrontEnd.Tokens("h|unknown|_"));
    }

    [Fact(Timeout = 120000)]
    public async Task InstalledEspeak_MatchesTheRecording_WhenItIsVersion1492()
    {
        await Task.Yield();
        // eSpeak NG is GPL and not shipped: this checks the binding only on machines with eSpeak NG 1.49.2 installed.
        if (!EspeakNg.IsAvailable || EspeakNg.Instance.Version != "1.49.2") return;
        var reference = Load("maskgct_g2p_reference.json");
        foreach (var call in reference.GetProperty("cases").EnumerateArray().SelectMany(c => c.GetProperty("calls").EnumerateArray()))
            Assert.Equal(call.GetProperty("phonemes").GetString(),
                EspeakNg.Instance.TextToPhonemes(call.GetProperty("voice").GetString()!, call.GetProperty("text").GetString()!));
    }
}
