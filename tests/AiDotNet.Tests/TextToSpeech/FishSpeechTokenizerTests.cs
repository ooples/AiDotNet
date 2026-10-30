using System;
using System.IO;
using System.Linq;
using AiDotNet.TextToSpeech.CodecBased;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// Fish Speech's tokenizers against the releases' (tools/reference-data/fishspeech_tokenizer_reference.py), where the
/// tokenizer files are present: <c>FISHSPEECH15_DIR</c> (fishaudio/fish-speech-1.5, <c>tokenizer.tiktoken</c>) and
/// <c>FISHSPEECH14_DIR</c> (fishaudio/fish-speech-1.4, <c>tokenizer.json</c>).
/// </summary>
public class FishSpeechTokenizerTests
{
    private static JObject Fixture()
    {
        const string fileName = "fishspeech_tokenizer_reference.json";
        string path = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", fileName);
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); !File.Exists(path) && dir is not null; dir = dir.Parent)
            path = Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", fileName);
        return JObject.Parse(File.ReadAllText(path));
    }

    [Fact]
    public void Tokenizer15_MatchesTheRelease_WhenPresent()
    {
        string? dir = Environment.GetEnvironmentVariable("FISHSPEECH15_DIR");
        if (dir is null || !File.Exists(Path.Combine(dir, "tokenizer.tiktoken"))) return;
        var f = Fixture();
        var tokenizer = FishSpeechText.Load15(Path.Combine(dir, "tokenizer.tiktoken"));
        Assert.Equal((int)f["semantic_begin15"]!, tokenizer.SpecialTokenId("<|semantic:0|>"));
        Assert.Equal((int)f["im_end15"]!, tokenizer.SpecialTokenId("<|im_end|>"));
        var texts = f["texts"]!.Select(t => (string)t!).ToArray();
        var expected = (JArray)f["ids15"]!;
        for (int i = 0; i < texts.Length; i++)
            Assert.Equal(expected[i].Select(v => (int)v).ToArray(), tokenizer.Encode(texts[i]));
    }

    [Fact]
    public void Tokenizer14_MatchesTheRelease_WhenPresent()
    {
        string? dir = Environment.GetEnvironmentVariable("FISHSPEECH14_DIR");
        if (dir is null || !File.Exists(Path.Combine(dir, "tokenizer.json"))) return;
        var f = Fixture();
        var tokenizer = FishSpeechText.Load14(Path.Combine(dir, "tokenizer.json"));
        Assert.Equal((int)f["semantic14"]!, tokenizer.TokenId("<|semantic|>"));
        var texts = f["texts"]!.Select(t => (string)t!).ToArray();
        var expected = (JArray)f["ids14"]!;
        for (int i = 0; i < texts.Length; i++)
            Assert.Equal(expected[i].Select(v => (int)v).ToArray(), tokenizer.Encode(texts[i]));
    }
}
