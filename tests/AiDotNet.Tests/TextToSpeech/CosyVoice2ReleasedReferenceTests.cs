using System.Text.Json;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.TextToSpeech.VoiceCloning;
using AiDotNet.Tokenization.Algorithms;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public sealed class CosyVoice2ReleasedReferenceTests
{
    [Fact]
    public void ReleasedBlankEnTokenizer_AllTextAndSpecialTokenIdsMatchOriginal()
    {
        using var document = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory,
            "TextToSpeech", "ReferenceData", "cosyvoice2_tokenizer_reference.json")));
        var fixture = document.RootElement; var tokenizer = QwenTokenizer.CosyVoice2;
        Assert.Equal(fixture.GetProperty("tokenizer_vocabulary").GetInt32(), tokenizer.VocabularySize);
        foreach (var item in fixture.GetProperty("tokenizer_cases").EnumerateArray())
            Assert.Equal(item.GetProperty("ids").EnumerateArray().Select(x => x.GetInt32()),
                tokenizer.Encode(item.GetProperty("text").GetString()!));
        foreach (var item in fixture.GetProperty("added_tokens").EnumerateObject())
            Assert.Equal(item.Value.GetInt32(), tokenizer.AddedTokenId(item.Name));
    }

    [Fact]
    public void ReleasedFrontendPaths_CopyThroughBothCloningAndCosyVoice3Options()
    {
        var clone = new CosyVoiceCloneOptions { SpeechTokenizerPath = "v2.onnx", SpeakerEncoderOnnxPath = "cam.onnx", MaxReferenceSeconds = 17 };
        var cloned = new CosyVoiceCloneOptions(clone);
        Assert.Equal("v2.onnx", cloned.SpeechTokenizerPath); Assert.Equal("cam.onnx", cloned.SpeakerEncoderOnnxPath);
        Assert.Equal(17, cloned.MaxReferenceSeconds);
        var three = new CosyVoice3Options { SpeechTokenizerPath = "v3.onnx", SpeakerEncoderOnnxPath = "cam3.onnx", MaxReferenceSeconds = 23 };
        var copied = new CosyVoice3Options(three);
        Assert.Equal("v3.onnx", copied.SpeechTokenizerPath); Assert.Equal("cam3.onnx", copied.SpeakerEncoderOnnxPath);
        Assert.Equal(23, copied.MaxReferenceSeconds);
    }
}
