using System;
using System.IO;
using System.Linq;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.Tokenization.Algorithms;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// CosyVoice's text front end against FunAudioLLM/CosyVoice's <c>text_normalize</c> without an external normalizer
/// (<c>tools/reference-data/cosyvoice_frontend_reference.py</c>): the sentences it splits each text into, and the text it
/// normalizes unsplit.
/// </summary>
public class CosyVoiceFrontEndReferenceTests
{
    private static JObject Reference()
    {
        const string fileName = "cosyvoice_frontend_reference.json";
        string output = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", fileName);
        if (!File.Exists(output))
        {
            var dir = new DirectoryInfo(AppContext.BaseDirectory);
            while (dir is not null && !File.Exists(output))
            {
                output = Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", fileName);
                dir = dir.Parent;
            }
        }
        return JObject.Parse(File.ReadAllText(output));
    }

    [Fact]
    public void TextNormalize_MatchesCosyVoice()
    {
        var encoding = WhisperTiktoken.Multilingual(100);
        Func<string, int> tokens = s => encoding.Encode(s).Length;
        foreach (var c in Reference()["cases"]!)
        {
            string text = (string)c["text"]!;
            if (c["error"] is not null)
            {
                Assert.ThrowsAny<Exception>(() => CosyVoiceTextFrontEnd.Sentences(text, tokens));
                continue;
            }
            var expected = c["sentences"]!.Select(s => (string)s!).ToArray();
            var actual = CosyVoiceTextFrontEnd.Sentences(text, tokens).ToArray();
            Assert.True(expected.SequenceEqual(actual),
                $"'{text}': CosyVoice [{string.Join(" | ", expected)}], port [{string.Join(" | ", actual)}].");
            Assert.Equal((string)c["unsplit"]!, CosyVoiceTextFrontEnd.Normalize(text));
        }
    }
}
