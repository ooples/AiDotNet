using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Tokenization.Algorithms;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>Step-1's released SentencePiece unigram tokenizer (digits split, byte fallback, the audio tokens as user-defined
/// pieces) against sentencepiece (<c>tools/reference-data/stepaudio_tokenizer_reference.py</c>).</summary>
public class StepAudioTokenizerReferenceTests
{
    [Fact(Timeout = 60000)]
    public async Task Step1Tokenizer_MatchesSentencePiece()
    {
        await Task.Yield();
        const string name = "stepaudio_tokenizer_reference.json";
        string path = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", name);
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); !File.Exists(path) && dir is not null; dir = dir.Parent)
            path = Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", name);
        var f = JObject.Parse(File.ReadAllText(path));
        var model = new SentencePieceModel(Convert.FromBase64String((string)f["model_base64"]!));
        Assert.Equal(74752, model.Count);
        foreach (var c in f["cases"]!)
        {
            string text = (string)c["text"]!;
            var ids = model.Encode(text);
            Assert.Equal(c["ids"]!.Select(v => (int)v).ToArray(), ids);
            Assert.Equal((string)c["decoded"]!, model.Decode(ids));
        }
    }
}
