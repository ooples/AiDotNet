using System;
using System.IO;
using AiDotNet.TextToSpeech.FrontEnd;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// WeTextProcessing's Chinese and English normalizers against the package (tools/reference-data/wetext_reference.py):
/// the tagger's token string and the normalized text for texts covering the grammars' classes.
/// </summary>
public class WeTextNormalizerReferenceTests
{
    [Theory]
    [InlineData("chinese")]
    [InlineData("english")]
    public void TagAndNormalize_MatchWeTextProcessing(string language)
    {
        const string fileName = "wetext_reference.json";
        string path = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", fileName);
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); !File.Exists(path) && dir is not null; dir = dir.Parent)
            path = Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", fileName);
        var f = JObject.Parse(File.ReadAllText(path));
        var normalizer = language == "chinese" ? WeTextNormalizer.Chinese : WeTextNormalizer.English;
        foreach (var c in f[language]!)
        {
            string text = (string)c["text"]!;
            if (c["error"] is not null)
            {
                Assert.Throws<InvalidOperationException>(() => normalizer.Normalize(text));
                continue;
            }
            string tagged = normalizer.Tag(text);
            Assert.Equal((string)c["tagged"]!, tagged);
            Assert.Equal((string)c["normalized"]!, normalizer.Verbalize(tagged));
        }
    }
}
