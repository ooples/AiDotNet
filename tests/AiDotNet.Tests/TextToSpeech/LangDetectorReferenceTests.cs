using System;
using System.IO;
using System.Linq;
using AiDotNet.TextToSpeech.FrontEnd;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// langdetect 1.0.9 against the package itself (tools/reference-data/langdetect_reference.py): the language order, the
/// n-grams extracted from texts in many scripts, and the probabilities with two seeds.
/// </summary>
public class LangDetectorReferenceTests
{
    [Fact]
    public void NgramsAndProbabilities_MatchLangdetect()
    {
        const string fileName = "langdetect_reference.json";
        string path = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", fileName);
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); !File.Exists(path) && dir is not null; dir = dir.Parent)
            path = Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", fileName);
        var f = JObject.Parse(File.ReadAllText(path));
        Assert.Equal(f["languages"]!.Select(v => (string)v!), LangDetector.Languages);
        foreach (var c in f["cases"]!)
        {
            string text = (string)c["text"]!;
            var detector = new LangDetector();
            Assert.Equal(c["ngrams"]!.Select(v => (string)v!), detector.ExtractNgrams(text));
            foreach (int seed in new[] { 0, 7 })
            {
                detector.Seed = seed;
                var expected = c[$"probabilities_{seed}"]!;
                if (expected.Type == JTokenType.String)
                {
                    Assert.Throws<InvalidOperationException>(() => detector.Probabilities(text));
                    continue;
                }
                var actual = detector.Probabilities(text);
                Assert.Equal(expected.Select(p => (string)p[0]!), actual.Select(p => p.Language));
                foreach (var (e, a) in expected.Zip(actual, (e, a) => ((double)e[1]!, a.Probability)))
                    Assert.True(Math.Abs(e - a) <= 1e-10, $"{text} (seed {seed}): {e} vs {a}");
                if (actual.Count > 0) Assert.Equal(actual[0].Language, detector.Detect(text));
            }
        }
    }
}
