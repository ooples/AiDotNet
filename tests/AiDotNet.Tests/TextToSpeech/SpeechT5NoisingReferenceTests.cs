using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AiDotNet.TextToSpeech.MultiModal;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// SpeechT5's pre-training noise against the official code (tools/reference-data/speecht5_noising_reference.py):
/// <c>TextPretrainDataset.add_whole_word_mask</c> (BART's text infilling, with its insertions) and fairseq's
/// <c>compute_mask_indices</c>, each replaying the draws the official code made.
/// </summary>
public class SpeechT5NoisingReferenceTests
{
    private static JObject Fixture()
    {
        const string fileName = "speecht5_noising_reference.json";
        string path = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", fileName);
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); !File.Exists(path) && dir is not null; dir = dir.Parent)
            path = Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", fileName);
        return JObject.Parse(File.ReadAllText(path));
    }

    /// <summary>The recorded draws, handed back in order; each call must be the one the official code made.</summary>
    private sealed class Replay : ISpeechT5Draws
    {
        private readonly Queue<JArray> _draws;

        public Replay(JToken draws) => _draws = new Queue<JArray>(draws.Select(d => (JArray)d));

        private JToken Next(string kind)
        {
            Assert.True(_draws.Count > 0, $"an extra {kind} draw");
            var draw = _draws.Dequeue();
            Assert.Equal(kind, (string)draw[0]!);
            return draw;
        }

        public int[] Categorical(double[] weights, int count)
        {
            var d = Next("categorical");
            Assert.Equal(count, (int)d[1]!);
            return d[2]!.Select(v => (int)v).ToArray();
        }

        public int[] Permutation(int n)
        {
            var d = Next("permutation");
            Assert.Equal(n, (int)d[1]!);
            return d[2]!.Select(v => (int)v).ToArray();
        }

        public double[] Uniforms(int count)
        {
            var d = Next("uniforms");
            Assert.Equal(count, (int)d[1]!);
            return d[2]!.Select(v => (double)v).ToArray();
        }

        public int[] Integers(int low, int high, int count)
        {
            var d = Next("integers");
            Assert.Equal(new[] { low, high }, d[1]!.Select(v => (int)v).ToArray());
            var values = d[2]!.Select(v => (int)v).ToArray();
            Assert.Equal(count, values.Length);
            return values;
        }

        public int[] Choice(int n, int count)
        {
            var d = Next("choice");
            Assert.Equal(new[] { n, count }, d[1]!.Select(v => (int)v).ToArray());
            return d[2]!.Select(v => (int)v).ToArray();
        }

        public void AssertDone() => Assert.Empty(_draws);
    }

    [Fact]
    public void TextInfilling_MatchesOfficialCode()
    {
        var cases = Fixture()["text"]!;
        Assert.Contains(cases, c => c["draws"]!.Any(d => (string)d[0]! == "permutation" && (int)d[1]! != c["source"]!.Count() - 2));
        foreach (var c in cases)
        {
            var replay = new Replay(c["draws"]!);
            var noised = SpeechT5Noising.TextInfilling(c["source"]!.Select(v => (int)v).ToArray(), 0.3, 0.1, 3.5, 79, 81, replay);
            Assert.Equal(c["noised"]!.Select(v => (int)v).ToArray(), noised);
            replay.AssertDone();
        }
    }

    [Fact]
    public void SpanMask_MatchesOfficialCode()
    {
        foreach (var c in Fixture()["spans"]!)
        {
            var replay = new Replay(c["draws"]!);
            var mask = SpeechT5Noising.SpanMask((int)c["frames"]!, 0.8, 10, 2, replay);
            Assert.Equal(c["mask"]!.Select(v => (bool)v).ToArray(), mask);
            replay.AssertDone();
        }
    }

    [Fact]
    public void PoissonWeights_MatchFairseq()
    {
        // fairseq stops at the first probability below 1e-7: k = 0 … 18 for λ = 3.5.
        var weights = SpeechT5Noising.PoissonWeights(3.5);
        Assert.Equal(19, weights.Length);
        Assert.Equal((double)(float)Math.Exp(-3.5), weights[0]);
    }
}
