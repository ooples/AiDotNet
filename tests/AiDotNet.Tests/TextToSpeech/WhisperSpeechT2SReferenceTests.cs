using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.TextToSpeech.MultiModal;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>WhisperSpeech's text-to-semantic model against the release's TSARTransformer
/// (<c>tools/reference-data/whisperspeech_t2s_reference.py</c>): the training loss, the eval logits and greedy
/// generation with the KV caches.</summary>
public class WhisperSpeechT2SReferenceTests
{
    internal static JObject Load(string name)
    {
        string path = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", name);
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); !File.Exists(path) && dir is not null; dir = dir.Parent)
            path = Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", name);
        return JObject.Parse(File.ReadAllText(path));
    }

    internal static double[] Doubles(JToken t) => t.Select(v => (double)v).ToArray();

    [Fact(Timeout = 120000)]
    public async Task T2S_MatchesTheRelease()
    {
        await Task.Yield();
        var f = Load("whisperspeech_t2s_reference.json");
        var cfg = f["config"]!;
        var state = ((JObject)f["state"]!).Properties().ToDictionary(p => p.Name, p => Doubles(p.Value), StringComparer.Ordinal);
        var model = new WhisperSpeechT2S<double>(new CpuEngine(), new List<LayerBase<double>>(), new WhisperSpeechT2SConfiguration
        {
            Depth = (int)cfg["depth"]!, Heads = (int)cfg["n_head"]!, HeadWidth = (int)cfg["head_width"]!, FfnMult = (int)cfg["ffn_mult"]!,
            TextLength = (int)cfg["ttoks_len"]!, TextCodes = (int)cfg["ttoks_codes"]!, SemanticLength = (int)cfg["stoks_len"]!,
            SemanticCodes = (int)cfg["stoks_codes"]!, SemanticWidth = (int)cfg["stoks_width"]!,
        }, new Random(0));
        var used = new HashSet<string>(StringComparer.Ordinal);
        model.LoadTorchWeights(n =>
        {
            used.Add(n);
            return state.TryGetValue(n, out var v) ? v : null;
        });
        Assert.Empty(state.Keys.Where(k => !used.Contains(k) && !k.EndsWith("positional_embedding", StringComparison.Ordinal)));
        string text = (string)f["text"]!;
        var stoks = f["stoks"]!.Select(v => (int)v).ToArray();
        double loss = model.Loss(text, stoks, 0, 14.0)[0];
        Assert.True(Math.Abs(loss - (double)f["loss"]!) <= 1e-10 * Math.Max(1, Math.Abs(loss)), $"loss {(double)f["loss"]!:R} vs {loss:R}");

        var c = model.Configuration;
        var (inText, _) = model.TextIds(text);
        var input = Enumerable.Repeat(c.PaddingToken, c.SemanticLength).ToArray();
        for (int i = 0; i < stoks.Length; i++) input[i + 1] = stoks[i];
        var logits = model.Decode(input, Enumerable.Range(0, c.SemanticLength).ToArray(), model.Encode(inText, 0), model.Cps(14.0)).ToArray();
        var expected = Doubles(f["logits"]!);
        Assert.Equal(expected.Length, logits.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - logits[i]) <= 1e-10 * Math.Max(1, Math.Abs(expected[i])), $"logits[{i}]: {expected[i]:R} vs {logits[i]:R}");

        var generated = model.Generate(text, 0, 14.0, new[] { 3, 17 }, 20, 1e-5, null, new Random(0));
        Assert.Equal(f["generated"]!.Select(v => (int)v).ToArray(), generated);
    }
}
