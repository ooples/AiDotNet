using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.TextToSpeech.MultiModal;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>WhisperSpeech's semantic-to-acoustic model and its conditioned variant against the release
/// (<c>tools/reference-data/whisperspeech_s2a_reference.py</c>): the training loss, the eval logits and greedy delayed
/// generation with the KV caches.</summary>
public class WhisperSpeechS2AReferenceTests
{
    [Theory(Timeout = 120000)]
    [InlineData("plain")]
    [InlineData("cond")]
    public async Task S2A_MatchesTheRelease(string variant)
    {
        await Task.Yield();
        var f = WhisperSpeechT2SReferenceTests.Load("whisperspeech_s2a_reference.json");
        var cfg = f["config"]!;
        var state = ((JObject)f[variant + "_state"]!).Properties().ToDictionary(p => p.Name, p => WhisperSpeechT2SReferenceTests.Doubles(p.Value), StringComparer.Ordinal);
        bool conditioned = variant == "cond";
        var model = new WhisperSpeechS2A<double>(new CpuEngine(), new List<LayerBase<double>>(), new WhisperSpeechS2AConfiguration
        {
            Depth = (int)cfg["depth"]!, Context = (int)cfg["ctx_n"]!, SemanticLength = (int)cfg["stoks_len"]!, SemanticCodes = (int)cfg["stoks_codes"]!,
            SemanticWidth = (int)cfg["stoks_width"]!, SpeakerWidth = (int)cfg["spk_width"]!, Heads = (int)cfg["n_head"]!, HeadWidth = (int)cfg["head_width"]!,
            FfnMult = (int)cfg["ffn_mult"]!, Quantizers = (int)cfg["quantizers"]!, Conditioned = conditioned,
        }, new Random(0));
        var used = new HashSet<string>(StringComparer.Ordinal);
        model.LoadTorchWeights(n =>
        {
            used.Add(n);
            return state.TryGetValue(n, out var v) ? v : null;
        });
        // The shared special rows appear under every quantizer; the validation counters and rotary frequency tables are
        // not parameters (the frequencies are 10000^(−2i/d)).
        Assert.Empty(state.Keys.Where(k => !used.Contains(k) && !k.EndsWith(".special.weight", StringComparison.Ordinal)
            && !k.StartsWith("val_", StringComparison.Ordinal) && !k.EndsWith("rotary.inv_freq", StringComparison.Ordinal)));
        var data = f[variant]!;
        var stoks = data["stoks"]!.Select(v => (int)v).ToArray();
        var flat = data["atoks"]!.Select(v => (int)v).ToArray();
        int q = model.Configuration.Quantizers, n = flat.Length / q;
        var atoks = Enumerable.Range(0, q).Select(i => flat.Skip(i * n).Take(n).ToArray()).ToArray();
        var speakerValues = WhisperSpeechT2SReferenceTests.Doubles(data["speaker"]!);
        var speaker = new Tensor<double>(new[] { speakerValues.Length }, new Vector<double>(speakerValues));
        var conditions = conditioned ? new WhisperSpeechConditions(null, 25.0, 33.0) : null;
        double loss = model.Loss(model.SemanticInput(stoks), atoks, speaker, conditions)[0];
        Assert.True(Math.Abs(loss - (double)data["loss"]!) <= 1e-10 * Math.Max(1, Math.Abs(loss)), $"loss {(double)data["loss"]!:R} vs {loss:R}");
        var logits = model.Decode(atoks, Enumerable.Range(0, n).ToArray(), model.Encode(model.SemanticInput(stoks), speaker, conditions)).ToArray();
        var expected = WhisperSpeechT2SReferenceTests.Doubles(data["logits"]!);
        Assert.Equal(expected.Length, logits.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - logits[i]) <= 1e-10 * Math.Max(1, Math.Abs(expected[i])), $"logits[{i}]: {expected[i]:R} vs {logits[i]:R}");
        var generated = model.Generate(stoks, speaker, 18, 1e-5, null, new Random(0), conditioned ? new WhisperSpeechConditions(null, 60, 60) : null);
        Assert.Equal(data["generated"]!.Select(v => (int)v).ToArray(), generated.SelectMany(r => r).ToArray());
    }
}
