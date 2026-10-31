using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.MultiModal;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// GLM-4's decoder against THUDM's modeling_chatglm.py (tools/reference-data/glm_decoder_reference.py): a tiny
/// ChatGLMForConditionalGeneration's logits at 1e-9, and token-by-token cached decoding equal to the full forward.
/// </summary>
public class GlmDecoderReferenceTests
{
    private static JObject Fixture()
    {
        const string fileName = "glm_decoder_reference.json";
        string path = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", fileName);
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); !File.Exists(path) && dir is not null; dir = dir.Parent)
            path = Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", fileName);
        return JObject.Parse(File.ReadAllText(path));
    }

    private static double[] Doubles(JToken t) => t.Select(v => (double)v).ToArray();

    internal static GlmConfiguration Tiny() => new()
    {
        VocabSize = 50, Hidden = 32, Layers = 2, Heads = 4, KeyValueGroups = 2, HeadDim = 8, FeedForward = 40, NormEpsilon = 1e-5,
    };

    private static GlmDecoder<double> Load(JObject f, HashSet<string>? used = null)
    {
        var decoder = new GlmDecoder<double>(AiDotNet.Tensors.Engines.AiDotNetEngine.Current, new List<LayerBase<double>>(), Tiny(), new Random(1));
        var state = (JObject)f["state"]!;
        decoder.Load((name, _) =>
        {
            used?.Add(name);
            return Doubles(state[name] ?? throw new InvalidDataException(name));
        });
        return decoder;
    }

    private static void Close(double[] expected, Tensor<double> actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual.GetFlat(i)) <= 1e-9 * Math.Max(1, Math.Abs(expected[i])), $"{what}[{i}]: {expected[i]} vs {actual.GetFlat(i)}");
    }

    [Fact]
    public void Logits_MatchModelingChatGlm()
    {
        var f = Fixture();
        var used = new HashSet<string>();
        var decoder = Load(f, used);
        Assert.All(((JObject)f["state"]!).Properties(), p => Assert.Contains(p.Name, used));
        var ids = f["ids"]!.Select(v => (int)v).ToArray();
        using var _ = new NoGradScope<double>();
        Close(Doubles(f["logits"]!), decoder.Logits(decoder.Forward(decoder.Embed(ids))), "logits");
    }

    [Fact]
    public void CachedDecoding_MatchesTheFullForward()
    {
        var f = Fixture();
        var decoder = Load(f);
        var ids = f["ids"]!.Select(v => (int)v).ToArray();
        var expected = Doubles(f["logits"]!);
        using var _ = new NoGradScope<double>();
        var cache = decoder.NewCache();
        Close(expected.Take(4 * 50).ToArray(), decoder.Logits(decoder.Forward(decoder.Embed(ids.Take(4).ToArray()), cache)), "prefix");
        for (int t = 4; t < ids.Length; t++)
            Close(expected.Skip(t * 50).Take(50).ToArray(), decoder.Logits(decoder.Forward(decoder.Embed(new[] { ids[t] }), cache)), $"step {t}");
    }
}
