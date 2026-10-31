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
/// AudioPaLM's decoders against their references (tools/reference-data/palm_decoder_reference.py): PaLM as
/// lucidrains/PaLM-pytorch builds it (with PaLM §5's 1/√d logit scale) and Gemma as transformers builds it; and the
/// key/value cache, whose token-by-token logits must equal one full forward's.
/// </summary>
public class PalmDecoderReferenceTests
{
    private static JObject Fixture()
    {
        const string fileName = "palm_decoder_reference.json";
        string path = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", fileName);
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); !File.Exists(path) && dir is not null; dir = dir.Parent)
            path = Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", fileName);
        return JObject.Parse(File.ReadAllText(path));
    }

    private static double[] Doubles(JToken t) => t.Select(v => (double)v).ToArray();

    private static void Close(double[] expected, Tensor<double> actual, string what, double tolerance = 1e-9)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual.GetFlat(i)) <= tolerance * Math.Max(1, Math.Abs(expected[i])), $"{what}[{i}]: {expected[i]} vs {actual.GetFlat(i)}");
    }

    internal static PalmDecoder<double> Load(PalmDecoderKind kind, JObject f, HashSet<string>? used = null)
    {
        var config = kind == PalmDecoderKind.Palm
            ? new PalmConfiguration { Kind = kind, VocabSize = 50, Hidden = 32, Layers = 2, Heads = 4, KeyValueHeads = 1, HeadDim = 8, Intermediate = 128, NormEpsilon = 1e-5 }
            : new PalmConfiguration { Kind = kind, VocabSize = 50, Hidden = 32, Layers = 2, Heads = 4, KeyValueHeads = 1, HeadDim = 8, Intermediate = 64, NormEpsilon = 1e-6 };
        var decoder = new PalmDecoder<double>(AiDotNet.Tensors.Engines.AiDotNetEngine.Current, new List<LayerBase<double>>(), config, new Random(2));
        var state = (JObject)f[kind == PalmDecoderKind.Palm ? "palm_state" : "gemma_state"]!;
        double[] Read(string name, int[] _)
        {
            used?.Add(name);
            return Doubles(state[name] ?? throw new InvalidDataException(name));
        }
        if (kind == PalmDecoderKind.Palm) decoder.LoadPalmPytorch(Read);
        else decoder.LoadGemma(Read);
        return decoder;
    }

    [Theory]
    [InlineData(PalmDecoderKind.Palm)]
    [InlineData(PalmDecoderKind.Gemma)]
    public void Logits_MatchReference(PalmDecoderKind kind)
    {
        var f = Fixture();
        var used = new HashSet<string>();
        var decoder = Load(kind, f, used);
        string prefix = kind == PalmDecoderKind.Palm ? "palm" : "gemma";
        Assert.All(((JObject)f[prefix + "_state"]!).Properties(), p => Assert.Contains(p.Name, used));
        var ids = f[prefix + "_ids"]!.Select(v => (int)v).ToArray();
        using var _ = new NoGradScope<double>();
        Close(Doubles(f[prefix + "_logits"]!), decoder.Logits(decoder.Forward(decoder.Embed(ids))), prefix + " logits");
    }

    [Theory]
    [InlineData(PalmDecoderKind.Palm)]
    [InlineData(PalmDecoderKind.Gemma)]
    public void CachedDecoding_MatchesTheFullForward(PalmDecoderKind kind)
    {
        var f = Fixture();
        var decoder = Load(kind, f);
        string prefix = kind == PalmDecoderKind.Palm ? "palm" : "gemma";
        var ids = f[prefix + "_ids"]!.Select(v => (int)v).ToArray();
        var expected = Doubles(f[prefix + "_logits"]!);
        using var _ = new NoGradScope<double>();
        var cache = decoder.NewCache();
        // A three-token prefix, then one token at a time.
        var first = decoder.Logits(decoder.Forward(decoder.Embed(ids.Take(3).ToArray()), cache));
        Close(expected.Take(3 * 50).ToArray(), first, prefix + " prefix");
        for (int t = 3; t < ids.Length; t++)
        {
            var step = decoder.Logits(decoder.Forward(decoder.Embed(new[] { ids[t] }), cache));
            Close(expected.Skip(t * 50).Take(50).ToArray(), step, $"{prefix} step {t}");
        }
    }
}
