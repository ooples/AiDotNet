using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AiDotNet.Audio.Speaker;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Autodiff;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// speechbrain's x-vector extractor against speechbrain 0.5.16 (tools/reference-data/speechbrain_xvector_reference.py): the
/// Fbank features with the sentence mean removed and a tiny Xvector's embedding; with <c>SPKREC_XVECT_DIR</c> set
/// (speechbrain/spkrec-xvect-voxceleb), the released model's embedding.
/// </summary>
public class SpeechBrainXVectorReferenceTests
{
    private static JObject Fixture()
    {
        const string fileName = "speechbrain_xvector_reference.json";
        string path = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", fileName);
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); !File.Exists(path) && dir is not null; dir = dir.Parent)
            path = Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", fileName);
        return JObject.Parse(File.ReadAllText(path));
    }

    private static double[] Doubles(JToken t) => t.Select(v => (double)v).ToArray();

    private static void Close(double[] expected, IReadOnlyList<double> actual, string what, double tolerance)
    {
        Assert.Equal(expected.Length, actual.Count);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= tolerance * Math.Max(1, Math.Abs(expected[i])), $"{what}[{i}]: {expected[i]} vs {actual[i]}");
    }

    [Fact]
    public void Embedding_MatchesSpeechBrain()
    {
        var f = Fixture();
        var engine = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;
        var tiny = new SpeechBrainXVector<double>(engine, new List<LayerBase<double>>(),
            new SpeechBrainXVectorConfiguration { Channels = new[] { 8, 8, 8, 8, 12 }, EmbeddingSize = 6 }, new Random(1));
        var state = (JObject)f["state"]!;
        var buffers = (JObject)f["buffers"]!;
        tiny.Load((name, _) => Doubles((state[name] ?? buffers[name])!));
        var wave = Doubles(f["wave"]!);
        var features = tiny.Features(wave);
        Close(Doubles(f["features"]!), features.Cast<double>().ToArray(), "features", 1e-9);
        using var _ = new NoGradScope<double>();
        Close(Doubles(f["embedding"]!), tiny.Embed(wave, null, Doubles(f["noise"]!)).ToArray(), "embedding", 1e-9);

        string? dir = Environment.GetEnvironmentVariable("SPKREC_XVECT_DIR");
        string checkpoint = dir is null ? "" : Path.Combine(dir, "embedding_model.ckpt");
        if (!File.Exists(checkpoint)) return;
        var released = new SpeechBrainXVector<double>(engine, new List<LayerBase<double>>(), new SpeechBrainXVectorConfiguration(), new Random(1));
        var weights = new AiDotNet.ComputerVision.Weights.WeightLoader().LoadWeights(checkpoint);
        released.Load((name, _) => weights[name].ToVector().Select(v => (double)v).ToArray());
        Close(Doubles(f["released_embedding"]!), released.Embed(wave, null, Doubles(f["released_noise"]!)).ToArray(), "released embedding", 1e-6);
    }
}
