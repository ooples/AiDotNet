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
/// speechbrain's ECAPA-TDNN extractor against speechbrain 0.5.16 (tools/reference-data/speechbrain_ecapa_reference.py): the
/// Fbank features with the sentence mean removed and a tiny ECAPA_TDNN's embedding; with <c>SPKREC_ECAPA_DIR</c> set
/// (speechbrain/spkrec-ecapa-voxceleb), the released model's embedding.
/// </summary>
public class SpeechBrainEcapaReferenceTests
{
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
        var f = WhisperSpeechT2SReferenceTests.Load("speechbrain_ecapa_reference.json");
        var engine = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;
        var c = f["config"]!;
        int[] Ints(string k) => c[k]!.Select(v => (int)v).ToArray();
        var tiny = new SpeechBrainEcapa<double>(engine, new List<LayerBase<double>>(), new SpeechBrainEcapaConfiguration
        {
            Mels = (int)c["input_size"]!, Channels = Ints("channels"), Kernels = Ints("kernel_sizes"), Dilations = Ints("dilations"),
            AttentionChannels = (int)c["attention_channels"]!, Res2NetScale = (int)c["res2net_scale"]!, SeChannels = (int)c["se_channels"]!,
            EmbeddingSize = (int)c["lin_neurons"]!,
        }, new Random(1));
        var state = (JObject)f["state"]!;
        var buffers = (JObject)f["buffers"]!;
        var used = new HashSet<string>(StringComparer.Ordinal);
        tiny.Load((name, _) =>
        {
            used.Add(name);
            return Doubles((state[name] ?? buffers[name])!);
        });
        Assert.Empty(state.Properties().Select(p => p.Name).Where(n => !used.Contains(n)));
        var wave = Doubles(f["wave"]!);
        Close(Doubles(f["features"]!), tiny.Features(wave).Cast<double>().ToArray(), "features", 1e-9);
        using var _ = new NoGradScope<double>();
        Close(Doubles(f["embedding"]!), tiny.Embed(wave).ToArray(), "embedding", 1e-9);

        string? dir = Environment.GetEnvironmentVariable("SPKREC_ECAPA_DIR");
        string checkpoint = dir is null ? "" : Path.Combine(dir, "embedding_model.ckpt");
        if (!File.Exists(checkpoint)) return;
        var released = new SpeechBrainEcapa<double>(engine, new List<LayerBase<double>>(), new SpeechBrainEcapaConfiguration(), new Random(1));
        var weights = new AiDotNet.ComputerVision.Weights.WeightLoader().LoadWeights(checkpoint);
        released.Load((name, _) => weights[name].ToVector().Select(v => (double)v).ToArray());
        Close(Doubles(f["released_embedding"]!), released.Embed(wave).ToArray(), "released embedding", 1e-6);
    }
}
