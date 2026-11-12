using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Audio.Whisper;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.TextToSpeech.MultiModal;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>WhisperSpeech's semantic tokenizer against the release (<c>tools/reference-data/whisperspeech_vq_reference.py</c>):
/// a mel's codes, the dequantized states of codes, and one training step's loss with the codebook's moving averages after
/// it.</summary>
public class WhisperSpeechVqReferenceTests
{
    [Fact(Timeout = 300000)]
    public async Task Vq_MatchesTheRelease()
    {
        await Task.Yield();
        var f = WhisperSpeechT2SReferenceTests.Load("whisperspeech_vq_reference.json");
        var wd = f["whisper"]!;
        int I(string k) => (int)wd[k]!;
        var dimensions = new WhisperDimensions(I("n_mels"), I("n_audio_ctx"), I("n_audio_state"), I("n_audio_head"), I("n_audio_layer"), I("n_vocab"),
            I("n_text_ctx"), I("n_text_state"), I("n_text_head"), I("n_text_layer"));
        var engine = new CpuEngine();
        var layers = new List<LayerBase<double>>();
        var whisper = new OpenAiWhisperNetwork<double>(engine, layers, dimensions, new Random(0));
        var teacher = ((JObject)f["teacher"]!).Properties().ToDictionary(p => p.Name, p => WhisperSpeechT2SReferenceTests.Doubles(p.Value), StringComparer.Ordinal);
        whisper.LoadTorchWeights((name, shape) =>
        {
            var v = teacher[name];
            Assert.Equal(shape.Aggregate(1, (a, s) => a * s), v.Length);
            return v;
        });
        var vq = f["vq"]!;
        var model = new WhisperSpeechVq<double>(engine, layers, new WhisperSpeechVqConfiguration
        {
            Codes = (int)vq["vq_codes"]!, QuantizerDepth = (int)vq["q_depth"]!, Depth = (int)vq["depth"]!, Heads = (int)vq["n_head"]!,
            HeadWidth = (int)vq["head_width"]!, FfnMult = (int)vq["ffn_mult"]!, CodebookWidth = (int)vq["codebook_dim"]!,
            Downsample = (int)vq["downsample"]!, DeadCodeThreshold = (double)vq["threshold_ema_dead_code"]!,
        }, whisper, new Random(0));
        var state = ((JObject)f["state"]!).Properties().ToDictionary(p => p.Name, p => WhisperSpeechT2SReferenceTests.Doubles(p.Value), StringComparer.Ordinal);
        var used = new HashSet<string>(StringComparer.Ordinal);
        model.LoadTorchWeights(n =>
        {
            used.Add(n);
            return state.TryGetValue(n, out var v) ? v : null;
        });
        // The validation counters, the "initted" flag (no k-means) and the rotary frequency tables (10000^(−2i/d)) are not weights.
        Assert.Empty(state.Keys.Where(k => !used.Contains(k) && !k.StartsWith("val_", StringComparison.Ordinal)
            && !k.EndsWith("initted", StringComparison.Ordinal) && !k.EndsWith("rotary.inv_freq", StringComparison.Ordinal)));

        var melValues = WhisperSpeechT2SReferenceTests.Doubles(f["mel"]!);
        var mel = new Tensor<double>(new[] { dimensions.Mels, melValues.Length / dimensions.Mels }, new Vector<double>(melValues));
        var codes = model.Quantize(whisper.Encode(mel))[0];
        Assert.Equal(f["stoks"]!.Select(v => (int)v).ToArray(), codes);

        int melFrames = (int)f["mel_frames"]!;
        var dequantized = model.Dequantize(codes.Take(melFrames / 2 / 2).ToArray()).ToArray();
        AssertClose(WhisperSpeechT2SReferenceTests.Doubles(f["dequantized"]!), dequantized, "dequantized");

        int maskFrames = (int)f["mask_frames"]!;
        var mask = Enumerable.Range(0, dimensions.AudioContext).Select(i => i < maskFrames).ToArray();
        double loss = model.Loss(mel, mask, f["input_toks"]!.Select(v => (int)v).ToArray(), f["output_toks"]!.Select(v => (int)v).ToArray())[0];
        Assert.True(Math.Abs(loss - (double)f["loss"]!) <= 1e-10 * Math.Max(1, Math.Abs(loss)), $"loss {(double)f["loss"]!:R} vs {loss:R}");
        var after = f["after"]!;
        AssertClose(WhisperSpeechT2SReferenceTests.Doubles(after["embed"]!), model.Embed[0].Weights.ToArray(), "embed");
        AssertClose(WhisperSpeechT2SReferenceTests.Doubles(after["embed_avg"]!), model.EmbedAverage[0].Weights.ToArray(), "embed_avg");
        AssertClose(WhisperSpeechT2SReferenceTests.Doubles(after["cluster_size"]!), model.ClusterSize[0].Weights.ToArray(), "cluster_size");
    }

    private static void AssertClose(double[] expected, double[] actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-10 * Math.Max(1, Math.Abs(expected[i])), $"{what}[{i}]: {expected[i]:R} vs {actual[i]:R}");
    }
}
