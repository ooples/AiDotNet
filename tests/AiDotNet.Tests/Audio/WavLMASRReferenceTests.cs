using System;
using System.IO;
using System.Linq;
using System.Text.Json;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.SpeechRecognition.Foundation;
using AiDotNet.SpeechRecognition.Robust;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.Audio;

/// <summary>
/// WavLM (Chen et al. 2022) fine-tuned for CTC: WavLM-ASR in the base-plus form and WavLM-Robust in the Large form must
/// match transformers' <c>WavLMForCTC</c> (tools/reference-data/wavlm_reference.py) (#2321).
/// </summary>
public class WavLMASRReferenceTests
{
    private static JsonElement Fixture()
    {
        const string name = "wavlm_reference.json";
        string relative = Path.Combine("Audio", "ReferenceData", name);
        string path = Path.Combine(AppContext.BaseDirectory, relative);
        if (!File.Exists(path))
        {
            var dir = new DirectoryInfo(AppContext.BaseDirectory);
            while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", relative))) dir = dir.Parent;
            path = Path.Combine(dir?.FullName ?? throw new FileNotFoundException(name), "tests", "AiDotNet.Tests", relative);
        }
        return JsonDocument.Parse(File.ReadAllText(path)).RootElement;
    }

    // The reference script's fill: 0.05 sin(0.7 (i + 1) + 0.013 s), s the sum of the name's character codes, with norm
    // gains and the weight-norm magnitude offset by 1.
    private static double[] Fill(string name, int[] shape)
    {
        int count = shape.Aggregate(1, (a, b) => a * b);
        int s = name.Sum(c => (int)c);
        double offset = name.EndsWith("norm.weight", StringComparison.Ordinal) || name.EndsWith("original0", StringComparison.Ordinal) ? 1.0 : 0.0;
        var values = new double[count];
        for (int i = 0; i < count; i++) values[i] = offset + 0.05 * Math.Sin(0.7 * (i + 1) + 0.013 * s);
        return values;
    }

    // Current torch saves weight norm under parametrizations.weight.original0/1.
    private static bool Has(string name) => !name.EndsWith("weight_g", StringComparison.Ordinal) && !name.EndsWith("weight_v", StringComparison.Ordinal);

    private static NeuralNetworkArchitecture<double> Architecture(int vocab)
        => new(InputType.OneDimensional, NeuralNetworkTaskType.SequenceToSequence, inputSize: 1600, outputSize: vocab);

    private static void AssertMatches(JsonElement k, Func<Tensor<double>, Tensor<double>> predict)
    {
        double[] audio = k.GetProperty("audio").EnumerateArray().Select(v => v.GetDouble()).ToArray();
        double[] expected = k.GetProperty("logits").EnumerateArray().Select(v => v.GetDouble()).ToArray();
        int frames = k.GetProperty("frames").GetInt32(), vocab = k.GetProperty("vocab").GetInt32();
        for (int b = 0; b < 2; b++)
        {
            var logits = predict(new Tensor<double>(new[] { 1600 }, new Vector<double>(audio.Skip(b * 1600).Take(1600).ToArray())));
            Assert.Equal(new[] { frames, vocab }, logits.Shape.ToArray());
            for (int i = 0; i < frames * vocab; i++)
            {
                double want = expected[b * frames * vocab + i];
                Assert.True(Math.Abs(want - logits[i]) < 1e-9, $"sample {b} logit {i}: {want} vs {logits[i]}");
            }
        }
    }

    [Fact]
    public void WavLMASR_MatchesWavLMForCtc_InTheBasePlusForm()
    {
        var k = Fixture().GetProperty("cases")[0];
        int vocab = k.GetProperty("vocab").GetInt32();
        var model = new WavLMASR<double>(Architecture(vocab), new WavLMASROptions
        {
            EncoderDim = 16, NumEncoderLayers = 2, NumAttentionHeads = 2, FeedForwardDim = 32, PositionalConvGroups = 16,
            VocabSize = vocab, DropoutRate = 0, ActivationDropout = 0, FeatureProjectionDropout = 0, FinalDropout = 0,
            LayerDrop = 0, MaskTimeProbability = 0,
        });
        model.LoadTorchWeights(Fill, Has);
        AssertMatches(k, model.Predict);
    }

    [Fact]
    public void WavLMRobust_MatchesWavLMForCtc_InTheLargeForm()
    {
        var k = Fixture().GetProperty("cases")[1];
        int vocab = k.GetProperty("vocab").GetInt32();
        var model = new WavLMRobust<double>(Architecture(vocab), new WavLMRobustOptions
        {
            EncoderDim = 16, NumEncoderLayers = 2, NumAttentionHeads = 2, FeedForwardDim = 32, PositionalConvGroups = 16,
            VocabSize = vocab, DropoutRate = 0, ActivationDropout = 0, FeatureProjectionDropout = 0, FinalDropout = 0,
            LayerDrop = 0, MaskTimeProbability = 0,
        });
        model.LoadTorchWeights(Fill, Has);
        AssertMatches(k, model.Predict);
    }

    [Fact]
    public void Defaults_AreTheReleasedWavLMConfigurations()
    {
        var basePlus = new WavLMASROptions();
        Assert.Equal((768, 12, 12, 3072, false), (basePlus.EncoderDim, basePlus.NumEncoderLayers, basePlus.NumAttentionHeads,
            basePlus.FeedForwardDim, basePlus.ToConfiguration().StableLayerNorm));
        var large = new WavLMRobustOptions();
        Assert.Equal((1024, 24, 16, 4096, true), (large.EncoderDim, large.NumEncoderLayers, large.NumAttentionHeads,
            large.FeedForwardDim, large.ToConfiguration().StableLayerNorm));
    }
}
