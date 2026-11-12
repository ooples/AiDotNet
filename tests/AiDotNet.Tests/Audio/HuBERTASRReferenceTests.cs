using System;
using System.IO;
using System.Linq;
using System.Text.Json;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.SpeechRecognition.Foundation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.Audio;

/// <summary>
/// HuBERT (Hsu et al. 2021) fine-tuned for CTC is the wav2vec 2.0 encoder with a CTC head: in the BASE form its logits
/// must match transformers' <c>HubertForCTC</c> (tools/reference-data/hubert_reference.py) (#2321).
/// </summary>
public class HuBERTASRReferenceTests
{
    private static JsonElement Fixture()
    {
        const string name = "hubert_reference.json";
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

    [Fact]
    public void Logits_MatchHubertForCtc_InTheBaseForm()
    {
        var fixture = Fixture();
        double[] audio = fixture.GetProperty("audio").EnumerateArray().Select(v => v.GetDouble()).ToArray();
        double[] expected = fixture.GetProperty("logits").EnumerateArray().Select(v => v.GetDouble()).ToArray();
        int frames = fixture.GetProperty("frames").GetInt32(), vocab = fixture.GetProperty("vocab").GetInt32();

        var model = new HuBERTASR<double>(
            new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.SequenceToSequence,
                inputSize: 1600, outputSize: vocab),
            new HuBERTASROptions
            {
                EncoderDim = 16, NumEncoderLayers = 2, NumAttentionHeads = 2, FeedForwardDim = 32, PositionalConvGroups = 16,
                VocabSize = vocab, DropoutRate = 0, ActivationDropout = 0, FeatureProjectionDropout = 0, FinalDropout = 0,
                LayerDrop = 0, MaskTimeProbability = 0,
            });
        model.LoadTorchWeights(Fill);

        for (int b = 0; b < 2; b++)
        {
            var waveform = new Tensor<double>(new[] { 1600 }, new Vector<double>(audio.Skip(b * 1600).Take(1600).ToArray()));
            var logits = model.Predict(waveform);
            Assert.Equal(new[] { frames, vocab }, logits.Shape.ToArray());
            for (int i = 0; i < frames * vocab; i++)
            {
                double want = expected[b * frames * vocab + i];
                Assert.True(Math.Abs(want - logits[i]) < 1e-9, $"sample {b} logit {i}: {want} vs {logits[i]}");
            }
        }
    }

    [Fact]
    public void Defaults_AreHubertBase()
    {
        var options = new HuBERTASROptions();
        Assert.Equal((768, 12, 12, 3072), (options.EncoderDim, options.NumEncoderLayers, options.NumAttentionHeads, options.FeedForwardDim));
        var configuration = options.ToConfiguration();
        Assert.False(configuration.StableLayerNorm || configuration.LayerNormFeatureEncoder || configuration.ConvBias);
        Assert.True(new HuBERTASROptions { StableLayerNorm = true }.ToConfiguration().StableLayerNorm);
    }
}
