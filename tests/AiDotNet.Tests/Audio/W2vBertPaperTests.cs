using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Audio.Foundations;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.Audio;

/// <summary>
/// w2v-BERT (Chung et al. 2021): the XL configuration of Table 1, the feature encoder's 4× subsampling to 25 frames a
/// second, the span masking of §4.2, the joint contrastive and masked-prediction pre-training of §3.2, and AudioLM's
/// semantic tokens. Google released no code, so these check the paper directly.
/// </summary>
public class W2vBertPaperTests
{
    private static W2vBertOptions Tiny() => new()
    {
        ModelDim = 16, NumHeads = 2, FeedForwardMultiplier = 2, ContrastiveLayers = 1, MaskedLayers = 1,
        CodebookSize = 64, CodeDim = 16, Negatives = 4, MaskProbability = 0.3, MaskLength = 2, SemanticLayer = 1,
        SemanticClusters = 4, WarmupSteps = 1, LearningRate = 1e-3, Seed = 5,
    };

    private static W2vBert<double> Create(W2vBertOptions? options = null) =>
        new(new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.SequenceToSequence,
            inputSize: 8000, outputSize: 16), options ?? Tiny());

    private static Tensor<double> Voice(int samples, double pitch)
    {
        var audio = new Tensor<double>(new[] { samples });
        for (int i = 0; i < samples; i++)
        {
            double t = i / 16000.0;
            audio[i] = 0.3 * Math.Sin(2 * Math.PI * pitch * t) + 0.15 * Math.Sin(2 * Math.PI * 3 * pitch * t) * Math.Sin(2 * Math.PI * 4 * t);
        }
        return audio;
    }

    [Fact(Timeout = 60000)]
    public async Task Defaults_AreW2vBertXl()
    {
        await Task.Yield();
        var o = new W2vBertOptions();
        // Table 1: 12 contrastive and 12 masked-prediction Conformer blocks, width 1024, 8 heads, kernel 5, a codebook
        // of 1024 entries 1024 wide; §4.1: 80-bin log-mel; §4.2: mask p = 0.065 over 10 steps, peak lr 2e-3, 25k warm-up.
        Assert.Equal((12, 12, 1024, 8, 5), (o.ContrastiveLayers, o.MaskedLayers, o.ModelDim, o.NumHeads, o.ConvKernel));
        Assert.Equal((1024, 1024, 80), (o.CodebookSize, o.CodeDim, o.MelBins));
        Assert.Equal((0.065, 10, 2e-3, 25_000), (o.MaskProbability, o.MaskLength, o.LearningRate, o.WarmupSteps));
        Assert.Equal(30, W2vBertOptions.Xxl().MaskedLayers);
    }

    [Fact(Timeout = 60000)]
    public async Task FeatureEncoder_SubsamplesTenMillisecondFramesFourTimes()
    {
        await Task.Yield();
        using var model = Create();
        var network = model.Network ?? throw new InvalidOperationException("native model");
        var mel = network.Features(Voice(16000, 150));
        Assert.Equal(80, mel.Shape[2]);
        Assert.InRange(mel.Shape[1], 98, 102);                                           // 100 frames a second
        var encoded = network.Encode(mel);
        Assert.Equal(network.EncodedLength(mel.Shape[1]), encoded.Shape[1]);
        Assert.InRange(encoded.Shape[1], 23, 25);                                        // 25 frames a second (AudioLM's rate)
    }

    [Fact(Timeout = 60000)]
    public async Task Masking_StartsSpansOfTheConfiguredLength()
    {
        await Task.Yield();
        using var model = Create(new W2vBertOptions { ModelDim = 16, NumHeads = 2, ContrastiveLayers = 1, MaskedLayers = 1, MaskLength = 10 });
        var network = model.Network ?? throw new InvalidOperationException("native model");
        var mask = network.DrawMask(1, 400, 0.065, AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(3));
        // Every maximal run before the last 10 steps is at least one span long, and the masked share is near
        // 1 − (1 − p)^10 ≈ 0.49 for overlapping spans.
        int run = 0, masked = 0;
        for (int t = 0; t < 400; t++)
        {
            if (mask[0, t]) { run++; masked++; continue; }
            if (run > 0) Assert.True(run >= 10, $"A masked run of {run} steps ended at {t}.");
            run = 0;
        }
        Assert.InRange(masked / 400.0, 0.3, 0.7);
    }

    [Fact(Timeout = 120000)]
    public async Task PretrainingStep_IsFiniteAndMovesBothModules()
    {
        await Task.Yield();
        using var model = Create();
        var network = model.Network ?? throw new InvalidOperationException("native model");
        var contrastive = network.ContrastiveBlocks[0].GetParameters().ToArray();
        var masked = network.MaskedBlocks[0].GetParameters().ToArray();
        var audio = Voice(8000, 180);
        model.Train(audio, audio);
        var (c, d, m) = model.LastPretrainingLosses;
        Assert.True(double.IsFinite(c) && double.IsFinite(d) && double.IsFinite(m), $"Losses {c}, {d}, {m}.");
        Assert.True(m > 0, "The masked-prediction loss must cover the masked steps.");
        Assert.NotEqual(contrastive, network.ContrastiveBlocks[0].GetParameters().ToArray());
        Assert.NotEqual(masked, network.MaskedBlocks[0].GetParameters().ToArray());
    }

    [Fact(Timeout = 120000)]
    public async Task SemanticTokens_AreOnePerEncoderFrame_FromTheFittedClusters()
    {
        await Task.Yield();
        using var model = Create();
        var recordings = new[] { Voice(8000, 120), Voice(8000, 220), Voice(8000, 320) };
        model.FitSemanticTokenizer(recordings);
        var network = model.Network ?? throw new InvalidOperationException("native model");
        var tokens = model.SemanticTokens(recordings[1]);
        Assert.Equal(network.EncodedLength(network.Features(recordings[1]).Shape[1]), tokens.Length);
        Assert.All(tokens, token => Assert.InRange(token, 0, 3));
        Assert.Equal(tokens, model.SemanticTokens(recordings[1]));
    }
}
