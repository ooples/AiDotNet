using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.TextToSpeech.Classic;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// Tacotron 2's spectrogram prediction network trains and synthesizes as its paper specifies (Shen et al. 2018, §2.2):
/// a convolutional + bidirectional-LSTM encoder, location-sensitive attention, a pre-net and two zoneout LSTMs per
/// frame, a stop token, a residual post-net, and the summed MSE before and after the post-net plus the stop loss.
/// </summary>
/// <remarks>
/// Before this change the "decoder LSTMs" were dense layers with no recurrent state, the location features were the
/// previous weights truncated to 32 values instead of a convolution over them, and the decoder emitted 2 frames per step
/// although the paper uses none.
/// </remarks>
public class Tacotron2PaperTests
{
    private const int MelBins = 8;

    private static Tacotron2<double> CreateModel(int seed = 0) => new(
        new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
            inputSize: 8, outputSize: MelBins) { RandomSeed = 11 },
        new Tacotron2Options
        {
            VocabSize = 32, EmbeddingDim = 16, EncoderDim = 16, AttentionRnnDim = 16, DecoderRnnDim = 16, AttentionDimension = 8,
            AttentionLocationChannels = 4, AttentionKernelSize = 5, PrenetDim = 8, PostnetDim = 8, NumEncoderLayers = 3,
            PostnetLayers = 5, MelChannels = MelBins, OutputsPerStep = 1, MaxMelLength = 12, ConvolutionDropout = 0.0,
            ZoneoutProbability = 0.0, PrenetDropout = 0.5, SamplingSeed = seed,
            // Never stop early: the first step reads the all-zero GO frame, on which the pre-net's dropout has no effect.
            StopThreshold = 1.0,
        },
        new AiDotNet.Optimizers.AdamOptimizer<double, Tensor<double>, Tensor<double>>(null));

    private static (Tensor<double> Tokens, Tensor<double> Mel) Example()
    {
        var tokens = new Tensor<double>(new[] { 1, 5 });
        for (int i = 0; i < 5; i++) tokens[0, i] = 3 + i * 5;
        var mel = new Tensor<double>(new[] { 1, 10, MelBins });
        for (int f = 0; f < 10; f++)
            for (int c = 0; c < MelBins; c++) mel[0, f, c] = Math.Sin(0.4 * f + 0.3 * c);
        return (tokens, mel);
    }

    [Fact(Timeout = 180000)]
    public async Task Training_ReducesTheMelAndStopObjective()
    {
        await Task.Yield();
        var model = CreateModel();
        var (tokens, mel) = Example();
        model.Train(tokens, mel);
        double first = Convert.ToDouble(model.GetLastLoss());
        for (int i = 0; i < 40; i++) model.Train(tokens, mel);
        double last = Convert.ToDouble(model.GetLastLoss());
        Assert.True(last < first, $"Objective did not fall ({first} -> {last}).");
    }

    [Fact(Timeout = 120000)]
    public async Task Synthesis_DecodesOneFramePerStep_AndThePrenetDropoutVariesWithTheSeed()
    {
        await Task.Yield();
        var (tokens, _) = Example();
        var model = CreateModel(seed: 1);
        var first = model.Predict(tokens);
        Assert.Equal(3, first.Rank);
        Assert.Equal(MelBins, first.Shape[2]);
        Assert.Equal(12, first.Shape[1]);
        Assert.Equal(first.ToVector().ToArray(), model.Predict(tokens).ToVector().ToArray());

        // The pre-net's dropout stays on at inference (§2.2), so another sampling seed gives another output.
        var other = CreateModel(seed: 2);
        other.SetParameters(model.GetParameters());
        var second = other.Predict(tokens);
        int shared = Math.Min(first.Shape[1], second.Shape[1]);
        Assert.True(Enumerable.Range(0, shared * MelBins).Any(i => Math.Abs(first[i] - second[i]) > 1e-12));
    }

    [Fact(Timeout = 60000)]
    public async Task LocationSensitiveAttention_IsADistribution_AndReadsThePreviousWeights()
    {
        await Task.Yield();
        var attention = new LocationSensitiveAttentionLayer<double>(queryDim: 4, memoryDim: 3, attentionDim: 5, filters: 2, kernelSize: 3);
        var memory = new Tensor<double>(new[] { 6, 3 });
        for (int i = 0; i < memory.Length; i++) memory[i] = Math.Sin(i);
        var query = new Tensor<double>(new[] { 1, 4 });
        for (int i = 0; i < 4; i++) query[0, i] = 0.1 * i;
        var projected = attention.ProjectMemory(memory);
        var zero = new Tensor<double>(new[] { 1, 6 });
        var (_, weights) = attention.Attend(query, memory, projected, zero, zero);
        Assert.Equal(1.0, Enumerable.Range(0, 6).Sum(j => weights[0, j]), 9);

        var focused = new Tensor<double>(new[] { 1, 6 });
        focused[0, 2] = 1.0;
        var (_, moved) = attention.Attend(query, memory, projected, focused, focused);
        Assert.True(Enumerable.Range(0, 6).Any(j => Math.Abs(moved[0, j] - weights[0, j]) > 1e-12),
            "The location features had no effect on the attention weights.");
    }
}
