using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Audio.Codecs;
using AiDotNet.Audio.Generation;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.Audio;

/// <summary>
/// SoundStream encodes, quantizes, decodes and trains as its paper specifies (Zeghidour et al. 2021).
/// </summary>
/// <remarks>
/// Before this change SoundStream was a stack of generic layers whose "residual vector quantization" bucketed each latent
/// value by magnitude; it had no codebooks, no quantizer dropout, no discriminators and trained by regression.
/// </remarks>
public class SoundStreamPaperTests
{
    private static NeuralNetworkArchitecture<double> Architecture()
        => new(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1) { RandomSeed = 13 };

    private static SoundStreamOptions TinyOptions() => new()
    {
        Filters = 4,
        Dimension = 8,
        Ratios = [4, 2],
        NumQuantizers = 2,
        CodebookSize = 16,
        TargetBandwidthKbps = 24.0,
        SegmentSize = 256,
        KMeansIterations = 5,
        MelScales = [5, 6],
        MelBins = 8,
        WaveDiscriminatorWidthDivisor = 16,
        StftWindow = 128,
        StftHop = 32,
        StftChannels = 2,
        SamplingSeed = 5,
    };

    private static Tensor<double> Noise(int samples, int seed)
    {
        var random = new Random(seed);
        var audio = new Tensor<double>(new[] { 1, 1, samples });
        for (int i = 0; i < samples; i++) audio[i] = random.NextDouble() - 0.5;
        return audio;
    }

    [Fact(Timeout = 300000)]
    public async Task Bitrates_FollowTheFrameRateAndCodebooks()
    {
        await Task.Yield();
        // Strides (2, 4, 5, 8) = 320 at 24 kHz: 75 frames per second; 10-bit codes give 0.75 kbps per quantizer, so 6 kbps is
        // N_q = 8 (§III-C) and the scalable model's 18 kbps is 24.
        using var model = new SoundStream<double>(Architecture(), new SoundStreamOptions
        {
            WaveDiscriminatorWidthDivisor = 64, StftChannels = 2,
        });
        Assert.Equal(320, model.HopLength);
        Assert.Equal(75, model.TokenFrameRate);
        Assert.Equal(8, model.QuantizersForBandwidth(6.0));
        Assert.Equal(24, model.QuantizersForBandwidth(18.0));
        Assert.Equal(24, model.NumQuantizers);
    }

    [Fact(Timeout = 300000)]
    public async Task Causal_OutputDoesNotDependOnFutureInput()
    {
        await Task.Yield();
        // §III-B: "all convolutions are causal ... padding is only applied to the past".
        using var model = new SoundStream<double>(Architecture(), TinyOptions());
        var audio = Noise(256, 1);
        var changed = new Tensor<double>(audio._shape, audio.ToVector());
        for (int i = 240; i < 256; i++) changed[i] += 0.7;
        var a = model.DecodeEmbeddings(model.EncodeEmbeddings(audio));
        var b = model.DecodeEmbeddings(model.EncodeEmbeddings(changed));
        for (int i = 0; i < 240; i++) Assert.Equal(a[i], b[i], 12);
        Assert.True(Enumerable.Range(240, 16).Any(i => Math.Abs(a[i] - b[i]) > 1e-9), "Changing the input's end did not change the output's end.");
    }

    [Fact(Timeout = 300000)]
    public async Task Film_StartsAsTheIdentity()
    {
        await Task.Yield();
        // Eq. 7 with (γ, β) = (1, 0) for both flags: an untrained codec codes the same whether or not denoising is asked for.
        using var model = new SoundStream<double>(Architecture(), TinyOptions());
        var audio = Noise(256, 2);
        model.Denoise = false;
        var plain = model.EncodeEmbeddings(audio);
        model.Denoise = true;
        var denoised = model.EncodeEmbeddings(audio);
        for (int i = 0; i < plain.Length; i++) Assert.Equal(plain[i], denoised[i], 12);
    }

    [Fact(Timeout = 60000)]
    public async Task StftDiscriminator_HasThePapersResidualUnits()
    {
        await Task.Yield();
        // Fig. 4: a 7x7 convolution and six residual units; the time axis halves at the three (2, 2) units and the F = W/2
        // frequency bins halve six times, so the final 1 x F/2^6 convolution leaves one logit per down-sampled frame.
        var layers = new List<LayerBase<double>>();
        var d = new SoundStreamStftDiscriminator<double>(AiDotNet.Tensors.Engines.AiDotNetEngine.Current, 1024, 256, 4, layers);
        var audio = new Tensor<double>(new[] { 1024 + 63 * 256 });               // 64 frames
        for (int i = 0; i < audio.Length; i++) audio[i] = Math.Sin(i * 0.05);
        var (logits, features) = d.Forward(audio);
        Assert.Equal(7, features.Count);
        Assert.Equal(new[] { 1, 1, 8, 1 }, logits.Shape.ToArray());           // 64 frames / 2^3 = 8
        Assert.Equal(16 * 4, features[^1].Shape[1]);
        Assert.Equal(8, features[^1].Shape[3]);
        Assert.Equal(1 + 6 * 3 + 1, layers.Count);
    }

    [Fact(Timeout = 600000)]
    public async Task Training_ReducesTheReconstructionLoss()
    {
        await Task.Yield();
        using var model = new SoundStream<double>(Architecture(), TinyOptions());
        var provider = (AiDotNet.Interfaces.ITrainingObjectiveProvider<double>)model;
        var audio = new Tensor<double>(new[] { 256 });
        for (int i = 0; i < audio.Length; i++) audio[i] = 0.5 * Math.Sin(2 * Math.PI * 1500 * i / 24000.0);
        double before = provider.EvaluateTrainingObjective(audio, audio);
        for (int i = 0; i < 30; i++) model.Train(audio, audio);
        double after = provider.EvaluateTrainingObjective(audio, audio);
        Assert.True(after < before, $"The reconstruction loss did not fall ({before} -> {after}).");
    }

    [Fact(Timeout = 600000)]
    public async Task Denoising_TrainsOnInputTargetTuples()
    {
        await Task.Yield();
        // §III-F: (inputs, targets, denoise) tuples - a noisy input with its clean target under denoise = true.
        using var model = new SoundStream<double>(Architecture(), TinyOptions());
        var clean = new Tensor<double>(new[] { 1, 1, 256 });
        for (int i = 0; i < 256; i++) clean[i] = 0.5 * Math.Sin(2 * Math.PI * 1500 * i / 24000.0);
        var noise = Noise(256, 9);
        var noisy = new Tensor<double>(clean._shape, clean.ToVector());
        for (int i = 0; i < 256; i++) noisy[i] += 0.2 * noise[i];
        var plain = model.EncodeEmbeddings(noisy);
        for (int i = 0; i < 10; i++) model.TrainDenoising(noisy, clean, denoise: true);
        model.Denoise = false;
        var off = model.EncodeEmbeddings(noisy);
        model.Denoise = true;
        var on = model.EncodeEmbeddings(noisy);
        Assert.True(Enumerable.Range(0, on.Length).Any(i => Math.Abs(on[i] - off[i]) > 1e-9),
            "Training with denoise = true left the denoising conditioning unused.");
        Assert.Equal(plain.Length, on.Length);
    }
}
