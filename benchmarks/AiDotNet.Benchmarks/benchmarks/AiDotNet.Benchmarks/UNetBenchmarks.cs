using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Diffusion.NoisePredictors;

namespace AiDotNet.Benchmarks;

/// <summary>
/// Benchmarks the full UNet forward pass at various scales.
/// SD15 paper dimensions: base=320, [1,2,4,4], input=[1,4,64,64].
/// </summary>
[SimpleJob(RuntimeMoniker.Net10_0)]
[MemoryDiagnoser]
[RankColumn]
public class UNetBenchmarks
{
    // Small UNet (test scale)
    private UNetNoisePredictor<double> _smallUnet = null!;
    private Tensor<double> _smallInput = null!;

    // Medium UNet (practical scale)
    private UNetNoisePredictor<double> _mediumUnet = null!;
    private Tensor<double> _mediumInput = null!;

    [GlobalSetup]
    public void Setup()
    {
        // Small: 64 base channels, [1,2,4], 8x8 spatial
        _smallUnet = new UNetNoisePredictor<double>(
            inputChannels: 4, outputChannels: 4,
            baseChannels: 64, channelMultipliers: [1, 2, 4],
            numResBlocks: 1, attentionResolutions: [1, 2],
            contextDim: 0, numHeads: 4, inputHeight: 8, seed: 42);
        _smallInput = CreateRandom(1, 4, 8, 8);

        // Medium: 128 base channels, [1,2,2,2], 16x16 spatial (DDPM CIFAR-10 scale)
        _mediumUnet = new UNetNoisePredictor<double>(
            inputChannels: 4, outputChannels: 4,
            baseChannels: 128, channelMultipliers: [1, 2, 2, 2],
            numResBlocks: 1, attentionResolutions: [1],
            contextDim: 0, numHeads: 4, inputHeight: 16, seed: 42);
        _mediumInput = CreateRandom(1, 4, 16, 16);
    }

    [Benchmark(Description = "UNet Small (64ch, 8x8) single forward")]
    public Tensor<double> UNet_Small_Forward()
        => _smallUnet.PredictNoise(_smallInput, 500, null);

    [Benchmark(Description = "UNet Medium (128ch, 16x16) single forward")]
    public Tensor<double> UNet_Medium_Forward()
        => _mediumUnet.PredictNoise(_mediumInput, 500, null);

    [Benchmark(Description = "UNet Small construction")]
    public UNetNoisePredictor<double> UNet_Small_Construction()
        => new(inputChannels: 4, outputChannels: 4,
            baseChannels: 64, channelMultipliers: [1, 2, 4],
            numResBlocks: 1, attentionResolutions: [1, 2],
            contextDim: 0, numHeads: 4, inputHeight: 8, seed: 42);

    private static Tensor<double> CreateRandom(int b, int c, int h, int w)
    {
        var rng = new Random(42);
        var tensor = new Tensor<double>([b, c, h, w]);
        for (int i = 0; i < tensor.Length; i++)
            tensor[i] = rng.NextDouble() - 0.5;
        return tensor;
    }
}
