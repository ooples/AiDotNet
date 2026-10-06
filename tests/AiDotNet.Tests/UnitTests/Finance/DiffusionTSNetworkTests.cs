using System;
using AiDotNet.Finance.Probabilistic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Finance;

/// <summary>
/// Diffusion-TS's seasonal block keeps, per channel, the int(ln K) strongest of the K retained frequencies and rebuilds
/// the signal from them. Checked against a known sum of sinusoids.
/// </summary>
public class DiffusionTSNetworkTests
{
    private const int Window = 32;
    private const int Width = 2;

    private static double Wave(int frequency, int t) => Math.Sin(2 * Math.PI * frequency * t / Window);

    [Fact]
    public void SeasonalBlock_KeepsTheStrongestFrequencies_AndDropsTheRest()
    {
        var network = new DiffusionTSNetwork<double>(1, Window, Width, 1, 1, 1, 1, 0.0, 1);
        // Window 32 retains frequencies 1..15 (DC and Nyquist excluded), and int(ln 15) = 2 of them are kept.
        Assert.Equal(2, network.FourierTopK);

        var input = new Tensor<double>(new[] { 1, Window, Width });
        var expected = new double[Window * Width];
        for (int t = 0; t < Window; t++)
            for (int c = 0; c < Width; c++)
            {
                input[t * Width + c] = 3.0 * Wave(3, t) + 2.0 * Wave(7, t) + 0.5 * Wave(11, t);
                expected[t * Width + c] = 3.0 * Wave(3, t) + 2.0 * Wave(7, t);
            }

        var seasonal = network.SeasonalComponent(AiDotNetEngine.Current, input);

        Assert.Equal(new[] { 1, Window, Width }, seasonal.Shape.ToArray());
        for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i], seasonal[i], 9);
    }

    private static Tensor<double> Window12(int seed)
    {
        var noisy = new Tensor<double>(new[] { 1, 12, 2 });
        for (int i = 0; i < noisy.Length; i++) noisy[i] = Math.Sin(0.37 * i + seed);
        return noisy;
    }

    private static Tensor<double> Step(double k)
    {
        var steps = new Tensor<double>(new[] { 1, 1 });
        steps[0] = k;
        return steps;
    }

    /// <summary>Attention mixes time steps, never rows: a window is denoised the same alone or within a batch.</summary>
    [Fact]
    public void Denoiser_PredictsEachRowIndependentlyOfTheBatch()
    {
        var engine = AiDotNetEngine.Current;
        var network = new DiffusionTSNetwork<double>(2, 12, 8, 2, 1, 1, 2, 0.0, 1);
        var rows = new[] { Window12(0), Window12(1), Window12(2) };
        var batch = engine.TensorConcatenate(rows, axis: 0);
        var steps = new Tensor<double>(new[] { 3, 1 });
        for (int i = 0; i < 3; i++) steps[i] = 2 * i + 1;

        var together = network.PredictCleanWindow(engine, batch, steps);

        for (int r = 0; r < 3; r++)
        {
            var alone = network.PredictCleanWindow(engine, rows[r], Step(2 * r + 1));
            for (int i = 0; i < alone.Length; i++) Assert.Equal(alone[i], together[r * alone.Length + i], 10);
        }
    }

    /// <summary>The diffusion step reaches the prediction through AdaLN: one window at two noise levels gives two answers.</summary>
    [Fact]
    public void Denoiser_IsConditionedOnTheDiffusionStep()
    {
        var engine = AiDotNetEngine.Current;
        var network = new DiffusionTSNetwork<double>(2, 12, 8, 2, 1, 1, 2, 0.0, 1);
        var window = Window12(3);

        var early = network.PredictCleanWindow(engine, window, Step(1));
        var late = network.PredictCleanWindow(engine, window, Step(9));

        double difference = 0;
        for (int i = 0; i < early.Length; i++) difference = Math.Max(difference, Math.Abs(early[i] - late[i]));
        Assert.True(difference > 1e-6, $"The prediction ignored the diffusion step (max difference {difference:G3}).");
    }
}