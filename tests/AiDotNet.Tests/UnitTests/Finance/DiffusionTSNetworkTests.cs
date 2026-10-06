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

    [Fact]
    public void Denoiser_PredictsAWindowOfTheInputShape()
    {
        var network = new DiffusionTSNetwork<double>(2, 12, 8, 2, 1, 1, 2, 0.0, 1);
        var noisy = new Tensor<double>(new[] { 3, 12, 2 });
        for (int i = 0; i < noisy.Length; i++) noisy[i] = Math.Sin(0.37 * i);
        var steps = new Tensor<double>(new[] { 3, 1 });
        for (int i = 0; i < 3; i++) steps[i] = i;

        var clean = network.PredictCleanWindow(AiDotNetEngine.Current, noisy, steps);

        Assert.Equal(new[] { 3, 12, 2 }, clean.Shape.ToArray());
        for (int i = 0; i < clean.Length; i++) Assert.False(double.IsNaN(clean[i]) || double.IsInfinity(clean[i]));
    }
}
