using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.Vocoders;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public class InverseStftAllocationTests
{
#if NET6_0_OR_GREATER // GC.GetTotalAllocatedBytes does not exist on .NET Framework (the compat build's net471)
    [Fact]
    public void WhisperSpeechTransformDoesNotAllocateQuadraticTables()
    {
        var engine = AiDotNetEngine.Current;
        _ = new InverseStft<double>(engine, 8, 2, 8);
        long before = GC.GetTotalAllocatedBytes(true);
        var transform = new InverseStft<double>(engine, 10240, 2560, 10240, samePadding: true);
        long allocated = GC.GetTotalAllocatedBytes(true) - before;
        GC.KeepAlive(transform);
        Assert.True(allocated < 1024 * 1024, $"10,240-point construction allocated {allocated:N0} bytes.");
    }
#endif

    [Theory]
    [InlineData(15, 3, false)]
    [InlineData(15, 3, true)]
    [InlineData(1280, 320, true)]
    [InlineData(10240, 2560, true)]
    public void SparseSpectrumMatchesAnalyticInverseAndOverlapNormalization(int fft, int hop, bool same)
    {
        const int frames = 3;
        int bin = Math.Min(13, fft / 2 - 1);
        var magnitude = new Tensor<double>(new[] { 1, fft / 2 + 1, frames });
        var phase = new Tensor<double>(magnitude.Shape.ToArray());
        for (int f = 0; f < frames; f++)
        {
            magnitude[0, 0, f] = .2;
            magnitude[0, bin, f] = .7 + .1 * f;
            phase[0, bin, f] = .3 + .1 * f;
        }
        var actual = new InverseStft<double>(AiDotNetEngine.Current, fft, hop, fft, same).Forward(magnitude, phase);
        int length = (frames - 1) * hop + fft;
        var expected = new double[length];
        var envelope = new double[length];
        for (int f = 0; f < frames; f++)
            for (int n = 0; n < fft; n++)
            {
                double window = .5 - .5 * Math.Cos(2 * Math.PI * n / fft);
                expected[f * hop + n] += window * (.2 + 2 * magnitude[0, bin, f]
                    * Math.Cos(2 * Math.PI * bin * n / fft + phase[0, bin, f])) / fft;
                envelope[f * hop + n] += window * window;
            }
        int start = same ? (fft - hop) / 2 : fft / 2;
        Assert.Equal(same ? length - 2 * start : (frames - 1) * hop, actual.Length);
        for (int i = 0; i < actual.Length; i++)
            Assert.True(Math.Abs(actual[i] - expected[start + i] / envelope[start + i]) < 1e-10,
                $"FFT {fft}, sample {i}: {actual[i]} differs from the analytic inverse.");
    }

    [Fact]
    public void MagnitudeAndPhaseBackwardMatchFiniteDifferences()
    {
        var engine = AiDotNetEngine.Current;
        var magnitude = new Tensor<double>(new[] { 1, 8, 3 });
        var phase = new Tensor<double>(magnitude.Shape.ToArray());
        for (int i = 0; i < magnitude.Length; i++)
        {
            magnitude[i] = .3 + .1 * Math.Sin(i);
            phase[i] = .2 * Math.Cos(i);
        }
        var transform = new InverseStft<double>(engine, 15, 3, 15, samePadding: true);
        double Loss()
        {
            using var noGrad = new NoGradScope<double>();
            var wave = transform.Forward(magnitude, phase);
            return wave.Sum(value => value * value);
        }
        using var tape = new GradientTape<double>();
        var output = transform.Forward(magnitude, phase);
        var loss = engine.ReduceSum(engine.TensorMultiply(output, output), new[] { 0 }, false);
        var gradients = tape.ComputeGradients(loss, sources: new[] { magnitude, phase });
        foreach (var source in new[] { magnitude, phase })
            for (int i = 0; i < source.Length; i++)
            {
                double value = source[i];
                const double epsilon = 1e-6;
                source[i] = value + epsilon;
                double plus = Loss();
                source[i] = value - epsilon;
                double minus = Loss();
                source[i] = value;
                double expected = (plus - minus) / (2 * epsilon);
                Assert.True(Math.Abs(gradients[source][i] - expected) < 1e-7,
                    $"Derivative {i}: {gradients[source][i]} versus {expected}.");
            }
    }
}
