#if NET5_0_OR_GREATER
using System.Reflection;
using System.Runtime.Serialization;
using AiDotNet.Audio.Enhancement;
using AiDotNet.NeuralNetworks;
using Xunit;

namespace AiDotNet.Tests.Audio;

public class DccrnTransformMemoryTests
{
    private static DCCRN<T> Model<T>(int size)
    {
        var model = (DCCRN<T>)FormatterServices.GetUninitializedObject(typeof(DCCRN<T>));
        (typeof(NeuralNetworkBase<T>).GetField("NumOps", BindingFlags.Instance | BindingFlags.NonPublic)
            ?? throw new InvalidOperationException("NumOps")).SetValue(model, MathHelper.GetNumericOperations<T>());
        foreach (var name in new[] { "_fftSize", "_hopSize" })
            (typeof(DCCRN<T>).GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)
                ?? throw new InvalidOperationException(name)).SetValue(model, size);
        return model;
    }

    private static Tensor<T> Transform<T>(DCCRN<T> model, string name, Tensor<T> input)
        => Assert.IsType<Tensor<T>>((typeof(DCCRN<T>).GetMethod(name, BindingFlags.Instance | BindingFlags.NonPublic)
            ?? throw new InvalidOperationException(name)).Invoke(model, new object[] { input }));

    [Theory]
    [InlineData(32)]
    [InlineData(16384)]
    public void StftAndInverse_LargeFftMatchesAnalyticImpulseAndSingleFrequency(int size)
    {
        var model = Model<double>(size);
        var audio = new Tensor<double>(new[] { size });
        int impulse = size / 4;
        audio.SetFlat(impulse, 1);
        var spectrum = Transform(model, "ComputeComplexSTFT", audio);
        int bins = size / 2 + 1;
        double window = 0.5 - 0.5 * Math.Cos(2 * Math.PI * impulse / (size - 1));
        Assert.Equal(new[] { 1, 2, bins, 1 }, spectrum.Shape);
        for (int k = 0; k < bins; k++)
        {
            double phase = 2 * Math.PI * ((long)k * impulse % size) / size;
            Assert.Equal(window * Math.Cos(phase), spectrum.GetFlat(k), 10);
            Assert.Equal(-window * Math.Sin(phase), spectrum.GetFlat(bins + k), 10);
        }

        var singleFrequency = new Tensor<double>(new[] { 1, 2, bins, 1 });
        singleFrequency[0, 0, 3, 0] = 1;
        singleFrequency[0, 1, 3, 0] = 0.25;
        var reconstructed = Transform(model, "ComputeInverseSTFT", singleFrequency);
        Assert.Equal(size * 2, reconstructed.Length);
        for (int n = 0; n < size; n++)
        {
            double phase = 2 * Math.PI * (3L * n % size) / size;
            double weight = 0.5 - 0.5 * Math.Cos(2 * Math.PI * n / (size - 1));
            double expected = (Math.Cos(phase) - 0.25 * Math.Sin(phase)) * weight / size;
            Assert.Equal(expected, reconstructed.GetFlat(n), 12);
        }
        Assert.All(reconstructed.ToArray().Skip(size), value => Assert.Equal(0, value));
    }

    [Fact]
    public void HalfStft_PhaseProductsDoNotOverflowAtOrdinaryFftSize()
    {
        const int size = 512, bins = size / 2 + 1;
        var model = Model<Half>(size);
        var audio = new Tensor<Half>(new[] { size });
        audio.SetFlat(size / 4, (Half)1);
        var spectrum = Transform(model, "ComputeComplexSTFT", audio);
        double weight = (double)(Half)(0.5 - 0.5 * Math.Cos(2 * Math.PI * (size / 4) / (size - 1)));
        for (int k = 0; k < bins; k++)
        {
            double phase = 2 * Math.PI * ((long)k * (size / 4) % size) / size;
            Assert.InRange(Math.Abs((double)spectrum.GetFlat(k) - weight * Math.Cos(phase)), 0, 0.001);
            Assert.InRange(Math.Abs((double)spectrum.GetFlat(bins + k) + weight * Math.Sin(phase)), 0, 0.001);
        }
    }
}
#endif
