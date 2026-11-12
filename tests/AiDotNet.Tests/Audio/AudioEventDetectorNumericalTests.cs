#if NET5_0_OR_GREATER
using System.Reflection;
using System.Runtime.Serialization;
using AiDotNet.Audio.Classification;
using Xunit;

namespace AiDotNet.Tests.Audio;

public class AudioEventDetectorNumericalTests
{
    private static double Statistic<T>(string name, Tensor<T> values, params object[] extra)
    {
        var model = (AudioEventDetector<T>)FormatterServices.GetUninitializedObject(typeof(AudioEventDetector<T>));
        var arguments = new object[extra.Length + 1];
        arguments[0] = values;
        extra.CopyTo(arguments, 1);
        return Assert.IsType<double>((typeof(AudioEventDetector<T>).GetMethod(name, BindingFlags.Instance | BindingFlags.NonPublic)
            ?? throw new InvalidOperationException(name)).Invoke(model, arguments));
    }

    [Fact]
    public void HalfEnergyAndSpectralReductions_RetainFiniteReferenceValues()
    {
        var spectrum = new Tensor<Half>(Enumerable.Repeat((Half)60000, 12).ToArray(), new[] { 3, 4 });
        Assert.Equal(3600000000.0, Statistic("ComputeEnergy", spectrum));
        Assert.Equal(1.5, Statistic("ComputeSpectralCentroid", spectrum), 12);
        Assert.Equal(1.0, Statistic("ComputeSpectralFlatness", spectrum), 12);
        Assert.Equal(60000.0, Statistic("ComputeBandEnergy", spectrum, 1, 3));
    }

    [Fact]
    public void HalfSilentSpectrum_FlatnessKeepsPositiveLogFloor()
    {
        Assert.Equal(1.0, Statistic("ComputeSpectralFlatness", new Tensor<Half>(new[] { 2, 4 })), 12);
    }

    [Fact]
    public void EmptyStatistics_PreserveExistingBoundarySemantics()
    {
        var empty = new Tensor<double>(new[] { 0, 4 });
        Assert.True(double.IsNaN(Statistic("ComputeEnergy", empty)));
        Assert.Equal(0.0, Statistic("ComputeSpectralCentroid", empty));
        Assert.Equal(0.0, Statistic("ComputeSpectralFlatness", empty));
        Assert.Equal(0.0, Statistic("ComputeBandEnergy", empty, 0, 4));
    }
}
#endif
