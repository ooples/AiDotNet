using System;
using System.Reflection;
using AiDotNet.Safety.Multimodal;
using AiDotNet.Safety.Watermarking;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Safety;

public class SafetyConstructorCompatibilityTests
{
    [Theory]
    [InlineData(typeof(TextImageAlignmentChecker<double>), 1)]
    [InlineData(typeof(AudioWatermarker<double>), 2)]
    [InlineData(typeof(LexicalWatermarker<double>), 1)]
    [InlineData(typeof(SpectralAudioWatermarker<double>), 1)]
    [InlineData(typeof(SyntacticWatermarker<double>), 1)]
    [InlineData(typeof(WatermarkDetector<double>), 0)]
    public void OriginalPublicConstructorRemainsCallable(Type type, int count)
    {
        var types = new Type[count];
        var values = new object[count];
        for (int i = 0; i < count; i++) { types[i] = typeof(double); values[i] = 0.5; }
        var constructor = type.GetConstructor(types);
        Assert.NotNull(constructor);
        Assert.IsType(type, constructor.Invoke(values));
    }

    [Theory]
    [InlineData(typeof(AudioWatermarkerBase<double>))]
    [InlineData(typeof(TextWatermarkerBase<double>))]
    public void OriginalProtectedConstructorRemainsAvailable(Type type)
    {
        Assert.NotNull(type.GetConstructor(BindingFlags.NonPublic | BindingFlags.Instance,
            null, new[] { typeof(double) }, null));
    }

    [Fact]
    public void OriginalMixedNumericConstructorsRemainCallable()
    {
        var audio = typeof(AudioSealWatermarker<double>).GetConstructor(new[] { typeof(double), typeof(int) });
        var text = typeof(SamplingWatermarker<double>).GetConstructor(new[] { typeof(double), typeof(int), typeof(double) });
        Assert.NotNull(audio);
        Assert.NotNull(text);
        Assert.IsType<AudioSealWatermarker<double>>(audio.Invoke(new object[] { 0.5, 4096 }));
        Assert.IsType<SamplingWatermarker<double>>(text.Invoke(new object[] { 0.5, 500, 0.25 }));
    }
}
