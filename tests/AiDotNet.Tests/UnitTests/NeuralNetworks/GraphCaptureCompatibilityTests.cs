using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Engines.Optimization;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

public sealed class GraphCaptureCompatibilityTests
{
    [Fact]
    public void CreateOptionsWithoutCompilation_PreservesEveryOtherPublicOption()
    {
        var source = new TensorCodecOptions();
        var properties = typeof(TensorCodecOptions).GetProperties()
            .Where(property => property.CanRead && property.CanWrite)
            .ToArray();

        foreach (var property in properties)
        {
            if (property.Name == nameof(TensorCodecOptions.EnableCompilation))
                continue;

            property.SetValue(source, DifferentValue(property.PropertyType, property.GetValue(source)));
        }

        var copy = GraphCaptureCompatibility.CreateOptionsWithoutCompilation(source);

        Assert.NotSame(source, copy);
        Assert.False(copy.EnableCompilation);
        foreach (var property in properties)
        {
            if (property.Name == nameof(TensorCodecOptions.EnableCompilation))
                continue;

            Assert.Equal(property.GetValue(source), property.GetValue(copy));
        }
    }

    [Fact]
    public void Enter_RestoresNestedCompatibilityState()
    {
        Assert.False(GraphCaptureCompatibility.IsActive);

        using (GraphCaptureCompatibility.Enter())
        {
            Assert.True(GraphCaptureCompatibility.IsActive);
            using (GraphCaptureCompatibility.Enter())
            {
                Assert.True(GraphCaptureCompatibility.IsActive);
            }

            Assert.True(GraphCaptureCompatibility.IsActive);
        }

        Assert.False(GraphCaptureCompatibility.IsActive);
    }

    private static object DifferentValue(Type type, object? current)
    {
        if (type == typeof(bool))
            return !(bool)current!;
        if (type == typeof(int))
            return (int)current! + 7;
        if (type == typeof(float))
            return (float)current! + 0.25f;
        if (type.IsEnum)
        {
            var values = Enum.GetValues(type);
            return values.Cast<object>().First(value => !Equals(value, current));
        }

        throw new NotSupportedException(
            $"TensorCodecOptions added unsupported writable property type {type.FullName}.");
    }
}
