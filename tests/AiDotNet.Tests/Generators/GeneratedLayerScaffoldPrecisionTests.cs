using AiDotNet.Tests.ModelFamilyTests.Base;
using Xunit;

namespace AiDotNet.Tests.Generators;

public sealed class GeneratedLayerScaffoldPrecisionTests
{
    [Fact]
    public void SVTRNumericallyConditionedScaffold_UsesDoublePrecision()
    {
        var scaffold = typeof(
            global::AiDotNet.Tests.ModelFamilyTests.Generated.SVTRThinPlateSplineLayerTests);

        Type? current = scaffold.BaseType;
        while (current is not null &&
               (!current.IsGenericType ||
                current.GetGenericTypeDefinition() != typeof(LayerTestBase<>)))
        {
            current = current.BaseType;
        }

        Assert.NotNull(current);
        Assert.Equal(typeof(double), current.GetGenericArguments()[0]);
    }
}
