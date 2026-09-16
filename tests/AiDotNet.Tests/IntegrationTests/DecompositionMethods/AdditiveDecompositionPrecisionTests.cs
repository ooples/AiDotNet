using AiDotNet.DecompositionMethods.TimeSeriesDecomposition;
using AiDotNet.Enums;
using AiDotNet.Enums.AlgorithmTypes;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.IntegrationTests.DecompositionMethods;

/// <summary>
/// Regression tests for the LOESS bandwidth in <see cref="AdditiveDecomposition{T}"/>'s STL path.
/// The cycle-subseries LOESS bandwidth was <c>windowSize / 2</c> in integer arithmetic, so a
/// subseries of 2 points (windowSize = 1) got a bandwidth of 0 and every weight became 0/0 = NaN.
/// </summary>
public class AdditiveDecompositionPrecisionTests
{
    [Fact(Timeout = 120000)]
    public async Task STL_TwoFullSeasons_ProducesFiniteComponents()
    {
        // STL uses a fixed seasonal period of 12, so 24 points give exactly 2 points per cycle-subseries.
        int n = 24;
        var data = new double[n];
        for (int i = 0; i < n; i++)
        {
            data[i] = 10.0 + 0.5 * i + 5.0 * Math.Sin(2.0 * Math.PI * i / 12.0);
        }

        var decomposition = new AdditiveDecomposition<double>(new Vector<double>(data), AdditiveDecompositionAlgorithmType.STL);

        foreach (var component in new[] { DecompositionComponentType.Trend, DecompositionComponentType.Seasonal, DecompositionComponentType.Residual })
        {
            var values = decomposition.GetComponentAsVector(component);
            Assert.Equal(n, values.Length);
            for (int i = 0; i < values.Length; i++)
            {
                Assert.False(double.IsNaN(values[i]) || double.IsInfinity(values[i]),
                    $"{component}[{i}] = {values[i]} is not finite.");
            }
        }
    }
}
