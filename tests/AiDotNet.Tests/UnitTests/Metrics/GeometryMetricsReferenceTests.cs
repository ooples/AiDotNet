using System.Threading.Tasks;
using AiDotNet.Metrics;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Metrics;

/// <summary>
/// Reference values for the point-cloud metrics: Sinkhorn EMD against the scaling-form iteration of POT's
/// ot.sinkhorn (uniform marginals, Euclidean cost), the far-apart case where exp(-C / epsilon) underflows, and
/// batched Chamfer distance against per-cloud Chamfer distance.
/// </summary>
public class GeometryMetricsReferenceTests
{
    private static Tensor<double> Points(int count, params double[] coordinates) =>
        new Tensor<double>(new[] { count, coordinates.Length / count }, new Vector<double>(coordinates));

    [Fact(Timeout = 60000)]
    public async Task Emd_MatchesSinkhornReference()
    {
        await Task.Yield();
        var emd = new EarthMoversDistance<double>(iterations: 100, epsilon: 0.1);
        var a = Points(3, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0);
        var b = Points(4, 0.2, 0.1, 1.1, 0.3, 0.4, 0.9, 0.5, 0.5);

        // numpy: u = a / (K v); v = b / (K^T u) for 100 iterations; sum(u K v * C).
        Assert.Equal(0.4161287563085932, emd.Compute(a, b), 1e-9);
    }

    [Fact(Timeout = 60000)]
    public async Task Emd_TranslatedCloud_IsTheTranslationLength_EvenWhenTheKernelUnderflows()
    {
        await Task.Yield();
        // exp(-10 / 0.01) underflows to zero in double; W1 between a cloud and its translate is |t| = 10.
        var emd = new EarthMoversDistance<double>(iterations: 200, epsilon: 0.01);
        var a = Points(3, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0);
        var b = Points(3, 10.0, 0.0, 11.0, 0.0, 10.0, 1.0);

        Assert.Equal(10.0, emd.Compute(a, b), 2);
    }

    [Fact(Timeout = 60000)]
    public async Task ChamferBatch_EqualsPerCloudChamfer()
    {
        await Task.Yield();
        foreach (bool squared in new[] { true, false })
        {
            var chamfer = new ChamferDistance<double>(squared);
            var batchA = new Tensor<double>(new[] { 2, 2, 2 },
                new Vector<double>(new[] { 0.0, 0.0, 2.0, 0.0, 1.0, 1.0, 3.0, 4.0 }));
            var batchB = new Tensor<double>(new[] { 2, 3, 2 },
                new Vector<double>(new[] { 1.0, 0.0, 10.0, 0.0, 0.0, 3.0, 0.0, 0.0, 1.0, 2.0, 5.0, 5.0 }));

            var result = chamfer.ComputeBatch(batchA, batchB);

            Assert.Equal(chamfer.Compute(Points(2, 0.0, 0.0, 2.0, 0.0), Points(3, 1.0, 0.0, 10.0, 0.0, 0.0, 3.0)), result[0], 1e-12);
            Assert.Equal(chamfer.Compute(Points(2, 1.0, 1.0, 3.0, 4.0), Points(3, 0.0, 0.0, 1.0, 2.0, 5.0, 5.0)), result[1], 1e-12);
        }
    }

    [Fact(Timeout = 60000)]
    public async Task ChamferEuclidean_IsMeanNearestDistanceInEachDirection()
    {
        await Task.Yield();
        var chamfer = new ChamferDistance<double>(squared: false);
        var a = Points(2, 0.0, 0.0, 2.0, 0.0);
        var b = Points(3, 1.0, 0.0, 10.0, 0.0, 0.0, 3.0);

        // A -> B: (1 + 1) / 2; B -> A: (1 + 8 + 3) / 3.
        Assert.Equal(1.0 + 4.0, chamfer.Compute(a, b), 1e-12);
    }
}
