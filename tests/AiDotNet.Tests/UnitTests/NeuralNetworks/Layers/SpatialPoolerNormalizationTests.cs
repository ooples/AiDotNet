using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Layers;

/// <summary>
/// The spatial pooler min-max normalizes its column activations before thresholding, unless they are identical. The
/// identical-activations test was an absolute 1e-10. That let a one-ulp difference through: 2^-29 at 1e7 in double,
/// and any nonzero range in Half, where 1e-10 rounds to 0. The normalization then stretched rounding noise to the full
/// [0, 1]. The test is now relative to the largest activation and to T's precision. The activations here are crafted
/// and exactly representable, so the cases do not depend on matmul rounding.
/// </summary>
public class SpatialPoolerNormalizationTests
{
    private static double[] Normalize<T>(params double[] activations)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var tensor = new Tensor<T>(activations.Select(a => ops.FromDouble(a)).ToArray(), new[] { activations.Length });
        return SpatialPoolerLayer<T>.NormalizeActivations(AiDotNetEngine.Current, tensor).ToArray()
            .Select(v => ops.ToDouble(v)).ToArray();
    }

    [Fact(Timeout = 60000)]
    public async Task Double_one_ulp_apart_at_ten_million_is_identical_and_one_to_two_is_spread()
    {
        await Task.Yield();
        const double big = 1e7;
        double nextUp = BitConverter.Int64BitsToDouble(BitConverter.DoubleToInt64Bits(big) + 1);
        Assert.True(nextUp - big > 1e-10, "the fixture needs a one-ulp range above the old absolute guard");

        Assert.Equal(new[] { big, nextUp }, Normalize<double>(big, nextUp));
        Assert.Equal(new[] { 0.0, 1.0 }, Normalize<double>(1.0, 2.0));
    }

    [Fact(Timeout = 60000)]
    public async Task Half_one_ulp_apart_near_one_is_identical_and_one_to_two_is_spread()
    {
        await Task.Yield();
        double nextUp = 1.0 + Math.Pow(2, -10);   // the Half after 1, exactly representable
        var half = MathHelper.GetNumericOperations<Half>();
        Assert.Equal(nextUp, half.ToDouble(half.FromDouble(nextUp)));

        Assert.Equal(new[] { 1.0, nextUp }, Normalize<Half>(1.0, nextUp));
        Assert.Equal(new[] { 0.0, 1.0 }, Normalize<Half>(1.0, 2.0));
    }

    [Fact(Timeout = 60000)]
    public async Task Equal_and_all_zero_activations_are_returned_unchanged()
    {
        await Task.Yield();
        Assert.Equal(new[] { 3.0, 3.0, 3.0 }, Normalize<double>(3.0, 3.0, 3.0));
        Assert.Equal(new[] { 0.0, 0.0 }, Normalize<double>(0.0, 0.0));
        Assert.Equal(new[] { 3.0, 3.0 }, Normalize<Half>(3.0, 3.0));
    }
}
