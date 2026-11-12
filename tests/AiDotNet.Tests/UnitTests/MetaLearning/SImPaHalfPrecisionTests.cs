using System;
using AiDotNet.MetaLearning.Algorithms;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.Helpers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.MetaLearning;

/// <summary>
/// <see cref="SImPaAlgorithm{T, TInput, TOutput}"/>'s Box-Muller sampler in Half: the uniforms and the 1e-12 floor
/// must stay in double, because 1e-12 rounds to 0 in Half and ln 0 = -inf made the radius infinite.
/// </summary>
public class SImPaHalfPrecisionTests
{
    [Fact]
    public void BoxMuller_FloorsAZeroUniformInHalf()
    {
        var algorithm = new SImPaAlgorithm<Half, Tensor<Half>, Tensor<Half>>(
            new SImPaOptions<Half, Tensor<Half>, Tensor<Half>>(new MockFullModel<Half>(numParameters: 4)));

        // Pairs (u1, u2): (0, 0) -> sqrt(-2 ln 1e-12) cos 0 = 7.434; (1e-9, 0) -> sqrt(-2 ln 1e-9) = 6.438
        // (1e-9 is 0 in Half too); (0.25, 0.5) -> sqrt(-2 ln 0.25) cos(pi) = -1.665.
        var normals = algorithm.BoxMuller(new Vector<double>(new[] { 0.0, 0.0, 1e-9, 0.0, 0.25, 0.5 }));

        Assert.Equal(Math.Sqrt(-2.0 * Math.Log(1e-12)), (double)(float)normals[0], 1);
        Assert.Equal(Math.Sqrt(-2.0 * Math.Log(1e-9)), (double)(float)normals[1], 1);
        Assert.Equal(-Math.Sqrt(-2.0 * Math.Log(0.25)), (double)(float)normals[2], 2);
    }
}
