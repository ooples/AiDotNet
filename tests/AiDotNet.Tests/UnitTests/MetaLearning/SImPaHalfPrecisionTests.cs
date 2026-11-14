using System;
using AiDotNet.MetaLearning.Algorithms;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Tensors.Helpers;
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

    [Fact]
    public void MetaLevelKL_DoesNotOverflowInHalf()
    {
        // psi = (200, 200, 200, 200), sigma_q = sigma_p = 1: KL = 0.5 ||psi||^2 = 80000. The Half dot product summed
        // 4 * 40000 past 65504 to inf, so the meta-level KL, and with it the bound, was inf.
        var algorithm = new SImPaAlgorithm<Half, Tensor<Half>, Tensor<Half>>(
            new SImPaOptions<Half, Tensor<Half>, Tensor<Half>>(new MockFullModel<Half>(numParameters: 4)));
        var psi = new Vector<Half>(new[] { (Half)200f, (Half)200f, (Half)200f, (Half)200f });

        Assert.Equal(80000.0, algorithm.GaussianKLToZeroMeanPrior(psi, 1.0, 1.0), 6);
    }

    [Fact]
    public void MetaParameterDraw_RoundsPsiPlusScaledNoiseOnce()
    {
        // theta = psi + sigma_0 g is rounded to Half once. Formed in Half it was rounded three times (g, sigma_0 g, the sum).
        const double sigma = 0.37;
        var options = new SImPaOptions<Half, Tensor<Half>, Tensor<Half>>(new MockFullModel<Half>(numParameters: 4))
        {
            Seed = 7,
            MetaPosteriorStdDev = sigma,
        };
        var algorithm = new SImPaAlgorithm<Half, Tensor<Half>, Tensor<Half>>(options);
        var psi = algorithm.Posterior.GetParameters();

        // The constructor takes two seeds from the learner's generator; the draw then reads two uniforms per entry.
        var rng = RandomHelper.CreateSeededRandom(7);
        rng.Next();
        rng.Next();
        var uniforms = new double[2 * psi.Length];
        for (int i = 0; i < uniforms.Length; i++) uniforms[i] = rng.NextDouble();
        var normals = algorithm.StandardNormals(uniforms).ToArray();

        var theta = algorithm.SampleMetaParameter();

        Assert.Equal(psi.Length, theta.Length);
        for (int i = 0; i < psi.Length; i++)
            Assert.Equal((Half)((double)psi[i] + normals[i] * sigma), theta[i]);
    }

    [Fact]
    public void PriorDraw_RoundsTheScaledNormalOnce()
    {
        const double sigma = 0.3;
        var options = new SImPaOptions<Half, Tensor<Half>, Tensor<Half>>(new MockFullModel<Half>(numParameters: 64))
        {
            Seed = 11,
            PriorStdDev = sigma,
        };
        var algorithm = new SImPaAlgorithm<Half, Tensor<Half>, Tensor<Half>>(options);

        var rng = RandomHelper.CreateSeededRandom(11);
        rng.Next();
        rng.Next();
        var uniforms = new double[2 * 2 * 64];
        for (int i = 0; i < uniforms.Length; i++) uniforms[i] = rng.NextDouble();
        var normals = algorithm.StandardNormals(uniforms).ToArray();

        var samples = algorithm.SamplePrior(2);

        for (int s = 0; s < 2; s++)
            for (int d = 0; d < 64; d++)
                Assert.Equal((Half)(normals[s * 64 + d] * sigma), samples[s][d]);
    }
}
